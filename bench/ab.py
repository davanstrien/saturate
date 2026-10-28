"""A/B benchmark: two saturate versions, same machine, same stub servers, runs interleaved.

    UV_NO_CONFIG=1 python bench/ab.py --base origin/main --head .            # working tree vs main
    UV_NO_CONFIG=1 python bench/ab.py --base origin/main --head origin/main  # A/A: the noise floor

Each ref is installed (not editable) into its own virtualenv; `.` means the working tree as it
is, uncommitted changes included. The stub servers run in their own processes. Every rep runs
each scenario once per arm, in a random order, so drift on the machine (thermal throttling,
background load) lands on both arms. The report gives each arm's median and the head/base ratio
of medians with a 95% bootstrap interval; an interval that contains 1.0 is no measurable
change. Run an A/A first: its interval width is the smallest change this machine can detect.

The workload (bench/workload.py) uses public API only, so it runs unchanged against any version.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import random
import shutil
import statistics
import subprocess
import sys
import tempfile

BENCH = pathlib.Path(__file__).resolve().parent
REPO = BENCH.parent

# name -> (stub latency ms, workload arguments)
SUITES = {
    "quick": {
        # flush-every 5000: the pipeline's own overhead, not parquet writes
        "overhead-w64": (0, ["--rows", "10000", "--window", "64", "--flush-every", "5000"]),
        "overhead-w512": (0, ["--rows", "10000", "--window", "512", "--flush-every", "5000"]),
        # the default local flush (1000 rows or 10 s): what an out-of-the-box run pays
        "default-flush-w64": (0, ["--rows", "3000", "--window", "64"]),
        "latency50-w256": (50, ["--rows", "4000", "--window", "256"]),
        "prep-cpu-4workers": (10, ["--rows", "1500", "--window", "64", "--prep-ms", "4",
                                   "--prepare-workers", "4"]),
        "slow-sink": (0, ["--rows", "6000", "--window", "256", "--output", "slowmem",
                          "--write-delay-ms", "200", "--flush-every", "500"]),
        "auto-ramp-200ms": (200, ["--rows", "2000", "--window", "auto"]),
    },
    # against a real engine (--endpoint); the prompt varies per row so the prefix cache cannot
    # answer for the engine
    # the HTTP client under load (#55): short outputs at windows past the engine's batch size,
    # and a long-output check that the client stays out of the way when generation dominates
    "transport": {
        "short-auto": (None, ["--rows", "3000", "--window", "auto", "--max-tokens", "64"]),
        "short-fixed512": (None, ["--rows", "6000", "--window", "512", "--max-tokens", "16"]),
        "short-fixed2048": (None, ["--rows", "12000", "--window", "2048", "--max-tokens", "16"]),
        "long-fixed256": (None, ["--rows", "1500", "--window", "256", "--max-tokens", "512"]),
    },
    "vllm": {
        "short-auto": (None, ["--rows", "3000", "--window", "auto", "--max-tokens", "64"]),
        "short-fixed256": (None, ["--rows", "3000", "--window", "256", "--max-tokens", "64"]),
        "long-auto-initial4": (None, ["--rows", "600", "--window", "auto", "--initial", "4",
                                      "--max-tokens", "512"]),
    },
    "smoke": {  # seconds, for checking the harness itself
        "overhead-w64": (0, ["--rows", "1000", "--window", "64"]),
        "slow-sink": (0, ["--rows", "600", "--window", "64", "--output", "slowmem",
                          "--write-delay-ms", "50", "--flush-every", "200"]),
    },
}

# metric -> True when higher is better (only used to label the report)
METRICS = {"rows_per_s": True, "tokens_per_s": True, "cpu_ms_per_row": False, "loop_lag_max_s": False,
           "loop_lag_mean_s": False, "first_durable_s": False, "peak_rss_mib": False}


def sh(*cmd: str, **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def build_arm(name: str, ref: str, workdir: pathlib.Path, python: str) -> dict:
    """Install `ref` into its own virtualenv; returns {"name", "ref", "sha", "python"}."""
    if ref == ".":
        src = REPO
        sha = sh("git", "-C", str(REPO), "rev-parse", "--short", "HEAD") + "+worktree"
    else:
        src = workdir / f"src-{name}"
        sh("git", "-C", str(REPO), "worktree", "add", "--detach", str(src), ref)
        sha = sh("git", "-C", str(src), "rev-parse", "--short", "HEAD")
    venv = workdir / f"venv-{name}"
    sh("uv", "venv", "-q", "--python", python, str(venv))
    py = str(venv / "bin" / "python")
    # [hf] so an hf:// output works; compiled bytecode so no run pays the .pyc compile
    sh("uv", "pip", "install", "-q", "--compile-bytecode", "--python", py, f"{src}[hf]")
    return {"name": name, "ref": ref, "sha": sha, "python": py, "src": str(src)}


def start_server(python: str, latency_ms: float) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen([python, str(BENCH / "serve.py"), str(latency_ms)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    endpoint = proc.stdout.readline().strip()
    if not endpoint.startswith("http"):
        proc.kill()
        raise RuntimeError(f"stub server did not start (got {endpoint!r})")
    return proc, endpoint


def run_workload(arm: dict, endpoint: str, args: list[str], cwd: str) -> dict:
    # TMPDIR=cwd: a local output lands in the A/B's own work dir, removed with it
    out = subprocess.run([arm["python"], str(BENCH / "workload.py"), "--endpoint", endpoint, *args],
                         capture_output=True, text=True, cwd=cwd, timeout=900,
                         env={**os.environ, "TMPDIR": cwd})
    if out.returncode != 0:
        raise RuntimeError(f"{arm['name']} workload failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def paired_ratios(base: list[dict], head: list[dict], metric: str) -> list[float]:
    """head/base for each rep: the two arms of a rep ran back to back, so machine drift between
    reps cancels within each ratio."""
    return [h[metric] / b[metric] for b, h in zip(base, head, strict=True)
            if b.get(metric) and h.get(metric) is not None]


def bootstrap_ratio(base: list[float], head: list[float], n: int = 2000, seed: int = 0):
    """95% interval of median(head) / median(base), resampling each arm's runs. Secondary: with a
    handful of reps it takes few distinct values and under-covers."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(n):
        b = statistics.median(rng.choices(base, k=len(base)))
        h = statistics.median(rng.choices(head, k=len(head)))
        if b:
            ratios.append(h / b)
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]


def summarise(results: dict) -> tuple[list[str], dict]:
    """Per scenario and metric: each arm's median, the median of the per-rep paired ratios with
    their range (the headline), and the bootstrap interval of the ratio of medians (secondary)."""
    lines = ["| scenario | metric | base median | head median | paired head/base (min–max) "
             "| bootstrap 95% |", "|---|---|---|---|---|---|"]
    summary: dict = {}
    for scenario, arms in results.items():
        for metric in METRICS:
            base = [r[metric] for r in arms["base"] if r.get(metric) is not None]
            head = [r[metric] for r in arms["head"] if r.get(metric) is not None]
            pairs = paired_ratios(arms["base"], arms["head"], metric)
            if not base or not head or not pairs:
                continue
            mb, mh = statistics.median(base), statistics.median(head)
            lo, hi = bootstrap_ratio(base, head)
            summary.setdefault(scenario, {})[metric] = {
                "base_median": mb, "head_median": mh, "paired_median": statistics.median(pairs),
                "paired_min": min(pairs), "paired_max": max(pairs), "bootstrap_95": [lo, hi]}
            lines.append(f"| {scenario} | {metric} | {mb:.3g} | {mh:.3g} | "
                         f"{statistics.median(pairs):.3f} ({min(pairs):.3f}–{max(pairs):.3f}) | "
                         f"{lo:.3f}–{hi:.3f} |")
    return lines, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="git ref, or '.' for the working tree")
    ap.add_argument("--head", required=True, help="git ref, or '.' for the working tree")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--suite", choices=sorted(SUITES), default="quick")
    ap.add_argument("--python", default="3.13")
    ap.add_argument("--out", default=None, help="JSON results path (default: bench/results/...)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded runs per scenario and arm before the measured reps (first-request "
                         "costs: imports, an engine's CUDA-graph capture)")
    ap.add_argument("--endpoint", default=None,
                    help="a running engine to measure against instead of the stub servers")
    ap.add_argument("--model", default=None, help="model name for --endpoint")
    ap.add_argument("--extra", default="", help="extra workload arguments for every scenario, e.g. "
                    "'--output-uri hf://buckets/me/scratch'")
    args = ap.parse_args()
    if os.environ.get("UV_NO_CONFIG") != "1":
        print("warning: UV_NO_CONFIG is not 1: a user-level uv config may change what gets installed",
              file=sys.stderr)

    suite = SUITES[args.suite]
    if args.endpoint is None and any(latency is None for latency, _ in suite.values()):
        ap.error(f"suite {args.suite!r} runs against a real engine: pass --endpoint (and --model)")
    rng = random.Random(args.seed)
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="saturate-ab-"))
    arms, servers = [], {}
    try:
        arms = [build_arm("base", args.base, workdir, args.python),
                build_arm("head", args.head, workdir, args.python)]
        print(f"base {arms[0]['sha']}  head {arms[1]['sha']}  suite {args.suite}  reps {args.reps}",
              file=sys.stderr)
        if args.endpoint is None:
            for latency in sorted({lat for lat, _ in suite.values()}):
                servers[latency] = start_server(arms[1]["python"], latency)
        extra = args.extra.split()
        if args.model:
            extra += ["--model", args.model, "--prompt", "Write one sentence about the number {i}."]
        results: dict = {name: {"base": [], "head": []} for name in suite}
        for _ in range(args.warmup):
            for latency, wl_args in suite.values():
                for arm in arms:
                    endpoint = args.endpoint or servers[latency][1]
                    run_workload(arm, endpoint, wl_args + extra, cwd=str(workdir))
            print(f"warmup done ({args.warmup} per scenario and arm)", file=sys.stderr)
        for rep in range(args.reps):
            for name, (latency, wl_args) in suite.items():
                order = arms[:]
                rng.shuffle(order)
                for arm in order:
                    endpoint = args.endpoint or servers[latency][1]
                    r = run_workload(arm, endpoint, wl_args + extra, cwd=str(workdir))
                    results[name][arm["name"]].append(r)
                    print(f"rep {rep + 1}/{args.reps} {name:<20} {arm['name']:<4} "
                          f"{r['rows_per_s']:>9} rows/s  {r['cpu_ms_per_row']:.3f} cpu ms/row",
                          file=sys.stderr)
        lines, summary = summarise(results)
        print("\n".join(lines))
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = pathlib.Path(args.out or BENCH / "results" / f"{stamp}-{arms[0]['sha']}-{arms[1]['sha']}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "base": {k: arms[0][k] for k in ("ref", "sha")}, "head": {k: arms[1][k] for k in ("ref", "sha")},
            "suite": args.suite, "reps": args.reps, "warmup": args.warmup, "python": args.python,
            "platform": sys.platform, "machine": os.uname().machine, "summary": summary,
            "runs": results}, indent=1))
        print(f"\nwrote {out}", file=sys.stderr)
    finally:
        for proc, _ in servers.values():
            proc.stdin.close()
            proc.wait(5)
        for arm in arms:
            if arm["ref"] != ".":
                subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", arm["src"]],
                               capture_output=True)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
