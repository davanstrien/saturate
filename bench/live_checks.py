"""Live checks against a real engine: does a run write every row, once?

    ENGINE_START="vllm serve M --port 8000" python bench/live_checks.py \\
        --endpoint http://127.0.0.1:8000/v1 --model M

Runs four checks and prints one JSON line per check.
The engine runs in its own process group, whose id is in ENGINE_PGID_FILE; the restart check
stops that group and starts ENGINE_START in a new one (an engine's worker processes go with it).

- complete: a normal run writes every row once, with no error rows.
- high-window: 2048 requests in flight with short outputs; counts error rows.
- resume: a run interrupted (SIGINT) part-way, then run again on the same output, ends with
  every row written once.
- engine-restart: the engine is stopped mid-run and started again; the breaker opens, closes,
  and the run still writes every row once.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request


def run(args) -> None:
    """One pump() run, in this process; prints its Stats as JSON (used by the checks below)."""
    from saturate import Auto, Fixed, pump

    rows = [{"id": str(i), "text": f"Write one sentence about the number {i}."} for i in range(args.rows)]
    window = Auto() if args.window == "auto" else Fixed(int(args.window))
    stats = pump(rows, lambda r: {"model": args.model, "max_tokens": args.max_tokens,
                                  "messages": [{"role": "user", "content": r["text"]}]},
                 lambda r, body: {"reply": body["choices"][0]["message"]["content"]},
                 endpoint=args.endpoint, output=args.output, window=window, id_key="id")
    print(json.dumps({"rows_processed": stats.rows_processed, "rows_failed": stats.rows_failed,
                      "breaker_opens": stats.breaker_opens}), flush=True)


def written(output: str) -> dict:
    """What the output holds: rows, distinct ids, error rows, empty replies."""
    import pyarrow.parquet as pq

    ids, errors, empty = [], 0, 0
    for path in sorted(glob.glob(f"{output}/part-*.parquet")):
        table = pq.read_table(path).to_pylist()
        ids += [r["id"] for r in table]
        errors += sum(r["error"] is not None for r in table)
        empty += sum(r["error"] is None and not r.get("reply") for r in table)
    return {"rows": len(ids), "distinct_ids": len(set(ids)), "error_rows": errors, "empty_replies": empty}


def start_run(args, output: str, rows: int, window: str, max_tokens: int) -> subprocess.Popen:
    cmd = [sys.executable, __file__, "run", "--endpoint", args.endpoint, "--model", args.model,
           "--output", output, "--rows", str(rows), "--window", window, "--max-tokens", str(max_tokens)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)


def finish(proc: subprocess.Popen) -> dict:
    out, _ = proc.communicate()
    lines = [line for line in out.splitlines() if line.startswith("{")]
    return {"exit": proc.returncode, **(json.loads(lines[-1]) if lines else {})}


def wait_healthy(endpoint: str, deadline_s: float = 900) -> None:
    base = endpoint.removesuffix("/v1")
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"{base}/health", timeout=5).status == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("engine did not come back")


def report(check: str, rows: int, t0: float, result: dict, output: str) -> None:
    w = written(output)
    ok = w["rows"] == rows and w["distinct_ids"] == rows and w["error_rows"] == 0 and result.get("exit") == 0
    record = {"check": check, "pass": ok, "wall_s": round(time.time() - t0, 1), **w, **result}
    print(json.dumps(record), flush=True)


def checks(args) -> None:
    def fresh() -> str:
        return tempfile.mkdtemp(prefix="live-")

    out, t0 = fresh(), time.time()
    report("complete", 2000, t0, finish(start_run(args, out, 2000, "auto", 64)), out)

    out, t0 = fresh(), time.time()
    report("high-window", 8000, t0, finish(start_run(args, out, 8000, "2048", 16)), out)

    out, t0 = fresh(), time.time()
    first = start_run(args, out, 4000, "auto", 64)
    time.sleep(args.interrupt_after_s)
    first.send_signal(signal.SIGINT)
    interrupted = finish(first)
    partial = written(out)["rows"]
    result = finish(start_run(args, out, 4000, "auto", 64))
    extra = {"first_exit": interrupted["exit"], "rows_before_resume": partial}
    report("resume", 4000, t0, {**result, **extra}, out)

    out, t0 = fresh(), time.time()
    proc = start_run(args, out, 4000, "auto", 64)
    time.sleep(args.interrupt_after_s)
    stop_engine()
    time.sleep(10)
    start_engine()
    wait_healthy(args.endpoint)
    report("engine-restart", 4000, t0, finish(proc), out)


PGID_FILE = os.environ.get("ENGINE_PGID_FILE", "/tmp/engine.pgid")


def stop_engine() -> None:
    pgid = int(open(PGID_FILE).read())
    os.killpg(pgid, signal.SIGTERM)
    for _ in range(30):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    os.killpg(pgid, signal.SIGKILL)
    time.sleep(5)  # GPU memory is released once the processes are gone


def start_engine() -> None:
    log = open("/tmp/engine-restart.log", "w")
    engine = subprocess.Popen(os.environ["ENGINE_START"], shell=True, stdout=log, stderr=log,
                              start_new_session=True)
    with open(PGID_FILE, "w") as f:
        f.write(str(engine.pid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="checks", choices=["checks", "run"])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--interrupt-after-s", type=float, default=8.0)
    ap.add_argument("--output")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--window", default="auto")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    run(args) if args.mode == "run" else checks(args)


if __name__ == "__main__":
    main()
