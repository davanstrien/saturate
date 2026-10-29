"""Live checks against a real engine: does a run write every row, once?

    ENGINE_START="vllm serve M --port 8000" python bench/live_checks.py \\
        --endpoint http://127.0.0.1:8000/v1 --model M

    ENGINE_START="..." python bench/live_checks.py --endpoint ... --model M --engine sglang

Runs five checks and prints one JSON line per check. The complete check measures the engine's
rows/s; the later runs are sized from it, so an interrupt lands mid-run on a fast engine too.
The engine runs in its own process group, whose id is in ENGINE_PGID_FILE; the restart check
stops that group and starts ENGINE_START in a new one (an engine's worker processes go with it).

- signals (with --engine): the probe names the engine and reads its capacity where it reports
  one, and the run's telemetry carries the engine's gauges (EXPECT says which).
- complete: a normal run writes every row once, with no error rows.
- high-window: 2048 requests in flight with short outputs; counts error rows.
- resume: a run interrupted (SIGINT) part-way, then run again on the same output, ends with
  every row written once.
- engine-restart: the engine is stopped mid-run and started again; the breaker opens, closes,
  and the run still writes every row once.
"""

from __future__ import annotations

import argparse
import asyncio
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
    if args.kind == "embeddings":
        to_request = lambda r: {"model": args.model, "input": r["text"]}  # noqa: E731
        parse = lambda r, body: {"reply": str(len(body["data"][0]["embedding"]))}  # noqa: E731
        route = "/embeddings"
    else:
        to_request = lambda r: {
            "model": args.model,
            "max_tokens": args.max_tokens,  # noqa: E731
            "messages": [{"role": "user", "content": r["text"]}],
        }
        parse = lambda r, body: {"reply": body["choices"][0]["message"]["content"]}  # noqa: E731
        route = "/chat/completions"
    stats = pump(
        rows,
        to_request,
        parse,
        endpoint=args.endpoint,
        output=args.output,
        window=window,
        id_key="id",
        route=route,
    )
    print(
        json.dumps(
            {
                "rows_processed": stats.rows_processed,
                "rows_failed": stats.rows_failed,
                "breaker_opens": stats.breaker_opens,
            }
        ),
        flush=True,
    )


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
    cmd = [
        sys.executable,
        __file__,
        "run",
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--output",
        output,
        "--rows",
        str(rows),
        "--window",
        window,
        "--max-tokens",
        str(max_tokens),
        "--kind",
        args.kind,
    ]
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


def report(
    check: str, rows: int, t0: float, result: dict, output: str, failures: list[str] | None = None
) -> None:
    """One JSON line: the check passes when every row is written once, with no error rows, the
    run exits 0, and the check's own conditions (`failures`) are empty."""
    w = written(output)
    failures = list(failures or [])
    if not (w["rows"] == rows and w["distinct_ids"] == rows and w["error_rows"] == 0):
        failures.append(f"wrote {w['rows']} rows, {w['distinct_ids']} distinct, {w['error_rows']} errors")
    if result.get("exit") != 0:
        failures.append(f"run exited {result.get('exit')}")
    record = {
        "check": check,
        "pass": not failures,
        "failures": failures,
        "rows_asked": rows,
        "wall_s": round(time.time() - t0, 1),
        **w,
        **result,
    }
    print(json.dumps(record), flush=True)


SIGINT_EXITS = (-signal.SIGINT, 128 + signal.SIGINT)  # killed by SIGINT, or exited 130 on it


def rows_for(rows_per_s: float, seconds: float, minimum: int) -> int:
    """Enough rows to keep this engine busy for `seconds`: an interrupt must land mid-run."""
    return max(minimum, int(rows_per_s * seconds))


# What each engine must say about itself, as bench/jobs_live.sh starts it. probe: the engine the
# probe names (vLLM has no probe endpoint: None). capacity: the requests it runs at once, as
# reported (True: any number). gauges: the telemetry fields a tick must carry at least once.
EXPECT = {
    "vllm": {"probe": None, "capacity": None, "gauges": ("waiting", "running", "kv", "hits")},
    "sglang": {"probe": "sglang", "capacity": True, "gauges": ("waiting", "running", "kv")},
    "llamacpp": {"probe": "llamacpp", "capacity": 8, "gauges": ("waiting", "running")},
    "tei": {"probe": "tei", "capacity": True, "gauges": ("waiting",)},
}


async def probe_engine(endpoint: str) -> dict:
    """What the probe and one scrape read, as the pump reads them before its first request."""
    from saturate.http import HttpClient
    from saturate.signals import HttpScrape, probe

    async with HttpClient() as client:
        found = await probe(client, endpoint)
        scrape = HttpScrape(client, endpoint)
        scrape.identify(found)
        gauges = await scrape.read()
    capacity = found.capacity
    return {
        "probe_engine": found.engine,
        "metrics_on": found.metrics_on,
        "capacity": None if capacity is None else capacity.requests,
        "queues": None if capacity is None else capacity.queues,
        "scrape_url": scrape.url,
        "dialect": None if gauges is None else gauges["dialect"],
    }


def telemetry_ticks(output: str) -> list[dict]:
    ticks = []
    for path in glob.glob(f"{output}/**/telemetry-shard*.jsonl", recursive=True):
        with open(path) as f:
            ticks += [json.loads(line) for line in f if line.strip()]
    return ticks


def capacity_matches(reported: int | None, expected: int | bool | None) -> bool:
    """None: the engine reports no capacity. True: it reports one, any number. A number: exactly it."""
    if expected is None:
        return reported is None
    if expected is True:
        return reported is not None
    return reported == expected


def signals_check(args, out: str, rows: int) -> None:
    """The probe, one scrape, and the gauges in a loaded run's telemetry, against EXPECT."""
    t0 = time.time()
    proc = start_run(args, out, rows, "auto", args.max_tokens)
    time.sleep(5)  # past startup and the first tick: the scrape reads gauges under load
    loaded = proc.poll() is None
    found = asyncio.run(probe_engine(args.endpoint))
    loaded = loaded and proc.poll() is None  # the run outlasted the probe and the scrape
    result = finish(proc)
    ticks = telemetry_ticks(out)
    fields = ("waiting", "running", "kv", "hits")
    seen = {field: any(t.get(field) is not None for t in ticks) for field in fields}
    expect = EXPECT[args.engine]
    failures = []
    if found["probe_engine"] != expect["probe"]:
        failures.append(f"probe named {found['probe_engine']}, expected {expect['probe']}")
    if not capacity_matches(found["capacity"], expect["capacity"]):
        failures.append(f"capacity {found['capacity']}, expected {expect['capacity']}")
    if found["dialect"] != args.engine:
        failures.append(f"scrape dialect {found['dialect']}, expected {args.engine}")
    failures += [f"no tick carried {field}" for field in expect["gauges"] if not seen[field]]
    if result.get("exit") != 0:
        failures.append(f"run exited {result.get('exit')}")
    ok, failed = result.get("rows_processed"), result.get("rows_failed")
    if (ok, failed) != (rows, 0):
        failures.append(f"{ok} of {rows} rows ok, {failed} failed")
    if not loaded:
        failures.append("the run ended before the probe and scrape: they did not read a loaded engine")
    record = {
        "check": "signals",
        "engine": args.engine,
        "pass": not failures,
        "failures": failures,
        "wall_s": round(time.time() - t0, 1),
        "ticks": len(ticks),
        "seen": seen,
        **found,
        **result,
    }
    print(json.dumps(record), flush=True)


def checks(args) -> None:
    def fresh() -> str:
        return tempfile.mkdtemp(prefix="live-")

    out, t0 = fresh(), time.time()
    report("complete", 2000, t0, finish(start_run(args, out, 2000, "auto", 64)), out)
    rows_per_s = 2000 / max(time.time() - t0, 1.0)  # sizes the runs below to this engine

    if args.engine:
        # 60 s by the complete check's rate, which counts startup: a fast engine (TEI) runs
        # several times faster in steady state, and the probe at 5 s must still see a busy engine
        signals_check(args, fresh(), rows_for(rows_per_s, 60, 3000))

    out, t0 = fresh(), time.time()
    report("high-window", 8000, t0, finish(start_run(args, out, 8000, "2048", 16)), out)

    rows = rows_for(rows_per_s, 4 * args.interrupt_after_s, 4000)
    out, t0 = fresh(), time.time()
    first = start_run(args, out, rows, "auto", 64)
    time.sleep(args.interrupt_after_s)
    first.send_signal(signal.SIGINT)
    interrupted = finish(first)
    partial = written(out)["rows"]
    result = finish(start_run(args, out, rows, "auto", 64))
    extra = {"first_exit": interrupted["exit"], "rows_before_resume": partial}
    missed = []
    if interrupted["exit"] not in SIGINT_EXITS:
        missed.append(f"the first run exited {interrupted['exit']}, not by the interrupt")
    if not 0 < partial < rows:
        missed.append(f"{partial} of {rows} rows before the resume: the interrupt did not land mid-run")
    report("resume", rows, t0, {**result, **extra}, out, missed)

    out, t0 = fresh(), time.time()
    proc = start_run(args, out, rows, "auto", 64)
    time.sleep(args.interrupt_after_s)
    stop_engine()
    time.sleep(10)
    start_engine()
    wait_healthy(args.endpoint)
    result = finish(proc)
    missed = [] if result.get("breaker_opens") else ["the breaker never opened: the restart missed the run"]
    report("engine-restart", rows, t0, result, out, missed)


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
    engine = subprocess.Popen(
        os.environ["ENGINE_START"], shell=True, stdout=log, stderr=log, start_new_session=True
    )
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
    ap.add_argument("--kind", choices=["chat", "embeddings"], default="chat")
    ap.add_argument("--engine", choices=sorted(EXPECT), help="run the signals check against EXPECT")
    args = ap.parse_args()
    run(args) if args.mode == "run" else checks(args)


if __name__ == "__main__":
    main()
