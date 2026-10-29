"""One benchmark scenario: pump() against an endpoint, measured from outside the library.

Uses only the public API (pump, Fixed, Auto) and the documented outputs (Stats, the telemetry
file), so the same script measures any saturate version, before and after a rewrite of the
plumbing. Prints one JSON line on stdout.

    python bench/workload.py --endpoint http://127.0.0.1:8000/v1 --rows 5000 --window 256

What it reports:
- rows_per_s, wall_s: successful rows per second as the caller sees it (rows_failed is reported;
  a run with failures is not comparable).
- cpu_ms_per_row: client CPU time (all threads) per row. At zero endpoint latency this is the
  library's own overhead, independent of how fast the server answers.
- loop_lag_max_s, loop_lag_mean_s: from the per-tick `loop_lag_s` telemetry (CONTRACT §6), the
  time the event loop was blocked beyond the tick. A flush or a parse on the loop shows here.
  Ticks are 2 s, so a run has few; loop_lag_p99_s is reported only from 20 ticks on.
- first_durable_s: seconds from start until the first part file exists in the output.
- peak_rss_mib: the process's peak resident memory.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import tempfile
import threading
import time
import uuid

import fsspec
from fsspec.implementations.memory import MemoryFileSystem

from saturate import Auto, Fixed, pump


class SlowMemoryFileSystem(MemoryFileSystem):
    """An in-memory "remote" store whose every write takes `write_delay_s` to close: stands in
    for an object store, where each flush is a round trip. Its own store, not memory://'s."""

    protocol = "slowmem"
    store: dict = {}
    pseudo_dirs = [""]
    write_delay_s = 0.0

    def _open(self, path, mode="rb", **kwargs):
        f = super()._open(path, mode=mode, **kwargs)
        if "w" in mode and self.write_delay_s:
            close = f.close

            def slow_close():
                time.sleep(self.write_delay_s)
                close()

            f.close = slow_close
        return f


def busy_ms(ms: float) -> None:
    """Hold the CPU (and the GIL) for `ms` milliseconds: pure-Python work like tokenising."""
    end = time.perf_counter() + ms / 1000
    while time.perf_counter() < end:
        pass


def make_to_request(prep_ms: float, prep_kind: str, model: str, max_tokens: int, kind: str = "chat"):
    def to_request(row: dict) -> dict:
        if prep_ms:
            if prep_kind == "cpu":
                busy_ms(prep_ms)
            else:
                time.sleep(prep_ms / 1000)  # releases the GIL, like image decoding in C
        if kind == "embeddings":
            return {"model": model, "input": row["text"]}
        messages = [{"role": "user", "content": row["text"]}]
        return {"model": model, "max_tokens": max_tokens, "messages": messages}

    return to_request


def make_parse(parse_ms: float, kind: str = "chat"):
    def parse(row: dict, resp: dict) -> dict:
        if parse_ms:
            busy_ms(parse_ms)
        if kind == "embeddings":
            return {"dim": len(resp["data"][0]["embedding"])}
        return {"reply": resp["choices"][0]["message"]["content"]}

    return parse


class FirstPartWatcher:
    """Polls the output for its first part file, to time how long results take to be durable.
    Every 10 ms on a local or in-memory store; every 0.5 s on a remote one, where each poll is a
    listing request (rate limits, and load on the store being measured)."""

    def __init__(self, fs, root: str, t0: float, interval_s: float):
        self.fs, self.root, self.t0, self.interval_s = fs, root, t0, interval_s
        self.first_s: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.fs.glob(f"{self.root}/part-*.parquet"):
                    self.first_s = time.perf_counter() - self.t0
                    return
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self) -> FirstPartWatcher:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(1)


def read_ticks(fs, root: str) -> list[dict]:
    ticks = []
    # telemetry lives under __saturate/telemetry/; outputs in the old layout keep it at the root
    paths = fs.glob(f"{root}/__saturate/telemetry/telemetry-*.jsonl") + fs.glob(f"{root}/telemetry-*.jsonl")
    for path in paths:
        with fs.open(path, "rb") as f:
            ticks += [json.loads(line) for line in f.read().decode().splitlines() if line]
    return ticks


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--window", default="64", help="an integer for Fixed(n), or 'auto'")
    ap.add_argument("--initial", type=int, default=None, help="Auto(initial=...) when --window auto")
    ap.add_argument("--model", default="stub")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--prompt", default="row {i}", help="row text; {i} is the row number")
    ap.add_argument("--prep-ms", type=float, default=0.0)
    ap.add_argument("--prep-kind", choices=["cpu", "sleep"], default="cpu")
    ap.add_argument("--parse-ms", type=float, default=0.0)
    ap.add_argument("--prepare-workers", type=int, default=0)
    ap.add_argument("--output", choices=["local", "slowmem"], default="local")
    ap.add_argument(
        "--output-uri",
        default=None,
        help="any fsspec URI (e.g. hf://buckets/...); a fresh subdirectory per run",
    )
    ap.add_argument(
        "--write-delay-ms", type=float, default=0.0, help="slowmem: milliseconds each write takes"
    )
    ap.add_argument("--flush-every", type=int, default=None)
    ap.add_argument(
        "--kind",
        choices=["chat", "embeddings"],
        default="chat",
        help="chat completions, or /embeddings with the row text as input",
    )
    ap.add_argument(
        "--signal-source",
        choices=["auto", "none"],
        default="auto",
        help="'none' never reads the engine's metrics (blind mode)",
    )
    args = ap.parse_args()

    if args.window == "auto":
        window = Auto() if args.initial is None else Auto(initial=args.initial)
    else:
        window = Fixed(int(args.window))
    if args.output_uri:
        out_uri = f"{args.output_uri.rstrip('/')}/{uuid.uuid4().hex[:12]}"  # a fresh run, never a resume
    elif args.output == "slowmem":
        fsspec.register_implementation("slowmem", SlowMemoryFileSystem, clobber=True)
        SlowMemoryFileSystem.write_delay_s = args.write_delay_ms / 1000
        out_uri = "slowmem://bench/out"
    else:
        out_uri = tempfile.mkdtemp(prefix="saturate-bench-")
    fs, root = fsspec.url_to_fs(out_uri)
    rows = [{"text": args.prompt.format(i=i)} for i in range(args.rows)]

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    local = args.output_uri is None
    watcher = FirstPartWatcher(fs, root, t0, interval_s=0.01 if local else 0.5).start()
    stats = pump(
        rows,
        make_to_request(args.prep_ms, args.prep_kind, args.model, args.max_tokens, args.kind),
        make_parse(args.parse_ms, args.kind),
        route="/embeddings" if args.kind == "embeddings" else "/chat/completions",
        endpoint=args.endpoint,
        output=out_uri,
        window=window,
        flush_every=args.flush_every,
        prepare_workers=args.prepare_workers,
        fail_fast=0,
        signal_source=args.signal_source,
    )
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    watcher.stop()

    lags = [t["loop_lag_s"] for t in read_ticks(fs, root) if t.get("loop_lag_s") is not None]
    import saturate

    print(
        json.dumps(
            {
                "saturate_version": saturate.__version__,
                "rows": args.rows,
                "rows_ok": stats.rows_processed,
                "rows_failed": stats.rows_failed,
                "wall_s": round(wall, 3),
                "rows_per_s": round(stats.rows_processed / wall, 1),
                "cpu_ms_per_row": round(cpu * 1000 / args.rows, 4),
                "loop_lag_max_s": max(lags) if lags else None,
                "loop_lag_mean_s": round(statistics.fmean(lags), 4) if lags else None,
                "loop_lag_p99_s": percentile(lags, 99) if len(lags) >= 20 else None,
                "ticks": len(lags),
                "first_durable_s": round(watcher.first_s, 3) if watcher.first_s is not None else None,
                "final_limit": stats.final_limit,
                "tokens_per_s": stats.tokens_per_sec,
                "peak_rss_mib": round(peak_rss_mib(), 1),
            }
        )
    )


if __name__ == "__main__":
    main()
