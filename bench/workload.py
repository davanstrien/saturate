"""One benchmark scenario: pump() against an endpoint, measured from outside the library.

Uses only the public API (pump, Fixed, Auto) and the documented outputs (Stats, the telemetry
file), so the same script measures any saturate version, before and after a rewrite of the
plumbing. Prints one JSON line on stdout.

    python bench/workload.py --endpoint http://127.0.0.1:8000/v1 --rows 5000 --window 256

What it reports:
- rows_per_s, wall_s: throughput as the caller sees it.
- cpu_ms_per_row: client CPU time (all threads) per row. At zero endpoint latency this is the
  library's own overhead, independent of how fast the server answers.
- loop_lag_p99_s, loop_lag_max_s: from the per-tick `loop_lag_s` telemetry (CONTRACT §6), the
  time the event loop was blocked beyond the tick. A flush or a parse on the loop shows here.
- first_durable_s: seconds from start until the first part file exists in the output.
- peak_rss_mb: the process's peak resident memory.
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


def make_to_request(prep_ms: float, prep_kind: str, model: str, max_tokens: int):
    def to_request(row: dict) -> dict:
        if prep_ms:
            if prep_kind == "cpu":
                busy_ms(prep_ms)
            else:
                time.sleep(prep_ms / 1000)  # releases the GIL, like image decoding in C
        messages = [{"role": "user", "content": row["text"]}]
        return {"model": model, "max_tokens": max_tokens, "messages": messages}

    return to_request


def make_parse(parse_ms: float):
    def parse(row: dict, resp: dict) -> dict:
        if parse_ms:
            busy_ms(parse_ms)
        return {"reply": resp["choices"][0]["message"]["content"]}

    return parse


class FirstPartWatcher:
    """Polls the output for its first part file, to time how long results take to be durable."""

    def __init__(self, fs, root: str, t0: float):
        self.fs, self.root, self.t0 = fs, root, t0
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
            time.sleep(0.01)

    def start(self) -> FirstPartWatcher:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(1)


def read_ticks(fs, root: str) -> list[dict]:
    ticks = []
    for path in fs.glob(f"{root}/telemetry-*.jsonl"):
        with fs.open(path, "rb") as f:
            ticks += [json.loads(line) for line in f.read().decode().splitlines() if line]
    return ticks


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, KiB on Linux


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
    ap.add_argument("--output-uri", default=None,
                    help="any fsspec URI (e.g. hf://buckets/...); a fresh subdirectory per run")
    ap.add_argument("--write-delay-ms", type=float, default=0.0, help="slowmem: seconds each write takes")
    ap.add_argument("--flush-every", type=int, default=None)
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
    watcher = FirstPartWatcher(fs, root, t0).start()
    stats = pump(rows, make_to_request(args.prep_ms, args.prep_kind, args.model, args.max_tokens),
                 make_parse(args.parse_ms),
                 endpoint=args.endpoint, output=out_uri, window=window, flush_every=args.flush_every,
                 prepare_workers=args.prepare_workers, fail_fast=0)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    watcher.stop()

    lags = [t["loop_lag_s"] for t in read_ticks(fs, root) if t.get("loop_lag_s") is not None]
    import saturate

    print(json.dumps({
        "saturate_version": saturate.__version__,
        "rows": args.rows,
        "rows_ok": stats.rows_processed,
        "rows_failed": stats.rows_failed,
        "wall_s": round(wall, 3),
        "rows_per_s": round(args.rows / wall, 1),
        "cpu_ms_per_row": round(cpu * 1000 / args.rows, 4),
        "loop_lag_p99_s": percentile(lags, 99),
        "loop_lag_max_s": max(lags) if lags else None,
        "ticks": len(lags),
        "first_durable_s": round(watcher.first_s, 3) if watcher.first_s is not None else None,
        "final_limit": stats.final_limit,
        "tokens_per_s": stats.tokens_per_sec,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }))


if __name__ == "__main__":
    main()
