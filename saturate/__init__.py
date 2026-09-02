"""saturate — adaptive batch-inference pump.

Rows in -> any OpenAI-compatible endpoint -> resumable output.

pump() is the batteries-included front door: a composition of the functional
layer (stream -> skip_done -> through -> drain) over an AdaptiveClient, landing
on the storage CONTRACT (CONTRACT.md). Every stage is importable on its own.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.metadata
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Iterable

from saturate.controller import Auto, Fixed, Obs
from saturate.core import AdaptiveClient, AdaptiveLimiter, Done, through
from saturate.engine import Engine, wait_for_health
from saturate.signals import CEILING_FLAG
from saturate.sink import FileSink, ParquetSink, as_sink, drain, read_output
from saturate.source import content_id, shard_select, skip_done, stream
from saturate.sources import bucket_rows, dataset_rows
from saturate.telemetry import advise, cut_reasons
from saturate.transport import FatalTransportError, Request, make_json_request, make_multipart_request

__all__ = [
    "pump", "Stats", "Fixed", "Auto", "Obs", "Engine", "wait_for_health",
    "Request", "make_json_request", "make_multipart_request", "existing_ids",
    "AdaptiveClient", "AdaptiveLimiter", "Done", "through", "stream", "skip_done",
    "drain", "read_output", "ParquetSink", "FileSink", "shard_select", "content_id",
    "FatalTransportError", "dataset_rows", "bucket_rows"]
try:
    __version__ = importlib.metadata.version("saturate")
except importlib.metadata.PackageNotFoundError:  # source tree without an installed dist
    __version__ = "0.0.0+unknown"

USER_AGENT = f"saturate/{__version__}"
AGENT_ENV_VARS = ("CLAUDECODE", "CODEX_SANDBOX", "AI_AGENT")


def agent_mode() -> bool:
    return any(os.environ.get(v) for v in AGENT_ENV_VARS)


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)


def existing_ids(out_uri: str, retry_errors: bool = False) -> set[str]:
    """The resume anti-join set for an output directory (manifest-first, exact)."""
    return ParquetSink(out_uri).existing_ids(retry_errors=retry_errors)


def _adapt_parse(parse: Callable) -> Callable[[dict, dict], dict]:
    """parse(row, resp) with row passthrough; single-arg parse(resp) accepted."""
    params = [p for p in inspect.signature(parse).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return parse if len(params) >= 2 else (lambda row, resp: parse(resp))


@dataclasses.dataclass
class Stats:
    rows_total: int = 0
    rows_done_prior: int = 0
    rows_processed: int = 0
    rows_failed: int = 0
    rows_deduped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    final_limit: int = 0
    input_bound: bool = False
    breaker_opens: int = 0
    hints: list = dataclasses.field(default_factory=list)
    cut_reasons: dict = dataclasses.field(default_factory=dict)  # window reductions by controller reason

    @property
    def tokens_per_sec(self) -> float:
        total = self.prompt_tokens + self.completion_tokens
        return round(total / self.elapsed_s, 1) if self.elapsed_s else 0.0

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["tokens_per_sec"] = self.tokens_per_sec
        return json.dumps(d)


def pump(
    rows: Iterable,
    to_request: Callable[[dict], Request | dict],
    parse: Callable,
    endpoint: str,
    output,
    window: Fixed | Auto | None = None,
    shard: tuple[int, int] = (0, 1),
    flush_every: int = 10,
    read_timeout: float = 1800.0,
    route: str = "/chat/completions",
    headers: dict | None = None,
    retry_errors: bool = False,
    id_key: str | None = None,
    id_fn: Callable | None = None,
    signal_source: str = "auto",  # "auto" (scrape, blind fallback) | "none"
    schema=None,  # pa.Schema: declared immutable output schema (CONTRACT §8) — else dynamic/sparse
) -> Stats:
    """Run every row through the endpoint and land the results; safe to re-run (resumes).

    rows: iterable of dict rows (or (id, row) pairs) — the input to process.
    to_request: row -> Request (or a plain dict body posted to `route`).
    parse: (row, response_json) -> output dict; a single-arg parse(response_json) is accepted.
    endpoint: base URL of the OpenAI-compatible server, e.g. "http://host:8000/v1".
    output: output directory or URI (local, hf://, s3://...) or a sink object.
    window: Auto (default) adapts concurrency to the engine; Fixed(n) pins it.
    shard: (rank, world) label written into output file names and completion markers.
    flush_every: rows buffered before a parquet part is written.
    read_timeout: seconds to wait for a single response (long generations are legitimate).
    route: path appended to `endpoint` when to_request returns a plain dict.
    headers: extra HTTP headers merged over the default User-Agent.
    retry_errors: re-run rows whose stored record carries an error instead of skipping them.
    id_key: row field to use as the row id (default: content hash of the row).
    id_fn: callable row -> id, an alternative to id_key.
    signal_source: "auto" scrapes /metrics and falls back to blind mode; "none" never scrapes.
    schema: pa.Schema declaring an immutable output schema; None infers it per part.

    Warning: `shard=(rank, world)` labels output files and completion markers only. It does
    not select input rows — every shard given the same `rows` processes all of them. Pre-shard
    the input yourself: `pump(shard_select(stream(rows), rank, world), ..., shard=(rank, world))`.
    """
    return asyncio.run(_pump(rows, to_request, parse, endpoint, output, window, shard,
                             flush_every, read_timeout, route, headers, retry_errors,
                             id_key, id_fn, signal_source, schema))


async def _pump(rows, to_request, parse, endpoint, output, window, shard, flush_every,
                read_timeout, route, extra_headers, retry_errors, id_key, id_fn,
                signal_source, schema=None) -> Stats:
    stats = Stats()
    if shard[1] > 1:
        _log(f"shard={tuple(shard)}: labels output only — input must be pre-sharded with "
             "shard_select(stream(rows), rank, world)")
    sink = as_sink(output, flush_every, schema=schema)
    hdrs = {"User-Agent": USER_AGENT + (" (agent)" if agent_mode() else ""),
            **(extra_headers or {})}
    t0 = time.monotonic()

    pending = stream(rows, id_key=id_key, id_fn=id_fn)
    pending = skip_done(pending, sink, retry_errors=retry_errors, stats=stats)

    def write_telemetry(ticks: list[dict]) -> None:  # best-effort sidecar: never fails the run
        if ticks and hasattr(sink, "write_telemetry"):
            try:
                sink.write_telemetry(shard, [json.dumps(x) for x in ticks])
            except Exception as e:
                _log(f"telemetry write failed (non-fatal): {e}")

    tick_n = 0

    def on_tick(_rec: dict) -> None:  # every 30 ticks (~60 s) the trajectory so far lands on
        nonlocal tick_n  # storage, so a reader sees a partial run, not only a finished one
        tick_n += 1
        if tick_n % 30 == 0:
            write_telemetry(client.limiter.ticks)

    async with AdaptiveClient(endpoint, window=window, headers=hdrs,
                              read_timeout=read_timeout,
                              signal_source=signal_source, on_tick=on_tick) as client:
        results = through(client, pending, to_request, _adapt_parse(parse), route=route)
        await drain(results, sink, shard=shard, stats=stats)
        limiter = client.limiter
        dialect = client.dialect
        stats.breaker_opens = client.breaker.opens

    stats.elapsed_s = round(time.monotonic() - t0, 2)
    stats.final_limit = limiter.window.limit
    stats.input_bound = limiter.input_bound_ever
    stats.cut_reasons = cut_reasons(limiter.ticks)
    write_telemetry(limiter.ticks)  # final: the complete trajectory, same file
    stats.hints = advise(limiter.ticks, dialect, stats.final_limit, CEILING_FLAG)
    if stats.input_bound:
        stats.hints.append("run was INPUT-BOUND — the source, not the engine, was the bottleneck")
    for h in stats.hints:
        _log(f"advisor: {h}")
    if hasattr(sink, "write_stats"):  # console-facing exact summary (CONTRACT §5)
        try:
            sink.write_stats(shard, stats.to_json())
        except Exception as e:
            _log(f"stats write failed (non-fatal): {e}")
    if hasattr(sink, "write_marker"):  # last: the marker certifies stats/telemetry landed
        sink.write_marker(shard)
    _log(f"done: {stats.rows_processed} ok, {stats.rows_failed} failed, "
         f"{stats.tokens_per_sec} tok/s, window settled at {stats.final_limit}")
    if agent_mode():
        print(stats.to_json(), flush=True)  # data on stdout, one line
        print("Hint: re-run the same command to resume — it is always safe.", file=sys.stderr)
    return stats
