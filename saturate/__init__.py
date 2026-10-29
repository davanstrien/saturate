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
import json
import logging
import os
import time
from collections.abc import Callable, Iterable

from saturate._logs import default_output
from saturate.controller import Auto, Fixed, Obs
from saturate.core import AdaptiveClient, AdaptiveLimiter, Done
from saturate.engine import Engine, wait_for_health
from saturate.pipeline import through
from saturate.signals import CEILING_FLAG
from saturate.sink import FileSink, ParquetSink, as_sink, read_output
from saturate.source import (
    check_shard,
    content_id,
    prepare_ahead,
    shard_of,
    shard_select,
    skip_done,
    stream,
)
from saturate.sources import bucket_rows, dataset_rows
from saturate.telemetry import advise, advise_input, bound_by_counts, cut_reasons
from saturate.transport import (
    Attempt,
    FatalTransportError,
    Request,
    make_json_request,
    make_multipart_request,
)
from saturate.writer import Sink, drain, sink_hooks

log = logging.getLogger(__name__)

__all__ = [
    "pump", "Stats", "RowsLostError", "Fixed", "Auto", "Obs", "Attempt", "Engine", "wait_for_health",
    "Request", "make_json_request", "make_multipart_request", "existing_ids",
    "AdaptiveClient", "AdaptiveLimiter", "Done", "through", "stream", "skip_done",
    "drain", "read_output", "Sink", "sink_hooks", "ParquetSink", "FileSink", "shard_select",
    "shard_of", "content_id", "prepare_ahead",
    "FatalTransportError", "dataset_rows", "bucket_rows"]
try:
    __version__ = importlib.metadata.version("saturate")
except importlib.metadata.PackageNotFoundError:  # source tree without an installed dist
    __version__ = "0.0.0+unknown"

USER_AGENT = f"saturate/{__version__}"
AGENT_ENV_VARS = ("CLAUDECODE", "CODEX_SANDBOX", "AI_AGENT")


def agent_mode() -> bool:
    return any(os.environ.get(v) for v in AGENT_ENV_VARS)



def existing_ids(out_uri: str, retry_errors: bool = False) -> set[str]:
    """The resume anti-join set for an output directory (manifest-first, exact)."""
    return ParquetSink(out_uri, create=False).existing_ids(retry_errors=retry_errors)


class RowsLostError(RuntimeError):
    """A run ended with rows that were read but neither skipped nor written."""


@dataclasses.dataclass
class Stats:
    rows_total: int = 0
    rows_done_prior: int = 0
    rows_errored_prior: int = 0  # of rows_done_prior: skipped rows whose only record is an error
    rows_processed: int = 0
    rows_failed: int = 0
    rows_deduped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    final_limit: int = 0  # the window at the end (maybe mid-trial); Auto(initial=final_limit) starts near it
    input_bound: bool = False
    breaker_opens: int = 0
    retries: int = 0  # retry attempts over the run (a row that failed three times and then succeeded: 3)
    hints: list = dataclasses.field(default_factory=list)
    cut_reasons: dict = dataclasses.field(default_factory=dict)  # window reductions by controller reason
    bound_by: dict = dataclasses.field(default_factory=dict)  # ticks per bottleneck verdict (CONTRACT §6)

    @property
    def rows_unaccounted(self) -> int:
        """Rows read but neither skipped nor handed to the sink; 0 after every run that ends
        normally."""
        accounted = self.rows_done_prior + self.rows_deduped + self.rows_processed + self.rows_failed
        return self.rows_total - accounted

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
    flush_every: int | None = None,
    read_timeout: float = 1800.0,
    route: str = "/chat/completions",
    headers: dict | None = None,
    retry_errors: bool = False,
    id_key: str | None = None,
    id_fn: Callable | None = None,
    signal_source: str = "auto",  # "auto" (scrape, blind fallback) | "none"
    schema=None,  # pa.Schema: declared immutable output schema (CONTRACT §8) — else dynamic/sparse
    prepare_workers: int = 0,
    fail_fast: int = 32,
    retry_budget_s: float | None = None,
    timeouts: dict[str, float] | None = None,
    proxy: str | bool | None = None,
) -> Stats:
    """Run every row through the endpoint and land the results; safe to re-run (resumes).

    rows: iterable of dict rows (or (id, row) pairs) — the input to process.
    to_request: row -> Request (or a plain dict body posted to `route`).
    parse: (row, response_json) -> output dict; a single-arg parse(response_json) is accepted.
    endpoint: base URL of the OpenAI-compatible server, e.g. "http://host:8000/v1".
    output: output directory or URI (local, hf://, s3://...) or a sink object.
    window: Auto (default) adapts concurrency to the engine; Fixed(n) pins it.
    shard: (rank, world) for a fan-out over `world` Jobs into one output. The run processes only
        the rows whose id belongs to shard `rank` — `shard_of(id, world) == rank`, a hash of the
        id (CONTRACT §2) — and labels its telemetry, stats and completion marker with it. Give
        every Job the same `rows` and its own rank; the shards are disjoint and together cover
        the input. For large objects, pass the same `shard`
        to the source too (`bucket_rows(..., shard=)`, `dataset_rows(..., shard=)`): it then
        skips the other shards' rows before it fetches or decodes them. A source sharded
        differently from pump() leaves the run nothing to process. Input you split yourself
        must be split by `shard_of` (or `shard_select`); split any other way, pass no `shard`
        and give each Job its own output.
    flush_every: rows buffered before a parquet part is written. None (default) = 1000 rows,
        or sooner once the oldest buffered row is 10 s old (60 s for a remote output, hf://,
        s3://..., where each write is slow), whichever comes first: a row's output appears
        within that long, and a crash loses at most that long of rows, which a resume
        re-requests. Rows are counted, not bytes: lower it if parse returns large payloads.
    read_timeout: seconds to wait for a single response (long generations are legitimate).
    retry_budget_s: active-work budget for one row's attempts, retries and backoff included.
        Time paused on an open circuit breaker does not count, so it is not a deadline: a row
        can outlive it while the run waits for the server. None (default) = max(300, 2 *
        read_timeout): a retry must be able to wait as long as the first attempt did. Each
        retry's phase timeouts are capped to what is left of it.
    timeouts: the other HTTP phases, any of {"connect", "write", "pool"} in seconds (30 each by
        default). "pool" is the whole wait for a connection to send on, opening a new one
        included. A request's total limit is pool + write + read_timeout.
    proxy: None (default) uses the proxy the environment sets for the endpoint's host
        (HTTPS_PROXY, NO_PROXY; never for localhost); False uses none; a URL uses that proxy.
    route: path appended to `endpoint` when to_request returns a plain dict.
    headers: extra HTTP headers merged over the default User-Agent.
    retry_errors: re-run rows whose stored record carries an error instead of skipping them.
    id_key: row field to use as the row id (default: content hash of the row).
    id_fn: callable row -> id, an alternative to id_key.
    signal_source: "auto" scrapes /metrics and falls back to blind mode; "none" never scrapes.
    schema: pa.Schema declaring an immutable output schema; None infers it per part.
    prepare_workers: threads running `to_request` ahead of admission (0 = on the event loop).
        Use it when `to_request` does real work per row (image decode, PDF render, tokenising):
        on the loop that work serialises the whole pump and idles the engine — the run-end
        advisor says "PREP-BOUND" and quotes the ms/row when that happened. It helps only
        if the work releases the GIL (PIL, numpy, IO); `parse` always stays on the loop.
        For pure-Python CPU work, or to own the pool, prepare rows before the pump instead:
        `pump(prepare_ahead(rows, fn, executor=ProcessPoolExecutor()), ...)`.
    fail_fast: stop the run if the first N completed rows all fail (a wrong model name, a
        bad token, a parse bug), before any of them is stored, so a fixed re-run retries
        them. 0 disables the check. Off when retry_errors=True: that input is mostly rows that
        already failed, and some rows fail for good.
    """
    default_output()
    shard = check_shard(shard)
    select_shard = shard[1] > 1
    if select_shard:
        log.info(f"shard={shard}: processing only the rows whose id hashes to shard {shard[0]} "
                 f"of {shard[1]}")
    return asyncio.run(_pump(rows, to_request, parse, endpoint, output, window, shard,
                             flush_every, read_timeout, route, headers, retry_errors,
                             id_key, id_fn, signal_source, schema, prepare_workers,
                             fail_fast, retry_budget_s, timeouts, proxy, select_shard))


async def _pump(rows, to_request, parse, endpoint, output, window, shard, flush_every,
                read_timeout, route, extra_headers, retry_errors, id_key, id_fn,
                signal_source, schema=None, prepare_workers=0, fail_fast=32,
                retry_budget_s=None, timeouts=None, proxy=None, select_shard=False) -> Stats:
    stats = Stats()
    sink = as_sink(output, flush_every, schema=schema)
    hdrs = {"User-Agent": USER_AGENT + (" (agent)" if agent_mode() else ""),
            **(extra_headers or {})}
    t0 = time.monotonic()

    pending = stream(rows, id_key=id_key, id_fn=id_fn)
    if select_shard:
        pending = shard_select(pending, *shard)
    pending = skip_done(pending, sink, retry_errors=retry_errors, stats=stats)

    hooks = sink_hooks(sink)
    if hooks.begin_run is not None:
        hooks.begin_run(shard)  # this run's telemetry gets its own file, whatever the last run did
    lines: list[str] = []  # one JSON line per tick, serialised once (a run is many ticks)
    every = hooks.telemetry_every_ticks
    next_at = every
    writer: asyncio.Task | None = None
    write_lines = hooks.write_telemetry

    async def write_telemetry(write: Callable, snapshot: list[str]) -> None:
        try:  # a remote rewrite is a commit (seconds): never on the loop, never overlapping
            await asyncio.to_thread(write, shard, snapshot)
        except Exception as e:
            log.warning(f"telemetry write failed (non-fatal): {e}")

    def on_tick(rec: dict) -> None:  # periodically the trajectory so far lands on storage, so a
        nonlocal writer, next_at  # reader sees a partial run, not only a finished one (CONTRACT §6)
        lines.append(json.dumps(rec))
        if len(lines) >= next_at and write_lines is not None and (writer is None or writer.done()):
            writer = asyncio.get_running_loop().create_task(write_telemetry(write_lines, lines[:]))
            # every rewrite re-sends the whole trajectory: stretch the cadence with the run
            # (a quarter of its length so far) so total bytes stay under ~5x the final file
            next_at = len(lines) + max(every, len(lines) // 4)

    async with AdaptiveClient(endpoint, window=window, headers=hdrs,
                              read_timeout=read_timeout,
                              signal_source=signal_source, on_tick=on_tick,
                              retry_budget_s=retry_budget_s, timeouts=timeouts,
                              proxy=proxy) as client:
        async with through(client, pending, to_request, parse, route=route,
                           prepare_workers=prepare_workers,
                           fail_fast=0 if retry_errors else fail_fast) as results:
            await drain(results, sink, shard=shard, stats=stats)
        limiter = client.limiter
        dialect = client.dialect
        stats.breaker_opens = client.breaker.opens

    if writer is not None:
        await writer  # never two writers on one file
    if stats.rows_unaccounted:  # a lost row is a bug: the run must not certify itself complete
        raise RowsLostError(f"{stats.rows_unaccounted} of {stats.rows_total} rows were read but "
                            "neither skipped nor handed to the sink; no completion marker was "
                            "written, and a re-run resumes them")
    stats.elapsed_s = round(time.monotonic() - t0, 2)
    stats.final_limit = limiter.limit
    stats.retries = getattr(limiter, "retries_total", 0)  # an embedder's own limiter may count none
    stats.bound_by = bound_by_counts(limiter.ticks)
    stats.input_bound = any(t.get("input_bound") for t in limiter.ticks)  # CONTRACT §7: from the ticks
    stats.cut_reasons = cut_reasons(limiter.ticks)
    if lines and write_lines is not None:
        await write_telemetry(write_lines, lines)  # final: the complete trajectory
    stats.hints = advise(limiter.ticks, dialect, stats.final_limit, CEILING_FLAG)
    stats.hints += advise_input(limiter.ticks, prepare_workers)
    if stats.rows_errored_prior:
        stats.hints.append(f"RETRY-ERRORS: {stats.rows_errored_prior} rows were skipped because their "
                           "only record is an error; once the cause is fixed, re-run with "
                           "retry_errors=True to retry them")
    for h in stats.hints:
        log.info(f"advisor: {h}")
    if hooks.write_stats is not None:  # console-facing exact summary (CONTRACT §5)
        try:
            await asyncio.to_thread(hooks.write_stats, shard, stats.to_json())
        except Exception as e:
            log.warning(f"stats write failed (non-fatal): {e}")
    if hooks.write_marker is not None:  # last: the marker certifies stats/telemetry landed
        await asyncio.to_thread(hooks.write_marker, shard)
    if stats.rows_done_prior:
        log.info(f"resume: skipped {stats.rows_done_prior} rows already in the output "
                 f"({stats.rows_errored_prior} of them only as errors)")
    if stats.rows_deduped:
        log.info(f"dedup: {stats.rows_deduped} rows had the id of an earlier row in this input and were "
                 "skipped (the default id is a content hash: identical rows share it; to keep repeats, "
                 "give each its own id, e.g. id_fn=lambda row: f\"{row['prompt_id']}-{row['sample']}\")")
    log.info(f"done: {stats.rows_processed} ok, {stats.rows_failed} failed, "
             f"{stats.tokens_per_sec} tok/s, window settled at {stats.final_limit}")
    if agent_mode():
        print(stats.to_json(), flush=True)  # data on stdout, one line
        log.info("Hint: re-run the same command to resume — it is always safe.")
    return stats
