"""drain(): finished rows into the sink, written off the event loop. Design: docs/pipeline.md.

    results ──▶ drain ──▶ [queue] ──▶ writer thread ──▶ sink

drain takes finished rows and hands them to the writer. The writer is one thread that owns the
sink: every call that touches the sink's state runs there, in order, so nothing races and a slow
write never blocks the requests in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import sys
from collections.abc import AsyncIterable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import anyio

from saturate.core import Done
from saturate.sink import UNSTORABLE, error_row
from saturate.transport import usage_tokens

WRITER_QUEUE = 256  # finished rows waiting for the writer; a full queue makes drain wait
_END = object()  # put in the writer's queue after the last row


class Sink(Protocol):
    """What drain needs from an output. Anything with these three methods is a sink; the
    optional extras are read by sink_hooks()."""

    def existing_ids(self, retry_errors: bool = False) -> set[str]: ...

    def append(self, record: dict) -> None: ...

    def flush(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class SinkHooks:
    """The optional parts of the sink contract, read in one place. A hook the sink does not
    have is None."""

    probe: Callable[[dict], None] | None  # check a record fits before it is buffered
    flush_interval_s: float | None  # flush buffered rows once none have arrived for this long
    begin_run: Callable | None
    write_telemetry: Callable | None
    write_stats: Callable | None
    write_marker: Callable | None
    telemetry_every_ticks: int


def sink_hooks(sink) -> SinkHooks:
    interval = getattr(sink, "flush_interval_s", None)
    return SinkHooks(probe=getattr(sink, "probe", None),
                     flush_interval_s=interval if interval and interval > 0 else None,  # 0: no timer
                     begin_run=getattr(sink, "begin_run", None),
                     write_telemetry=getattr(sink, "write_telemetry", None),
                     write_stats=getattr(sink, "write_stats", None),
                     write_marker=getattr(sink, "write_marker", None),
                     telemetry_every_ticks=max(1, getattr(sink, "telemetry_every_ticks", 30)))


async def drain(results: AsyncIterable[Done], sink, shard: tuple[int, int] = (0, 1), stats=None):
    """Write every finished row to the sink; returns the run's Stats.

    However the run ends, the rows that already finished are written. If you cancel, drain
    stops the run, takes the finished rows still waiting, and waits for them to be written
    before the cancellation goes on (a second cancel stops waiting; the writing goes on by
    itself). If writing fails, drain stops reading at once and raises the write error.
    Completion markers are pump()'s job."""
    from saturate import Stats  # local import: avoid cycle

    stats = stats if stats is not None else Stats()
    writer = _Writer(sink, stats)
    writer.start()
    caller_error: BaseException | None = None
    holding: Done | None = None  # taken from `results`, not yet with the writer
    async with _closing(results):
        with anyio.CancelScope() as reading:
            writer.on_failure = reading.cancel  # a failed write stops the reading straight away
            try:
                async for done in results:
                    holding = done
                    await writer.put(done)
                    holding = None
            except asyncio.CancelledError as e:
                if reading.cancel_called:  # the writer failed: the scope ends the reading
                    raise
                caller_error = e
            except BaseException as e:
                caller_error = e
        cancelled = isinstance(caller_error, asyncio.CancelledError)
        # one cleanup, in order: the finished rows still waiting (on a cancel), then the end
        cleanup = asyncio.ensure_future(_close(results, writer, holding, hand_over=cancelled,
                                               failed=caller_error is not None))
        await _wait_out(cleanup, cancelled_before=cancelled)
    # The first failure is the error. A failed write stops the reading at once, so drain ends
    # without an error of its own: an error the caller hit (a cancel included) came first.
    if caller_error is not None:
        if writer.failure is not None:
            _log(f"writing also failed: {writer.failure!r}")
        raise caller_error
    if writer.failure is not None:
        raise writer.failure
    return stats


async def _close(results: AsyncIterable[Done], writer: _Writer, holding: Done | None,
                 hand_over: bool, failed: bool) -> None:
    """Finish the writing, in order: on a cancel, stop the run and hand over every finished row
    still waiting; then tell the writer there are no more, and wait for its last flush."""
    if hand_over:
        try:
            stop = getattr(results, "stop", None)
            if stop is not None:
                await stop()
            if holding is not None:
                await writer.put(holding)
            take_waiting = getattr(results, "take_waiting", None)
            for done in take_waiting() if take_waiting is not None else []:
                await writer.put(done)
        except Exception as e:
            _log(f"stopped before every finished row was handed over: {e!r}")
    await writer.finish(failed=failed)


def _closing(results):
    """Close the results stream on the way out when it can be closed: a sink error must stop
    the pipeline feeding it, or its requests keep running with no consumer."""
    return contextlib.aclosing(results) if hasattr(results, "aclose") else contextlib.nullcontext()


async def _wait_out(task: asyncio.Future, cancelled_before: bool) -> None:
    """Wait for `task`. The first cancel of the drain does not stop the waiting (it is raised
    once the task is done); any later one does, and the task goes on by itself."""
    cancelled_here = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if cancelled_before or cancelled_here:
                raise
            cancelled_here = True
    if cancelled_here:
        raise asyncio.CancelledError


class _Writer:
    """One thread owns the sink. Rows arrive on a queue; the writer task takes every row
    waiting and appends them in one call to the thread; when none arrive for the sink's flush
    interval, what the sink holds is flushed."""

    def __init__(self, sink, stats):
        self.sink, self.stats = sink, stats
        self.hooks = sink_hooks(sink)
        self.failure: BaseException | None = None
        self.on_failure: Callable[[], None] = lambda: None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=WRITER_QUEUE)
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="saturate-writer")
        self._task: asyncio.Task | None = None
        self._next_row: asyncio.Future | None = None  # a get waiting for the next row
        self._ended = False  # drain said there are no more rows
        self._run_failed = False
        self._appended_since_flush = 0
        self._demoted_before = getattr(sink, "rows_demoted", 0)

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._run())

    async def put(self, done: Done) -> None:
        """Give the writer a finished row; waits while its queue is full. A row either is in
        the queue when this returns or is not if it raised: never half."""
        if self._task is not None and self._task.done():
            _log(f"the writer has finished; row {done.id} was not written")
            return
        await self._queue.put(done)

    async def finish(self, failed: bool) -> None:
        """No more rows: let the writer write what it holds, flush, and stop."""
        self._run_failed = self._run_failed or failed
        await self._queue.put(_END)
        if self._task is not None:
            await self._task
        self._thread.shutdown(wait=False)  # idle by now: the thread exits on its own

    async def _run(self) -> None:
        """The writer task: write each batch, flush when quiet, flush at the end. After a failed
        write it keeps taking rows (and drops them), so a sender never waits on a full queue."""
        try:
            while (batch := await self._next_batch()) is not None:
                if self.failure is not None:
                    continue
                if batch:
                    await self._call(self._append, batch)
                elif self._appended_since_flush:  # quiet for the flush interval
                    await self._call(self._flush)
        finally:
            if self._next_row is not None:
                self._next_row.cancel()  # a pending get: the row it would take stays in the queue
        await self._call(self._finish)

    async def _next_batch(self) -> list[Done] | None:
        """Every row waiting now (at least one); [] when none arrived for the sink's flush
        interval; None once drain has said there are no more."""
        if self._ended:
            return None
        if self._next_row is None:  # kept across quiet intervals, so no row is ever dropped
            self._next_row = asyncio.ensure_future(self._queue.get())
        done, _ = await asyncio.wait({self._next_row}, timeout=self.hooks.flush_interval_s)
        if not done:
            return []
        batch = [self._next_row.result(), *self._take_waiting()]
        self._next_row = None
        if _END in batch:
            self._ended = True
            batch = [d for d in batch if d is not _END]
            return batch or None
        return batch

    def _take_waiting(self) -> list:
        taken = []
        while True:
            try:
                taken.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return taken

    async def _call(self, call: Callable[..., None], *args) -> None:
        """Run `call` in the writer thread; a failure is recorded, stops the reading, and from
        then on rows are dropped."""
        try:
            await asyncio.get_running_loop().run_in_executor(self._thread, call, *args)
        except Exception as e:
            if self.failure is None:
                self.failure = e
                self.on_failure()

    # --- in the writer thread ------------------------------------------------------------------

    def _append(self, batch: list[Done]) -> None:
        for done in batch:
            self.sink.append(_record(done, self.hooks.probe, self.stats))
        self._appended_since_flush += len(batch)

    def _flush(self) -> None:
        self.sink.flush()
        self._appended_since_flush = 0

    def _finish(self) -> None:
        """The last flush, and the demoted-row count. After a failure, a flush that fails too is
        logged: the first error is the one raised."""
        try:
            self._flush()
        except Exception as e:
            if not self._run_failed and self.failure is None:
                raise
            _log(f"flush after a failed run also failed (the first error is raised): {e!r}")
        finally:
            _count_demoted(self.sink, self._demoted_before, self.stats)


def _record(done: Done, probe, stats) -> dict:
    """The record one finished row leaves in the sink, with its counts added to `stats`."""
    if done.error is not None:
        stats.rows_failed += 1
        return {"id": done.id, "error": done.error}
    assert done.out is not None  # a row without an error has an output
    # id/error are reserved columns and ALWAYS win over parse output —
    # a parse returning e.g. the OpenAI response id must not break resume
    rec = {**done.out, "id": done.id, "error": None}
    if probe is not None:
        try:  # validate against the sink's ACTUAL schema before buffering —
            probe(rec)  # a bad value must become an error row, not a flush crash
        except UNSTORABLE as e:
            rec = error_row(done.id, e)
    # tokens were spent even when the sink rejects the output
    prompt, completion = usage_tokens({"usage": done.usage})
    stats.prompt_tokens += prompt
    stats.completion_tokens += completion
    if rec["error"] is None:
        stats.rows_processed += 1
    else:
        stats.rows_failed += 1
    return rec


def _count_demoted(sink, demoted_before: int, stats) -> None:
    """Rows a flush demoted landed as error rows, not successes. Capped at what this drain
    counted: rows buffered before it started were never in these stats."""
    demoted = min(getattr(sink, "rows_demoted", 0) - demoted_before, stats.rows_processed)
    stats.rows_processed -= demoted
    stats.rows_failed += demoted


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)
