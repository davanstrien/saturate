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
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import anyio

from saturate.core import Done
from saturate.sink import UNSTORABLE, _error_row
from saturate.transport import usage_tokens

WRITER_QUEUE = 256  # finished rows waiting for the writer; a full queue makes drain wait


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
    return SinkHooks(probe=getattr(sink, "probe", None),
                     flush_interval_s=getattr(sink, "flush_interval_s", None),
                     begin_run=getattr(sink, "begin_run", None),
                     write_telemetry=getattr(sink, "write_telemetry", None),
                     write_stats=getattr(sink, "write_stats", None),
                     write_marker=getattr(sink, "write_marker", None),
                     telemetry_every_ticks=max(1, getattr(sink, "telemetry_every_ticks", 30)))


async def drain(results: AsyncIterable[Done], sink, shard: tuple[int, int] = (0, 1), stats=None):
    """Write every finished row to the sink; returns the run's Stats. However the run ends, the
    rows that already finished are written. If you cancel, drain first stops the run, takes the
    rows already waiting in `results` and waits for them to be written, then the cancellation
    goes on. Completion markers are pump()'s job; drain writes none."""
    from saturate import Stats  # local import: avoid cycle

    stats = stats if stats is not None else Stats()
    async with _closing(results), _writer(sink, stats) as writer:
        holding: Done | None = None  # a row taken from `results` but not yet with the writer
        try:
            async for done in results:
                holding = done
                await writer.put(done)
                holding = None
        except asyncio.CancelledError:
            await asyncio.shield(_hand_over_the_rest(results, writer, holding))
            raise
    return stats


def _closing(results):
    """Close the results stream on the way out when it can be closed: a sink error must stop
    the pipeline feeding it, or its requests keep running with no consumer."""
    return contextlib.aclosing(results) if hasattr(results, "aclose") else contextlib.nullcontext()


async def _hand_over_the_rest(results: AsyncIterable[Done], writer: _Writer, holding: Done | None) -> None:
    """On cancel: stop the run, then give the writer every finished row already waiting. After
    a second cancel the writer may close first: the rows it did not take are logged as lost."""
    try:
        await _give_the_rest(results, writer, holding)
    except Exception as e:
        _log(f"stopped before every finished row was written: {e!r}")


async def _give_the_rest(results: AsyncIterable[Done], writer: _Writer, holding: Done | None) -> None:
    stop = getattr(results, "stop", None)
    if stop is not None:
        stop()
    if holding is not None:
        await writer.put(holding)
    take = getattr(results, "receive_nowait", None)
    while take is not None:
        try:
            done = take()
        except (anyio.WouldBlock, anyio.EndOfStream):
            return
        await writer.put(done)


@contextlib.asynccontextmanager
async def _writer(sink, stats) -> AsyncGenerator[_Writer, None]:
    """Start the writer; on the way out let it write what it was given, then join its thread."""
    writer = _Writer(sink, stats)
    caller_error: BaseException | None = None
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(writer.run)
            try:
                yield writer
            except BaseException as e:
                caller_error = e  # raised below, outside the task group, so it is not wrapped
            finally:
                writer.no_more_rows(failed=caller_error is not None)
    finally:
        writer.join()  # never abandon a write: it could leave half a file
    if caller_error is not None and caller_error is not writer.failure:
        if writer.failure is not None:  # the caller's error (a cancel, say) came first
            _log(f"writing also failed: {writer.failure!r}")
        raise caller_error
    if writer.failure is not None:
        raise writer.failure


class _Writer:
    """One thread owns the sink. Rows arrive on a queue; each batch is appended in one call to
    the thread; when none arrive for the sink's flush interval, what it holds is flushed."""

    def __init__(self, sink, stats):
        self.sink, self.stats = sink, stats
        self.hooks = sink_hooks(sink)
        self.failure: BaseException | None = None
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="saturate-writer")
        self._send, self._receive = anyio.create_memory_object_stream[Done](WRITER_QUEUE)
        self._run_failed = False
        self._appended_since_flush = 0
        self._demoted_before = getattr(sink, "rows_demoted", 0)

    async def put(self, done: Done) -> None:
        """Give the writer a finished row; waits while its queue is full. Raises the writer's
        error if it has stopped."""
        try:
            await self._send.send(done)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            raise self.failure or RuntimeError("the writer has stopped") from None

    def no_more_rows(self, failed: bool) -> None:
        self._run_failed = failed
        self._send.close()

    def join(self) -> None:
        self._thread.shutdown(wait=True)

    async def run(self) -> None:
        """The writer task: write each batch, flush when quiet, flush at the end."""
        with self._receive:
            try:
                while (batch := await self._next_batch()) is not None:
                    if batch:
                        await self._in_thread(self._append, batch)
                    elif self._appended_since_flush:
                        await self._in_thread(self._flush)
            except BaseException as e:
                self.failure = e
                self._run_failed = True
            await self._in_thread(self._finish)

    async def _next_batch(self) -> list[Done] | None:
        """The finished rows waiting now (at least one); [] when none arrived for the sink's
        flush interval; None when drain has no more."""
        quiet = self.hooks.flush_interval_s
        try:
            with anyio.fail_after(quiet) if quiet else contextlib.nullcontext():
                first = await self._receive.receive()
        except TimeoutError:
            return []
        except anyio.EndOfStream:
            return None
        batch = [first]
        while True:
            try:
                batch.append(self._receive.receive_nowait())
            except (anyio.WouldBlock, anyio.EndOfStream):
                return batch

    async def _in_thread(self, call: Callable[..., None], *args) -> None:
        await asyncio.get_running_loop().run_in_executor(self._thread, call, *args)

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
            if not self._run_failed:
                self.failure = self.failure or e
            else:
                _log(f"flush after a failed run also failed (the first error is raised): {e!r}")
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
            rec = _error_row(done.id, e)
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
