"""through(): rows in, finished rows out. The design, in plain terms: docs/pipeline.md.

    input ──▶ reader ──▶ feeder ──▶ one task per row ──▶ results ──▶ (drain)
            (thread)                  (request + parse)

The reader pulls rows from the input in a thread; the feeder starts a task per row when there is
room; each row task builds the request, waits for a slot in the window, sends it and parses the
reply; a finished row goes into `results`, which the caller reads.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import math
import sys
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator

import anyio
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from saturate.core import AdaptiveClient, AdaptiveLimiter, Done
from saturate.transport import FatalTransportError

READ_BATCH_ROWS = 64  # the reader hands over at most this many rows at a time...
READ_BATCH_S = 0.05  # ...and never holds a batch longer than this


def adapt_parse(parse: Callable) -> Callable[[dict, dict], dict]:
    """parse(row, resp) with row passthrough; a parse with one REQUIRED positional parameter
    is parse(resp). Parameters with defaults do not count: `lambda body, model=M: ...` takes
    the response, not the row."""
    try:
        sig = inspect.signature(parse)
    except (TypeError, ValueError):  # a callable with no introspectable signature: the documented form
        return parse
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    return parse if len(params) >= 2 else (lambda row, resp: parse(resp))


class Results:
    """The finished rows, in the order they finished. Taking one makes room for another row to
    start (the backlog limit), so read it to the end, or leave the `async with` to stop the run."""

    def __init__(self, receive: MemoryObjectReceiveStream[Done], run: _Run):
        self._receive, self._run = receive, run

    def __aiter__(self) -> Results:
        return self

    async def __anext__(self) -> Done:
        try:
            done = await self._receive.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None
        self._run.backlog.release()
        return done

    def receive_nowait(self) -> Done:
        """A finished row already waiting, without waiting for one. Raises anyio.WouldBlock when
        none is waiting, anyio.EndOfStream when the run is over."""
        done = self._receive.receive_nowait()
        self._run.backlog.release()
        return done

    def stop(self) -> None:
        """Stop starting and running rows; rows already finished stay readable."""
        self._run.stop()


@dataclasses.dataclass
class _Run:
    """What the feeder and the row tasks share for one run."""

    client: AdaptiveClient
    limiter: AdaptiveLimiter
    to_request: Callable
    parse: Callable[[dict, dict], dict]
    route: str
    fail_fast: int
    results: MemoryObjectSendStream[Done]
    prepare: anyio.Semaphore  # request bodies built but not yet admitted
    backlog: anyio.Semaphore  # rows started but not yet taken from `results`
    prepare_threads: anyio.CapacityLimiter | None  # None: to_request runs on the event loop
    reader_stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    failure: BaseException | None = None  # the first failure: the error the run raises
    held: list[Done] = dataclasses.field(default_factory=list)  # fail_fast: errors before a success
    succeeded: bool = False
    discarding: bool = False  # fail_fast gave up: finished rows are dropped, not stored
    scope: anyio.CancelScope | None = None  # the feeder's and row tasks' scope

    def fail(self, error: BaseException) -> None:
        """Record the run's first failure and stop the row tasks; later failures are logged."""
        if self.failure is None:
            self.failure = error
        else:
            print(f"[pump] after the first failure, also: {error!r}", file=sys.stderr, flush=True)
        self.stop()

    def stop(self) -> None:
        self.reader_stop.set()
        if self.scope is not None:
            self.scope.cancel()

    def publish(self, done: Done) -> None:
        """Put a finished row in `results`. No await: once a row is parsed it is never lost to
        a cancellation before it is published."""
        if self.discarding:
            self.backlog.release()
            return
        if self.fail_fast and not self.succeeded:
            if done.error is not None:  # hold errors back until the first success
                self.held.append(done)
                if len(self.held) >= self.fail_fast:
                    self._give_up_fail_fast()
                return
            self.succeeded = True
            for held in self.held:
                self.results.send_nowait(held)
            self.held.clear()
        self.results.send_nowait(done)

    def publish_held(self) -> None:
        """The input ended before fail_fast rows finished: the held errors are this run's results."""
        for held in self.held:
            self.results.send_nowait(held)
        self.held.clear()

    def _give_up_fail_fast(self) -> None:
        first = self.held[0].error
        for _ in self.held:
            self.backlog.release()
        self.held.clear()
        self.discarding = True
        self.fail(FatalTransportError(
            f"the first {self.fail_fast} rows all failed, so the run stopped without storing "
            f"them. First error: {first}. Fix the cause and re-run (these rows will be retried), "
            "or pass fail_fast=0 to disable this check"))


@contextlib.asynccontextmanager
async def through(client: AdaptiveClient, rows: Iterable[tuple[str, dict]], to_request: Callable,
                  parse: Callable, route: str = "/chat/completions", prepare_workers: int = 0,
                  fail_fast: int = 32, backlog: int | None = None) -> AsyncGenerator[Results, None]:
    """Run every (id, row) through the endpoint; yields the finished rows as they finish.

        async with through(client, rows, to_request, parse) as results:
            async for done in results:
                ...

    `prepare_workers` > 0 runs `to_request` in that many threads (it must bound its own I/O: a
    running call cannot be interrupted). `fail_fast`: if the first N finished rows all failed,
    the run stops without storing them and raises. `backlog` caps rows started but not yet read
    from `results`; by default room for the largest window, the fail_fast hold-back and 2048
    rows for the sink to buffer before a flush.

    Leaving the `async with` early stops the run. When the run fails (the input raised, the
    endpoint is gone, fail_fast), every row that already finished can still be read, and the
    failure is raised when the `async with` is left."""
    send, receive = anyio.create_memory_object_stream[Done](math.inf)  # the backlog is the bound
    limiter = client.limiter
    if limiter is None:
        raise RuntimeError("through() needs the client entered: `async with AdaptiveClient(...) as client:`")
    limiter.prep_workers = prepare_workers
    run = _Run(client=client, limiter=limiter, to_request=to_request, parse=adapt_parse(parse), route=route,
               fail_fast=max(0, fail_fast), results=send,
               prepare=anyio.Semaphore(max(2 * prepare_workers, 4)),
               backlog=anyio.Semaphore(backlog or _default_backlog(limiter, fail_fast)),
               prepare_threads=anyio.CapacityLimiter(prepare_workers) if prepare_workers else None)
    caller_error: BaseException | None = None
    with receive:
        async with anyio.create_task_group() as group:
            group.start_soon(_produce, run, rows)
            try:
                yield Results(receive, run)
            except BaseException as e:
                caller_error = e  # raised below, outside the task group, so it is not wrapped
            finally:
                run.stop()
    raise_first_failure(caller_error, run.failure)


def raise_first_failure(caller_error: BaseException | None, run_failure: BaseException | None) -> None:
    """The run's error is its first failure. A failure recorded by the run happened before the
    caller's error (the caller's error stops the run at once), so it wins, except over a
    cancellation, which always goes on as itself."""
    if caller_error is not None:
        if run_failure is None or isinstance(caller_error, anyio.get_cancelled_exc_class()):
            raise caller_error
        print(f"[pump] after the first failure, also: {caller_error!r}", file=sys.stderr, flush=True)
    if run_failure is not None:
        raise run_failure


def _default_backlog(limiter: AdaptiveLimiter, fail_fast: int) -> int:
    window = getattr(limiter, "controller", None)
    most = getattr(window, "max", None) or getattr(window, "n", None) or 512  # Auto.max, Fixed.n
    return int(most) + max(0, fail_fast) + 2048


async def _produce(run: _Run, rows: Iterable[tuple[str, dict]]) -> None:
    """The feeder and its row tasks, in their own task group: a failure cancels them, not the
    caller reading `results`. Closing `results` at the end tells the caller there is no more."""
    with run.results:
        async with anyio.create_task_group() as row_tasks:
            run.scope = row_tasks.cancel_scope
            await _feed(run, rows, row_tasks)
        if not run.discarding and not isinstance(run.failure, FatalTransportError):
            run.publish_held()  # the input ended (or raised) before fail_fast rows finished


async def _feed(run: _Run, rows: Iterable[tuple[str, dict]], row_tasks: TaskGroup) -> None:
    """Start a row task per row, when there is room in the backlog and the prepare limit."""
    limiter = run.limiter
    source = iter(rows)
    while True:
        batch, error = await anyio.to_thread.run_sync(_read_batch, source, run.reader_stop,
                                                      abandon_on_cancel=True)
        for (id_, row), waited in batch:
            limiter.note_source_wait(waited)
            t = time.monotonic()
            await run.backlog.acquire()
            limiter.note_sink_wait(time.monotonic() - t)
            await run.prepare.acquire()
            row_tasks.start_soon(_row, run, id_, row)
        if error is not None:  # the input raised: rows already started finish, then the run fails
            if not isinstance(error, StopIteration):
                run.failure = run.failure or error
            return


def _read_batch(source: Iterator[tuple[str, dict]],
                stop: threading.Event) -> tuple[list[tuple[tuple[str, dict], float]], BaseException | None]:
    """In the reader thread: up to READ_BATCH_ROWS rows, or what READ_BATCH_S allows, each with
    the time the input took to yield it. Ends with the input's StopIteration or error, if any,
    after the rows read before it."""
    batch: list[tuple[tuple[str, dict], float]] = []
    started = time.monotonic()
    while len(batch) < READ_BATCH_ROWS and not stop.is_set():
        t = time.monotonic()
        try:
            item = next(source)
        except BaseException as e:  # StopIteration included: the input is done
            return batch, e
        batch.append((item, time.monotonic() - t))
        if time.monotonic() - started >= READ_BATCH_S:
            break
    return batch, (StopIteration() if stop.is_set() else None)


async def _row(run: _Run, id_: str, row: dict) -> None:
    """One row: build the request, wait for a slot, send it, parse the reply, publish."""
    holding_prepare = True

    def admitted() -> None:  # the body is in flight now, no longer waiting: free the prepare limit
        nonlocal holding_prepare
        if holding_prepare:
            run.prepare.release()
            holding_prepare = False

    try:
        request = await _build_request(run, row)
        body, error = await run.client.post(request, run.route, on_admit=admitted)
        if body is None:  # the request failed: `error` says why
            run.publish(Done(id_, row, None, error))
            return
        out = run.parse(row, body)
        if not isinstance(out, dict):  # storage contract: rows are dicts
            raise TypeError(f"parse must return a dict, got {type(out).__name__}")
        run.publish(Done(id_, row, out, None, body.get("usage") or {}))
    except FatalTransportError as e:  # the run cannot go on: stop it, never a row error
        run.backlog.release()
        run.fail(e)
    except Exception as e:  # to_request/parse bugs are the row's error, not a lost row
        run.publish(Done(id_, row, None, f"client: {type(e).__name__}: {e}"))
    finally:
        admitted()  # never strand the prepare limit: a failed or cancelled row frees it too


async def _build_request(run: _Run, row: dict):
    """`to_request(row)`, on the event loop or in a prepare thread; its time goes to telemetry."""
    def build():
        t = time.monotonic()
        return run.to_request(row), time.monotonic() - t

    if run.prepare_threads is None:
        request, took = build()
    else:
        request, took = await anyio.to_thread.run_sync(build, limiter=run.prepare_threads,
                                                       abandon_on_cancel=True)
    run.limiter.note_prep(took)
    return request
