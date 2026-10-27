"""The behaviour contract the 0.3 refactor must keep: what each kind of failure does to the run,
and what the concurrency primitives guarantee under cancellation and resizing.

The tests drive public entry points (pump, AdaptiveClient, AdaptiveLimiter, through, drain,
ParquetSink, Breaker) and observe public outputs (Stats, on_tick records, the output on disk, the
stub server's counters). Everything below the public API that they still need sits in the
"transitional seams" section, so a rewrite of the plumbing changes those lines, not the tests.

Pipeline tests run on a loop that outlives the coroutine (`run_checked`): asyncio.run cancels
stray tasks at shutdown, which would hide exactly the leaks these tests look for.

Preconditions are established by events (a request reached the server, a row was parsed), not by
sleeps; the remaining sleeps are deadlock guards or negative checks ("nothing else happened").

Failure policy (one test per row):

| failure                         | behaviour                                                    |
|---------------------------------|--------------------------------------------------------------|
| the source raises               | rows already submitted finish and land, then it raises       |
| fatal transport (breaker dead)  | in-flight rows are cancelled, completed rows land, it raises |
| fail_fast trips                 | the held rows are not stored, it raises                      |
| the sink raises                 | everything stops, the sink's own error raises                |
| the caller cancels              | everything stops, completed rows land                        |

Known gaps: where today's code breaks the contract, the test checks the contract with `gap()`,
which raises KnownGap, and is marked `known_gap(...)` = xfail(raises=KnownGap, strict=True). Only
that exception counts as the expected failure: a harness error, a leak or a timeout still fails
the test. When the fix lands the test passes, strict turns that into a failure, and the marker
comes off.
"""

import asyncio
import gc
import random
import threading
import time

import httpx2 as httpx
import pyarrow.parquet as pq
import pytest
from stub_server import StubServer

import saturate.core
import saturate.transport
from saturate import (
    AdaptiveClient,
    AdaptiveLimiter,
    FatalTransportError,
    Fixed,
    ParquetSink,
    drain,
    stream,
    through,
)
from saturate.core import Done
from saturate.transport import Breaker

TICK = 0.02


class KnownGap(Exception):
    """Raised by gap() only: the contract and today's code disagree here."""


def gap(ok: bool, message: str) -> None:
    if not ok:
        raise KnownGap(message)


def known_gap(reason: str):
    return pytest.mark.xfail(raises=KnownGap, strict=True, reason=reason)


# --- transitional seams ----------------------------------------------------------------------
# The only places these tests reach below the public API. A refactor that moves them changes
# these lines; the tests stay as they are.

@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    monkeypatch.setattr(saturate.core, "TICK_S", TICK)  # controller ticks fast enough to observe


@pytest.fixture
def no_retries(monkeypatch):
    monkeypatch.setattr(saturate.transport, "RETRY_ACTIVE", False)


def http_client_closed(client: AdaptiveClient) -> bool:
    return client._client.is_closed


async def _gated(results, gate: asyncio.Event):
    """Pass results on, but after the first one wait for `gate`: a consumer that is busy while
    the pipeline keeps completing rows behind it."""
    try:
        first = True
        async for done in results:
            yield done
            if first:
                first = False
                await gate.wait()
    finally:
        await results.aclose()


async def pipeline(endpoint, data, sink, *, window=None, to_request=None, parse=None, clients=None,
                   gate: asyncio.Event | None = None, **kw):
    """The functional layer end to end: AdaptiveClient -> through -> drain. With `gate`, the
    consumer stalls after the first result until the gate opens."""
    async with AdaptiveClient(endpoint, window=window or Fixed(4), signal_source="none") as client:
        if clients is not None:
            clients.append(client)
        results = through(client, stream(data), to_request or default_to_request, parse or default_parse,
                          **kw)
        if gate is not None:
            results = _gated(results, gate)
        return await drain(results, sink)


# --- helpers ---------------------------------------------------------------------------------

@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


def default_to_request(row):
    return {"model": "stub", "messages": [{"role": "user", "content": row["text"]}]}


def default_parse(row, resp):
    return {"reply": resp["choices"][0]["message"]["content"]}


def content(request: dict) -> str:
    """The row text a request carries (stub-side)."""
    return str((request.get("messages") or [{}])[-1].get("content", ""))


def rows(n, start=0):
    return [(f"r{i}", {"text": f"row {i}"}) for i in range(start, start + n)]


def on_disk(path) -> set[str]:
    return ParquetSink(str(path)).existing_ids()


async def until(condition, what: str, timeout: float = 10.0) -> None:
    """Wait for `condition()`; the timeout is a deadlock guard, not part of the scenario."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def _new_threads(before: set[int], include_runtime: bool) -> list[threading.Thread]:
    """Non-daemon threads started since `before`. The loop's own executor workers (asyncio_N)
    belong to the runtime until the loop shuts its executor down."""
    return [t for t in threading.enumerate()
            if t.ident not in before and not t.daemon and t.is_alive()
            and (include_runtime or not t.name.startswith("asyncio_"))]


def _wait_for_threads(before: set[int], include_runtime: bool, timeout: float = 2.0) -> list:
    deadline = time.monotonic() + timeout
    while _new_threads(before, include_runtime) and time.monotonic() < deadline:
        time.sleep(0.02)
    return _new_threads(before, include_runtime)


def run_checked(make_coro, timeout: float = 30.0):
    """Run `make_coro()` on a fresh loop that stays open afterwards, then check that nothing
    outlived it, before any cleanup of ours could hide it: no unfinished task, no library thread
    still alive (within 2 s), no error the loop was left to handle (a crashed task nobody
    awaited), and after the loop's executor shuts down, no thread at all. Then re-raise the
    coroutine's own exception, so a test can expect an error and still require a clean teardown."""
    before = {t.ident for t in threading.enumerate()}
    loop = asyncio.new_event_loop()
    loop_errors: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    error: BaseException | None = None
    result = None
    leaks: list[str] = []
    try:
        try:
            result = loop.run_until_complete(asyncio.wait_for(make_coro(), timeout))
        except BaseException as e:  # noqa: BLE001 — re-raised below, after the leak checks
            error = e
        loop.run_until_complete(asyncio.sleep(0.05))  # let done-callbacks and closes settle
        stray = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if stray:
            leaks.append(f"tasks outlived the run: {stray}")
        threads = _wait_for_threads(before, include_runtime=False)
        if threads:
            leaks.append(f"threads outlived the run: {threads}")
        gc.collect()  # a crashed task nobody awaited reports itself when it is collected
        if loop_errors:
            leaks.append(f"errors left to the loop: {[c.get('message') for c in loop_errors]}")
    finally:
        for t in asyncio.all_tasks(loop):
            t.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()
    threads = _wait_for_threads(before, include_runtime=True)
    if threads:
        leaks.append(f"threads outlived the loop: {threads}")
    if leaks:
        raise AssertionError("; ".join(leaks))
    if error is not None:
        raise error
    return result


class RecordingSink:
    """A duck-typed sink that remembers what it was given, and can be told to fail."""

    def __init__(self, fail_append_at: int | None = None, fail_flush: bool = False):
        self.appended: list[dict] = []
        self.flushed = 0
        self.fail_append_at, self.fail_flush = fail_append_at, fail_flush

    def existing_ids(self, retry_errors=False):
        return set()

    def append(self, record):
        if self.fail_append_at is not None and len(self.appended) == self.fail_append_at:
            raise SinkBoom("append failed")
        self.appended.append(record)

    def flush(self):
        if self.fail_flush:
            raise OSError("flush failed too")
        self.flushed += 1


class SinkBoom(RuntimeError):
    pass


class SourceBoom(RuntimeError):
    pass


# --- failure policy --------------------------------------------------------------------------

def test_source_failure_lands_the_submitted_rows_then_raises(stub, tmp_path):
    stub.latency_s = 0.01

    def source():
        yield from rows(10)
        raise SourceBoom("source died at row 10")

    with pytest.raises(SourceBoom):
        run_checked(lambda: pipeline(stub.endpoint, source(), ParquetSink(str(tmp_path), flush_every=1000)))
    assert on_disk(tmp_path) == {f"r{i}" for i in range(10)}


def test_fatal_cancels_in_flight_rows_instead_of_waiting_for_them(stub, tmp_path):
    """r0 is generating (5 s) when r1's to_request raises the fatal: the run ends now."""
    r0_arrived = threading.Event()

    def latency(request):
        if content(request) == "row 0":
            r0_arrived.set()
            return 5.0
        return 0.0

    stub.latency_for = latency

    def to_request_fatal_on_r1(row):
        if row["text"] == "row 1":
            r0_arrived.wait(10)  # in the prepare thread: r0 is in flight before the fatal
            raise FatalTransportError("breaker gave up")
        return default_to_request(row)

    t = time.monotonic()
    with pytest.raises(FatalTransportError):
        run_checked(lambda: pipeline(stub.endpoint, rows(2), ParquetSink(str(tmp_path)),
                                     to_request=to_request_fatal_on_r1, prepare_workers=1))
    assert r0_arrived.is_set()
    assert time.monotonic() - t < 3.0  # not the 5 s r0 would have taken
    assert on_disk(tmp_path) == set()  # r0 was cancelled, never completed


@known_gap("#35: results already queued behind a fatal error are discarded")
def test_fatal_lands_every_row_that_completed_before_it(stub, tmp_path):
    """r2's to_request raises the fatal once r1 has reached the server; r1 answers 1 s later,
    so it completes behind the fatal. The consumer is busy until r1 is parsed. r1 was paid for,
    so it must land."""
    r1_arrived = threading.Event()

    def latency(request):
        if content(request) == "row 1":
            r1_arrived.set()
            return 1.0
        return 0.0

    stub.latency_for = latency

    def to_request_fatal_on_r2(row):
        if row["text"] == "row 2":
            r1_arrived.wait(10)
            raise FatalTransportError("breaker gave up")
        return default_to_request(row)

    parsed: list[str] = []

    async def go():
        r1_parsed = asyncio.Event()

        def recording_parse(row, resp):
            parsed.append(f"r{row['text'].split()[-1]}")
            if row["text"] == "row 1":
                r1_parsed.set()
            return default_parse(row, resp)

        await pipeline(stub.endpoint, rows(3), ParquetSink(str(tmp_path), flush_every=1000),
                       to_request=to_request_fatal_on_r2, parse=recording_parse, prepare_workers=1,
                       gate=r1_parsed)

    with pytest.raises(FatalTransportError):
        run_checked(go)
    assert "r1" in parsed  # the scenario happened: r1 completed
    gap(set(parsed) <= on_disk(tmp_path), f"parsed {sorted(parsed)}, on disk {sorted(on_disk(tmp_path))}")


def test_fail_fast_stores_nothing_when_every_early_row_fails(stub, tmp_path):
    stub.status_for = lambda request: 400
    with pytest.raises(FatalTransportError, match="first 4 rows all failed"):
        run_checked(lambda: pipeline(stub.endpoint, rows(20), ParquetSink(str(tmp_path)),
                                     window=Fixed(1), fail_fast=4))
    assert on_disk(tmp_path) == set()


def test_fail_fast_releases_the_held_errors_once_a_row_succeeds(stub, tmp_path):
    """Errors before the first success are held, not dropped: when a success arrives they all
    land as ordinary error rows."""
    early = {"row 0", "row 1", "row 2"}
    stub.status_for = lambda request: 400 if content(request) in early else 200
    stats = run_checked(lambda: pipeline(stub.endpoint, rows(10), ParquetSink(str(tmp_path)),
                                         window=Fixed(1), fail_fast=8))
    assert (stats.rows_processed, stats.rows_failed) == (7, 3)
    sink = ParquetSink(str(tmp_path))
    assert sink.existing_ids() == {f"r{i}" for i in range(10)}
    assert sink.existing_ids(retry_errors=True) == {f"r{i}" for i in range(3, 10)}


def test_a_sink_failure_raises_its_own_error_and_stops_the_requests(stub):
    stub.latency_s = 0.01
    sink = RecordingSink(fail_append_at=3)
    with pytest.raises(SinkBoom):
        run_checked(lambda: pipeline(stub.endpoint, rows(500), sink))
    seen = stub.requests
    time.sleep(0.2)  # negative check: nothing keeps feeding the endpoint
    assert stub.requests == seen < 500
    assert sink.flushed == 1


def test_a_sink_failure_is_not_masked_by_a_failing_flush(stub):
    """A sink that failed once often fails its flush too: the first error is the one raised,
    and the pipeline still stops (no worker outlives the run)."""
    stub.latency_s = 0.01
    sink = RecordingSink(fail_append_at=3, fail_flush=True)
    with pytest.raises(SinkBoom):
        run_checked(lambda: pipeline(stub.endpoint, rows(500), sink))


@known_gap("#35: on cancel, results already completed but not yet consumed are discarded")
def test_caller_cancellation_lands_the_completed_rows_and_cleans_up(stub, tmp_path):
    """The consumer stalls after the first row while five more complete behind it; then the
    caller cancels. Every completed row was paid for, so every one must land; and the
    cancellation must leave nothing running and the HTTP client closed."""
    clients: list[AdaptiveClient] = []
    parsed: list[str] = []

    async def go():
        never = asyncio.Event()

        def recording_parse(row, resp):
            parsed.append(f"r{row['text'].split()[-1]}")
            return default_parse(row, resp)

        sink = ParquetSink(str(tmp_path), flush_every=1000)
        task = asyncio.ensure_future(pipeline(stub.endpoint, rows(200), sink, parse=recording_parse,
                                              clients=clients, gate=never))
        await until(lambda: len(parsed) >= 6, "six rows parsed")
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        run_checked(go)
    assert http_client_closed(clients[0])
    gap(set(parsed) <= on_disk(tmp_path), f"{len(parsed)} parsed, {len(on_disk(tmp_path))} on disk")


# --- the transport and the sink ------------------------------------------------

def test_any_2xx_is_a_success(stub, tmp_path, no_retries):
    stub.status_for = lambda request: 201
    stats = run_checked(lambda: pipeline(stub.endpoint, rows(3), ParquetSink(str(tmp_path)), fail_fast=0))
    gap((stats.rows_processed, stats.rows_failed) == (3, 0), f"{stats.rows_failed} of 3 rows failed")


@pytest.mark.parametrize("status, body", [(200, b"<html>down for maintenance</html>"),
                                          (200, b"[1, 2]"),  # JSON, but not an object
                                          (204, b"")])
def test_a_2xx_without_a_json_object_is_a_server_error_not_a_success(stub, no_retries, status, body):
    stub.status_for = lambda request: status
    stub.raw_for = lambda request: body
    ticks: list[dict] = []

    async def go():
        async with AdaptiveClient(stub.endpoint, window=Fixed(1), signal_source="none",
                                  on_tick=ticks.append) as client:
            try:
                got = await client.post(default_to_request({"text": "hi"}))
            except ValueError as e:
                raise KnownGap(f"post raised {e!r} instead of returning the row's error") from e
            await until(lambda: len(ticks) >= 3, "three ticks")
            return got

    body, err = run_checked(go)
    gap(body is None and err is not None and not err.startswith("client:"), f"got error {err!r}")
    gap(sum(t["ok"] for t in ticks) == 0, "the controller counted a success")


@known_gap("#29: the age flush runs only when another row arrives")
def test_a_buffered_row_is_flushed_by_age_without_new_arrivals(tmp_path):
    async def go():
        appended = asyncio.Event()

        async def one_row_then_quiet():
            yield Done("r0", {"text": "row 0"}, {"reply": "x"}, None)
            appended.set()  # drain asks for the next row only after appending this one
            await asyncio.sleep(3.0)

        sink = ParquetSink(str(tmp_path), flush_every=1000, flush_interval_s=0.1)
        task = asyncio.ensure_future(drain(one_row_then_quiet(), sink))
        await until(appended.is_set, "the row appended")
        await asyncio.sleep(1.0)  # ten flush intervals with no new row
        landed = list(tmp_path.glob("part-*.parquet"))
        await task
        return landed

    landed = run_checked(go)
    gap(bool(landed), "no part written while the stream was quiet")
    assert pq.read_table(landed[0])["id"].to_pylist() == ["r0"]


# --- concurrency primitives ------------------------------------------------------------------

class ShrinkTo:
    """A controller that starts at `initial` and asks for `n` from the first tick on."""

    last_reason = "test"

    def __init__(self, initial: int, n: int):
        self.initial, self.n = initial, n

    def decide(self, obs, limit):
        return self.n


def test_shrinking_the_window_holds_admissions_until_inflight_drops_below_it():
    """Four holders at once under limit 4; the limit drops to 2 while they hold; releasing them
    one by one admits a waiter only once fewer than 2 are in flight."""
    ticks: list[dict] = []

    async def go():
        holding: list[int] = []
        releases = {i: asyncio.Event() for i in range(6)}

        async def holder(limiter, i):
            async with limiter.slot():
                holding.append(i)
                await releases[i].wait()
                holding.remove(i)

        async with AdaptiveLimiter(ShrinkTo(4, 2), on_tick=ticks.append) as limiter:
            tasks = [asyncio.ensure_future(holder(limiter, i)) for i in range(6)]
            await until(lambda: len(holding) == 4, "four holders at once")
            first_four = list(holding)
            await until(lambda: any(t["limit"] == 2 for t in ticks), "the shrink to 2")
            await asyncio.sleep(0.05)
            assert len(holding) == 4  # a shrink never evicts a holder

            waiting = [i for i in range(6) if i not in first_four]
            for released, i in enumerate(first_four[:3], start=1):
                releases[i].set()
                if released < 3:  # 3, then 2 in flight: not below the limit, nobody admitted
                    await asyncio.sleep(0.05)
                    assert not any(w in holding for w in waiting), (released, holding)
            await until(lambda: any(w in holding for w in waiting), "one waiter admitted")
            await asyncio.sleep(0.05)
            assert len(holding) == 2  # 1 left of the first four + exactly one waiter

            for e in releases.values():
                e.set()
            await asyncio.gather(*tasks)

    run_checked(go)


def test_cancellation_churn_never_leaks_a_slot():
    """Holders and waiters cancelled at random points: afterwards the full window is still
    usable, and no more than the window can hold at once."""
    rng = random.Random(7)

    async def hold(limiter, seconds):
        async with limiter.slot():
            await asyncio.sleep(seconds)

    async def go():
        async with AdaptiveLimiter(Fixed(3)) as limiter:
            tasks = [asyncio.ensure_future(hold(limiter, rng.uniform(0, 0.02))) for _ in range(200)]
            for _ in range(100):
                await asyncio.sleep(rng.uniform(0, 0.002))
                rng.choice(tasks).cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            inside: list[int] = []
            release = asyncio.Event()

            async def occupy(i):
                async with limiter.slot():
                    inside.append(i)
                    await release.wait()

            occupants = [asyncio.ensure_future(occupy(i)) for i in range(4)]
            await until(lambda: len(inside) == 3, "all three slots back")
            await asyncio.sleep(0.05)
            assert len(inside) == 3  # and the fourth waits
            release.set()
            await asyncio.gather(*occupants)

    run_checked(go)


def test_the_controller_hears_backpressure_while_the_row_is_still_retrying(stub):
    """Pressure is reported on the first failed attempt, not when the row gives up: a row
    backing off for minutes must not hide an overload from the controller meanwhile."""
    stub.status_for = lambda request: 429
    ticks: list[dict] = []

    async def go():
        async with AdaptiveClient(stub.endpoint, window=Fixed(2), signal_source="none",
                                  on_tick=ticks.append) as client:
            post = asyncio.ensure_future(client.post(default_to_request({"text": "hi"})))
            await until(lambda: any(t["bp"] for t in ticks), "a tick with backpressure")
            still_retrying = not post.done()
            post.cancel()
            await asyncio.gather(post, return_exceptions=True)
            return still_retrying

    assert run_checked(go)


class CountingClient(httpx.AsyncClient):
    """Counts the breaker's probes in flight on the client side. The stub's own count would
    also include a cancelled probe the server is still answering, which the client has
    already dropped."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.inflight = self.peak = 0

    async def post(self, *a, **kw):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            return await super().post(*a, **kw)
        finally:
            self.inflight -= 1


def test_concurrent_callers_elect_one_prober_and_pass_only_when_it_closes(stub):
    healthy = threading.Event()
    stub.status_for = lambda request: 200 if healthy.is_set() else 503
    stub.latency_s = 0.01

    async def go():
        breaker = Breaker(threshold=1, probe_interval=0.01)
        breaker.fail()
        async with CountingClient() as client:
            url = f"{stub.endpoint}/chat/completions"
            gates = [asyncio.ensure_future(breaker.gate(client, url, "stub")) for _ in range(10)]
            await until(lambda: stub.probes >= 3, "three failed probes")
            assert not any(g.done() for g in gates)  # nobody passes an open circuit
            healthy.set()
            await asyncio.wait_for(asyncio.gather(*gates), 5.0)
        return breaker, client.peak

    breaker, peak = run_checked(go)
    assert breaker.opens == 1
    assert peak == 1  # one probe at a time, never ten


def test_a_cancelled_prober_hands_over_and_the_waiters_pass_only_when_it_closes(stub):
    healthy = threading.Event()
    stub.status_for = lambda request: 200 if healthy.is_set() else 503
    stub.latency_s = 0.01

    async def go():
        breaker = Breaker(threshold=1, probe_interval=0.01)
        breaker.fail()
        async with CountingClient() as client:
            url = f"{stub.endpoint}/chat/completions"
            prober = asyncio.ensure_future(breaker.gate(client, url, "stub"))
            await until(lambda: stub.probes >= 1, "the first caller probing")
            waiters = [asyncio.ensure_future(breaker.gate(client, url, "stub")) for _ in range(2)]
            await asyncio.sleep(0.02)
            prober.cancel()
            await asyncio.gather(prober, return_exceptions=True)
            at_cancel = stub.probes
            await until(lambda: stub.probes >= at_cancel + 2, "a waiter probing in its place")
            assert not any(w.done() for w in waiters)  # still open: nobody passes
            healthy.set()
            await asyncio.wait_for(asyncio.gather(*waiters), 5.0)
        return breaker, client.peak

    breaker, peak = run_checked(go)
    assert breaker.opens == 2  # re-opened by a waiter, not left unattended
    assert peak == 1
