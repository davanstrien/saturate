"""The behaviour contract the 0.3 refactor must keep: what each kind of failure does to the run,
and what the concurrency primitives guarantee under cancellation and resizing.

These tests use public entry points only (pump, AdaptiveClient, AdaptiveLimiter, through,
drain, ParquetSink, Breaker) so they survive the rewrite of the plumbing underneath. The one
seam the rewrite changes on purpose is the shape of through(); every pipeline test goes
through `pipeline()` below, so that change touches one helper, not the tests.

Pipeline tests run on a loop that outlives the coroutine (`run_checked`): asyncio.run cancels
stray tasks at shutdown, which would hide exactly the leaks these tests look for.

Failure policy (one test per row):

| failure                         | behaviour                                                    |
|---------------------------------|--------------------------------------------------------------|
| the source raises               | rows already submitted finish and land, then it raises       |
| fatal transport (breaker dead)  | in-flight rows are cancelled, completed rows land, it raises |
| fail_fast trips                 | the held rows are not stored, it raises                      |
| the sink raises                 | everything stops, the sink's own error raises                |
| the caller cancels              | everything stops, completed rows land                        |

xfail(strict=True) marks a known gap: the test flips to a pass when the fix lands, and
strict makes that flip fail loudly until the marker is removed.
"""

import asyncio
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


@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    monkeypatch.setattr(saturate.core, "TICK_S", TICK)


def to_request(row):
    return {"model": "stub", "messages": [{"role": "user", "content": row["text"]}]}


def parse(row, resp):
    return {"reply": resp["choices"][0]["message"]["content"]}


def rows(n, start=0):
    return [(f"r{i}", {"text": f"row {i}"}) for i in range(start, start + n)]


def on_disk(path) -> set[str]:
    return ParquetSink(str(path)).existing_ids()


# --- harness --------------------------------------------------------------------------------

def _new_non_daemon_threads(before: set[int]) -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.ident not in before and not t.daemon and t.is_alive()]


def run_checked(make_coro, timeout: float = 30.0):
    """Run `make_coro()` on a fresh loop that stays open afterwards, then assert nothing outlived
    it: no unfinished task, and (within 2 s) no new non-daemon thread. Re-raises the
    coroutine's own exception after the checks, so a test can both expect an error and
    require a clean teardown."""
    before = {t.ident for t in threading.enumerate()}
    loop = asyncio.new_event_loop()
    try:
        error: BaseException | None = None
        result = None
        try:
            result = loop.run_until_complete(asyncio.wait_for(make_coro(), timeout))
        except BaseException as e:  # noqa: BLE001 — re-raised below, after the leak checks
            error = e
        loop.run_until_complete(asyncio.sleep(0.05))  # let done-callbacks and closes settle
        stray = [t for t in asyncio.all_tasks(loop) if not t.done()]
        assert not stray, f"tasks outlived the run: {stray}"
    finally:
        for t in asyncio.all_tasks(loop):
            t.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
    deadline = time.monotonic() + 2.0
    while _new_non_daemon_threads(before) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _new_non_daemon_threads(before), f"threads outlived the run: {_new_non_daemon_threads(before)}"
    if error is not None:
        raise error
    return result


async def pipeline(endpoint, data, sink, *, window=None, to_request=to_request, parse=parse,
                   clients=None, **kw):
    """The functional layer end to end: AdaptiveClient -> through -> drain."""
    async with AdaptiveClient(endpoint, window=window or Fixed(4), signal_source="none") as client:
        if clients is not None:
            clients.append(client)
        return await drain(through(client, stream(data), to_request, parse, **kw), sink)


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
    """A fatal error ends the run now: a row still generating (5 s here) is not waited for."""
    stub.latency_s = 5.0

    def to_request_fatal_on_r1(row):
        if row["text"] == "row 1":
            time.sleep(0.05)  # in the prepare thread: r0 is already in flight
            raise FatalTransportError("breaker gave up")
        return to_request(row)

    t = time.monotonic()
    with pytest.raises(FatalTransportError):
        run_checked(lambda: pipeline(stub.endpoint, rows(2), ParquetSink(str(tmp_path)),
                                     to_request=to_request_fatal_on_r1, prepare_workers=1))
    assert time.monotonic() - t < 2.0
    assert on_disk(tmp_path) == set()  # r0 was cancelled, never completed


@pytest.mark.xfail(strict=True, reason="#35: results already queued behind a fatal error are discarded")
def test_fatal_lands_every_row_that_completed_before_it(stub):
    """r2's to_request raises the fatal at ~50 ms; r1 completes at ~100 ms, behind the fatal in
    the results queue; the consumer is busy until ~300 ms. r1 was paid for, so it must land."""
    stub.latency_for = lambda request: 0.1 if "row 1" in str(request) else 0.0
    parsed: list[str] = []
    yielded: list[str] = []

    def to_request_fatal_on_r2(row):
        if row["text"] == "row 2":
            time.sleep(0.05)
            raise FatalTransportError("breaker gave up")
        return to_request(row)

    def recording_parse(row, resp):
        parsed.append(row["text"])
        return parse(row, resp)

    async def consume():
        async with AdaptiveClient(stub.endpoint, window=Fixed(4), signal_source="none") as client:
            results = through(client, stream(rows(3)), to_request_fatal_on_r2, recording_parse,
                              prepare_workers=1)
            try:
                async for done in results:
                    yielded.append(done.row["text"])
                    if len(yielded) == 1:
                        await asyncio.sleep(0.3)  # a slow consumer: r1 completes meanwhile
            finally:
                await results.aclose()

    with pytest.raises(FatalTransportError):
        run_checked(consume)
    assert "row 1" in parsed
    assert set(parsed) <= set(yielded)


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
    stub.status_for = lambda request: 400 if request["messages"][0]["content"] in early else 200
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
    time.sleep(0.2)
    assert stub.requests == seen < 500  # nothing kept feeding the endpoint after the failure
    assert sink.flushed == 1


@pytest.mark.xfail(strict=True, reason="finding 1: drain's finally flush masks the error, skips aclose")
def test_a_sink_failure_is_not_masked_by_a_failing_flush(stub):
    stub.latency_s = 0.01
    sink = RecordingSink(fail_append_at=3, fail_flush=True)
    with pytest.raises(SinkBoom):
        run_checked(lambda: pipeline(stub.endpoint, rows(500), sink))


def test_caller_cancellation_lands_the_completed_rows_and_cleans_up(stub, tmp_path):
    stub.latency_s = 0.05
    clients: list[AdaptiveClient] = []
    appended: list[str] = []

    class Recording(ParquetSink):
        def append(self, record):
            appended.append(record["id"])
            super().append(record)

    async def go():
        sink = Recording(str(tmp_path), flush_every=1000)
        task = asyncio.ensure_future(pipeline(stub.endpoint, rows(200), sink, clients=clients))
        await asyncio.sleep(0.4)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        run_checked(go)
    assert appended and on_disk(tmp_path) == set(appended)
    assert clients[0]._client.is_closed


# --- known gaps in the transport and the sink ------------------------------------------------

@pytest.mark.xfail(strict=True, reason="finding 2: a 2xx other than 200 is classified as a 5xx")
def test_any_2xx_is_a_success(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(saturate.transport, "RETRY_ACTIVE", False)
    stub.status_for = lambda request: 201
    stats = run_checked(lambda: pipeline(stub.endpoint, rows(3), ParquetSink(str(tmp_path)), fail_fast=0))
    assert (stats.rows_processed, stats.rows_failed) == (3, 0)


@pytest.mark.xfail(strict=True, reason="finding 3: a 200 whose body is not JSON counts as a success")
def test_a_200_that_is_not_json_is_a_server_error_not_a_success(stub):
    stub.raw_for = lambda request: b"<html>down for maintenance</html>"
    ticks: list[dict] = []

    async def go():
        async with AdaptiveClient(stub.endpoint, window=Fixed(1), signal_source="none",
                                  on_tick=ticks.append) as client:
            got = await client.post(to_request({"text": "hi"}))
            await asyncio.sleep(5 * TICK)
            return got

    body, err = run_checked(go)
    assert body is None and err is not None and not err.startswith("client:")
    assert sum(t["ok"] for t in ticks) == 0  # the controller never heard a success


@pytest.mark.xfail(strict=True, reason="#29: the age flush runs only when another row arrives")
def test_a_buffered_row_is_flushed_by_age_without_new_arrivals(tmp_path):
    async def one_row_then_quiet():
        yield Done("r0", {"text": "row 0"}, {"reply": "x"}, None)
        await asyncio.sleep(1.0)

    async def go():
        sink = ParquetSink(str(tmp_path), flush_every=1000, flush_interval_s=0.2)
        task = asyncio.ensure_future(drain(one_row_then_quiet(), sink))
        await asyncio.sleep(0.6)
        landed = list(tmp_path.glob("part-*.parquet"))
        await task
        return landed

    landed = run_checked(go)
    assert landed and pq.read_table(landed[0])["id"].to_pylist() == ["r0"]


# --- concurrency primitives ------------------------------------------------------------------

class ShrinkTo:
    """A controller that starts at `initial` and asks for `n` from the first tick on."""

    last_reason = "test"

    def __init__(self, initial: int, n: int):
        self.initial, self.n = initial, n

    def decide(self, obs, limit):
        return self.n


def test_shrinking_the_window_holds_new_admissions_until_inflight_drops_below_it():
    admitted_at: list[int] = []
    inflight = 0

    async def holder(limiter):
        nonlocal inflight
        async with limiter.slot():
            inflight += 1
            admitted_at.append(inflight)
            await asyncio.sleep(0.1)  # longer than a tick: the shrink lands while all four hold
            inflight -= 1

    async def go():
        async with AdaptiveLimiter(ShrinkTo(4, 2)) as limiter:
            await asyncio.gather(*(holder(limiter) for _ in range(10)))

    run_checked(go)
    assert max(admitted_at[:4]) <= 4
    assert max(admitted_at[4:]) <= 2, admitted_at


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

            entered = asyncio.Event()
            count = 0
            release = asyncio.Event()

            async def occupy():
                nonlocal count
                async with limiter.slot():
                    count += 1
                    if count == 3:
                        entered.set()
                    await release.wait()

            occupants = [asyncio.ensure_future(occupy()) for _ in range(4)]
            await asyncio.wait_for(entered.wait(), 1.0)  # all three slots came back
            await asyncio.sleep(0.05)
            assert count == 3  # and the fourth waits
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
            post = asyncio.ensure_future(client.post(to_request({"text": "hi"})))
            deadline = time.monotonic() + 3.0
            while not any(t["bp"] for t in ticks) and time.monotonic() < deadline:
                await asyncio.sleep(TICK)
            still_retrying = not post.done()
            post.cancel()
            await asyncio.gather(post, return_exceptions=True)
            return still_retrying

    assert run_checked(go)
    assert any(t["bp"] for t in ticks)


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


def test_concurrent_callers_elect_one_prober_and_all_pass_when_it_closes(stub):
    probes = 0

    def recovering(request):
        nonlocal probes
        probes += 1
        return 503 if probes <= 3 else 200

    stub.status_for = recovering
    stub.latency_s = 0.02

    async def go():
        breaker = Breaker(threshold=1, probe_interval=0.01)
        breaker.fail()
        async with CountingClient() as client:
            url = f"{stub.endpoint}/chat/completions"
            await asyncio.gather(*(breaker.gate(client, url, "stub") for _ in range(10)))
        return breaker, client.peak

    breaker, peak = run_checked(go)
    assert breaker.opens == 1
    assert peak == 1  # one probe at a time, never ten


def test_a_cancelled_prober_hands_over_and_the_waiters_still_pass(stub):
    stub.status_for = lambda request: 503
    stub.latency_s = 0.02

    async def go():
        breaker = Breaker(threshold=1, probe_interval=0.01)
        breaker.fail()
        async with CountingClient() as client:
            url = f"{stub.endpoint}/chat/completions"
            prober = asyncio.ensure_future(breaker.gate(client, url, "stub"))
            await asyncio.sleep(0.05)  # the first caller is probing
            waiters = [asyncio.ensure_future(breaker.gate(client, url, "stub")) for _ in range(2)]
            await asyncio.sleep(0.05)
            prober.cancel()
            await asyncio.gather(prober, return_exceptions=True)
            await asyncio.sleep(0.1)  # a waiter takes over the probing
            stub.status_for = lambda request: 200
            await asyncio.wait_for(asyncio.gather(*waiters), 2.0)
        return breaker, client.peak

    breaker, peak = run_checked(go)
    assert breaker.opens >= 2  # the circuit was re-opened by a waiter, not left unattended
    assert peak == 1
