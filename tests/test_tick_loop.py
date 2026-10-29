"""The tick loop feeds the controller honest observations and records why the
window moved: zero delivered tokens is a reading, not a missing signal; every
tick carries the controller's reason and the observed request latency.

Ticks are driven directly (`limiter._tick()`), never by the clock."""

import asyncio
from typing import cast

import pytest

from saturate import AdaptiveLimiter, Auto, Fixed, Stats
from saturate.controller import Obs
from saturate.telemetry import cut_reasons
from saturate.transport import Attempt


class Recorder:
    """A controller that keeps every observation it was handed and names each tick. It has the
    `initial`, `decide` and `last_reason` the limiter uses, so the tests cast it to Fixed."""

    initial = 1

    def __init__(self):
        self.seen: list[Obs] = []
        self.last_reason = "hold"

    def decide(self, obs, limit):
        self.seen.append(obs)
        self.last_reason = f"tick:{len(self.seen)}"
        return limit


def run(coro):
    return asyncio.run(coro)


@pytest.mark.internals
def test_zero_tokens_with_requests_in_flight_is_a_reading():
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=cast(Fixed, ctrl))
        await lim._tick()  # idle: nothing in flight, nothing observable
        async with lim.slot():
            lim.observe(Attempt(ok=True, tokens=100))
            await lim._tick()  # tokens delivered this tick
            await lim._tick()  # a request sits in flight delivering nothing
        await lim._tick()
        return lim

    lim = run(go())
    idle, delivered, stalled = ctrl.seen[0], ctrl.seen[1], ctrl.seen[2]
    assert idle.tok_s is None and idle.inflight == 0
    assert delivered.tok_s is not None and delivered.tok_s > 0 and delivered.successes == 1
    assert stalled.tok_s == 0.0 and stalled.inflight == 1 and stalled.successes == 0
    assert [t["reason"] for t in lim.ticks] == ["tick:1", "tick:2", "tick:3", "tick:4"]


@pytest.mark.internals
def test_endpoint_without_token_usage_has_no_throughput_signal():
    """Completions that never report tokens leave tok_s None: the controller
    falls back to queue gauges rather than reading a permanent plateau."""
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=cast(Fixed, ctrl))
        async with lim.slot():
            lim.observe(Attempt(ok=True, tokens=0))
            await lim._tick()
            await lim._tick()

    run(go())
    assert [o.tok_s for o in ctrl.seen] == [None, None]


@pytest.mark.internals
def test_tick_carries_request_latency_and_the_oldest_inflight_age():
    """Latency is the p50 of successful attempts reported by the transport (or
    observe()); the oldest in-flight age comes from slot admission times."""
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=Auto(initial=4))
        lim.controller = cast(Fixed, ctrl)
        await lim._tick()
        assert ctrl.seen[-1].latency_s is None and ctrl.seen[-1].oldest_s is None
        async with lim.slot():
            lim.observe(Attempt(ok=True, tokens=10, latency_s=0.25))
        async with lim.slot():
            lim.observe(Attempt(ok=True, latency_s=0.75))  # what call_endpoint reports on a 200
            await lim._tick()
        return lim

    lim = run(go())
    obs = ctrl.seen[-1]
    assert obs.latency_s == 0.5  # median of the two attempts
    assert obs.oldest_s is not None and obs.oldest_s >= 0.0
    assert obs.tick_s is not None and obs.tick_s > 0
    assert lim.ticks[-1]["latency_s"] == 0.5 and lim._counts.latencies == []
    assert lim.oldest_s is None  # released


@pytest.mark.internals
def test_slot_exit_always_releases():
    """A second exit (or a lost admission record) must never leak an in-flight count."""

    async def go():
        lim = AdaptiveLimiter(window=Fixed(2))
        slot = lim.slot()
        await slot.__aenter__()
        lim._admitted.clear()
        await slot.__aexit__(None, None, None)
        return lim.inflight

    assert run(go()) == 0


@pytest.mark.internals
def test_fixed_controller_ticks_carry_hold():
    async def go():
        lim = AdaptiveLimiter(window=Fixed(2))
        await lim._tick()
        await lim._tick()
        return lim

    lim = run(go())
    assert {t["reason"] for t in lim.ticks} == {"hold"}


def test_cut_reasons_counted_into_stats():
    ticks = [{"reason": r} for r in ("hold", "cut:bp", "hold:cooldown", "cut:bp", "cut:stall", "grow")]
    assert cut_reasons(ticks) == {"cut:bp": 2, "cut:stall": 1}
    assert cut_reasons([]) == {}
    stats = Stats(cut_reasons=cut_reasons(ticks))
    assert '"cut_reasons": {"cut:bp": 2, "cut:stall": 1}' in stats.to_json()


class SuspendedScrape:
    """Signals whose read() waits until released: the window in which a scrape is in flight."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def read(self):
        self.started.set()
        await self.release.wait()
        return None


@pytest.mark.internals
def test_reports_during_a_scrape_count_in_that_tick():
    """The counters are read after the scrape, with no await in between: pressure reported while
    the scrape is in flight reaches this decision, not the next one."""
    ctrl = Recorder()

    async def go():
        signals = SuspendedScrape()
        lim = AdaptiveLimiter(window=Auto(initial=4), signals=signals)
        lim.controller = cast(Fixed, ctrl)
        tick = asyncio.ensure_future(lim._tick())
        await signals.started.wait()
        lim.observe(Attempt(pressure=True))
        lim.observe(Attempt(ok=True, tokens=5))
        lim.observe(Attempt(retry=True))
        signals.release.set()
        await tick
        return lim

    lim = run(go())
    assert (ctrl.seen[-1].backpressure, ctrl.seen[-1].successes) == (1, 1)
    assert lim.ticks[-1]["retries"] == 1


@pytest.mark.internals
def test_a_cancelled_scrape_loses_no_reports():
    ctrl = Recorder()

    async def go():
        signals = SuspendedScrape()
        lim = AdaptiveLimiter(window=Auto(initial=4), signals=signals)
        lim.controller = cast(Fixed, ctrl)
        lim.observe(Attempt(pressure=True))
        tick = asyncio.ensure_future(lim._tick())
        await signals.started.wait()
        tick.cancel()
        await asyncio.gather(tick, return_exceptions=True)
        signals.release.set()
        await lim._tick()
        return lim

    run(go())
    assert ctrl.seen[-1].backpressure == 1
