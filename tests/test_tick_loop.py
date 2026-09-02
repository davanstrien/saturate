"""The tick loop feeds the controller honest observations and records why the
window moved: zero delivered tokens is a reading, not a missing signal; every
tick carries the controller's reason and the observed request latency.

Ticks are driven directly (`limiter._tick()`), never by the clock."""

import asyncio

from saturate import AdaptiveLimiter, Auto, Fixed, Stats
from saturate.controller import Obs
from saturate.telemetry import cut_reasons


class Recorder:
    """A controller that keeps every observation it was handed."""

    initial = 1

    def __init__(self):
        self.seen: list[Obs] = []
        self.last_reason = "hold"

    def decide(self, obs, limit):
        self.seen.append(obs)
        self.last_reason = "cut:stall" if obs.inflight >= limit and obs.successes == 0 else "hold"
        return limit


def run(coro):
    return asyncio.run(coro)


def test_zero_tokens_with_requests_in_flight_is_a_reading():
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=ctrl)
        await lim._tick()  # idle: nothing in flight, nothing observable
        async with lim.slot():
            lim.observe(ok=True, tokens=100)
            await lim._tick()  # tokens delivered this tick
            await lim._tick()  # a request sits in flight delivering nothing
        await lim._tick()
        return lim

    lim = run(go())
    idle, delivered, stalled = ctrl.seen[0], ctrl.seen[1], ctrl.seen[2]
    assert idle.tok_s is None and idle.inflight == 0
    assert delivered.tok_s > 0 and delivered.successes == 1
    assert stalled.tok_s == 0.0 and stalled.inflight == 1 and stalled.successes == 0
    assert [t["reason"] for t in lim.ticks] == ["hold", "hold", "cut:stall", "hold"]


def test_endpoint_without_token_usage_has_no_throughput_signal():
    """Completions that never report tokens leave tok_s None: the controller
    falls back to queue gauges rather than reading a permanent plateau."""
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=ctrl)
        async with lim.slot():
            lim.observe(ok=True, tokens=0)
            await lim._tick()
            await lim._tick()

    run(go())
    assert [o.tok_s for o in ctrl.seen] == [None, None]


def test_tick_carries_request_latency_and_the_oldest_inflight_age():
    ctrl = Recorder()

    async def go():
        lim = AdaptiveLimiter(window=Auto(initial=4))
        lim.controller = ctrl
        await lim._tick()
        assert ctrl.seen[-1].latency_s is None and ctrl.seen[-1].oldest_s is None
        async with lim.slot():
            pass
        async with lim.slot():
            await lim._tick()
        return lim

    lim = run(go())
    obs = ctrl.seen[-1]
    assert obs.latency_s is not None and obs.latency_s >= 0.0 and obs.oldest_s >= 0.0
    assert obs.tick_s is not None and obs.tick_s > 0
    assert lim.ticks[-1]["latency_s"] == round(obs.latency_s, 3)
    assert lim.oldest_s is None  # released


def test_latency_window_covers_the_largest_window():
    assert AdaptiveLimiter(window=Fixed(4))._latencies.maxlen == 64
    assert AdaptiveLimiter(window=Auto(max_limit=1024))._latencies.maxlen == 1024


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
