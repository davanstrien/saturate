"""The tick loop feeds the controller honest observations and records why the
window moved: zero delivered tokens is a reading, not a missing signal; every
tick carries the controller's reason and the observed request latency."""

import asyncio

import saturate.core as core
from saturate import AdaptiveLimiter, Fixed, Stats
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


def test_zero_tokens_with_requests_in_flight_is_a_reading(monkeypatch):
    monkeypatch.setattr(core, "TICK_S", 0.01)
    ctrl = Recorder()

    async def go():
        async with AdaptiveLimiter(window=ctrl) as lim:
            await asyncio.sleep(0.03)  # idle: nothing in flight, nothing observable
            async with lim.slot():
                lim.observe(ok=True, tokens=100)  # tokens have been seen once...
                await asyncio.sleep(0.05)  # ...then a request sits in flight delivering nothing
            await asyncio.sleep(0.03)
        return lim

    lim = run(go())
    idle = [o for o in ctrl.seen if o.inflight == 0 and o.successes == 0]
    stalled = [o for o in ctrl.seen if o.inflight > 0 and o.successes == 0]
    assert idle and all(o.tok_s is None for o in idle)
    assert stalled and all(o.tok_s == 0.0 for o in stalled), [o.tok_s for o in stalled]
    assert any(t["reason"] == "cut:stall" for t in lim.ticks) and all("reason" in t for t in lim.ticks)


def test_endpoint_without_token_usage_has_no_throughput_signal(monkeypatch):
    """Completions that never report tokens leave tok_s None: the controller
    falls back to queue gauges rather than reading a permanent plateau."""
    monkeypatch.setattr(core, "TICK_S", 0.01)
    ctrl = Recorder()

    async def go():
        async with AdaptiveLimiter(window=ctrl) as lim:
            async with lim.slot():
                lim.observe(ok=True, tokens=0)
                await asyncio.sleep(0.04)

    run(go())
    assert all(o.tok_s is None for o in ctrl.seen)


def test_tick_records_request_latency(monkeypatch):
    monkeypatch.setattr(core, "TICK_S", 0.01)
    ctrl = Recorder()

    async def go():
        async with AdaptiveLimiter(window=ctrl) as lim:
            assert ctrl.seen == [] or ctrl.seen[-1].latency_s is None
            for _ in range(3):
                async with lim.slot():
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.03)
        return lim

    lim = run(go())
    assert ctrl.seen[-1].latency_s is not None and 0.015 <= ctrl.seen[-1].latency_s < 0.2
    assert lim.ticks[-1]["latency_s"] == round(ctrl.seen[-1].latency_s, 3)


def test_fixed_controller_ticks_carry_hold(monkeypatch):
    monkeypatch.setattr(core, "TICK_S", 0.01)

    async def go():
        async with AdaptiveLimiter(window=Fixed(2)) as lim:
            await asyncio.sleep(0.03)
        return lim

    lim = run(go())
    assert lim.ticks and {t["reason"] for t in lim.ticks} == {"hold"}


def test_cut_reasons_counted_into_stats():
    ticks = [{"reason": r} for r in ("hold", "cut:bp", "hold:cooldown", "cut:bp", "cut:stall", "grow")]
    assert cut_reasons(ticks) == {"cut:bp": 2, "cut:stall": 1}
    assert cut_reasons([]) == {}
    stats = Stats(cut_reasons=cut_reasons(ticks))
    assert '"cut_reasons": {"cut:bp": 2, "cut:stall": 1}' in stats.to_json()
