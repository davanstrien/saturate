"""Auto against simulated engines: the share of the best fixed window's throughput it reaches,
how fast it gets there, and how far past the engine's capacity it goes."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sim_engine import SGLANG_LIKE, Engine, run, share_of_best

from saturate.controller import Auto

LINEAR = [(256, 200.0)]  # throughput grows with the window up to the engine's cap
CURVES = {"sglang-like": SGLANG_LIKE, "linear": LINEAR}


@pytest.mark.parametrize("gauges", [False, True], ids=["blind", "gauges"])
@pytest.mark.parametrize("curve", list(CURVES), ids=list(CURVES))
def test_auto_reaches_most_of_the_best_fixed_window(curve, gauges):
    """Over a long run Auto delivers at least 90% of what the best fixed window would. Blind
    mode stopped at window 37 on SGLang's flat stretch (32-64): about 40% of the best."""
    engine = Engine(CURVES[curve], gauges=gauges)
    ticks = run(Auto(), engine, 300)
    assert share_of_best(ticks[100:], engine, 512) >= 0.9, [t.limit for t in ticks[-10:]]


@pytest.mark.parametrize("gauges", [False, True], ids=["blind", "gauges"])
@pytest.mark.parametrize("curve", list(CURVES), ids=list(CURVES))
def test_auto_ramps_within_the_first_minute(curve, gauges):
    """A short run must not spend itself ramping: within 30 ticks (a minute) Auto delivers at
    least 60% of the best fixed window on average. Blind growth by an eighth per evaluation
    took about 18 evaluations to go from 16 to 128."""
    engine = Engine(CURVES[curve], gauges=gauges)
    ticks = run(Auto(), engine, 30)
    assert share_of_best(ticks, engine, 512) >= 0.6, [t.limit for t in ticks]


@pytest.mark.parametrize("gauges", [False, True], ids=["blind", "gauges"])
@pytest.mark.parametrize("curve", list(CURVES), ids=list(CURVES))
def test_auto_does_not_settle_far_past_the_engine_cap(curve, gauges):
    """Requests past the engine's cap only queue. Auto may cross the cap while it searches, but
    it settles within an eighth of it."""
    engine = Engine(CURVES[curve], gauges=gauges)
    ticks = run(Auto(), engine, 300)
    settled = [t.limit for t in ticks[200:]]
    assert max(settled) <= engine.cap * 9 // 8 + 1, settled[-10:]


@st.composite
def engines(draw):
    """An engine whose throughput never falls as the window grows, with a cap of 32 to 512
    requests and readings one or two ticks late, in blind mode. A flat stretch may be up to three
    times as wide as the window it starts at; blind trials reach 8x the window."""
    cap = draw(st.sampled_from([32, 64, 128, 256, 512]))
    window, rate, curve, flat = 8, draw(st.floats(5.0, 50.0)), [], False
    while window < cap:
        # flat about a third of the time, never twice in a row: one flat stretch at a time
        gain = draw(st.sampled_from([1.3, 1.6, 2.0] if flat else [1.0, 1.0, 1.3, 1.6, 2.0]))
        flat = gain == 1.0
        widen = draw(st.sampled_from([2, 3]))
        window, rate = min(cap, window * widen), rate * gain
        curve.append((window, rate))
    # blind only: gauge mode steps its probes by +8 (see the xfails below)
    return Engine(curve, cap=cap, gauges=False, lag_ticks=draw(st.sampled_from([1, 2])))


@settings(max_examples=200, deadline=None)
@given(engines())
def test_blind_auto_reaches_most_of_the_best_fixed_window_on_any_engine(engine):
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine, 512) >= 0.85, (engine, [t.limit for t in ticks[-12:]])


@pytest.mark.xfail(strict=True, reason="gauge-mode probes step by +8; a capacity query (PR 6b) "
                   "lets Auto grow to the engine's cap while its queue is empty")
def test_gauges_cross_a_flat_stretch_three_times_wide():
    engine = Engine([(16, 5.0), (48, 5.0), (64, 6.5)], cap=64, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine, 512) >= 0.85, [t.limit for t in ticks[-12:]]


@pytest.mark.xfail(strict=True, reason="gauge-mode probes step by +8, which gains under 5% on a "
                   "gentle slope; a capacity query (PR 6b) lets Auto grow while the queue is empty")
def test_gauges_climb_a_gentle_slope_after_a_flat_stretch():
    engine = Engine([(16, 5.0), (32, 6.5), (64, 6.5), (128, 8.45)], cap=128, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine, 512) >= 0.85, [t.limit for t in ticks[-12:]]
