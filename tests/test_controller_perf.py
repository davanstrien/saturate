"""Auto against simulated engines (tests/sim_engine.py): the share of the best fixed window's
throughput it reaches, how fast it gets there, and how long it spends past the window the engine
can use. bench/controller_sim.py prints the same scenarios as a table."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sim_engine import (
    LINEAR,
    PREEMPTING,
    SGLANG_LIKE,
    Engine,
    failed_share,
    run,
    share_of_best,
    share_past_useful,
)

from saturate.controller import Auto

ENGINES = {
    "sglang-like": dict(curve=SGLANG_LIKE, cap=256),
    "linear-256": dict(curve=LINEAR, cap=256),
    "linear-100": dict(curve=[(100, 200.0)], cap=100),  # a cap off the doubling ladder
    "linear-48": dict(curve=[(48, 60.0)], cap=48),
}


@pytest.mark.parametrize("noise", [0.0, 0.07])
@pytest.mark.parametrize("gauges", [False, True], ids=["blind", "gauges"])
@pytest.mark.parametrize("name", list(ENGINES))
def test_auto_reaches_most_of_the_best_fixed_window(name, gauges, noise):
    """After the first 100 ticks, Auto delivers at least 90% of the best fixed window."""
    engine = Engine(**ENGINES[name], gauges=gauges, noise=noise)
    ticks = run(Auto(), engine, 600)
    assert share_of_best(ticks[100:], engine) >= 0.9, [t.limit for t in ticks[-10:]]


@pytest.mark.parametrize("gauges", [False, True], ids=["blind", "gauges"])
@pytest.mark.parametrize("name", list(ENGINES))
def test_auto_ramps_within_the_first_minute(name, gauges):
    """A short run must not spend itself ramping: over the first 30 ticks (a minute) Auto
    delivers at least 60% of the best fixed window."""
    engine = Engine(**ENGINES[name], gauges=gauges, noise=0.05)
    ticks = run(Auto(), engine, 30)
    assert share_of_best(ticks, engine) >= 0.6, [t.limit for t in ticks]


@pytest.mark.parametrize("name", ["sglang-like", "linear-100", "linear-48"])
def test_blind_auto_trims_a_window_past_what_the_engine_uses(name):
    """Blind, a doubling can confirm a gain and still overshoot the engine's cap (64 -> 128 for
    a cap of 100). Trims bring the window back: after warm-up it spends at most a quarter of
    the time more than an eighth past the useful window. Readings without noise: with noisy
    readings a trim's verdict is often wrong, and the window stays larger."""
    engine = Engine(**ENGINES[name], gauges=False)
    ticks = run(Auto(), engine, 600)
    assert share_past_useful(ticks[200:], engine) <= 0.25, [t.limit for t in ticks[-10:]]


@pytest.mark.parametrize("rate_limit", [64, 100, 127, 200])
def test_a_hosted_api_rate_limit_is_met_without_halving(rate_limit):
    """A hosted API rejects requests past its concurrency limit with 429s. Auto remembers the
    window that met them and grows no closer than an eighth below it. After a clean stretch it
    looks again (a provider's limit can rise); meeting the same limit then steps just under it,
    so after warm-up the window never halves, and few rows are rejected."""
    engine = Engine(curve=[(1024, 400.0)], cap=1024, rate_limit=rate_limit, gauges=False, noise=0.05)
    ticks = run(Auto(), engine, 600)
    assert share_of_best(ticks[100:]) >= 0.95, [t.limit for t in ticks[-10:]]
    assert min(t.limit for t in ticks[200:]) >= rate_limit * 3 // 4, [t.limit for t in ticks[200:]]
    assert failed_share(ticks) <= 0.04  # main: about 2%, halving and climbing back


def test_a_hosted_api_limit_that_rises_mid_run_is_found():
    """The limit doubles from 100 to 200 at tick 300: released ceilings let Auto find it."""
    kw = dict(curve=[(1024, 400.0)], cap=1024, gauges=False, noise=0.05)
    ticks = run(Auto(), Engine(rate_limit=100, **kw), 900, then=(300, Engine(rate_limit=200, **kw)))
    assert share_of_best(ticks[300:]) >= 0.85, [t.limit for t in ticks[-10:]]


def test_long_generations_survive_a_trial_past_the_read_timeout():
    """12 s requests at the engine's cap and a 60 s read timeout: a doubling trial past the
    timeout makes requests time out, which reports no latency. The cooldown keeps the last
    latency seen, so the late timeouts of the old window do not halve the window again."""
    engine = Engine(curve=[(100, 8.0)], cap=100, gauges=False, noise=0.05, read_timeout_s=60)
    ticks = run(Auto(), engine, 600)
    assert share_of_best(ticks[100:], engine) >= 0.9, [t.limit for t in ticks[-10:]]
    assert failed_share(ticks) <= 0.1  # one trial past the timeout: Auto does not know the timeout


def test_preemption_past_a_batch_size_is_not_worse_than_a_fixed_middle_window():
    """Throughput falls past 192 running requests (vLLM preempting): Auto must not sit far past
    that point."""
    engine = Engine(curve=PREEMPTING, cap=512, gauges=False)
    ticks = run(Auto(), engine, 600)
    assert share_of_best(ticks[100:], engine) >= 0.75, [t.limit for t in ticks[-10:]]


@st.composite
def engines(draw):
    """A blind engine whose throughput never falls as the window grows, with a cap on or off the
    doubling ladder and noisy readings. A flat stretch may be up to three times as wide as the
    window it starts at; blind probes reach 8x the window."""
    cap = draw(st.sampled_from([32, 48, 64, 100, 127, 200, 256, 300, 512]))
    window, rate, curve, flat = 8, draw(st.floats(5.0, 50.0)), [], False
    while window < cap:
        # flat about a third of the time, never twice in a row: one flat stretch at a time
        gain = draw(st.sampled_from([1.3, 1.6, 2.0] if flat else [1.0, 1.0, 1.3, 1.6, 2.0]))
        flat = gain == 1.0
        widen = draw(st.sampled_from([2, 3]))
        reach = min(cap, window * widen)  # at the cap, only the part of the step that fits
        rate *= gain ** (math.log(reach / window) / math.log(widen))
        window = reach
        curve.append((window, rate))
    return Engine(curve, cap=cap, gauges=False, noise=draw(st.sampled_from([0.0, 0.05])),
                  seed=draw(st.integers(0, 3)))


@settings(max_examples=200, deadline=None)
@given(engines())
def test_blind_auto_reaches_most_of_the_best_fixed_window_on_any_engine(engine):
    """Judged on the second half of 20 minutes: on a slow engine (25 s requests) each probe
    waits two latencies for its own readings, so the search takes minutes."""
    ticks = run(Auto(), engine, 600)
    assert share_of_best(ticks[300:], engine) >= 0.85, (engine, [t.limit for t in ticks[-12:]])


@pytest.mark.xfail(strict=True, reason="gauge-mode probes step by +8; a capacity query lets Auto "
                   "grow to the engine's cap while its queue is empty")
def test_gauges_cross_a_flat_stretch_three_times_wide():
    engine = Engine([(16, 5.0), (48, 5.0), (64, 6.5)], cap=64, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine) >= 0.85, [t.limit for t in ticks[-12:]]


def test_gauges_climb_a_gentle_slope_after_a_flat_stretch():
    """A +8 probe from 64 gains under 5% here; the probe stays until its own readings arrive."""
    engine = Engine([(16, 5.0), (32, 6.5), (64, 6.5), (128, 8.45)], cap=128, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine) >= 0.85, [t.limit for t in ticks[-12:]]
