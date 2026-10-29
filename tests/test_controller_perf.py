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

from saturate.controller import Auto, _above

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
    delivers at least 55% of the best fixed window. Secondary to the steady state."""
    engine = Engine(**ENGINES[name], gauges=gauges, noise=0.05)
    ticks = run(Auto(), engine, 30)
    assert share_of_best(ticks, engine) >= 0.55, [t.limit for t in ticks]


@pytest.mark.parametrize("name", ["sglang-like", "linear-100", "linear-48"])
def test_blind_auto_trims_a_window_past_what_the_engine_uses(name):
    """Blind, a doubling can confirm a gain and still overshoot the engine's cap (64 -> 128 for
    a cap of 100). x7/8 moves bring the window back: after warm-up it spends at most a quarter of
    the time more than an eighth past the useful window. Readings without noise: with noisy
    readings a x7/8 move's verdict is often wrong, and the window stays larger."""
    engine = Engine(**ENGINES[name], gauges=False)
    ticks = run(Auto(), engine, 600)
    assert share_past_useful(ticks[200:], engine) <= 0.25, [t.limit for t in ticks[-10:]]


SEEDS = range(10)  # each a different noise draw and trial schedule: one run can pass by luck


def hosted(rate_limit: int, seed: int, **over) -> Engine:
    return Engine(curve=[(1024, 400.0)], cap=1024, rate_limit=rate_limit, gauges=False, noise=0.05,
                  seed=seed, **over)


@pytest.mark.parametrize("rate_limit", [64, 100, 127, 200])
def test_a_hosted_api_rate_limit_is_met_without_halving(rate_limit):
    """A hosted API rejects requests past its concurrency limit with 429s. Auto remembers the
    window that met them and grows no closer than an eighth below it. After a clean stretch it
    looks again (a provider's limit can rise), an eighth past the old limit at most; meeting the
    same limit then steps just under it, so after warm-up the window never halves, and few rows
    are rejected. Judged over SEEDS: the mean, and the worst run."""
    best, failed = [], []
    for seed in SEEDS:
        ticks = run(Auto(seed=seed), hosted(rate_limit, seed), 600)
        best.append(share_of_best(ticks[100:]))
        failed.append(failed_share(ticks))
        # starting at 64, Auto can meet a low limit early and halve once: 3/5, not less
        tail = [t.limit for t in ticks[200:]]
        assert min(tail) >= rate_limit * 3 // 5, (seed, tail)
    assert sum(best) / len(best) >= 0.94 and min(best) >= 0.90, best
    assert sum(failed) / len(failed) <= 0.03 and max(failed) <= 0.04, failed  # about 1-2%


def test_a_hosted_api_limit_that_rises_mid_run_is_found():
    """The limit doubles from 100 to 200 at tick 300: released ceilings let Auto find it."""
    shares = []
    for seed in SEEDS:
        ticks = run(Auto(seed=seed), hosted(100, seed), 900, then=(300, hosted(200, seed)))
        shares.append(share_of_best(ticks[300:]))
    # over 100 seeds main reached a mean of 0.87 and a worst run of 0.72: the floor is main's tail
    assert sum(shares) / len(shares) >= 0.85 and min(shares) >= 0.72, shares


@given(cap=st.integers(2, 400), initial=st.integers(2, 512), seed=st.integers(0, 2**16),
       reject=st.sampled_from([4, 12, 50]), data=st.data())
def test_a_hard_429_wall_is_never_crossed_far_once_met(cap, initial, seed, reject, data):
    """An API that rejects everything past `cap` (1 in `reject` of the served rows come back as
    429s while over it). Once Auto has met the wall, every later window stays within two
    eighths of the lowest window that met it: growth stops an eighth below a ceiling, and a
    released ceiling is probed an eighth past (at least one), never by a move of x2 or x4. A
    floor at the wall (min_limit == cap) starts the probe at the ceiling itself."""
    min_limit = data.draw(st.integers(1, min(cap, 64)))
    ctrl = Auto(initial=initial, min_limit=min_limit, max_limit=512, seed=seed)
    limit, hist, met = ctrl.initial, [ctrl.initial], None
    for tick in range(1500):
        seen = hist[max(0, len(hist) - 2)]
        served = min(seen, cap)
        bp = max(1, served // reject) if seen > cap else 0
        obs = dict(waiting=None, running=None, inflight=limit, backpressure=bp, successes=served,
                   input_bound=False, kv=None, hits=None, tok_s=served * 100.0)
        limit = ctrl.decide(obs, limit)
        hist.append(limit)
        if ctrl.last_reason == "cut:bp" and seen > cap:
            met = tick if met is None else met
        if met is not None and tick > met + 60:  # past the first cuts down to the wall
            wall = ctrl.state.ceiling or ctrl.state.seen_ceiling
            assert wall is not None and limit <= _above(wall.limit), \
                (tick, limit, wall, hist[-20:])


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
    window it starts at; the x4 move crosses it."""
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


@settings(max_examples=200, deadline=None, derandomize=True)  # the same engines on every run
@given(engines())
def test_blind_auto_reaches_most_of_the_best_fixed_window_on_any_engine(engine):
    """Judged on the second half of 20 minutes: on a slow engine (25 s requests) each trial
    waits two latencies for its own readings, so the search takes minutes."""
    ticks = run(Auto(), engine, 600)
    # 0.75, not higher: past a flat stretch that starts below the initial window, the x4 move
    # can land on a gain just over `improve`, which can read as flat
    assert share_of_best(ticks[300:], engine) >= 0.75, (engine, [t.limit for t in ticks[-12:]])


def test_gauges_cross_a_flat_stretch_three_times_wide():
    """With gauges, the same moves apply: the x4 move crosses a flat stretch three times wide."""
    engine = Engine([(16, 5.0), (48, 5.0), (64, 6.5)], cap=64, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine) >= 0.85, [t.limit for t in ticks[-12:]]


def test_gauges_climb_a_gentle_slope_after_a_flat_stretch():
    """A gentle slope after a flat stretch: the x1.5 and x1.25 moves climb it."""
    engine = Engine([(16, 5.0), (32, 6.5), (64, 6.5), (128, 8.45)], cap=128, gauges=True)
    ticks = run(Auto(), engine, 400)
    assert share_of_best(ticks[200:], engine) >= 0.85, [t.limit for t in ticks[-12:]]
