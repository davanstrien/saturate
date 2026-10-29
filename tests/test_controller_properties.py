"""Properties of Auto that hold for any stream of observations, any configuration and any trial
schedule (seed). Hypothesis builds the streams, including ones no engine sends (gauges that
come and go, zero ticks, latencies that jump); each rule must still keep these promises."""

import dataclasses
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_controller import REASONS

from saturate.controller import Auto, Steady, _below
from saturate.signals import Capacity

maybe_count = st.one_of(st.none(), st.integers(0, 2000))
maybe_share = st.one_of(st.none(), st.floats(0.0, 1.0))
maybe_seconds = st.one_of(st.none(), st.floats(0.0, 600.0, allow_nan=False))


@st.composite
def observations(draw, limit: int):
    """One tick's observation for a window of `limit`."""
    successes = draw(st.integers(0, 4 * limit + 4))
    return dict(
        waiting=draw(maybe_count), running=draw(maybe_count),
        inflight=draw(st.integers(0, limit)), backpressure=draw(st.integers(0, limit + 4)),
        successes=successes, input_bound=draw(st.booleans()),
        kv=draw(maybe_share), hits=draw(maybe_share),
        tok_s=draw(st.one_of(st.none(), st.just(0.0), st.floats(0.0, 1e6, allow_nan=False))),
        latency_s=draw(maybe_seconds), oldest_s=draw(maybe_seconds),
        tick_s=draw(st.one_of(st.none(), st.floats(0.1, 10.0))),
    )


@st.composite
def autos(draw):
    min_limit = draw(st.integers(1, 64))
    max_limit = draw(st.integers(min_limit, 1024))
    auto = Auto(target_waiting=draw(st.integers(1, 64)),
                initial=draw(st.one_of(st.none(), st.integers(1, 2048))),
                min_limit=min_limit, max_limit=max_limit, step=draw(st.integers(1, 32)),
                bp_frac=draw(st.floats(0.0, 1.0)), seed=draw(st.integers(0, 2**32)))
    if draw(st.booleans()):
        auto.begin(Capacity(draw(st.integers(1, 1024)), draw(st.booleans()), "any"))
    return auto


@given(st.data())
def test_every_decision_keeps_the_window_in_bounds_and_names_a_known_reason(data):
    """For any observations: the window stays within [min_limit, max_limit]; a hold keeps it, a
    cut or revert never raises it, a trial moves it the way its name says; any rise stays under
    an active backpressure ceiling; a due trial in slow-start (backoff 1) is never delayed by
    jitter."""
    auto = data.draw(autos())
    limit = auto.initial
    for _ in range(data.draw(st.integers(1, 120))):
        before = auto.state
        new = auto.decide(data.draw(observations(limit)), limit)
        reason = auto.last_reason
        assert reason in REASONS, reason
        assert auto.min <= new <= auto.max, (new, auto.min, auto.max)
        if reason.startswith("hold"):
            assert new == limit, (reason, limit, new)
        if reason.startswith(("cut", "revert")):
            assert new <= limit, (reason, limit, new)
        if reason == "trial:up":
            assert new > limit
        if reason == "trial:down":
            assert new < limit
        ceiling = auto.state.ceiling
        if new > limit and ceiling is not None:
            assert new <= max(auto.min, _below(ceiling.limit)), (new, ceiling)
        slow_start = isinstance(before.phase, Steady) and before.phase.backoff == 1
        if slow_start and before.phase.jitter is None:
            assert reason != "hold:jitter", before.phase
        limit = new


@given(st.data())
def test_the_same_seed_makes_the_same_decisions(data):
    """Auto is a pure function of its configuration, its seed and the observations: two Autos
    built alike and fed the same stream decide alike, tick by tick."""
    seed = data.draw(st.integers(0, 2**32))
    first, second = Auto(seed=seed), Auto(seed=seed)
    limit = first.initial
    for _ in range(data.draw(st.integers(1, 120))):
        obs = data.draw(observations(limit))
        a, b = first.decide(obs, limit), second.decide(obs, limit)
        assert (a, first.last_reason) == (b, second.last_reason)
        assert first.state == second.state
        limit = a


@given(st.data())
def test_decide_never_changes_a_state_it_was_given(data):
    """The state is frozen dataclasses: decide() replaces it, so a state held from an earlier
    tick still reads the same afterwards (rules can be replayed from any tick)."""
    auto = data.draw(autos())
    limit = auto.initial
    for _ in range(data.draw(st.integers(1, 60))):
        before = auto.state
        snapshot = dataclasses.astuple(before)
        limit = auto.decide(data.draw(observations(limit)), limit)
        assert dataclasses.astuple(before) == snapshot


class _RecordingRandom(random.Random):
    """Wraps an Auto's own Random (keeping its seed) and records the jitter draws made from it."""

    def __init__(self, inner: random.Random):
        super().__init__()
        self.inner = inner
        self.drawn: list[int] = []

    def randint(self, a: int, b: int) -> int:
        self.drawn.append(self.inner.randint(a, b))
        return self.drawn[-1]


def check_jitter(auto: Auto, steps) -> None:
    """Drive `auto` through `steps` (a callable: limit -> observation, or None to stop) and check
    that every trial opened after a failure waited exactly the jitter drawn for it, in
    hold:jitter ticks since the previous trial or skipped move: a fresh draw each time, never
    more than JITTER_TICKS, and not spent by any other hold."""
    rng = _RecordingRandom(auto.rng)
    auto.rng = rng
    limit, draws_seen, jitter_holds = auto.initial, 0, 0
    while (obs := steps(limit)) is not None:
        before = auto.state.phase
        limit = auto.decide(obs, limit)
        reason = auto.last_reason
        if reason == "hold:jitter":
            jitter_holds += 1
        if reason.startswith("trial") and isinstance(before, Steady) and before.backoff > 1:
            assert len(rng.drawn) == draws_seen + 1, (rng.drawn, draws_seen)
            assert jitter_holds == rng.drawn[-1] <= auto.JITTER_TICKS, (jitter_holds, rng.drawn)
        if reason.startswith(("trial", "cut", "revert")) or reason in ("hold:ceiling", "hold:floor"):
            draws_seen, jitter_holds = len(rng.drawn), 0


@given(st.data())
def test_every_trial_after_a_failure_waits_its_own_jitter(data):
    """For any observations: see check_jitter."""
    auto = data.draw(autos())
    ticks_left = [data.draw(st.integers(1, 150))]

    def anything(limit):
        if ticks_left[0] == 0:
            return None
        ticks_left[0] -= 1
        return data.draw(observations(limit))

    check_jitter(auto, anything)


@given(cap=st.integers(8, 300), seed=st.integers(0, 2**16))
def test_trials_at_a_backpressure_wall_each_wait_their_own_jitter(cap, seed):
    """At a 429 wall most moves up are skipped (hold:ceiling) and most trials follow a failure:
    the case where a carried-over draw would show."""
    hist = [16]

    def wall(limit):
        if len(hist) > 1500:
            return None
        hist.append(limit)
        seen = hist[max(0, len(hist) - 3)]
        served = min(seen, cap)
        return dict(inflight=limit, backpressure=max(1, served // 12) if seen > cap else 0,
                    successes=served, tok_s=served * 100.0)

    check_jitter(Auto(initial=16, max_limit=512, seed=seed), wall)


def test_a_hold_every_client_sees_does_not_spend_the_jitter():
    """A queue in band holds every client sharing the engine; if those ticks spent the jitter,
    the clients would all open their next trial on the same tick after it."""

    class Longest(random.Random):
        """Always draws the longest jitter, JITTER_TICKS."""

        def randint(self, a: int, b: int) -> int:
            return b

    auto = Auto(initial=64)
    auto.rng = Longest()
    limit = auto.initial
    for _ in range(200):  # until a trial fails and the next one is due
        limit = auto.decide(dict(inflight=limit, successes=limit, tok_s=100.0 * 64, waiting=0, running=limit),
                            limit)
        if auto.last_reason == "hold:jitter":
            break
    assert auto.last_reason == "hold:jitter", auto.last_reason
    assert isinstance(auto.state.phase, Steady), auto.state.phase
    left = auto.state.phase.jitter
    assert left == auto.JITTER_TICKS - 1
    for _ in range(10):  # a queue in band: hold
        limit = auto.decide(dict(inflight=limit, successes=limit, tok_s=100.0 * 64, waiting=4, running=limit),
                            limit)
        assert auto.last_reason == "hold"
    assert auto.state.phase.jitter == left


whole = st.one_of(st.integers(-5, 2000), st.booleans(), st.floats(-5, 2000), st.just(float("nan")),
                  st.text(max_size=2), st.integers(10**300, 10**400))
real = st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=2), st.fractions())
SETTINGS = dict(target_waiting=whole, initial=st.one_of(st.none(), whole), min_limit=whole, max_limit=whole,
                step=whole, stall_ticks=whole, kv_hi=real, hits_lo=real, improve=real, bp_frac=real)


@given(st.data())
def test_any_configuration_is_refused_up_front_or_runs(data):
    """Auto either refuses a configuration with a ValueError when it is built, or runs any
    observations without raising: a bad setting never fails mid-run."""
    kwargs = data.draw(st.fixed_dictionaries({}, optional=SETTINGS))
    try:
        auto = Auto(seed=0, **kwargs)
    except ValueError:
        return
    limit = auto.initial
    for _ in range(data.draw(st.integers(1, 60))):
        limit = auto.decide(data.draw(observations(limit)), limit)
        assert auto.min <= limit <= auto.max


@pytest.mark.parametrize("kwargs", [
    dict(target_waiting=0), dict(initial=0), dict(initial=-3), dict(kv_hi=0.0), dict(kv_hi=1.5),
    dict(hits_lo=-0.1), dict(hits_lo=1.1), dict(improve=1.0), dict(improve=0.9),
    dict(improve=float("nan")), dict(bp_frac=float("nan")), dict(min_limit=0), dict(max_limit=1, min_limit=2),
    dict(step=0), dict(stall_ticks=0), dict(improve=float("inf")), dict(target_waiting=True),
    dict(kv_hi=True), dict(bp_frac=False), dict(seed=True), dict(max_limit=float("nan")),
    dict(stall_ticks=float("nan")), dict(step=2.5), dict(initial=float("inf")), dict(min_limit=4.0),
    dict(target_waiting="8"), dict(seed=1.5),
])
def test_a_setting_that_cannot_work_is_refused(kwargs):
    with pytest.raises(ValueError):
        Auto(**kwargs)
