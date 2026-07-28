"""Probe-and-revert: the controller must escape a FALSE plateau (throughput
reading lags the tick on real engines) yet stay bounded at a REAL knee."""

from pumpjack.controller import Auto


def drive(ctrl, limit, ticks, tok_of_limit, lag=1):
    """Simulate: tok_s observed at tick t reflects the limit from `lag` ticks ago."""
    history = [limit]
    traj = []
    for _ in range(ticks):
        seen = history[max(0, len(history) - 1 - lag)]
        obs = dict(waiting=0, running=min(limit, 200), inflight=limit,
                   backpressure=0, successes=seen, input_bound=False,
                   kv=0.2, hits=0.9, tok_s=tok_of_limit(seen))
        limit = ctrl.decide(obs, limit)
        history.append(limit)
        traj.append(limit)
    return traj


def test_escapes_false_plateau_with_lag():
    # true capacity is high: tok scales with limit up to 128, but LAGS one tick
    ctrl = Auto(target_waiting=8, initial=8, step=8)
    traj = drive(ctrl, 8, 40, lambda lim: min(lim, 128) * 100.0, lag=2)
    assert max(traj) >= 64, f"stuck below capacity despite headroom: {traj}"


def test_bounded_at_real_knee():
    # true knee at 16: tok never improves past it; probes must revert
    ctrl = Auto(target_waiting=8, initial=8, step=8)
    traj = drive(ctrl, 8, 40, lambda lim: min(lim, 16) * 100.0, lag=1)
    assert max(traj) <= 16 * 3, f"ran away past the knee: {traj}"


def test_probe_voided_by_gauge_loss():
    """Codex re-review #1: a probe begun in gauge mode must not survive blind
    ticks and fire a stale revert when gauges return."""
    ctrl = Auto(target_waiting=8, initial=8, step=8)
    limit = 32
    flat = dict(waiting=0, running=16, inflight=32, backpressure=0,
                successes=10, input_bound=False, tok_s=1000.0)
    ctrl._best_tok = 5000.0  # plateau-blocked
    for _ in range(ctrl._probe_cooldown):  # drive until a probe fires
        limit = ctrl.decide(dict(flat, inflight=limit), limit)
    assert ctrl._probe_from is not None  # probe active
    limit = ctrl.decide(dict(flat, waiting=None), limit)  # gauges vanish
    assert ctrl._probe_from is None  # voided, no stale revert possible


def test_initial_clamped_to_max():
    """Codex re-review #2: Auto(initial > max_limit) must clamp."""
    ctrl = Auto(initial=600, max_limit=512)
    assert ctrl.initial == 512


def test_cut_decays_best_tok():
    """Codex r3 #13: _best_tok was an all-time max retained after cuts, so the
    plateau gate could block recovery forever. A cut must decay it."""
    ctrl = Auto(target_waiting=8, initial=8, step=8)
    ctrl._best_tok = 8000.0
    ctrl.decide(dict(waiting=0, running=8, inflight=8, backpressure=5,
                     successes=8, input_bound=False, tok_s=1000.0), 64)
    assert ctrl._best_tok < 8000.0  # decayed: post-cut throughput can grow again


def test_cut_at_floor_keeps_best_tok():
    """Codex r4 blocker #3: a cut that cannot reduce the limit (already at
    min_limit) must not decay the baseline — flat throughput at the floor
    would otherwise read as growth and the window would climb right back."""
    ctrl = Auto(target_waiting=8, initial=8, step=8, min_limit=2)
    ctrl._best_tok = 8000.0
    ctrl.decide(dict(waiting=0, running=2, inflight=2, backpressure=5,
                     successes=2, input_bound=False, tok_s=1000.0), 2)
    assert ctrl._best_tok == 8000.0


def test_auto_invalid_bounds_raise():
    """Codex r4: Auto(min_limit=0) deadlocks admission exactly like Fixed(0)."""
    import pytest

    for kw in (dict(min_limit=0), dict(max_limit=1, min_limit=2), dict(step=0)):
        with pytest.raises(ValueError):
            Auto(**kw)
