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
