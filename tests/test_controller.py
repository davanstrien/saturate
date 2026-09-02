"""Auto controller scenarios, driven as Obs trajectories through decide().

Each test names the engine behaviour it pins: a wedged engine, a burst of
429s, a fast idle engine, long generations, a plateau that lags the tick."""

import pytest

from saturate.controller import Auto

STEP = 8


def healthy(limit, **over):
    """A saturated, well-served window: no queue, everything completing."""
    obs = dict(waiting=0, running=limit, inflight=limit, backpressure=0, successes=limit * 4,
               input_bound=False, kv=0.2, hits=0.9, tok_s=limit * 100.0)
    obs.update(over)
    return obs


def drive(ctrl, limit, obs_seq):
    """Feed a sequence of observations; return (limits, reasons) per tick."""
    traj, reasons = [], []
    for obs in obs_seq:
        limit = ctrl.decide(obs, limit)
        traj.append(limit)
        reasons.append(ctrl.last_reason)
    return traj, reasons


def drive_lagged(ctrl, limit, ticks, tok_of_limit, lag=1):
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


def warm(ctrl, limit, ticks=1):
    """Ticks with the queue in band (hold): the controller learns a throughput baseline at `limit`."""
    for _ in range(ticks):
        assert ctrl.decide(healthy(limit, waiting=4), limit) == limit
    return limit


# --- stalled engine ---------------------------------------------------------


def test_stalled_engine_is_cut_not_grown():
    """Window full, nothing completing, zero tokens: the signature of a wedged
    engine. The queue reads empty only because nothing new is admitted."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = warm(ctrl, 32)
    stalled = [healthy(limit, waiting=1, successes=0, tok_s=0.0)] * 8
    traj, reasons = drive(ctrl, limit, [dict(o, inflight=limit) for o in stalled])
    assert max(traj) <= limit, f"grew into a stalled engine: {traj}"
    assert not ctrl._slow_start
    assert traj[2] <= limit // 2 and "cut:stall" in reasons[:3], (traj, reasons)
    assert min(traj) >= ctrl.min


def test_stall_patience_scales_with_request_latency():
    """Three empty ticks are a stall for chat, but routine for 60 s generations
    on a window of two. Patience follows the observed request latency."""
    ctrl = Auto(target_waiting=8, initial=4, step=STEP)
    limit = warm(ctrl, 4)
    slow = healthy(limit, successes=0, tok_s=0.0, latency_s=20.0)
    traj, reasons = drive(ctrl, limit, [slow] * 10)  # 10 ticks = 20 s = one latency
    assert traj == [limit] * 10 and "cut:stall" not in reasons, (traj, reasons)
    traj, reasons = drive(ctrl, limit, [slow] * 15)  # past 2x latency: now it is a stall
    assert "cut:stall" in reasons, (traj, reasons)


# --- backpressure -----------------------------------------------------------


def test_backpressure_episode_cuts_once():
    """Requests admitted under the old window keep failing after the cut; one
    saturation episode is one halving, then a hold while the old window drains."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    burst = healthy(64, backpressure=1, successes=5, tok_s=5000.0, latency_s=10.0)
    traj, reasons = drive(ctrl, 64, [burst] * 5)
    assert traj == [32] * 5, traj
    assert reasons == ["cut:bp"] + ["hold:cooldown"] * 4, reasons


def test_backpressure_cooldown_follows_latency():
    burst = healthy(64, backpressure=1, successes=5, tok_s=5000.0)
    fast, _ = drive(Auto(initial=64), 64, [dict(burst, latency_s=0.5)] * 6)
    unknown, _ = drive(Auto(initial=64), 64, [burst] * 6)
    slow, _ = drive(Auto(initial=64), 64, [dict(burst, latency_s=30.0)] * 6)
    assert fast == [32, 32, 32, 16, 16, 16], fast  # 2-tick floor
    assert unknown == [32, 32, 32, 32, 16, 16], unknown  # fixed 3 when latency is unknown
    assert slow == [32] * 6, slow


def test_cut_rescales_throughput_baseline():
    """After a halving, throughput at the smaller window is compared against a
    proportionally smaller baseline, so recovery is not read as a plateau."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)  # baseline 6400 tok/s at 64
    limit = ctrl.decide(healthy(limit, backpressure=1), limit)
    assert limit == 32
    traj, reasons = drive(ctrl, limit, [healthy(32, tok_s=3500.0)] * 6)  # 3500 > 3200 * 1.05
    assert "grow" in reasons and max(traj) > 32, (traj, reasons)


def test_cut_at_floor_keeps_baseline():
    """A cut that cannot reduce the limit must not decay the baseline: flat
    throughput at the floor would otherwise read as growth."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    limit = warm(ctrl, 2)  # baseline 200 tok/s at 2
    limit = ctrl.decide(healthy(2, backpressure=1), 2)
    assert limit == 2
    traj, reasons = drive(ctrl, 2, [healthy(2)] * 8)  # flat 200 tok/s
    assert "grow" not in reasons, (traj, reasons)


def test_cut_decay_is_proportional_near_floor():
    """3 -> 2 is a one-third reduction; halving the baseline would let flat
    throughput at 2 read as growth (3 -> 2 -> 2 -> 2 -> 10)."""
    ctrl = Auto(target_waiting=8, initial=3, min_limit=2, step=STEP)
    warm(ctrl, 3)  # baseline 300 tok/s at 3
    assert ctrl.decide(healthy(3, backpressure=1), 3) == 2
    traj, reasons = drive(ctrl, 2, [healthy(2)] * 8)  # flat 200 tok/s: 2/3 of the old baseline
    assert "grow" not in reasons, (traj, reasons)


# --- ack-clocked growth -----------------------------------------------------


def test_growth_waits_for_a_window_of_completions():
    """Long generations: widen only once `limit` completions have arrived since
    the last widening, and only when throughput improved over that window."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    slow = healthy(16, successes=4, tok_s=400.0)  # 4 completions per tick at 16
    traj, reasons = drive(ctrl, 16, [slow] * 4)
    assert traj == [16, 16, 16, 32], (traj, reasons)  # 16 acks arrive on the 4th tick
    assert reasons[:3] == ["hold:ack"] * 3 and reasons[3] == "grow", reasons


def test_cheap_text_ramps_every_tick():
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj, reasons = drive(ctrl, 8, [healthy(lim) for lim in (8, 16, 32)])
    assert traj == [16, 32, 64] and reasons == ["grow"] * 3, (traj, reasons)


def test_cold_start_neither_grows_nor_stalls():
    """Before the first completion the request latency is unknown: a full window
    with nothing back yet is not evidence of headroom, nor yet of a wedge."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    traj, reasons = drive(ctrl, 16, [healthy(16, successes=0, tok_s=None)] * 5)
    assert traj == [16] * 5 and set(reasons) == {"hold:ack"}, (traj, reasons)


# --- idle engine ------------------------------------------------------------


def test_idle_fast_engine_holds():
    """Completions flowing with the window mostly empty: the source, not the
    engine, is the limit. Neither a stall (window full) nor headroom (window used)."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = warm(ctrl, 32)
    idle = healthy(limit, inflight=limit // 4, successes=40, tok_s=4000.0)
    traj, reasons = drive(ctrl, limit, [idle] * 8)
    assert traj == [limit] * 8 and set(reasons) == {"hold"}, (traj, reasons)


# --- queue-driven step-down -------------------------------------------------


def test_queue_step_down_rescales_throughput_baseline():
    """A step-down for a standing queue must move the baseline with it, else
    throughput at the smaller window is judged against the larger one."""
    ctrl = Auto(target_waiting=8, initial=40, step=STEP)
    limit = warm(ctrl, 40)  # baseline 4000 tok/s at 40
    limit = ctrl.decide(healthy(40, waiting=100), 40)
    assert limit == 32 and ctrl.last_reason == "cut:queue"
    traj, reasons = drive(ctrl, 32, [healthy(32, tok_s=3500.0)] * 6)  # 3500 > 3200 * 1.05, < 4000
    assert "grow" in reasons, (traj, reasons)


def test_queue_pressure_voids_active_probe():
    """An independent queue-driven reduction while a probe is settling voids the
    probe: the pending revert must not climb back over the reduction."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = warm(ctrl, 32)
    flat = healthy(32, tok_s=1000.0)
    traj, reasons = drive(ctrl, limit, [flat] * 4)
    assert reasons[-1] == "probe" and traj[-1] == 40, (traj, reasons)
    traj, reasons = drive(ctrl, 40, [healthy(40, waiting=100)] * 2 + [healthy(24, tok_s=1000.0)] * 3)
    assert traj == [32, 24, 24, 24, 24] and "revert" not in reasons, (traj, reasons)


# --- probe-and-revert -------------------------------------------------------


def test_escapes_false_plateau_with_lag():
    # true capacity is high: tok scales with limit up to 128, but LAGS one tick
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 40, lambda lim: min(lim, 128) * 100.0, lag=2)
    assert max(traj) >= 64, f"stuck below capacity despite headroom: {traj}"


def test_bounded_at_real_knee():
    # true knee at 16: tok never improves past it; probes must revert
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 40, lambda lim: min(lim, 16) * 100.0, lag=1)
    assert max(traj) <= 16 * 3, f"ran away past the knee: {traj}"


def test_probe_voided_by_gauge_loss():
    """A probe begun in gauge mode must not survive blind ticks and fire a
    stale revert when gauges return."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = warm(ctrl, 32)
    flat = healthy(32, tok_s=1000.0)
    traj, reasons = drive(ctrl, limit, [flat] * 4)
    assert reasons[-1] == "probe" and traj[-1] == 40, (traj, reasons)
    limit = ctrl.decide(dict(flat, waiting=None, inflight=40), 40)  # gauges vanish
    traj, reasons = drive(ctrl, limit, [healthy(40, tok_s=1000.0)] * 4)  # gauges return, flat
    assert "revert" not in reasons and min(traj) >= 40, (traj, reasons)


# --- construction -----------------------------------------------------------


def test_initial_clamped_to_max():
    assert Auto(initial=600, max_limit=512).initial == 512


def test_auto_invalid_bounds_raise():
    """Auto(min_limit=0) deadlocks admission exactly like Fixed(0)."""
    for kw in (dict(min_limit=0), dict(max_limit=1, min_limit=2), dict(step=0), dict(stall_ticks=0)):
        with pytest.raises(ValueError):
            Auto(**kw)


def test_decide_reports_a_reason_for_every_tick():
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    assert ctrl.last_reason == "hold"
    for obs in (healthy(16), healthy(32, backpressure=1), healthy(16, input_bound=True)):
        ctrl.decide(obs, obs["inflight"])
        assert ctrl.last_reason in {"hold", "grow", "probe", "revert", "cut:bp", "cut:kv", "cut:stall",
                                    "cut:queue", "hold:input_bound", "hold:cooldown", "hold:plateau",
                                    "hold:ack"}
