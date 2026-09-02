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


def probing(ctrl, limit):
    """Warm at `limit`, then flat throughput with headroom until a probe fires; returns the probed limit."""
    warm(ctrl, limit)
    traj, reasons = drive(ctrl, limit, [healthy(limit, tok_s=limit * 100.0 / 3)] * 8)  # 4 flat evaluations
    assert reasons[-1] == "probe" and traj[-1] == limit + STEP, (traj, reasons)
    return traj[-1]


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
    traj, reasons = drive(ctrl, limit, [slow] * 20)  # 20 ticks = 40 s = two latencies
    assert traj == [limit] * 20 and "cut:stall" not in reasons, (traj, reasons)
    traj, reasons = drive(ctrl, limit, [slow] * 15)  # past 3x latency: now it is a stall
    assert "cut:stall" in reasons, (traj, reasons)


def test_one_stall_cut_per_episode():
    """A wedged engine is halved once; the next stall cut needs a completion
    first. Otherwise the stall rule re-fires the tick each cooldown ends."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)
    stalled = healthy(limit, successes=0, tok_s=0.0, latency_s=2.0)
    traj, reasons = drive(ctrl, limit, [stalled] * 12)
    assert reasons.count("cut:stall") == 1 and traj[-1] == 8, (traj, reasons)
    recovered = healthy(8, successes=8, tok_s=800.0, latency_s=2.0)  # completions resume...
    traj, reasons = drive(ctrl, 8, [recovered] + [dict(stalled, inflight=8)] * 4)  # ...then wedge again
    assert reasons.count("cut:stall") == 1 and traj[-1] == 4, (traj, reasons)


def test_completion_during_cooldown_does_not_rearm_the_stall_rule():
    """The old window's requests keep completing (or sitting in backoff) after
    a cut. One of them landing during the cooldown is not a fresh completion,
    and their silence afterwards is not a stall of the new window."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    burst = healthy(limit, backpressure=1, successes=0, tok_s=0.0, latency_s=1.0)
    retry_ok = healthy(limit, successes=1, tok_s=50.0, latency_s=1.0)  # one old-window retry lands
    backoff = healthy(limit, successes=0, tok_s=0.0, latency_s=1.0)
    traj, reasons = drive(ctrl, limit, [burst, retry_ok] + [backoff] * 5)
    assert traj == [32] * 7 and "cut:stall" not in reasons, (traj, reasons)
    draining = [dict(backoff, oldest_s=age) for age in (3.0, 5.0, 7.0, 9.0)]  # old window still out
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    traj, reasons = drive(ctrl, warm(ctrl, 64), [burst, dict(retry_ok, oldest_s=3.0)] + draining)
    assert traj == [32] * 6 and "cut:stall" not in reasons, (traj, reasons)


def test_backpressure_cut_does_not_cascade_into_a_stall_cut():
    """After a 429 the retry ladder holds the slots in backoff: a full window
    with no completions, which is not a second reason to halve."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    burst = healthy(limit, backpressure=1, successes=0, tok_s=0.0, latency_s=1.0)
    backoff = healthy(limit, successes=0, tok_s=0.0, latency_s=1.0)
    traj, reasons = drive(ctrl, limit, [burst] + [backoff] * 8)
    assert traj == [32] * 9 and "cut:stall" not in reasons, (traj, reasons)


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


def test_cooldown_holds_while_the_old_window_drains():
    """The oldest request in flight predating the cut means the old window has
    not drained yet: keep holding, but never past the stall patience."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    burst = healthy(limit, backpressure=1, successes=5, tok_s=5000.0, latency_s=1.0)
    after = [healthy(32, successes=5, tok_s=1000.0, latency_s=1.0, oldest_s=age)
             for age in (3.0, 5.0, 7.0, 1.0, 1.0)]  # ages at 2 s ticks: predates the cut until the 4th
    traj, reasons = drive(ctrl, limit, [burst] + after)
    assert reasons == ["cut:bp"] + ["hold:cooldown"] * 3 + ["hold:ack"] * 2, (traj, reasons)
    stuck = [healthy(32, successes=5, tok_s=1000.0, latency_s=1.0, oldest_s=1000.0)] * 6
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    traj, reasons = drive(ctrl, warm(ctrl, 64), [burst] + stuck)
    assert reasons.count("hold:cooldown") == 3, reasons  # capped at the stall patience (3 ticks)


def test_tick_length_comes_from_the_observation():
    """Latency-derived waits use the measured tick, not the nominal TICK_S."""
    burst = healthy(64, backpressure=1, successes=5, tok_s=5000.0, latency_s=3.0)
    default, _ = drive(Auto(initial=64), 64, [burst] * 8)
    fast_ticks, _ = drive(Auto(initial=64), 64, [dict(burst, tick_s=0.5)] * 8)
    assert default == [32, 32, 32, 16, 16, 16, 8, 8], default  # ceil(3 / 2) = 2 ticks
    assert fast_ticks == [32] * 7 + [16], fast_ticks  # ceil(3 / 0.5) = 6 ticks


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


def test_reduction_at_the_floor_is_reported_as_a_hold():
    """Stats count actual reductions: a cut or step-down that cannot go below
    min_limit is a hold, not a cut."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    limit = warm(ctrl, 2)
    traj, reasons = drive(ctrl, limit, [healthy(2, backpressure=1)] + [healthy(2, waiting=100)] * 4)
    assert traj == [2] * 5, traj
    assert reasons[0] == "hold:floor" and reasons[-1] == "hold:floor", reasons
    assert "cut:bp" not in reasons and "cut:queue" not in reasons, reasons


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


def test_input_bound_ticks_are_not_evidence():
    """A window starved by the source completes little and delivers few tokens.
    Those ticks say nothing about the engine; the first evaluation after the
    source resumes must not read them as a plateau."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)  # baseline 1600 tok/s
    starved = healthy(16, inflight=4, successes=2, tok_s=50.0, input_bound=True)
    resumed = healthy(16, tok_s=2000.0)  # the engine got faster meanwhile
    traj, reasons = drive(ctrl, limit, [starved] * 6 + [resumed])
    assert reasons == ["hold:input_bound"] * 6 + ["grow"] and traj[-1] == 32, (traj, reasons)


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


def test_queue_step_down_holds_while_the_old_window_drains():
    """A standing queue with 30 s generations: the step-down cannot show in the
    queue until the old window drains, so stepping again each tick walks the
    limit to the floor. One reduction, then the same cooldown as a halving."""
    ctrl = Auto(target_waiting=8, initial=40, step=STEP)
    limit = warm(ctrl, 40)
    backlog = healthy(limit, waiting=20, latency_s=30.0)  # 20 > hi (16)
    traj, reasons = drive(ctrl, limit, [backlog] * 5)
    assert traj == [32] * 5 and reasons == ["cut:queue"] + ["hold:cooldown"] * 4, (traj, reasons)


def test_cold_engine_queueing_the_initial_window_is_not_a_backlog():
    """A cold engine queues the whole initial window while it warms up. Until
    something has completed, the queue gauge is not a backlog signal and
    slow-start must survive it."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    cold = healthy(32, waiting=32, running=0, successes=0, tok_s=None)
    traj, reasons = drive(ctrl, 32, [cold] * 3 + [healthy(32), healthy(64)])
    assert traj == [32, 32, 32, 64, 128], (traj, reasons)
    assert reasons == ["hold"] * 3 + ["grow"] * 2, reasons


def test_step_down_from_a_probe_keeps_the_probe_baseline():
    """The baseline was measured where the probe started, not at the probed
    limit; a step-down back to it must not scale the baseline down and then
    read the same throughput as growth."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = probing(ctrl, 32)  # baseline 3200 tok/s at 32, probing 40
    traj, reasons = drive(ctrl, limit, [healthy(limit, waiting=100)] + [healthy(32)] * 8)  # flat 3200 at 32
    assert reasons[0] == "cut:queue" and "grow" not in reasons and max(traj) <= 32, (traj, reasons)


def test_partially_used_window_is_not_evidence():
    """A source hiccup leaves the window a quarter full and throughput low.
    Those ticks say nothing about the engine; the next full-window evaluation
    must not read them as a plateau."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)  # baseline 1600 tok/s
    thin = healthy(16, inflight=4, successes=2, tok_s=200.0)
    resumed = healthy(16, tok_s=2000.0)  # the engine got faster meanwhile
    traj, reasons = drive(ctrl, limit, [thin] * 6 + [resumed])
    assert reasons == ["hold"] * 6 + ["grow"] and traj[-1] == 32, (traj, reasons)


def test_at_max_limit_reports_ceiling_and_opens_no_probe():
    ctrl = Auto(target_waiting=8, initial=32, max_limit=32, step=STEP)
    traj, reasons = drive(ctrl, 32, [healthy(32, tok_s=t) for t in (3200.0, 4000.0, 5000.0, 6000.0)])
    assert traj == [32] * 4 and set(reasons) == {"hold:ceiling"}, (traj, reasons)
    traj, reasons = drive(ctrl, 32, [healthy(32, tok_s=6000.0)] * 10)  # flat: a probe cannot open either
    assert traj == [32] * 10 and "probe" not in reasons and ctrl._probe_from is None, (traj, reasons)
    assert "hold:ceiling" in reasons and ctrl._probe_cooldown == 4, reasons


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
    limit = probing(ctrl, 32)
    traj, reasons = drive(ctrl, limit, [healthy(40, waiting=100)] + [healthy(32, tok_s=1000.0)] * 6)
    assert reasons[0] == "cut:queue" and "revert" not in reasons, (traj, reasons)
    assert max(traj) <= 32 and set(reasons[4:]) <= {"hold:ack", "hold:plateau"}, (traj, reasons)


# --- probe-and-revert -------------------------------------------------------


def test_escapes_false_plateau_with_lag():
    """Completions and throughput lag admission by two ticks, so the reading
    after each widening still describes the old window. Slow-start must ride
    through that false plateau and reach a capacity of 512 in a few tens of
    seconds, not creep there a step at a time."""
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 40, lambda lim: min(lim, 512) * 100.0, lag=2)
    assert 512 in traj[:16], f"stuck below capacity despite headroom: {traj}"


def test_bounded_at_real_knee():
    # true knee at 16: tok never improves past it; probes must revert
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 40, lambda lim: min(lim, 16) * 100.0, lag=1)
    assert max(traj) <= 16 * 3, f"ran away past the knee: {traj}"


def test_probe_voided_by_gauge_loss():
    """A probe begun in gauge mode must not survive blind ticks and fire a
    stale revert when gauges return."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = probing(ctrl, 32)
    limit = ctrl.decide(healthy(limit, waiting=None, tok_s=1000.0), limit)  # gauges vanish
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
                                    "hold:ack", "hold:floor", "hold:ceiling"}
