"""Auto controller scenarios, driven as Obs trajectories through decide().

Each test names the engine behaviour it pins: a wedged engine, a burst of
429s, a fast idle engine, long generations, a plateau that lags the tick, and
the hill climber's own trial, judge and ceiling rules."""

import typing
from typing import Any

import pytest

from saturate.controller import BP_CEILING_RELEASE_S, TICK_S, Auto, Reason, Settling, Trial

STEP = 8
REASONS = set(typing.get_args(Reason))


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


def climb(ctrl, limit, ticks, tok_of_limit, **over):
    """Feed `ticks` healthy observations of the current window, delivering tok_of_limit(window)."""
    traj, reasons = [], []
    for _ in range(ticks):
        limit = ctrl.decide(healthy(limit, tok_s=tok_of_limit(limit), **over), limit)
        traj.append(limit)
        reasons.append(ctrl.last_reason)
    return traj, reasons


def climb_until(ctrl, limit, reason, tok_of_limit, ticks=2000, **over):
    """Climb until `reason` fires; returns (window before, window after, ticks taken, reasons so far)."""
    reasons = []
    for n in range(1, ticks + 1):
        new = ctrl.decide(healthy(limit, tok_s=tok_of_limit(limit), **over), limit)
        reasons.append(ctrl.last_reason)
        if ctrl.last_reason == reason:
            return limit, new, n, reasons
        limit = new
    raise AssertionError(f"{reason} never fired: {reasons[-20:]}")


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
    """Ticks with the queue in band (hold): completions arrive and the window is evidence."""
    for _ in range(ticks):
        assert ctrl.decide(healthy(limit, waiting=4), limit) == limit
    return limit


def flat(_limit):
    return 1000.0


def knee(at):
    """Throughput that grows with the window up to `at`, then stays flat."""
    return lambda limit: min(limit, at) * 100.0


# --- stalled engine ---------------------------------------------------------


def test_stalled_engine_is_cut_not_grown():
    """Window full, nothing completing, zero tokens: the signature of a wedged
    engine. The queue reads empty only because nothing new is admitted."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    limit = warm(ctrl, 32)
    stalled = [healthy(limit, waiting=1, successes=0, tok_s=0.0)] * 8
    traj, reasons = drive(ctrl, limit, [dict(o, inflight=limit) for o in stalled])
    assert max(traj) <= limit, f"grew into a stalled engine: {traj}"
    assert "trial:up" not in reasons, reasons
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
    recovered = healthy(8, successes=4, tok_s=400.0, latency_s=2.0)  # completions resume...
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
    """The cooldown after a cut lasts max(2 ticks, the latency), or 3 ticks while the latency is unknown."""
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


def test_reduction_at_the_floor_is_reported_as_a_hold():
    """Stats count actual reductions: a cut or step-down that cannot go below
    min_limit is a hold, not a cut."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    limit = warm(ctrl, 2)
    traj, reasons = drive(ctrl, limit, [healthy(2, backpressure=1)] + [healthy(2, waiting=100)] * 4)
    assert traj == [2] * 5, traj
    assert reasons[0] == "hold:floor" and reasons[-1] == "hold:floor", reasons
    assert "cut:bp" not in reasons and "cut:queue" not in reasons, reasons


def test_queue_step_down_at_the_floor_is_reported_as_a_hold():
    """A step-down for a standing queue that cannot go below min_limit is hold:floor, not a cut."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    traj, reasons = drive(ctrl, warm(ctrl, 2), [healthy(2, waiting=100)] * 4)
    assert traj == [2] * 4 and set(reasons) == {"hold:floor"}, (traj, reasons)


def test_backpressure_cuts_on_a_share_of_outcomes_not_on_any_event():
    """One row that always 500s is one backpressure event among hundreds of successes at a
    large window: not overload, so no cut. 14 events in 270 outcomes (5.2%) is overload."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    assert ctrl.decide(healthy(limit, backpressure=1, successes=256), limit) >= limit
    assert ctrl.last_reason != "cut:bp"
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    assert ctrl.decide(healthy(limit, backpressure=14, successes=256), limit) == limit // 2
    assert ctrl.last_reason == "cut:bp"


def test_backpressure_with_no_successes_always_cuts():
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = warm(ctrl, 64)
    assert ctrl.decide(healthy(limit, backpressure=1, successes=0), limit) == limit // 2
    assert ctrl.last_reason == "cut:bp"


def test_bp_frac_zero_cuts_on_any_event():
    ctrl = Auto(target_waiting=8, initial=64, step=STEP, bp_frac=0.0)
    limit = warm(ctrl, 64)
    assert ctrl.decide(healthy(limit, backpressure=1, successes=256), limit) == limit // 2
    assert ctrl.last_reason == "cut:bp"


def test_a_fixed_set_of_failing_rows_cuts_once_but_sustained_failures_keep_cutting():
    """Across ticks. Rows that fail and then only retry count once (poison-shaped): 13 of 269
    outcomes is under bp_frac, so no cut; with a lower bp_frac they cut once, not again on
    every retry. Failures on 6% of NEW rows every tick (overload-shaped) keep cutting."""
    def run(bp_frac, first, rest):
        ctrl = Auto(target_waiting=8, initial=128, max_limit=128, step=STEP, bp_frac=bp_frac)
        limit = warm(ctrl, 128)
        return drive(ctrl, limit, [first] + rest)

    stuck, clean = healthy(128, backpressure=13, successes=256), [healthy(128)] * 8
    _, reasons = run(0.05, stuck, clean)
    assert "cut:bp" not in reasons, reasons
    _, reasons = run(0.04, stuck, clean)
    assert reasons.count("cut:bp") == 1, reasons

    overload = healthy(128, backpressure=16, successes=250)
    traj, reasons = run(0.05, overload, [overload] * 11)
    assert reasons.count("cut:bp") >= 2 and min(traj) < 64, (traj, reasons)


# --- evidence windows -------------------------------------------------------


def test_trial_waits_for_a_window_of_completions():
    """Long generations: a trial opens only once `limit` completions (and two throughput
    samples) have arrived at the current window."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    slow = healthy(16, successes=4, tok_s=400.0)  # 4 completions per tick at 16
    traj, reasons = drive(ctrl, 16, [slow] * 4)
    assert traj == [16, 16, 16, 32], (traj, reasons)  # 16 acks arrive on the 4th tick
    assert reasons == ["hold:ack"] * 3 + ["trial:up"], reasons


def test_input_bound_ticks_are_not_evidence():
    """A window starved by the source completes little and delivers few tokens. Those ticks
    say nothing about the engine: the next trial's reference reading excludes them."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)
    starved = healthy(16, inflight=4, successes=2, tok_s=50.0, input_bound=True)
    resumed = healthy(16, tok_s=2000.0)
    traj, reasons = drive(ctrl, limit, [starved] * 6 + [resumed])
    assert reasons == ["hold:input_bound"] * 6 + ["trial:up"] and traj[-1] == 32, (traj, reasons)
    assert isinstance(ctrl.state.phase, Trial), ctrl.state.phase
    assert ctrl.state.phase.ref.tok_s == 2000.0, ctrl.state.phase


def test_partially_used_window_is_not_evidence():
    """A source hiccup leaves the window a quarter full and throughput low. Those ticks say
    nothing about the engine: the next trial's reference reading excludes them."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)
    thin = healthy(16, inflight=4, successes=2, tok_s=200.0)
    resumed = healthy(16, tok_s=2000.0)
    traj, reasons = drive(ctrl, limit, [thin] * 6 + [resumed])
    assert reasons == ["hold"] * 6 + ["trial:up"] and traj[-1] == 32, (traj, reasons)
    assert isinstance(ctrl.state.phase, Trial), ctrl.state.phase
    assert ctrl.state.phase.ref.tok_s == 2000.0, ctrl.state.phase


def test_cold_start_neither_grows_nor_stalls():
    """Before the first completion the request latency is unknown: a full window
    with nothing back yet is not evidence of headroom, nor yet of a wedge."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    traj, reasons = drive(ctrl, 16, [healthy(16, successes=0, tok_s=None)] * 5)
    assert traj == [16] * 5 and set(reasons) == {"hold:ack"}, (traj, reasons)


def test_no_trial_on_a_zero_reading():
    """Completions arriving with zero tokens delivered give no basis for a trial."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    traj, reasons = drive(ctrl, 16, [healthy(16, tok_s=0.0)] * 10)
    assert traj == [16] * 10 and set(reasons) == {"hold:ack"}, (traj, reasons)


def test_no_trial_while_the_window_is_silent():
    """An evidence window is in, but the window is full with nothing completing this tick:
    no trial opens until completions resume."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = warm(ctrl, 16)
    traj, reasons = drive(ctrl, limit, [healthy(16, successes=0, tok_s=1600.0)])
    assert (traj, reasons) == ([16], ["hold:ack"])
    traj, reasons = drive(ctrl, limit, [healthy(16)])
    assert (traj, reasons) == ([32], ["trial:up"])


def test_completions_per_second_stand_in_when_tokens_are_unreported():
    """With tok_s=None the throughput reading is completions per second: completions that
    grow with the window keep a wider trial; flat completions revert it."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    traj, reasons = drive(ctrl, 16, [healthy(16, tok_s=None, successes=32)] * 2)
    assert reasons[-1] == "trial:up" and isinstance(ctrl.state.phase, Trial), (reasons, ctrl.state.phase)
    assert ctrl.state.phase.ref.rows_s == 32 / TICK_S, ctrl.state.phase
    traj, reasons = drive(ctrl, 32, [healthy(32, tok_s=None, successes=64)] * 2)
    assert reasons[-1] == "hold:kept" and traj[-1] == 32, reasons

    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit, reasons = 16, []
    while "revert" not in reasons:
        limit = ctrl.decide(healthy(limit, tok_s=None, successes=64), limit)  # flat completions
        reasons.append(ctrl.last_reason)
    assert reasons.count("trial:up") == 1 and limit == 16, reasons


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
    something has completed, the queue gauge is not a backlog signal."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    cold = healthy(32, waiting=32, running=0, successes=0, tok_s=None)
    traj, reasons = drive(ctrl, 32, [cold] * 3 + [healthy(32)])
    assert traj == [32, 32, 32, 64], (traj, reasons)
    assert reasons == ["hold"] * 3 + ["trial:up"], reasons


# --- the climber: trials, judging, backoff ----------------------------------


def test_a_kept_trial_restarts_moves_at_x2():
    """A trial that gains is kept and the next trial is x2 again, so while each doubling pays
    the window keeps doubling (slow-start)."""
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj, reasons = climb(ctrl, 8, 16, knee(512))
    trials = [lim for lim, r in zip(traj, reasons, strict=True) if r == "trial:up"]
    assert trials == [16, 32, 64, 128], (traj, reasons)
    assert "revert" not in reasons and reasons.count("hold:kept") >= 3, reasons


def test_a_gain_is_kept_before_the_trial_settles():
    """A gain from a wider window counts as soon as its evidence is in, even with a long
    latency still to wait out."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    traj, reasons = climb(ctrl, 16, 4, knee(512), latency_s=30.0)
    assert reasons == ["hold:ack", "trial:up", "hold:ack", "hold:kept"] and traj[-1] == 32, reasons


def test_a_trial_without_gain_is_reverted_only_once_settled():
    """A wider trial that does not gain is reverted only after three ticks and two latencies
    scaled by new/old window (Little's law); until then it holds its window."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    origin, trial, _, _ = climb_until(ctrl, 16, "trial:up", flat, latency_s=5.0)
    assert (origin, trial) == (16, 32)
    settle_s = 2 * 5.0 * (32 / 16)  # 20 s
    traj, reasons = climb(ctrl, trial, 20, flat, latency_s=5.0)
    judged = reasons.index("revert") + 1  # ticks since the trial opened
    assert judged * TICK_S >= settle_s and judged == settle_s / TICK_S, (judged, reasons)
    assert set(reasons[:judged - 1]) == {"hold:ack"} and traj[:judged - 1] == [32] * (judged - 1)
    assert traj[judged - 1] == 16

    ctrl = Auto(target_waiting=8, initial=16, step=STEP)  # no latency: three ticks
    climb_until(ctrl, 16, "trial:up", flat)
    _, reasons = climb(ctrl, 32, 3, flat)
    assert reasons == ["hold:ack", "hold:ack", "revert"], reasons


def test_reverts_back_off_and_walk_through_moves():
    """Each revert moves on to the next factor of MOVES (x2, x4, x1.5, x1.25, x7/8) and doubles
    the evidence windows waited before the next trial (1, 2, 4 ... capped at 32)."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit, trials, plateaus, waited = 16, [], [], 0
    for _ in range(3000):
        new = ctrl.decide(healthy(limit, tok_s=knee(16)(limit)), limit)
        if ctrl.last_reason == "hold:plateau":
            waited += 1
        if ctrl.last_reason in ("trial:up", "trial:down"):
            trials.append(new)
            plateaus.append(waited)
            waited = 0
        limit = new
        if len(trials) == 7:
            break
    assert trials == [32, 64, 24, 20, 14, 32, 64], trials
    assert plateaus == [0, 1, 3, 7, 15, 31, 31], plateaus  # backoff - 1 windows each
    assert limit == 64  # the last trial is still open


def test_a_trial_holds_its_window_until_judged():
    """While a trial is pending nothing else moves the window: no further trial, no idle hold,
    only hold:ack until its verdict."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    climb_until(ctrl, 16, "trial:up", flat, latency_s=20.0)
    traj, reasons = climb(ctrl, 32, 39, flat, latency_s=20.0)  # settles after 2 * 20 * 2 = 80 s
    assert traj == [32] * 39 and set(reasons) == {"hold:ack"}, (traj, reasons)
    traj, reasons = climb(ctrl, 32, 1, flat, latency_s=20.0)
    assert (traj, reasons) == ([16], ["revert"])


def test_a_cut_during_a_trial_goes_no_higher_than_its_origin():
    """A cut while a trial is open fails the trial: the window drops to at most where the trial
    began, no revert follows, and the next trial uses the next move after a doubled backoff."""
    ctrl = Auto(target_waiting=8, initial=32, step=STEP)
    origin, trial, _, _ = climb_until(ctrl, 32, "trial:up", flat, latency_s=1.0)
    assert (origin, trial) == (32, 64)
    traj, reasons = drive(ctrl, trial, [healthy(trial, waiting=100, latency_s=1.0)])  # step 64 -> 56
    assert (traj, reasons) == ([32], ["cut:queue"])
    _, after, _, seen = climb_until(ctrl, 32, "trial:up", flat, latency_s=1.0)
    assert "revert" not in seen and seen.count("hold:plateau") == 1, seen
    assert after == 128, after  # x4, the move after the failed x2

    ctrl = Auto(target_waiting=8, initial=16, max_limit=1024, step=STEP)
    climb_until(ctrl, 16, "revert", flat, latency_s=1.0)  # x2 reverted: next is x4
    origin, trial, _, _ = climb_until(ctrl, 16, "trial:up", flat, latency_s=1.0)
    assert (origin, trial) == (16, 64)
    new = ctrl.decide(healthy(trial, backpressure=50, successes=50, latency_s=1.0), trial)
    assert (ctrl.last_reason, new) == ("cut:bp", 16)  # not half the trial window


def test_a_down_trial_is_kept_when_throughput_holds():
    """x7/8 is kept if throughput falls by less than `improve`: a narrower window that delivers
    the same is better."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    origin, trial, _, _ = climb_until(ctrl, 16, "trial:down", flat, latency_s=1.0)
    assert (origin, trial) == (16, 14)
    _, after, _, seen = climb_until(ctrl, trial, "hold:kept", flat, latency_s=1.0)
    assert after == 14 and "revert" not in seen, seen


def test_a_run_of_down_trials_is_judged_against_where_it_began():
    """Each kept x7/8 is judged against the reading where the run of moves down began, so small
    losses cannot add up: on a curve where every 7/8 step loses ~3%, the window stops within
    `improve` (5%) of the anchor instead of creeping toward the floor."""
    def gentle(limit):  # ~3% less throughput per 7/8 step below 128
        return 10000.0 * (min(limit, 128) / 128) ** 0.23

    ctrl = Auto(target_waiting=8, initial=128, max_limit=128, step=STEP)
    traj, reasons = climb(ctrl, 128, 1500, gentle, latency_s=1.0)
    judged = ("trial:down", "hold:kept", "revert")
    moves = [(lim, r) for lim, r in zip(traj, reasons, strict=True) if r in judged]
    assert moves[:3] == [(112, "trial:down"), (112, "hold:kept"), (98, "trial:down")], moves
    assert moves[3] == (112, "revert"), moves  # 98 loses 6% against 128, though only 3% against 112
    kept = {lim for lim, r in zip(traj, reasons, strict=True) if r == "hold:kept"}
    assert min(kept) >= 98 and min(traj) >= 86, (kept, sorted(set(traj)))
    assert traj[-200:].count(128) > 0, traj[-200:]  # and the climber finds its way back up


def test_at_max_limit_reports_hold_ceiling_and_opens_no_wider_trial():
    """At max_limit every wider move is empty: each is skipped as hold:ceiling, never a trial:up."""
    ctrl = Auto(target_waiting=8, initial=32, max_limit=32, step=STEP)
    traj, reasons = climb(ctrl, 32, 8, knee(512))
    assert traj == [32] * 8 and "trial:up" not in reasons, (traj, reasons)
    assert reasons.count("hold:ceiling") == 4, reasons  # x2, x4, x1.5, x1.25
    assert max(climb(ctrl, 32, 200, knee(512))[0]) == 32


def test_at_min_limit_reports_hold_floor_and_opens_no_narrower_trial():
    """At min_limit the x7/8 move is empty: it is skipped as hold:floor, never a trial:down."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    traj, reasons = climb(ctrl, 2, 400, flat)
    assert "trial:down" not in reasons and "hold:floor" in reasons, reasons
    assert min(traj) == 2


# --- the climber: backpressure ceiling ---------------------------------------


def ceiling_at_64(ctrl):
    """Warm at 64, meet backpressure there (ceiling 64, cut to 32), and wait out the cooldown."""
    limit = warm(ctrl, 64)
    limit = ctrl.decide(healthy(64, backpressure=64, latency_s=1.0), limit)
    assert (ctrl.last_reason, limit) == ("cut:bp", 32)
    traj, _ = climb(ctrl, limit, 2, knee(512), latency_s=1.0)
    return traj[-1]


def test_a_backpressure_ceiling_stops_growth_an_eighth_below_it():
    """After backpressure at 64, growth stops at 56 even while throughput would still rise."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    traj, reasons = climb(ctrl, limit, 100, knee(512), latency_s=1.0)
    assert max(traj) == 56 and traj[-1] == 56, (traj, reasons)
    assert "hold:ceiling" in reasons


def test_no_down_trial_at_a_backpressure_ceiling():
    """Sitting just under a backpressure ceiling, the x7/8 move is skipped: nothing waits there."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    traj, reasons = climb(ctrl, limit, 120, knee(512), latency_s=1.0)
    assert traj[-1] == 56 and "trial:down" not in reasons, (traj, reasons)


def test_a_skipped_down_move_at_a_backpressure_ceiling_is_not_reported_as_the_floor():
    """hold:floor means the window is at min_limit; skipping x7/8 under a ceiling is hold:ceiling."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    traj, reasons = climb(ctrl, limit, 100, knee(512), latency_s=1.0)
    assert "hold:floor" not in reasons, reasons


def test_re_meeting_a_known_ceiling_steps_under_it():
    """Backpressure again at a ceiling met before steps the window an eighth down, not in half."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    traj, _ = climb(ctrl, limit, 100, knee(512), latency_s=1.0)
    assert traj[-1] == 56
    new = ctrl.decide(healthy(56, backpressure=64, latency_s=1.0), 56)
    assert (ctrl.last_reason, new) == ("cut:bp", 49)


def test_a_ceiling_is_released_after_the_release_time():
    """With no backpressure for BP_CEILING_RELEASE_S, growth past the old ceiling resumes, and
    meeting that same limit again steps under it rather than halving."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    ticks = int(BP_CEILING_RELEASE_S / TICK_S)
    traj, _ = climb(ctrl, limit, ticks - 5, knee(512), latency_s=1.0)
    assert max(traj) == 56, max(traj)
    traj, _ = climb(ctrl, traj[-1], 200, knee(512), latency_s=1.0)
    assert max(traj) > 64, max(traj)

    ctrl = Auto(target_waiting=8, initial=64, step=STEP, seed=0)
    limit = ceiling_at_64(ctrl)
    traj, _ = climb(ctrl, limit, ticks - 5, knee(512), latency_s=1.0)  # held at 56, not yet released
    _, limit, _, _ = climb_until(ctrl, traj[-1], "trial:up", knee(512), latency_s=1.0)
    assert 64 < limit <= 72, limit  # the first probe past the released ceiling goes an eighth past it
    new = ctrl.decide(healthy(limit, backpressure=limit, latency_s=1.0), limit)
    assert (ctrl.last_reason, new) == ("cut:bp", 56)


def test_a_released_ceiling_does_not_cap_a_climb_under_a_higher_active_one():
    """The probe cap is for the first move past a released ceiling. Once a higher ceiling is
    active (met since, so the engine ran above the old one), only the active ceiling bounds a
    climb: a cut back under the old one does not make the climb stop an eighth past it."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = ceiling_at_64(ctrl)
    ticks = int(BP_CEILING_RELEASE_S / TICK_S)
    traj, _ = climb(ctrl, limit, ticks + 100, knee(512), latency_s=1.0)  # released, and climbed past it
    assert ctrl.state.ceiling is None and ctrl.state.seen_ceiling is not None and max(traj) > 128, traj
    high = max(traj)
    ctrl.decide(healthy(high, backpressure=high, latency_s=1.0), high)  # a new, higher ceiling
    assert ctrl.state.ceiling is not None and ctrl.state.ceiling.limit == high
    _, new, _, _ = climb_until(ctrl, 40, "trial:up", knee(512), latency_s=1.0)  # from under the old one
    assert new > 72, new  # not stopped at an eighth past the released ceiling of 64


def test_a_released_ceiling_at_the_floor_is_probed_an_eighth_past():
    """min_limit at the ceiling: the cut cannot go under it, so the window is still at the
    ceiling when it is released. The first probe still goes only an eighth past it."""
    ctrl = Auto(target_waiting=8, initial=64, min_limit=64, step=STEP)
    limit = warm(ctrl, 64)
    limit = ctrl.decide(healthy(64, backpressure=64, latency_s=1.0), limit)
    assert ctrl.state.ceiling is not None
    assert (ctrl.last_reason, limit, ctrl.state.ceiling.limit) == ("hold:floor", 64, 64)
    ticks = int(BP_CEILING_RELEASE_S / TICK_S)
    traj, reasons = climb(ctrl, limit, ticks + 60, knee(512), latency_s=1.0)
    first_up = next(new for new, reason in zip(traj, reasons, strict=True) if reason == "trial:up")
    assert first_up == 72, (first_up, traj)


# --- whole runs -------------------------------------------------------------


def test_escapes_false_plateau_with_lag():
    """Completions and throughput lag admission by two ticks, so the first readings after
    each widening still describe the old window. The climber must still reach a capacity
    of 512, not stall at a false plateau."""
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 60, lambda lim: min(lim, 512) * 100.0, lag=2)
    assert 512 in traj, f"stuck below capacity despite headroom: {traj}"


def test_bounded_at_real_knee():
    """True knee at 16: throughput never improves past it, so wider trials revert."""
    ctrl = Auto(target_waiting=8, initial=8, step=STEP)
    traj = drive_lagged(ctrl, 8, 200, lambda lim: min(lim, 16) * 100.0, lag=1)
    assert max(traj) <= 16 * 4, f"ran away past the knee: {traj}"
    assert sum(t == 16 for t in traj[50:]) >= 0.7 * len(traj[50:]), traj


@pytest.mark.parametrize("cap", [32, 64, 256])
def test_blind_growth_reaches_the_engine_capacity(cap):
    """No gauges: growth is judged on throughput alone. The climber reaches capacity and
    spends most of its time within an eighth of it, trials included."""
    ctrl, limit, hist = Auto(initial=16, max_limit=512), 16, [16]
    for _ in range(400):
        served = min(hist[max(0, len(hist) - 2)], cap)  # throughput lags the window by a tick
        obs = dict(waiting=None, running=None, inflight=limit, backpressure=0, successes=served,
                   input_bound=False, kv=None, hits=None, tok_s=served * 100.0)
        limit = ctrl.decide(obs, limit)
        hist.append(limit)
    tail = hist[-200:]
    near = sum(cap * 7 // 8 <= h <= cap + cap // 8 + 1 for h in tail)
    assert near >= 0.8 * len(tail), tail
    assert max(tail) <= 4 * cap, tail


@pytest.mark.parametrize("cap", [64, 256])
def test_blind_growth_against_a_hard_429_wall_stays_bounded(cap):
    """An API that 429s as soon as concurrency passes its limit: blind growth reaches the
    wall, cuts, and grows back. It must never run away past the wall, and on average use a
    good share of it."""
    ctrl, limit, hist = Auto(initial=16, max_limit=512), 16, [16]
    for _ in range(3000):
        seen = hist[max(0, len(hist) - 2)]
        served = min(seen, cap)
        bp = max(1, served // 12) if seen > cap else 0  # ~8% of outcomes rejected above the wall
        obs = dict(waiting=None, running=None, inflight=limit, backpressure=bp, successes=served,
                   input_bound=False, kv=None, hits=None, tok_s=served * 100.0)
        limit = ctrl.decide(obs, limit)
        hist.append(limit)
    tail = hist[500:]
    assert max(tail) <= 2 * cap, max(tail)
    assert sum(min(h, cap) for h in tail) / len(tail) >= 0.6 * cap, sum(tail) / len(tail)


# --- construction -----------------------------------------------------------


def test_initial_clamped_to_max():
    assert Auto(initial=600, max_limit=512).initial == 512
    assert Auto(initial=1, min_limit=2).initial == 2


def test_auto_invalid_bounds_raise():
    """Auto(min_limit=0) deadlocks admission exactly like Fixed(0)."""
    cases: tuple[dict[str, Any], ...] = (
        dict(min_limit=0), dict(max_limit=1, min_limit=2), dict(step=0), dict(stall_ticks=0),
        dict(bp_frac=-0.1), dict(bp_frac=1.5))
    for kw in cases:
        with pytest.raises(ValueError):
            Auto(**kw)


def test_decide_reports_a_reason_for_every_tick():
    """Every tick names one reason from the documented set."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    assert ctrl.last_reason == "hold"
    seen = set()
    limit = 16
    obs_seq = ([healthy(16)] * 6 + [healthy(32, backpressure=1, successes=0)]
               + [healthy(16, input_bound=True)] + [healthy(16, kv=0.95, hits=0.1)]
               + [healthy(16, waiting=100)] * 3 + [healthy(8)] * 40)
    for obs in obs_seq:
        limit = ctrl.decide(dict(obs, inflight=limit), limit)
        assert ctrl.last_reason in REASONS, ctrl.last_reason
        seen.add(ctrl.last_reason)
    assert len(seen) >= 5, seen


def open_trial(ctrl, limit, obs):
    """Feed `obs` (its inflight following the window) until the first trial opens; the new limit."""
    for _ in range(10):
        new = ctrl.decide(dict(obs, inflight=limit), limit)
        if ctrl.last_reason == "trial:up":
            return new
        limit = new
    raise AssertionError("no trial opened")


def test_a_trial_is_judged_on_its_own_window_readings():
    """30 s requests: readings from the first two latencies after a move still describe the old
    window. A real 6% gain is kept once the trial window's own readings arrive, not averaged
    away with the old ones (then the next trial opens)."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = open_trial(ctrl, 16, healthy(16, tok_s=100.0, latency_s=30.0, waiting=None))
    assert limit == 32
    reasons = []
    for _ in range(28):  # 56 s: still the old window's completions
        limit = ctrl.decide(healthy(limit, tok_s=100.0, latency_s=30.0, waiting=None), limit)
        reasons.append(ctrl.last_reason)
    for _ in range(32):  # the trial window's own readings: up 6%
        limit = ctrl.decide(healthy(limit, tok_s=106.0, latency_s=30.0, waiting=None), limit)
        reasons.append(ctrl.last_reason)
    assert "hold:kept" in reasons and "revert" not in reasons[:reasons.index("hold:kept")], reasons


def test_a_latency_first_seen_mid_trial_extends_the_wait():
    """The trial opened before any latency was known; a 30 s latency reported afterwards makes it
    wait two latencies at the trial window before a verdict, not three ticks."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = open_trial(ctrl, 16, healthy(16, tok_s=100.0, waiting=None))
    traj, reasons = drive(ctrl, limit, [healthy(limit, tok_s=100.0, latency_s=30.0, waiting=None)] * 20)
    assert "revert" not in reasons and traj == [32] * 20, (traj, reasons)


def test_throughput_units_are_compared_like_with_like():
    """The reference was taken in completions/s (no tokens reported); tokens appearing during the
    trial are not a gain: the trial is judged in completions/s."""
    ctrl = Auto(target_waiting=8, initial=16, step=STEP)
    limit = open_trial(ctrl, 16, healthy(16, tok_s=None, waiting=None, successes=16))
    traj, reasons = drive(ctrl, limit, [healthy(limit, tok_s=100.0, waiting=None, successes=16)] * 8)
    assert "hold:kept" not in reasons, reasons


def test_a_backpressure_ceiling_is_released_after_a_clean_stretch():
    """However much backpressure came before (a falling limit, an outage), a ceiling is released
    BP_CEILING_RELEASE_S after it was last met: a provider's limit can rise. The old ceiling is
    kept as seen_ceiling, so meeting it again steps under it rather than halving."""
    ctrl = Auto(target_waiting=8, initial=2, min_limit=2, step=STEP)
    limit = 2
    for _ in range(40):
        limit = ctrl.decide(healthy(limit, backpressure=1, successes=0, waiting=None), limit)
    assert ctrl.state.ceiling is not None
    for _ in range(int(BP_CEILING_RELEASE_S / TICK_S) + 2):
        limit = ctrl.decide(healthy(limit, waiting=None), limit)
    assert ctrl.state.ceiling is None and ctrl.state.seen_ceiling is not None

def test_a_queue_cut_steps_down_by_the_excess_within_bounds():
    """The gauge says how far past the engine's capacity the window is: the cut steps down by
    the excess queue, at least `step` and at most half the window (another client's requests
    may be queued too)."""
    for waiting, expected in [(20, 64 - STEP), (40, 64 - 24), (500, 32)]:
        ctrl = Auto(target_waiting=8, initial=64, step=STEP)
        limit = warm(ctrl, 64)
        assert ctrl.decide(healthy(limit, waiting=waiting), limit) == expected, waiting
        assert ctrl.last_reason == "cut:queue"


def test_evidence_counts_only_the_completions_of_the_samples_kept():
    """Samples beyond the most recent MAX_SAMPLES ticks are dropped, and so are their
    completions: a burst long before cannot qualify an evidence window whose kept ticks
    delivered nothing."""
    from saturate.controller import MAX_SAMPLES

    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    limit = 64
    slow = dict(waiting=None, latency_s=60.0)  # 60 s requests: a quiet window is no stall yet
    ctrl.decide(healthy(limit, successes=64, tok_s=6400.0, **slow), limit)  # one burst
    for _ in range(MAX_SAMPLES + 2):  # then nothing completes
        ctrl.decide(healthy(limit, successes=0, tok_s=0.0, **slow), limit)
    assert ctrl.evidence(ctrl.state, limit) is None


FULL = dict(kv=0.99, hits=0.0)  # a full KV cache, unique prompts


def test_preemptions_are_counted_from_the_first_reading_and_a_reset_is_not_a_preemption():
    """The engine's counter is cumulative: its first reading is the baseline (preemptions
    before this run are not this window's), and a counter that goes down (a restarted engine)
    counts none. A full cache with a preemption this tick halves the window."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    ctrl.decide(healthy(64, preempts=500, **FULL), 64)
    assert ctrl.state.preempted == 0 and ctrl.last_reason != "cut:kv"
    ctrl.decide(healthy(64, preempts=3, **FULL), 64)
    assert ctrl.state.preempted == 0 and ctrl.last_reason != "cut:kv"
    new = ctrl.decide(healthy(64, preempts=4, **FULL), 64)
    assert (ctrl.state.preempted, ctrl.last_reason, new) == (1, "cut:kv", 32)


def test_a_full_kv_cache_without_preemptions_is_not_a_cut_where_they_are_counted():
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    for _ in range(5):
        ctrl.decide(healthy(64, preempts=7, **FULL), 64)
        assert ctrl.last_reason != "cut:kv"


def test_a_preemption_with_room_in_the_cache_is_not_a_cut():
    """A preemption while KV is low (another client's burst passed) is not this window's limit."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    ctrl.decide(healthy(64, preempts=0), 64)
    ctrl.decide(healthy(64, preempts=5), 64)
    assert ctrl.last_reason != "cut:kv"


def test_preemptions_during_a_cooldown_are_not_charged_to_the_next_window():
    """A cut's cooldown sees the old window drain: preemptions then belong to it. The counter's
    baseline moves every tick, so the first tick after the cooldown counts only its own."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    ctrl.decide(healthy(64, preempts=0, **FULL), 64)
    limit = ctrl.decide(healthy(64, preempts=10, **FULL), 64)
    assert (ctrl.last_reason, limit) == ("cut:kv", 32)
    count, cooled = 10, 0
    while isinstance(ctrl.state.phase, Settling) and ctrl.state.clock + TICK_S <= ctrl.state.phase.until:
        count += 10  # the draining old window keeps preempting
        limit = ctrl.decide(healthy(limit, preempts=count, **FULL), limit)
        assert ctrl.last_reason == "hold:cooldown"
        cooled += 1
    assert cooled >= 1
    limit = ctrl.decide(healthy(limit, preempts=count, **FULL), limit)  # the new window: none
    assert ctrl.last_reason != "cut:kv" and limit == 32


def test_a_gap_in_the_counter_starts_a_new_baseline():
    """A reading missing (the scrape failed) between two counts: what happened in the gap is
    not charged to the window that follows it."""
    ctrl = Auto(target_waiting=8, initial=64, step=STEP)
    ctrl.decide(healthy(64, preempts=100, **FULL), 64)
    ctrl.decide(healthy(64, preempts=None, kv=None, hits=None), 64)
    ctrl.decide(healthy(64, preempts=900, **FULL), 64)
    assert ctrl.state.preempted == 0 and ctrl.last_reason != "cut:kv"
