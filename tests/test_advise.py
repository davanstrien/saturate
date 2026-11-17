"""The run-end window hints. The server-ceiling hint: `running` pinned below the window while a queue stands
names the engine's relaunch flag. A `running` gauge that stays at 0 under a queue does not
count this work (vLLM pooling routes: /rerank, /classify), so it gives no hint. The
window-cap hint: Auto steady at the caller's `max_limit`, with every wider move skipped."""

from saturate.controller import Auto, Fixed
from saturate.signals import CEILING_FLAG
from saturate.telemetry import advise, advise_window


def ticks(running: int, waiting: int, n: int = 10) -> list[dict]:
    return [{"t": float(i), "running": running, "waiting": waiting} for i in range(n)]


def test_running_pinned_below_the_window_with_a_queue_names_the_relaunch_flag():
    hints = advise(ticks(running=48, waiting=20), "vllm", final_limit=128, ceiling_flags=CEILING_FLAG)
    assert len(hints) == 1
    assert hints[0].startswith("server ceiling: running pinned at 48")
    assert hints[0].endswith("--max-num-seqs 96")


def test_running_pinned_at_zero_gives_no_hint():
    assert advise(ticks(running=0, waiting=20), "vllm", final_limit=64, ceiling_flags=CEILING_FLAG) == []


def test_running_mostly_at_zero_with_some_positive_ticks_gives_no_hint():
    mixed = ticks(running=0, waiting=20, n=7) + ticks(running=48, waiting=20, n=3)
    assert advise(mixed, "vllm", final_limit=128, ceiling_flags=CEILING_FLAG) == []


def test_no_standing_queue_gives_no_hint():
    assert advise(ticks(running=48, waiting=0), "vllm", final_limit=128, ceiling_flags=CEILING_FLAG) == []


def run(ctrl: Auto, limit: int, ticks: int, tok_of_limit, **over) -> tuple[int, list[dict]]:
    """Drive a real Auto with a full, well-served window delivering tok_of_limit(window), and
    record each tick as pump does: the window the tick ran at, and the rule that decided."""
    out = []
    for _ in range(ticks):
        obs = dict(waiting=None, running=None, inflight=limit, backpressure=0, successes=limit * 4)
        obs |= dict(input_bound=False, kv=None, hits=None, tok_s=tok_of_limit(limit)) | over
        new = ctrl.decide(obs, limit)
        out.append({"limit": limit, "reason": ctrl.last_reason})
        limit = new
    return limit, out


def linear(limit: int) -> float:
    return limit * 100.0


def test_a_run_held_at_max_limit_says_to_raise_it():
    """Throughput still rising at the cap: Auto stays there and skips each wider move."""
    _, ticks = run(Auto(max_limit=64), 64, 200, linear)
    (hint,) = advise_window(ticks, 64)
    assert hint.startswith("WINDOW-CAP: the window was at max_limit=64 for ")
    assert hint.endswith("raise max_limit to find out whether a wider window is faster")
    assert advise_window(ticks, None) == []  # Fixed, or an engine with a capacity of its own
    assert advise_window(ticks[:3], 64) == []  # too short: not every wider move skipped yet
    assert advise_window([], 64) == []


def test_a_backpressure_ceiling_at_max_limit_is_not_the_callers_cap():
    """Backpressure met at 64 holds the window at 56: raising max_limit would change nothing."""
    ctrl = Auto(max_limit=64)
    limit, _ = run(ctrl, 64, 6, linear)
    limit, cut = run(ctrl, limit, 1, linear, backpressure=64)
    assert cut[0]["reason"] == "cut:bp"
    _, ticks = run(ctrl, limit, 100, linear, latency_s=1.0)  # the ceiling is still in force
    assert max(t["limit"] for t in ticks) == 56
    assert any(t["reason"] == "hold:ceiling" for t in ticks)
    assert advise_window(ticks, 64) == []


def test_a_window_auto_chose_just_under_max_limit_is_not_held_by_it():
    """Throughput flat past 56: Auto tries 64, comes back, and settles under the cap."""
    _, ticks = run(Auto(initial=56, max_limit=64), 56, 400, lambda limit: min(limit, 56) * 100.0)
    assert sum(t["limit"] >= 56 for t in ticks) / len(ticks) > 0.5  # near the cap most of the run
    assert advise_window(ticks, 64) == []


def test_a_full_window_with_nothing_completing_gives_no_hint():
    """A few slow rows fill the window: no completions, so Auto has no evidence and skips no move."""
    _, ticks = run(Auto(max_limit=6), 6, 30, lambda limit: 0.0, successes=0)
    assert advise_window(ticks, 6) == []


def test_only_a_cap_on_auto_with_no_engine_capacity_is_the_callers_own():
    from saturate import _own_cap
    from saturate.signals import Capacity

    assert _own_cap(Auto(max_limit=64)) == 64
    assert _own_cap(Fixed(64)) is None
    queued = Auto(max_limit=64)
    queued.begin(Capacity(requests=32, queues=True, engine="sglang"))  # queues the rest: no hard cap
    assert _own_cap(queued) == 64
    for capacity in (32, 64, 128):  # rejects past its capacity, whatever max_limit is
        hard = Auto(max_limit=64)
        hard.begin(Capacity(requests=capacity, queues=False, engine="tei"))
        assert _own_cap(hard) is None
