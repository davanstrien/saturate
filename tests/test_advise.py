"""The run-end window hints. The server-ceiling hint: `running` pinned below the window while a queue stands
names the engine's relaunch flag. A `running` gauge that stays at 0 under a queue does not
count this work (vLLM pooling routes: /rerank, /classify), so it gives no hint. The
window-cap hint: a window held full at the caller's `max_limit` with nothing pushing back."""

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


def window_ticks(limit: int, n: int, **over) -> list[dict]:
    tick = {"limit": limit, "inflight": limit, "bp": 0, "waiting": None, "input_bound": False}
    return [tick | {"bound_by": "engine"} | over for _ in range(n)]


def test_a_window_held_full_at_max_limit_says_to_raise_it():
    # a hosted API with no gauges: at the cap, with Auto's x7/8 trials below it
    run = window_ticks(64, 62) + window_ticks(56, 26) + window_ticks(32, 2)
    run += window_ticks(64, 10, bound_by="prep")
    (hint,) = advise_window(run, 64)
    assert hint == (
        "WINDOW-CAP: the window was full at or just under max_limit=64 for 88% of the run, with "
        "no backpressure and no queue seen at the engine — Auto could not try a wider window, "
        "so raise max_limit to find out whether one is faster"
    )


def test_no_window_cap_hint_when_something_else_set_the_pace():
    assert advise_window(window_ticks(64, 10), None) == []  # Fixed, or an engine's own hard capacity
    assert advise_window(window_ticks(64, 4), 64) == []  # under five ticks: too short to judge
    assert advise_window(window_ticks(64, 10), 512) == []  # Auto settled well under the cap
    assert advise_window(window_ticks(64, 10, inflight=20), 64) == []  # the window was not full
    assert advise_window(window_ticks(64, 10, waiting=12), 64) == []  # a queue at the engine
    assert advise_window(window_ticks(64, 10, bp=3), 64) == []  # the endpoint pushed back
    assert advise_window(window_ticks(64, 10, input_bound=True), 64) == []
    assert advise_window(window_ticks(64, 10, bound_by="prep"), 64) == []  # slow input has its own hint
    assert advise_window(window_ticks(64, 4) + window_ticks(24, 6), 64) == []  # 40% at the cap


def test_only_a_cap_the_caller_set_on_auto_counts_as_their_own():
    from saturate import _own_cap
    from saturate.signals import Capacity

    assert _own_cap(Auto(max_limit=64)) == 64
    hard = Auto(max_limit=64)
    hard.begin(Capacity(requests=32, queues=False, engine="tei"))  # rejects past 32: the engine's cap
    assert _own_cap(hard) is None
    assert _own_cap(Fixed(64)) is None
