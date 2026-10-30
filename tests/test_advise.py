"""The run-end server-ceiling hint: `running` pinned below the window while a queue stands
names the engine's relaunch flag. A `running` gauge that stays at 0 while rows complete does
not count this work (vLLM pooling routes: /rerank, /classify), so it gives no hint."""

from saturate.signals import CEILING_FLAG
from saturate.telemetry import advise


def ticks(running: int, waiting: int, n: int = 10) -> list[dict]:
    return [{"t": float(i), "running": running, "waiting": waiting} for i in range(n)]


def test_running_pinned_below_the_window_with_a_queue_names_the_relaunch_flag():
    hints = advise(ticks(running=48, waiting=20), "vllm", final_limit=128, ceiling_flags=CEILING_FLAG)
    assert len(hints) == 1
    assert hints[0].startswith("server ceiling: running pinned at 48")


def test_running_pinned_at_zero_gives_no_hint():
    assert advise(ticks(running=0, waiting=20), "vllm", final_limit=64, ceiling_flags=CEILING_FLAG) == []


def test_no_standing_queue_gives_no_hint():
    assert advise(ticks(running=48, waiting=0), "vllm", final_limit=128, ceiling_flags=CEILING_FLAG) == []
