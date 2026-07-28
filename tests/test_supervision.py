"""Codex r3 findings #10/#23: background tasks are awaited, not abandoned — a
crashed tick loop surfaces at limiter exit; invalid public inputs fail at
construction (Fixed(0) previously deadlocked admission forever)."""

import asyncio

import pytest

import pumpjack.core as core
from pumpjack import AdaptiveLimiter, Fixed, ParquetSink
from pumpjack.sink import as_sink


def test_crashed_tick_loop_surfaces(monkeypatch):
    monkeypatch.setattr(core, "TICK_S", 0.01)

    class BadSignals:
        async def read(self):
            raise RuntimeError("scrape bug")

    async def go():
        async with AdaptiveLimiter(signals=BadSignals()):
            await asyncio.sleep(0.1)

    with pytest.raises(RuntimeError, match="scrape bug"):
        asyncio.run(go())


def test_fixed_zero_raises():
    with pytest.raises(ValueError, match=">= 1"):
        Fixed(0)


def test_as_sink_accepts_path(tmp_path):
    assert isinstance(as_sink(tmp_path), ParquetSink)  # Path was mistaken for a custom sink
