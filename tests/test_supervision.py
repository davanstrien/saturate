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


def test_client_closed_when_tick_loop_crashed(monkeypatch):
    """Codex r4 blocker #6: surfacing a crashed tick loop at limiter exit must
    not leak the HTTP client."""
    from pumpjack import AdaptiveClient

    monkeypatch.setattr(core, "TICK_S", 0.01)

    class BadSignals:
        async def read(self):
            raise RuntimeError("scrape bug")

    holder = []

    async def go():
        async with AdaptiveClient("http://127.0.0.1:1", signal_source="none") as c:
            holder.append(c)
            c.limiter.signals = BadSignals()
            await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="scrape bug"):
        asyncio.run(go())
    assert holder[0]._client.is_closed


def test_admission_fails_fast_after_tick_death(monkeypatch):
    """Codex r5 blocker #4a: a long run must not continue with a dead
    controller — the next admission raises run-fatally (never a row error)."""
    from pumpjack import FatalTransportError

    monkeypatch.setattr(core, "TICK_S", 0.01)

    class BadSignals:
        async def read(self):
            raise RuntimeError("scrape bug")

    async def go():
        async with AdaptiveLimiter(signals=BadSignals()) as lim:
            await asyncio.sleep(0.05)  # let the tick loop die
            async with lim.slot():
                pass

    with pytest.raises(FatalTransportError, match="controller loop died"):
        asyncio.run(go())


def test_blocked_admission_fails_after_tick_death(monkeypatch):
    """Codex r6 blocker #3: a waiter already queued behind a full window passed
    the pre-acquire check, then proceeded after the controller died. The check
    re-runs post-acquire (and releases the slot before raising)."""
    from pumpjack import FatalTransportError

    monkeypatch.setattr(core, "TICK_S", 0.01)

    class BadSignals:
        async def read(self):
            raise RuntimeError("scrape bug")

    async def go():
        async with AdaptiveLimiter(window=Fixed(1), signals=BadSignals()) as lim:
            first = lim.slot()
            await first.__aenter__()  # fill the window
            waiter = asyncio.create_task(lim.slot().__aenter__())
            await asyncio.sleep(0.05)  # controller dies while the waiter is queued
            await first.__aexit__(None, None, None)  # waiter acquires now
            await waiter

    with pytest.raises(FatalTransportError, match="controller loop died"):
        asyncio.run(go())


def test_body_exception_wins_over_tick_crash(monkeypatch):
    """Codex r5 blocker #4b: a tick-loop exception at exit must not mask the
    primary body exception."""
    monkeypatch.setattr(core, "TICK_S", 0.01)

    class BadSignals:
        async def read(self):
            raise RuntimeError("scrape bug")

    async def go():
        async with AdaptiveLimiter(signals=BadSignals()):
            await asyncio.sleep(0.05)
            raise ValueError("primary")

    with pytest.raises(ValueError, match="primary"):
        asyncio.run(go())


def test_fixed_zero_raises():
    with pytest.raises(ValueError, match=">= 1"):
        Fixed(0)


def test_as_sink_accepts_path(tmp_path):
    assert isinstance(as_sink(tmp_path), ParquetSink)  # Path was mistaken for a custom sink
