"""Background tasks are awaited, not abandoned: a crashed tick loop surfaces at limiter
exit. Invalid public inputs fail at construction (Fixed(0) would deadlock admission forever)."""

import asyncio

import pytest

import saturate.core as core
from saturate import AdaptiveLimiter, Fixed, ParquetSink
from saturate.sink import as_sink
from saturate.telemetry import Tick


class CrashingWindow(Fixed):
    """A window whose controller fails at its first decision: the way these tests crash the
    tick loop."""

    def decide(self, obs, limit):
        raise RuntimeError("controller bug")


@pytest.mark.internals
def test_crashed_tick_loop_surfaces(monkeypatch):
    monkeypatch.setattr(core, "TICK_S", 0.01)

    async def go():
        async with AdaptiveLimiter(CrashingWindow(4)):
            await asyncio.sleep(0.1)

    with pytest.raises(RuntimeError, match="controller bug"):
        asyncio.run(go())


@pytest.mark.internals
def test_client_closed_when_tick_loop_crashed(monkeypatch):
    """Surfacing a crashed tick loop at limiter exit must
    not leak the HTTP client."""
    from saturate import AdaptiveClient

    monkeypatch.setattr(core, "TICK_S", 0.01)

    holder = []

    async def go():
        async with AdaptiveClient("http://127.0.0.1:1", CrashingWindow(4), signal_source="none") as c:
            holder.append(c)
            await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="controller bug"):
        asyncio.run(go())
    assert holder[0]._client.closed


@pytest.mark.internals
def test_admission_fails_fast_after_tick_death(monkeypatch):
    """A long run does not continue with a dead controller: the next admission raises
    run-fatally (never a row error)."""
    from saturate import FatalTransportError

    monkeypatch.setattr(core, "TICK_S", 0.01)

    async def go():
        async with AdaptiveLimiter(CrashingWindow(4)) as lim:
            await asyncio.sleep(0.05)  # let the tick loop die
            async with lim.slot():
                pass

    with pytest.raises(FatalTransportError, match="controller loop died"):
        asyncio.run(go())


@pytest.mark.internals
def test_blocked_admission_fails_after_tick_death(monkeypatch):
    """A waiter already queued behind a full window passes the pre-acquire check while the
    controller is alive. The check re-runs after acquire, so the waiter raises once the
    controller has died (and releases the slot before raising)."""
    from saturate import FatalTransportError

    monkeypatch.setattr(core, "TICK_S", 0.01)

    async def go():
        async with AdaptiveLimiter(CrashingWindow(1)) as lim:
            first = lim.slot()
            await first.__aenter__()  # fill the window
            waiter = asyncio.create_task(lim.slot().__aenter__())
            await asyncio.sleep(0.05)  # controller dies while the waiter is queued
            await first.__aexit__(None, None, None)  # waiter acquires now
            await waiter

    with pytest.raises(FatalTransportError, match="controller loop died"):
        asyncio.run(go())


@pytest.mark.internals
def test_body_exception_wins_over_tick_crash(monkeypatch):
    """A tick-loop exception at exit must not mask the
    primary body exception."""
    monkeypatch.setattr(core, "TICK_S", 0.01)

    async def go():
        async with AdaptiveLimiter(CrashingWindow(4)):
            await asyncio.sleep(0.05)
            raise ValueError("primary")

    with pytest.raises(ValueError, match="primary"):
        asyncio.run(go())


@pytest.mark.internals
def test_a_failing_signal_source_costs_the_gauges_not_the_run(monkeypatch, caplog):
    """Gauges are advice: a signal source that raises leaves its ticks without gauges and the
    controller running. The source here fails twice, reads twice, then fails for good: each run
    of failures is logged once, and so is the recovery between them."""
    monkeypatch.setattr(core, "TICK_S", 0.01)
    readings = iter([OSError("scrape failed 1"), OSError("scrape failed 2"), {"waiting": 3}, {"waiting": 3}])

    class FlakySignals:
        async def read(self):
            reading = next(readings, OSError("scrape failed again"))
            if isinstance(reading, Exception):
                raise reading
            return reading

    async def go():
        enough = asyncio.Event()
        ticks: list[Tick] = []

        def on_tick(tick):
            ticks.append(tick)
            if len(ticks) >= 6:
                enough.set()

        async with AdaptiveLimiter(Fixed(4), signals=FlakySignals(), on_tick=on_tick) as limiter:
            await asyncio.wait_for(enough.wait(), 10)
            async with limiter.slot():  # admission still works: the controller loop is alive
                pass
        return ticks[:6]

    with caplog.at_level("INFO", logger="saturate"):
        ticks = asyncio.run(go())
    assert [tick["waiting"] for tick in ticks] == [None, None, 3, 3, None, None]
    assert [record.message for record in caplog.records if "signal source" in record.message] == [
        "signal source failed: OSError('scrape failed 1'); running without gauges until it recovers",
        "signal source reads again",
        "signal source failed: OSError('scrape failed again'); running without gauges until it recovers",
    ]


def test_fixed_zero_raises():
    with pytest.raises(ValueError, match=">= 1"):
        Fixed(0)


def test_as_sink_accepts_path(tmp_path):
    assert isinstance(as_sink(tmp_path), ParquetSink)  # a Path is a ParquetSink target, not a custom sink


@pytest.mark.internals
def test_a_cancelled_controller_loop_fails_admissions():
    """The tick loop cancelled from outside while the limiter is open: admissions fail the run
    instead of carrying on with a frozen window."""
    from saturate.transport import FatalTransportError

    async def go():
        async with AdaptiveLimiter(Fixed(4)) as limiter:
            limiter._task.cancel()
            await asyncio.sleep(0)
            async with limiter.slot():
                pass

    with pytest.raises(FatalTransportError, match="cancelled"):
        asyncio.run(go())


@pytest.mark.internals
def test_a_cancel_of_the_caller_during_limiter_exit_is_not_swallowed():
    """Exit stops the tick loop; a cancel of the caller that arrives meanwhile goes on."""

    async def slow_to_stop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # a loop that takes a moment to wind down
            raise

    async def body(limiter, exiting):
        async with limiter:
            limiter._task.cancel()
            limiter._task = asyncio.create_task(slow_to_stop())
            await asyncio.sleep(0)
            exiting.set()
        return "finished"

    async def go():
        limiter, exiting = AdaptiveLimiter(Fixed(4)), asyncio.Event()
        task = asyncio.create_task(body(limiter, exiting))
        await exiting.wait()
        await asyncio.sleep(0.05)  # inside __aexit__, waiting for the loop to stop
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert limiter._task is not None
        assert limiter._task.done()  # the loop finished its cleanup before exit returned

    asyncio.run(go())


@pytest.mark.internals
def test_a_limiter_entered_again_still_supervises_its_controller_loop():
    from saturate.transport import FatalTransportError

    async def go():
        limiter = AdaptiveLimiter(Fixed(4))
        async with limiter:
            pass
        async with limiter:
            assert limiter._task is not None
            limiter._task.cancel()
            await asyncio.sleep(0)
            async with limiter.slot():
                pass

    with pytest.raises(FatalTransportError, match="cancelled"):
        asyncio.run(go())
