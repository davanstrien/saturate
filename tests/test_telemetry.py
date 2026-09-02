"""Telemetry sidecar: one file per run, rewritten periodically during the run and
finalised at exit (CONTRACT §6)."""

import asyncio
import json

import saturate
from saturate import ParquetSink


def test_write_telemetry_rewrites_one_file_per_run(tmp_path):
    """Repeated writes from one sink instance land in ONE file; the latest write wins.
    A second sink instance (another run) gets its own file."""
    sink = ParquetSink(str(tmp_path))
    sink.write_telemetry((0, 1), ['{"t": 2}'])
    sink.write_telemetry((0, 1), ['{"t": 2}', '{"t": 4}'])
    files = list(tmp_path.glob("telemetry-shard0-*.jsonl"))
    assert len(files) == 1
    assert files[0].read_text().splitlines() == ['{"t": 2}', '{"t": 4}']
    ParquetSink(str(tmp_path)).write_telemetry((0, 1), ['{"t": 2}'])
    assert len(list(tmp_path.glob("telemetry-shard0-*.jsonl"))) == 2


def test_pump_writes_telemetry_periodically_and_at_exit(tmp_path, monkeypatch):
    """pump() hands the controller an on_tick hook that rewrites the telemetry file
    every 30 ticks; the final write at exit completes the same file."""
    captured = {}

    class FakeLimiter:
        def __init__(self):
            self.ticks = []
            self.input_bound_ever = False
            self.window = type("W", (), {"limit": 16})()

    class FakeClient:
        def __init__(self, endpoint, on_tick=None, **kw):
            captured["on_tick"] = on_tick
            self.limiter = FakeLimiter()
            self.dialect = None
            self.breaker = type("B", (), {"opens": 0})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def fake_through(client, rows, *a, **kw):
        for i in range(60):  # 60 controller ticks, no rows
            client.limiter.ticks.append({"t": 2 * i, "limit": 16})
            captured["on_tick"]({"t": 2 * i, "limit": 16})
        return
        yield

    writes = []

    class CountingSink(ParquetSink):
        def write_telemetry(self, shard, lines):
            writes.append(len(lines))
            super().write_telemetry(shard, lines)

    monkeypatch.setattr(saturate, "AdaptiveClient", FakeClient)
    monkeypatch.setattr(saturate, "through", fake_through)
    sink = CountingSink(str(tmp_path))
    asyncio.run(saturate._pump([], lambda r: r, lambda r: r, "http://x", sink, None, (0, 1),
                               10, 1.0, "/chat/completions", None, False, None, None, "none"))
    assert captured["on_tick"] is not None
    assert writes == [30, 60, 60]  # tick 30, tick 60, final
    files = list(tmp_path.glob("telemetry-shard0-*.jsonl"))
    assert len(files) == 1
    assert [json.loads(line)["t"] for line in files[0].read_text().splitlines()] == [2 * i for i in range(60)]


def test_pump_periodic_telemetry_failure_is_not_fatal(tmp_path, monkeypatch, capsys):
    """A failing periodic write is logged and the run continues (same rule as the final write)."""
    captured = {}

    class FakeClient:
        def __init__(self, endpoint, on_tick=None, **kw):
            captured["on_tick"] = on_tick
            self.limiter = type("L", (), {"ticks": [], "input_bound_ever": False,
                                          "window": type("W", (), {"limit": 16})()})()
            self.dialect = None
            self.breaker = type("B", (), {"opens": 0})()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

    async def fake_through(client, rows, *a, **kw):
        for i in range(30):
            client.limiter.ticks.append({"t": 2 * i})
            captured["on_tick"]({"t": 2 * i})
        return
        yield

    class BrokenSink(ParquetSink):
        def write_telemetry(self, shard, lines):
            raise OSError("remote store down")

    monkeypatch.setattr(saturate, "AdaptiveClient", FakeClient)
    monkeypatch.setattr(saturate, "through", fake_through)
    stats = asyncio.run(saturate._pump([], lambda r: r, lambda r: r, "http://x", BrokenSink(str(tmp_path)),
                                       None, (0, 1), 10, 1.0, "/chat/completions", None, False, None,
                                       None, "none"))
    assert stats.rows_processed == 0 and (tmp_path / "completions" / "shard-0.done").exists()
    assert capsys.readouterr().err.count("telemetry write failed (non-fatal)") == 2  # periodic + final
