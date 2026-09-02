"""pump() end to end over real HTTP against the local stub server (tests/stub_server.py):
the full composition — stream -> skip_done -> AdaptiveClient (retry ladder, breaker,
metrics scrape, adaptive window) -> drain -> the storage CONTRACT on disk."""

import json
import os

import pyarrow.parquet as pq
import pytest
from stub_server import StubServer

import saturate.core
from saturate import Auto, FatalTransportError, Fixed, existing_ids, pump, read_output
from saturate.source import content_id
from saturate.transport import Breaker

TELEMETRY_KEYS = {"t", "limit", "inflight", "waiting", "running", "bp", "ok", "input_bound",
                  "tok_s", "kv", "hits", "preempts"}  # CONTRACT §6
STATS_KEYS = {"rows_total", "rows_done_prior", "rows_processed", "rows_failed", "rows_deduped",
              "prompt_tokens", "completion_tokens", "elapsed_s", "final_limit", "input_bound",
              "breaker_opens", "hints", "tokens_per_sec"}  # CONTRACT §7


def to_request(row):
    return {"model": "stub", "messages": [{"role": "user", "content": row["text"]}]}


def parse(row, resp):
    return {"text": resp["choices"][0]["message"]["content"],
            "prompt_tokens": resp["usage"]["prompt_tokens"]}


def rows(n, start=0):
    return [{"text": f"row {i}"} for i in range(start, start + n)]


@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


@pytest.fixture(autouse=True)
def fast_ticks(monkeypatch):
    monkeypatch.setattr(saturate.core, "TICK_S", 0.1)  # controller ticks fast enough to matter in <10s


def test_pump_happy_path(stub, tmp_path):
    stub.latency_s = 0.02  # long enough for several controller ticks
    stats = pump(rows(200), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                 window=Auto(initial=4, max_limit=32))

    assert (stats.rows_total, stats.rows_done_prior) == (200, 0)
    assert (stats.rows_processed, stats.rows_failed) == (200, 0)
    assert stub.requests == 200
    assert (stats.prompt_tokens, stats.completion_tokens) == (stub.prompt_tokens, stub.completion_tokens)
    assert stats.prompt_tokens > 0

    parts = sorted(p.name for p in tmp_path.glob("part-*.parquet"))
    assert parts
    assert sorted(p.name[4:] for p in (tmp_path / "_manifest").glob("ids-part-*.parquet")) == parts
    assert (tmp_path / "completions" / "shard-0.done").read_bytes() == b"done"
    written = json.loads((tmp_path / "completions" / "stats-0.json").read_text())
    assert written["rows_processed"] == 200 and (written["rank"], written["world"]) == (0, 1)
    assert STATS_KEYS <= written.keys()

    out = dict(read_output(str(tmp_path)))
    assert len(out) == 200
    assert out[content_id({"text": "row 7"})] == {"text": "echo: row 7", "prompt_tokens": len("row 7")}

    (telemetry,) = list(tmp_path.glob("telemetry-shard0-*.jsonl"))
    ticks = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert ticks
    assert all(TELEMETRY_KEYS <= t.keys() for t in ticks)  # frozen keys; additive keys allowed
    assert all(t["running"] is not None for t in ticks)  # the /metrics scrape fed the controller


def test_pump_resume_is_exact(stub, tmp_path):
    first = pump(rows(100), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                 window=Fixed(8))
    assert (first.rows_processed, stub.requests) == (100, 100)

    second = pump(rows(150), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                  window=Fixed(8))
    assert second.rows_total == 150
    assert second.rows_done_prior == 100
    assert (second.rows_processed, second.rows_failed) == (50, 0)
    assert stub.requests == 150  # not one durable row was re-paid
    assert len(dict(read_output(str(tmp_path)))) == 150


def test_error_rows_and_heal(stub, tmp_path):
    def poison_is_400(request):
        return 400 if "poison" in request["messages"][0]["content"] else 200

    stub.status_for = poison_is_400
    data = rows(30)
    for i in (3, 11, 12, 20, 29):
        data[i]["text"] = f"poison {i}"
    poisoned = {content_id(r) for r in data if "poison" in r["text"]}
    healthy = {content_id(r) for r in data} - poisoned

    stats = pump(data, to_request, parse, endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(4))
    assert (stats.rows_processed, stats.rows_failed) == (25, 5)
    assert existing_ids(str(tmp_path)) == poisoned | healthy  # error rows are durable...
    assert existing_ids(str(tmp_path), retry_errors=True) == healthy  # ...and healable
    errors = [e for p in tmp_path.glob("part-*.parquet")
              for e in pq.read_table(p, columns=["error"])["error"].to_pylist() if e]
    assert len(errors) == 5 and all(e.startswith("http 400") for e in errors)
    assert set(dict(read_output(str(tmp_path)))) == healthy

    stub.status_for = lambda request: 200
    healed = pump(data, to_request, parse, endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(4),
                  retry_errors=True)
    assert healed.rows_done_prior == 25
    assert (healed.rows_processed, healed.rows_failed) == (5, 0)
    assert stub.requests == 35
    out = dict(read_output(str(tmp_path)))
    assert set(out) == poisoned | healthy  # the reader rule: error IS NULL wins
    assert all(out[i]["text"].startswith("echo: poison") for i in poisoned)


def test_agent_mode_stdout_is_one_json_line(stub, tmp_path, monkeypatch, capsys):
    for var in ("CLAUDECODE", "CODEX_SANDBOX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AI_AGENT", "1")
    stats = pump(rows(5), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(2))

    out, err = capsys.readouterr()
    assert out.endswith("\n") and out.count("\n") == 1
    line = json.loads(out)
    assert STATS_KEYS <= line.keys()
    assert line["rows_processed"] == stats.rows_processed == 5
    assert "[pump] done:" in err
    assert "re-run the same command to resume" in err


def test_breaker_aborts_run_when_server_dies(stub, tmp_path, monkeypatch):
    stub.status_for = lambda request: 503

    class QuickBreaker(Breaker):
        def __init__(self):
            # threshold=2: a row's own second failure trips the circuit, so no row can burn
            # through its retry ladder and land as a durable error before the breaker gives up
            super().__init__(threshold=2, probe_interval=0.1, max_open_s=1.0)

    monkeypatch.setattr(saturate.core, "Breaker", QuickBreaker)
    with pytest.raises(FatalTransportError, match="not coming back"):
        pump(rows(50), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(4))

    assert stub.probes >= 1  # the open circuit probed and never saw a live server
    assert existing_ids(str(tmp_path)) == set()  # nothing durable: every row is re-admitted next run
    assert not list(tmp_path.glob("part-*.parquet"))
    assert not os.path.exists(tmp_path / "completions" / "shard-0.done")


def test_window_ramps_and_endpoint_sees_concurrency(stub, tmp_path):
    stub.latency_s = 0.1
    stats = pump(rows(300), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                 window=Auto(initial=4, max_limit=64), flush_every=50)
    assert (stats.rows_processed, stats.rows_failed) == (300, 0)
    assert stub.peak_inflight > 4  # the window widened and the endpoint actually saw it
    assert stats.final_limit > 4
