"""pump() end to end over real HTTP against the local stub server (tests/stub_server.py):
the full composition — stream -> skip_done -> AdaptiveClient (retry ladder, breaker,
metrics scrape, adaptive window) -> drain -> the storage CONTRACT on disk."""

import json
import os
import time

import pyarrow.parquet as pq
import pytest
from stub_server import StubServer

import saturate.core
import saturate.transport
from saturate import Auto, FatalTransportError, Fixed, existing_ids, pump, read_output
from saturate.source import content_id
from saturate.transport import Breaker

TELEMETRY_KEYS = {"t", "limit", "inflight", "waiting", "running", "bp", "ok", "input_bound",
                  "tok_s", "kv", "hits", "preempts", "reason", "latency_s", "bound_by", "source_s",
                  "prep_s", "prep_n", "prep_workers", "loop_lag_s"}  # CONTRACT §6
STATS_KEYS = {"rows_total", "rows_done_prior", "rows_processed", "rows_failed", "rows_deduped",
              "prompt_tokens", "completion_tokens", "elapsed_s", "final_limit", "input_bound",
              "breaker_opens", "hints", "tokens_per_sec", "cut_reasons", "bound_by"}  # CONTRACT §7


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


@pytest.fixture
def quick_breaker(monkeypatch):
    """threshold=2 trips the circuit on a row's own second failure; the probe retries fast;
    the retry ladder's jitter is pinned so the test's cost is fixed."""

    class QuickBreaker(Breaker):
        def __init__(self):
            super().__init__(threshold=2, probe_interval=0.05, max_open_s=1.5)

    monkeypatch.setattr(saturate.core, "Breaker", QuickBreaker)
    monkeypatch.setattr(saturate.transport.random, "uniform", lambda a, b: 0.05)


def down_then_up(n):
    """status_for: 503 for the first n POSTs (probes included), 200 after."""
    calls = 0

    def status(request):
        nonlocal calls
        calls += 1
        return 503 if calls <= n else 200

    return status


def test_breaker_probes_the_request_route_with_a_tiny_body(stub, tmp_path, quick_breaker):
    """A non-chat workload whose server 503s for a while: the recovery probe hits the route
    the run uses (/embeddings) with the tiny generic body carrying the run's model — never
    /chat/completions, never the caller's own body."""
    stub.status_for = down_then_up(6)
    stats = pump(rows(12), lambda r: {"model": "stub", "input": [r["text"]]},
                 lambda r, resp: {"dim": len(resp["data"][0]["embedding"])},
                 endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(4), route="/embeddings")

    assert stats.breaker_opens >= 1 and stub.probes >= 1
    assert {path for path, _ in stub.probe_log} == {"/v1/embeddings"}
    assert all(body["model"] == "stub" and body["max_tokens"] == 1 and body["input"] == "hi"
               for _, body in stub.probe_log)
    assert (stats.rows_processed, stats.rows_failed) == (12, 0)  # the run recovered


def test_breaker_probe_answers_inside_its_timeout_when_generations_are_long(stub, tmp_path,
                                                                             quick_breaker, monkeypatch):
    """Rows ask for long generations the server takes longer than the probe timeout to answer;
    the probe (max_tokens=1) answers at once, so a healthy server closes the circuit."""
    monkeypatch.setattr(saturate.transport, "PROBE_TIMEOUT_S", 0.3)
    stub.latency_for = lambda request: 0.0 if request.get("max_tokens") == 1 else 1.0
    stub.status_for = down_then_up(4)
    long_request = lambda r: {**to_request(r), "max_tokens": 4096}  # noqa: E731
    t = time.monotonic()
    stats = pump(rows(6), long_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                 window=Fixed(4))
    elapsed = time.monotonic() - t

    assert stats.breaker_opens >= 1 and stub.probes >= 1
    assert all(body["max_tokens"] == 1 for _, body in stub.probe_log)  # a row body would have timed out
    assert (stats.rows_processed, stats.rows_failed) == (6, 0)
    print(f"probe-timeout run: {elapsed:.2f}s, {stub.probes} probes")


def test_poison_rows_do_not_hold_the_circuit_open(stub, tmp_path, quick_breaker, monkeypatch):
    """Every row makes the server 500 (poison) while the server itself is healthy: the probe
    must not re-send a poison body, or the circuit never closes and the run aborts. The
    probe closes it and the poison rows land as error rows."""
    monkeypatch.setattr(saturate.transport, "RETRY_ACTIVE", False)
    stub.status_for = lambda request: 500 if "poison" in str(request.get("messages")) else 200
    data = [{"text": f"poison {i}"} for i in range(6)]
    stats = pump(data, to_request, parse, endpoint=stub.endpoint, output=str(tmp_path), window=Fixed(2))

    assert stats.breaker_opens >= 1 and stub.probes >= 1
    assert (stats.rows_processed, stats.rows_failed) == (0, 6)
    assert existing_ids(str(tmp_path), retry_errors=True) == set()  # all six durable as error rows


def test_a_slow_parse_reads_loop_bound(stub, tmp_path):
    """parse blocks the event loop like on-loop to_request does; the window looks full, but
    the tick arrives late by the blocked time and says `loop` — with a hint."""
    stub.latency_s = 0.02

    def slow_parse(row, resp):
        time.sleep(0.04)
        return parse(row, resp)

    stats = pump(rows(60), to_request, slow_parse, endpoint=stub.endpoint, output=str(tmp_path / "slow"),
                 window=Fixed(6))
    assert stats.rows_processed == 60
    assert stats.bound_by.get("loop", 0) > sum(stats.bound_by.values()) / 2, stats.bound_by
    assert any(h.startswith("LOOP-BOUND: the event loop was blocked") for h in stats.hints), stats.hints
    assert not stats.input_bound  # the window was full: not starved

    stats = pump(rows(60), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path / "fast"),
                 window=Fixed(6))
    assert stats.bound_by.get("loop", 0) == 0 and not any(h.startswith("LOOP-BOUND") for h in stats.hints)
    assert stats.bound_by, stats.bound_by  # the control run did tick


def test_prepared_bodies_are_bounded_independently_of_the_feeder(stub, tmp_path):
    """With prepare_workers=2 and Fixed(2), at most max(2*2, 4) prepared bodies wait for a slot
    plus 2 in flight: 6 to_request calls can have completed before the first response."""
    stub.latency_s = 0.15
    prepared = 0
    at_first_response = []

    def counting_to_request(row):
        nonlocal prepared
        prepared += 1
        return to_request(row)

    def first_parse(row, resp):
        if not at_first_response:
            at_first_response.append(prepared)
        return parse(row, resp)

    stats = pump(rows(24), counting_to_request, first_parse, endpoint=stub.endpoint,
                 output=str(tmp_path), window=Fixed(2), prepare_workers=2)
    assert stats.rows_processed == 24
    assert at_first_response[0] <= 6, at_first_response


def slow_to_request(row):
    time.sleep(0.05)  # image decode + encode stands in: real work that releases the GIL
    return to_request(row)


def test_prepare_stage_offloads_to_request_and_the_verdict_names_it(stub, tmp_path, monkeypatch):
    """to_request at 50 ms/row on the loop serialises the pump: the endpoint idles and the
    ticks say `prep`. prepare_workers=4 runs it ahead in threads, the window fills, the run is
    at least 2x faster, and the ticks stop blaming prep."""
    stub.latency_s = 0.2
    timings = {}
    for workers in (0, 4):
        # on the loop a tick must span many 50 ms calls; threaded, the run is short: tick often
        monkeypatch.setattr(saturate.core, "TICK_S", 0.25 if workers == 0 else 0.1)
        out = tmp_path / f"w{workers}"
        t = time.monotonic()
        stats = pump(rows(60), slow_to_request, parse, endpoint=stub.endpoint, output=str(out),
                     window=Fixed(14), prepare_workers=workers)
        timings[workers] = time.monotonic() - t
        assert (stats.rows_processed, stats.rows_failed) == (60, 0)
        (telemetry,) = list(out.glob("telemetry-shard0-*.jsonl"))
        ticks = [json.loads(line) for line in telemetry.read_text().splitlines()]
        prep_ticks = sum(t["bound_by"] == "prep" for t in ticks)
        assert sum(t["prep_s"] for t in ticks) >= 60 * 0.05 * 0.8  # the measurement itself (tail excluded)
        assert all(t["prep_workers"] == workers for t in ticks)
        if workers == 0:
            assert prep_ticks > len(ticks) / 2, [t["bound_by"] for t in ticks]
            assert stats.input_bound and stats.bound_by.get("prep", 0) == prep_ticks
            assert any(h.startswith("PREP-BOUND: to_request took") for h in stats.hints), stats.hints
        else:
            assert prep_ticks <= len(ticks) / 2, [t["bound_by"] for t in ticks]
            assert not any(h.startswith("PREP-BOUND") for h in stats.hints), stats.hints
    print(f"prepare_workers=0: {timings[0]:.2f}s  prepare_workers=4: {timings[4]:.2f}s  "
          f"speedup {timings[0] / timings[4]:.1f}x")
    assert timings[0] >= 2 * timings[4], timings


def test_window_ramps_and_endpoint_sees_concurrency(stub, tmp_path):
    stub.latency_s = 0.1
    stats = pump(rows(300), to_request, parse, endpoint=stub.endpoint, output=str(tmp_path),
                 window=Auto(initial=4, max_limit=64), flush_every=50)
    assert (stats.rows_processed, stats.rows_failed) == (300, 0)
    assert stub.peak_inflight > 4  # the window widened and the endpoint actually saw it
    assert stats.final_limit > 4
