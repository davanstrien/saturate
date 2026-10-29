"""TEI is the first queue-only dialect: one real gauge, no running, no KV.

The TEI body below is a verbatim excerpt of `/metrics` on the MAIN HTTP port
of `ghcr.io/huggingface/text-embeddings-inference:cpu-latest`
(bge-small-en-v1.5, job 6a69f7ad4497041dbfc387a4, 2026-07-29) — including the
`te_batch_next_size` histogram, which is the near-miss the queue pattern must
not match.
"""

from typing import cast

from saturate.http import HttpClient, TransportError
from saturate.signals import CEILING_FLAG, parse_gauges

TEI_METRICS = """\
# TYPE te_request_success counter
te_request_success{method="single"} 1

# TYPE te_embed_count counter
te_embed_count 1

# TYPE te_request_count counter
te_request_count{method="single"} 1

# TYPE te_embed_success counter
te_embed_success 1

# TYPE te_queue_size gauge
te_queue_size 3

# TYPE te_batch_next_size histogram
te_batch_next_size_bucket{le="1"} 2
te_batch_next_size_bucket{le="+Inf"} 2
te_batch_next_size_count 2
te_batch_next_size_sum 1
"""

VLLM_METRICS = """\
vllm:num_requests_waiting{model_name="x"} 5.0
vllm:num_requests_running{model_name="x"} 12.0
vllm:gpu_cache_usage_perc{model_name="x"} 0.42
vllm:gpu_prefix_cache_hit_rate{model_name="x"} 0.9
"""


def test_tei_queue_only():
    g = parse_gauges(TEI_METRICS)
    assert g is not None
    assert g["dialect"] == "tei"
    assert g["waiting"] == 3
    assert g["running"] is None  # TEI exposes no in-flight gauge
    assert g["kv"] is None and g["hits"] is None  # and no KV cache at all


def test_tei_dual_spelling():
    g = parse_gauges("te:queue_size 7\n")
    assert g is not None
    assert g["waiting"] == 7


def test_tei_ceiling_flag_is_a_relaunch_flag():
    assert "{n}" in CEILING_FLAG["tei"]


def test_vllm_still_parses():
    g = parse_gauges(VLLM_METRICS)
    assert g is not None
    assert g["dialect"] == "vllm"
    assert (g["waiting"], g["running"], g["kv"], g["hits"]) == (5, 12, 0.42, 0.9)


class _FlakyClient:
    """GET raises `fail_first` times, then serves vLLM gauges. It stands in for HttpClient with
    only the `get` that HttpScrape uses, so the test casts it to HttpClient."""

    def __init__(self, fail_first: int):
        self.fail_first = fail_first
        self.calls = 0

    async def get(self, url, timeouts=None):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TransportError("metrics not up yet")
        return type("R", (), {"status_code": 200, "text": VLLM_METRICS})()


def test_scrape_retries_periodically_after_going_blind(caplog):
    """Five failures put the scrape in blind mode; it then skips requests for
    retry_every-1 reads and probes again, recovering when /metrics comes up."""
    import asyncio

    from saturate.signals import HttpScrape

    client = _FlakyClient(fail_first=5)
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1", max_fails=5, retry_every=30)

    async def run():
        results = []
        for _ in range(35):
            results.append((await scrape.read(), client.calls))
        return results

    results = asyncio.run(run())
    assert [calls for _, calls in results[:5]] == [1, 2, 3, 4, 5]  # reads 1-5 each attempt
    assert all(g is None for g, _ in results[:34])
    assert all(calls == 5 for _, calls in results[5:34])  # reads 6-34: no request
    gauges, calls = results[34]
    assert calls == 6 and gauges["dialect"] == "vllm"  # read 35 probes and recovers
    assert scrape.dialect == "vllm"
    err = caplog.text
    assert "running blind" in err and "http://x/metrics (unreachable)" in err
    assert "metrics scrape recovered (vllm)" in err


def test_preemption_counters_are_summed_over_their_series_and_sglang_retractions_count():
    vllm = (
        'vllm:num_preemptions_total{engine="0"} 5\n'
        'vllm:num_preemptions_total{engine="1"} 0\n'
        "vllm:num_requests_running 3\n"
    )
    g = parse_gauges(vllm)
    assert g is not None and g["preempts"] == 5
    g = parse_gauges("\n".join(reversed(vllm.strip().splitlines())))
    assert g is not None and g["preempts"] == 5  # the order of the series does not matter
    g = parse_gauges("sglang:num_retracted_requests_total 7\nsglang:num_running_reqs 2\n")
    assert g is not None and (g["preempts"], g["dialect"]) == (7, "sglang")


TRTLLM_METRICS = "trtllm_num_requests_waiting 4\ntrtllm_num_requests_running 60\n"


class _ScriptedClient:
    """GET answers from a table of path -> (status, body); every requested path is recorded.
    Stands in for HttpClient like _FlakyClient."""

    def __init__(self, routes: dict[str, tuple[int, str]]):
        self.routes = routes
        self.paths: list[str] = []

    async def get(self, url, timeouts=None):
        path = url.removeprefix("http://x")
        self.paths.append(path)
        status, text = self.routes.get(path, (404, "Not Found"))
        return type(
            "R",
            (),
            {"status_code": status, "text": text, "json": lambda self: __import__("json").loads(text)},
        )()


def _reads(scrape, n):
    import asyncio

    async def run():
        return [await scrape.read() for _ in range(n)]

    return asyncio.run(run())


def _probe(client):
    import asyncio

    from saturate.signals import probe

    return asyncio.run(probe(cast(HttpClient, client), "http://x/v1"))


def test_a_probed_trtllm_is_scraped_at_prometheus_metrics_and_never_at_metrics():
    """trtllm-serve's /metrics is JSON iteration stats that each read clears: once /server_info
    names TensorRT-LLM, the scrape never reads it."""
    from saturate.signals import Capacity, HttpScrape

    client = _ScriptedClient(
        {
            "/server_info": (200, '{"disaggregated_params": null, "max_batch_size": 64}'),
            "/prometheus/metrics": (200, TRTLLM_METRICS),
            "/metrics": (200, "[]"),
        }
    )
    found = _probe(client)
    assert (found.engine, found.capacity) == ("trtllm", Capacity(64, True, "trtllm"))
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    scrape.identify(found)
    gauges = _reads(scrape, 3)
    assert all(g is not None and (g["waiting"], g["running"]) == (4, 60) for g in gauges)
    assert "/metrics" not in client.paths


def test_json_at_metrics_is_read_once_then_the_scrape_moves_to_prometheus_metrics(caplog):
    """A TensorRT-LLM without /server_info (1.2 and earlier) is named by its first /metrics reply,
    which is JSON: that read is the only one."""
    from saturate.signals import HttpScrape

    client = _ScriptedClient(
        {"/metrics": (200, '[{"iter": 1}]'), "/prometheus/metrics": (200, TRTLLM_METRICS)}
    )
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    gauges = _reads(scrape, 3)
    assert gauges[0] is None and gauges[1] is not None and gauges[1]["dialect"] == "trtllm"
    assert client.paths == ["/metrics", "/prometheus/metrics", "/prometheus/metrics"]
    assert "TensorRT-LLM; scraping http://x/prometheus/metrics instead" in caplog.text


def test_a_llamacpp_that_reports_metrics_off_is_never_scraped_and_the_warning_names_the_flag(caplog):
    from saturate.signals import HttpScrape

    client = _ScriptedClient(
        {
            "/props": (
                200,
                '{"total_slots": 4, "default_generation_settings": {}, "endpoint_metrics": false}',
            ),
        }
    )
    found = _probe(client)
    assert (found.engine, found.metrics_on) == ("llamacpp", False)
    client.paths.clear()
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    scrape.identify(found)
    assert _reads(scrape, 40) == [None] * 40
    assert client.paths == []
    assert caplog.text.count("running blind") == 1
    assert "llamacpp reports its metrics endpoint off" in caplog.text and "--metrics" in caplog.text
    assert not caplog.text.rstrip().endswith(";")


def test_blind_mode_says_why_the_reads_missed_and_names_the_engines_flag(caplog):
    """A 404 and a reply with no engine gauges each name themselves (an unreachable server: see
    above); for a probed engine the warning adds what turns its gauges on."""
    from saturate.signals import HttpScrape, Probe

    scrape = HttpScrape(cast(HttpClient, _ScriptedClient({})), "http://x/v1")
    scrape.identify(Probe(engine="sglang"))
    _reads(scrape, 6)  # five misses, then the warning
    assert "(HTTP 404)" in caplog.text and "start SGLang with --enable-metrics" in caplog.text
    caplog.clear()
    scrape = HttpScrape(cast(HttpClient, _ScriptedClient({"/metrics": (200, "up 1\n")})), "http://x/v1")
    _reads(scrape, 6)
    assert "(no engine gauges in the reply)" in caplog.text
    assert scrape.reason == "no engine gauges in the reply"


def test_a_probe_of_vllm_or_a_hosted_api_names_no_engine():
    """JSON answers on every probed path that match no engine's shape name none: a hosted API's
    error bodies, and SGLang's own /server_info, which lacks disaggregated_params."""
    from saturate.signals import CAPACITY_SOURCES, Probe

    error = (200, '{"error": {"message": "not found"}}')
    assert _probe(_ScriptedClient({path: error for path, _, _, _ in CAPACITY_SOURCES})) == Probe()
    assert _probe(_ScriptedClient({"/server_info": (200, '{"max_running_requests": 8}')})) == Probe()


def test_json_after_gauges_named_the_engine_is_a_miss_and_the_scrape_recovers(caplog):
    """A JSON error body from a vLLM whose gauges were read does not turn it into TensorRT-LLM;
    its blind-mode warning names vLLM's flag."""
    from saturate.signals import HttpScrape

    client = _ScriptedClient({"/metrics": (200, VLLM_METRICS)})
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    assert _reads(scrape, 1)[0] is not None
    client.routes["/metrics"] = (200, '{"error": "temporarily unavailable"}')
    assert _reads(scrape, 6) == [None] * 6
    assert "--disable-log-stats" in caplog.text and "TensorRT-LLM" not in caplog.text
    client.routes["/metrics"] = (200, VLLM_METRICS)
    scrape._skipped = scrape.retry_every - 1  # the next read is the periodic probe
    gauges = _reads(scrape, 1)[0]
    assert gauges is not None and gauges["dialect"] == "vllm"
    assert set(client.paths) == {"/metrics"}


def test_json_that_is_not_trtllms_list_is_an_ordinary_miss():
    """Only a 200 JSON list at /metrics, from a server nothing has named, is TensorRT-LLM's: a
    JSON object (a gateway's error body), a non-200 JSON body (llama.cpp's 501 with --metrics
    off) and any JSON from a probed engine each stay at /metrics."""
    from saturate.signals import HttpScrape, Probe

    for status, body, probed in [
        (200, '{"error": "upstream restarting"}', None),
        (501, '[{"error": "metrics off"}]', None),
        (200, "[]", "sglang"),
    ]:
        client = _ScriptedClient({"/metrics": (status, body)})
        scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
        if probed:
            scrape.identify(Probe(engine=probed))
        _reads(scrape, 3)
        assert client.paths == ["/metrics"] * 3 and scrape.dialect != "trtllm"


def test_a_probe_names_the_engine_from_its_shape_without_a_capacity():
    from saturate.signals import Probe

    assert _probe(_ScriptedClient({"/get_server_info": (200, '{"max_prefill_tokens": 16384}')})) == Probe(
        engine="sglang"
    )
    assert _probe(_ScriptedClient({"/info": (200, '{"model_id": "m", "max_batch_tokens": 16384}')})) == Probe(
        engine="tei"
    )


def test_a_trtllm_named_by_its_json_names_its_flag_when_prometheus_metrics_is_off(caplog):
    from saturate.signals import HttpScrape

    scrape = HttpScrape(cast(HttpClient, _ScriptedClient({"/metrics": (200, "[]")})), "http://x/v1")
    _reads(scrape, 6)
    assert "/prometheus/metrics (HTTP 404)" in caplog.text and "return_perf_metrics" in caplog.text


def _vllm_prefix(queries: float, hits: float, engine1: tuple[float, float] | None = None) -> str:
    """vLLM's prefix-cache counters as it publishes them: one series per engine, with the
    `_created` sample prometheus_client adds beside each `_total`. Engine 1 holds a quarter of
    each count unless given its own (queries, hits)."""
    q1, h1 = engine1 if engine1 is not None else (queries * 0.25, hits * 0.25)
    return (
        f'vllm:prefix_cache_queries_total{{engine="0",model_name="x"}} {queries - q1}\n'
        f'vllm:prefix_cache_queries_total{{engine="1",model_name="x"}} {q1}\n'
        'vllm:prefix_cache_queries_created{engine="0",model_name="x"} 1.7e9\n'
        f'vllm:prefix_cache_hits_total{{engine="0",model_name="x"}} {hits - h1}\n'
        f'vllm:prefix_cache_hits_total{{engine="1",model_name="x"}} {h1}\n'
        'vllm:prefix_cache_hits_created{engine="0",model_name="x"} 1.7e9\n'
        'vllm:external_prefix_cache_queries_total{engine="0",model_name="x"} 99999\n'
        'vllm:kv_cache_usage_perc{engine="0",model_name="x"} 0.97\n'
    )


def test_vllm_prefix_counters_are_kept_per_series_and_name_no_rate():
    g = parse_gauges(_vllm_prefix(1000, 600))
    assert g is not None and g["dialect"] == "vllm"
    assert g["prefix_counts"] == {  # _created and the external cache's counter ignored
        '{engine="0",model_name="x"}': {"queries": 750, "hits": 450},
        '{engine="1",model_name="x"}': {"queries": 250, "hits": 150},
    }
    assert g["hits"] is None  # a rate needs two readings: parse_gauges sees one
    g = parse_gauges("vllm:gpu_prefix_cache_queries_total 10\nvllm:gpu_prefix_cache_hits_total 4\n")
    assert g is not None and g["prefix_counts"] == {"": {"queries": 10, "hits": 4}}  # vLLM 0.8-0.9


def _rates(texts: list[str | None]) -> list:
    """The hit rate each reading gives, in order; None in `texts` is a missed read (a 404)."""
    from saturate.signals import HttpScrape

    client = _ScriptedClient({})
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    rates = []
    for text in texts:
        client.routes["/metrics"] = (404, "") if text is None else (200, text)
        g = _reads(scrape, 1)[0]
        rates.append("miss" if g is None else g["hits"])
    return rates


def test_the_scrape_derives_vllms_hit_rate_from_two_readings_of_its_counters():
    """New hits over new queries since the last read, summed over engines; no rate from the
    first reading, from a tick with no new query, or across a counter reset."""
    readings = [(1000, 600), (2000, 1500), (2000, 1500), (100, 10), (300, 50)]
    assert _rates([_vllm_prefix(q, h) for q, h in readings]) == [None, 0.9, None, None, 0.2]


def test_one_engines_reset_gives_no_rate_where_a_sum_would_fake_a_high_one():
    """Engine 1 restarts (its counters drop) while engine 0 gains 120 queries and 24 hits: the
    summed deltas would read 24 / 20 (clamped to 1.0); per series, no rate."""
    before = _vllm_prefix(1100, 100, engine1=(100, 0))
    after = _vllm_prefix(1120, 124, engine1=(0, 0))
    assert _rates([before, after]) == [None, None]
    # hits down while queries rise: a reset too
    assert _rates([_vllm_prefix(1000, 500), _vllm_prefix(1200, 400)]) == [None, None]


def test_a_counter_missing_from_a_reading_is_no_baseline():
    """A reading without the hits series is not taken as zero hits: the next rate would credit
    every earlier hit to the last tick."""
    queries_only = 'vllm:prefix_cache_queries_total{engine="0"} 1000\nvllm:num_requests_running 1\n'
    both = 'vllm:prefix_cache_queries_total{engine="0"} 1100\nvllm:prefix_cache_hits_total{engine="0"} 900\n'
    assert _rates([queries_only, both]) == [None, None]


def test_the_baseline_follows_every_read_and_restarts_after_a_miss():
    """A reading that carries a hit-rate gauge (vLLM V0, SGLang) still moves the counter
    baseline, and a missed read starts a new one, so no rate spans a reading it did not see."""
    gauge = "vllm:gpu_prefix_cache_hit_rate 0.5\n"
    one, two, three = _vllm_prefix(1000, 900), _vllm_prefix(2000, 1900), _vllm_prefix(2100, 1900)
    assert _rates([one, two + gauge, three]) == [None, 0.5, 0.0]
    assert _rates([one, None, two, three]) == [None, "miss", None, 0.0]


def test_a_healthy_vllm_prefix_cache_lets_a_full_kv_cache_with_a_preemption_stand():
    """The point of #15: with the rate derived, vLLM's high KV is verified benign by a high hit
    rate, as the two-condition cut intends; a low rate still counts as full."""
    from saturate import Auto
    from saturate.controller import Obs, State, kv_full

    auto = Auto()
    state = State(preempts=3, preempted=1)  # a preemption this tick: the KV check applies
    assert not kv_full(auto, state, Obs(kv=0.97, hits=0.9, preempts=4), 64)
    assert kv_full(auto, state, Obs(kv=0.97, hits=0.2, preempts=4), 64)
    assert kv_full(auto, state, Obs(kv=0.97, hits=None, preempts=4), 64)


def test_the_metrics_table_is_well_formed():
    """A row that names an unknown field, or a key where its field takes none (or none where it
    does), would be read wrongly without failing: each is checked here, as is a name that two
    rows or two spellings share."""
    from saturate.signals import _BY_NAME, FIELDS, METRICS

    for m in METRICS:
        assert m.field in FIELDS, m
        assert (m.key is not None) == (FIELDS[m.field] == "series"), m
        assert m.key in (None, "queries", "hits"), m
    assert len(_BY_NAME) == 2 * len(METRICS)


def test_every_metric_row_is_read_in_both_spellings_into_its_field():
    """One row of METRICS is all a metric needs: each accepted spelling names its engine and
    lands in its own field."""
    from saturate.signals import _BY_NAME, FIELDS

    for full, m in _BY_NAME.items():
        g = parse_gauges(f'{full}{{engine="0"}} 3\n')
        assert g is not None and g["dialect"] == m.engine, full
        want = {'{engine="0"}': {m.key: 3.0}} if FIELDS[m.field] == "series" else 3
        assert g[m.field] == want, full


def test_the_sample_parser_skips_what_is_not_a_finite_sample():
    """Comments, a non-finite value (NaN, +Inf) and a metric no row names are skipped; a label
    value holding a brace and a trailing timestamp are read correctly."""
    text = (
        "# HELP vllm:num_requests_waiting Number of requests waiting.\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        'vllm:num_requests_waiting{model_name="a}b"} 4 1730000000000\n'
        "vllm:num_requests_running NaN\n"
        "vllm:kv_cache_usage_perc +Inf\n"
        "vllm:num_requests_swapped 9\n"
    )
    g = parse_gauges(text)
    assert g is not None and (g["waiting"], g["running"], g["kv"]) == (4, None, None)
    assert parse_gauges("# only comments\n\nup 1\n") is None
    g = parse_gauges('vllm:prefix_cache_queries_total{model_name="a}b"} 10\n')
    assert g is not None and g["prefix_counts"] == {'{model_name="a}b"}': {"queries": 10.0}}


def test_the_sample_parser_reads_real_exposition_shapes():
    """Shapes the Prometheus text format allows: a tab separator, an exemplar after the value
    (its braces are not the labels'), an escaped quote and a brace in a label value, a line
    separator other than a newline inside a label value. A line starting with whitespace, or
    a value with no whitespace before it, is not a sample."""
    exemplar = 'vllm:num_preemptions_total{engine="0"} 7 # {trace_id="abc"} 1.0\n'
    g = parse_gauges(exemplar)
    assert g is not None and g["preempts"] == 7
    g = parse_gauges(
        'vllm:prefix_cache_queries_total{engine="0"} 5 # {trace_id="x"} 1\n'
        'vllm:prefix_cache_hits_total{engine="0"} 2\n'
    )
    assert g is not None and g["prefix_counts"] == {'{engine="0"}': {"queries": 5, "hits": 2}}
    g = parse_gauges("vllm:num_requests_waiting\t7\n")
    assert g is not None and g["waiting"] == 7
    g = parse_gauges('vllm:num_requests_waiting{a="x\\"}y",b="p\u2028q"} 3 1730000000000\n')
    assert g is not None and g["waiting"] == 3
    assert parse_gauges("  vllm:num_requests_waiting 7\n") is None
    assert parse_gauges('vllm:num_requests_waiting{a="b"}7\n') is None
