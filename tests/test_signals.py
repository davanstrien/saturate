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
    vllm = ('vllm:num_preemptions_total{engine="0"} 5\n'
            'vllm:num_preemptions_total{engine="1"} 0\n'
            'vllm:num_requests_running 3\n')
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
        return type("R", (), {"status_code": status, "text": text,
                              "json": lambda self: __import__("json").loads(text)})()


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

    client = _ScriptedClient({
        "/server_info": (200, '{"disaggregated_params": null, "max_batch_size": 64}'),
        "/prometheus/metrics": (200, TRTLLM_METRICS),
        "/metrics": (200, "[]"),
    })
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

    client = _ScriptedClient({"/metrics": (200, '[{"iter": 1}]'),
                              "/prometheus/metrics": (200, TRTLLM_METRICS)})
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    gauges = _reads(scrape, 3)
    assert gauges[0] is None and gauges[1] is not None and gauges[1]["dialect"] == "trtllm"
    assert client.paths == ["/metrics", "/prometheus/metrics", "/prometheus/metrics"]
    assert "TensorRT-LLM; scraping http://x/prometheus/metrics instead" in caplog.text


def test_a_llamacpp_that_reports_metrics_off_is_never_scraped_and_the_warning_names_the_flag(caplog):
    from saturate.signals import HttpScrape

    client = _ScriptedClient({
        "/props": (200, '{"total_slots": 4, "default_generation_settings": {}, "endpoint_metrics": false}'),
    })
    found = _probe(client)
    assert (found.engine, found.metrics_on) == ("llamacpp", False)
    client.paths.clear()
    scrape = HttpScrape(cast(HttpClient, client), "http://x/v1")
    scrape.identify(found)
    assert _reads(scrape, 40) == [None] * 40
    assert client.paths == []
    assert caplog.text.count("running blind") == 1
    assert "llamacpp reports its metrics endpoint off" in caplog.text and "--metrics" in caplog.text


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
    from saturate.signals import Probe

    assert _probe(_ScriptedClient({"/version": (200, '{"version": "0.32.0"}')})) == Probe()
    # SGLang has a /server_info too; without disaggregated_params it is not TensorRT-LLM's
    assert _probe(_ScriptedClient({"/server_info": (200, '{"max_running_requests": 8}')})) == Probe()
