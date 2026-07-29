"""TEI is the first queue-only dialect: one real gauge, no running, no KV.

The TEI body below is a verbatim excerpt of `/metrics` on the MAIN HTTP port
of `ghcr.io/huggingface/text-embeddings-inference:cpu-latest`
(bge-small-en-v1.5, job 6a69f7ad4497041dbfc387a4, 2026-07-29) — including the
`te_batch_next_size` histogram, which is the near-miss the queue pattern must
not match.
"""

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
    assert g["dialect"] == "tei"
    assert g["waiting"] == 3
    assert g["running"] is None  # TEI exposes no in-flight gauge
    assert g["kv"] is None and g["hits"] is None  # and no KV cache at all


def test_tei_dual_spelling():
    assert parse_gauges("te:queue_size 7\n")["waiting"] == 7


def test_tei_ceiling_flag_is_a_relaunch_flag():
    assert "{n}" in CEILING_FLAG["tei"]


def test_vllm_still_parses():
    g = parse_gauges(VLLM_METRICS)
    assert g["dialect"] == "vllm"
    assert (g["waiting"], g["running"], g["kv"], g["hits"]) == (5, 12, 0.42, 0.9)
