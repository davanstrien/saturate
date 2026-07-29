# WHY.md — why this exists, and why not just use X

Every claim below carries a receipt: a job id, a source file and line, or a URL. If a claim has
no receipt it does not belong here. Where an alternative genuinely wins, that is stated as an
honest counter rather than argued away.

Receipt conventions: `RESULTS.md §x` refers to `spikes/RESULTS.md` in this repo;
`history/…` paths are in `docs/history/`; receipts marked **(internal)** cite the author's
private measurement notes — those claims are measured but not reproducible from this repo,
which is a weaker standard and flagged as such. External source line numbers were read at the
commits noted (mostly 2026-07-28/29).

---

## 0. The one-sentence position

saturate exists for one loop: **data → model → nicer data**. It is the client in the
middle — it sits between a row source and any OpenAI-compatible endpoint, decides at runtime
how many requests to keep in flight, and writes the result as crash-safe resumable parquet
that is itself a dataset. Everything below is about doing that one shape well.

The short version of every comparison below:

| alternative | verdict in one line | where it wins |
|---|---|---|
| vLLM `AsyncLLM` in-process (§1) | same throughput, but no persistence/resume, one engine's API, can't reach remote endpoints | single owned engine, one shot, no resume needed |
| offline `LLM.generate` (§2) | all-or-nothing output; loses on vision (1.19× for serving) | text, moderate scale, in-memory input |
| datatrove `InferenceRunner` (§3) | no speed tax — the difference is a hand-tuned fixed semaphore vs an adaptive one; complementary via `AdaptiveLimiter` | multi-stage pipelines, rollouts, executor family |
| Curator / cookbook script / lm-deluge (§4) | built for published API quotas; against self-hosted engines they fall back to guessed static caps | commercial APIs with real rate-limit headers |
| Ray Data LLM / Daft (§5) | scale their own worker pools, don't observe the endpoint | heterogeneous CPU+GPU stages, multi-replica clusters |
| NVIDIA DataDesigner (§6) | adapts, but only *down* from a user cap on 429s — a signal self-hosted engines never send | hosted APIs that actually 429 |
| llm-d / k8s inference gateways (§7) | same gauges, different layer — platform-scale Go/k8s infrastructure | many tenants, many pods, a cluster you already run |
| just picking a number (§8) | the right number moves (32↔257 on one GPU), OOMs when too high, idles when too low | fully measured, never-changing single workload |
| JSONL or a database (§9) | parquet+manifest gives typed arrays, exact resume, any-reader output | a few thousand rows: JSONL is fine |
| a pipeline/DAG framework (§10) | rows, not nodes, are the recompute unit for billed LLM calls | genuine DAGs: heterogeneous stages, fan-in, branches |

The narrow, defensible version of the novelty claim:

> saturate is the only standalone client whose controller *discovers* endpoint capacity from
> delivered throughput plus server gauges. Other adaptive clients throttle *down* from a
> user-supplied cap on rate-limit signals, which self-hosted engines never send.

(The unqualified "nobody adapts" claim is false — see §6 for the alternative that does.)

---

## 1. Why not use vLLM's `AsyncLLM` in-process?

The suggestion (raised in review by a vLLM maintainer): skip HTTP and `/metrics`
entirely, drive `AsyncLLM` in-process, manage concurrency with plain asyncio.

**At the throughput level, this is right, and we measured it.** Offline `LLM.generate`,
naive `AsyncLLM` (all 2,000 tasks at once), and windowed `AsyncLLM` (semaphore 512) all landed
within ~5% of each other: 31,095 / 29,620 / 30,202 tok/s. The engine's scheduler keeps itself
fed however requests arrive. Offline was marginally the fastest arm.

> Receipt: RESULTS.md §AsyncLLM spike — job `6a67be2d6026358f64018db6`, a10g-small,
> Qwen2.5-0.5B-Instruct, 2,000 prompts, 200 forced output tokens, `max_num_seqs=256`.

So the reasons to have a client at all are not throughput. They are:

- **Nothing in the `AsyncLLM` API persists anything.** No output format, no crash-safe flush, no
  resume. A killed run re-pays from zero.
  > Receipt: the storage contract this replaces is `CONTRACT.md` §1–§3; the resume proof is
  > RESULTS.md §A — job `6a67c52b` cancelled at 20 durable rows, identical relaunch
  > (job `6a67c606`) reported `rows_done_prior: 20, rows_processed: 30, rows_failed: 0`.
- **In-process binds you to one engine's Python API.** The engine's offline Python surface drifts
  (`--task embed` became `--runner pooling --convert embed` between our own runs); the HTTP
  route did not.
  > Receipt: RESULTS.md §Embeddings — "Flag churn: `--task embed` is gone".
- **It cannot reach an endpoint you do not own.** A colleague's server, an exposed Job behind the
  proxy, TEI, a hosted API. We ran all three shapes through one client.
  > Receipt: RESULTS.md §TEI — TEI in an exposed GPU Job, client hitting `/v1/embeddings`
  > through the Jobs proxy with auth headers, 200/200 batch-rows at 23k tok/s, zero gauges.
- **Lifecycle separation.** In-process means engine crash equals client crash equals lost run.

**Honest counter — where in-process wins outright.** A single engine you own, one shot, text-only,
max throughput, no crash-resume requirement, no second consumer: in-process is simpler and
marginally faster, and you should just do it. That is why `Transport` is a protocol from day 0
and in-process is the reserved post-v1 transport, not a rejected idea.

> Receipt: [history/decisions.md](history/decisions.md) #15 — "transport option, not redesign".

**Honest scoping of the adaptive controller.** The controller is a *remote/shared-endpoint*
thesis. In-process it degenerates to a fixed feed-ahead window, and the spike shows the window
costs nothing there (windowed ≈ naive), so it is free insurance rather than a win. Separately,
the client layer only becomes decisive at high request rates: at ~15 req/s, client architecture
is measurably irrelevant (§3).

**Caveats on our own spike.** Single run per arm, 0.5B text-only model, short prompts, one A10G.
Fusion-cliff effects (§8) will not show at that model size. Do not extrapolate the flatness to
large models or to multimodal payloads — and note the spike deliberately did not test the case
where naive in-process fan-out blows up RSS (1M rows or image payloads held in-process).

---

## 2. Why not offline `LLM.generate` / `sglang.Engine` batch?

Offline batch APIs are the natural answer to "run a model over a list". Two reasons they are not
the whole answer.

**Vision workloads: serving beats offline, and we know the mechanism.** Pumping a served endpoint
ran 1.19× offline `LLM()` on vision OCR, same job, same GPU, same model.

> Receipt: evidence ledger **(internal)**, 2026-07-17 — 1.19× (v3 clean; 1.26× v2;
> 1.51× v1 superseded), marked launch-grade.

The mechanism is not hand-waved: a `max_tokens=1` probe A/B isolates preprocessing from
generation, and offline's probe took 62.3s against the pump's 20.6s — a 3.0× gap, replicated
twice. Offline serializes multimodal preprocessing; a client streaming requests overlaps its own
preprocessing with the GPU's work.

> Receipt: same ledger, "Mechanism (probe A/B, max_tokens=1) — offline probe 62.3s vs pump probe
> 20.6s (3.0×), confirmed twice".

Independently, the uv-scripts OCR sweep measured server-mode at 1.2–1.8× offline across recipes.

> Receipt: OCR server-mode sweep notes **(internal)**, 2026-07-16; the shipped recipes are uv-scripts PRs #93/#94.

**All-or-nothing.** `LLM.generate` hands back results when the whole list finishes. Crash at 90%
and you have nothing. There is no partial-output story and no resume.

**Honest counter.** On text, offline can win, and did: our own short-run text benchmark showed
offline ahead by 1.76×. That was a client slow-start artifact and was fixed the same day, but the
honest reading stands — for text at moderate scale, in-memory input, and a single owned engine,
offline is a perfectly good answer and the pump's margin is not the reason to adopt it.

> Receipt: evidence ledger **(internal)**, 2026-07-17 — "Text short run: offline **wins
> 1.76×** (honest loss → slow-start fix)".

Also honest: the 1.19× is one model, one GPU class, vision. It is not a general "serving is
faster" claim.

---

## 3. Why not datatrove's `InferenceRunner`?

**First, the claim we must not resurrect.** An earlier measurement suggested datatrove's client
path cost 2.1×–2.8× throughput versus a bare httpx harness. **That is REFUTED.** A controlled GPU
A/B — one identically launched vLLM, three arms back-to-back at the exact original regime —
found bare-pooled-httpx 163.5s, datatrove-raw 162.4s, datatrove-pooled 162.5s. All 1.00×, batch
equally full (effective concurrency ≈ 240/256 in every arm). datatrove's client path costs
nothing at its normal operating point.

> Receipt: datatrove-overhead probe notes **(internal)**, 2026-07-17 §GPU A/B — job `6a59f921d216bd6f3a1fad87`,
> a100-large, Qwen3.5-4B, 2,500 rows, conc 256, max_tokens 512.

The original 2.8× was almost certainly node contention during a busy overnight fleet run, not
code. There is no datatrove client tax. The one reproducible client-overhead finding is
narrow: connect-per-request costs ~1.64× at roughly 3,000 req/s, about 65× above that workload's
actual rate. At ≤216 req/s the two are identical.

> Receipt: same note §CPU results — 1.0s/req (216 req/s): bare 7.39s vs datatrove 7.31s, no gap;
> 0.05s/req (~3000 req/s): 9.55s vs 15.64s, 1.64×.

**So the real reasons are shape, not speed.**

- **Fixed semaphore.** `asyncio.Semaphore(self.config.max_concurrent_generations)`, created once,
  default 500. It is a number you pick.
  > Receipt: `run_inference.py:560` (semaphore), `:87` (`max_concurrent_generations: int = 500`),
  > datatrove @ `9f044f1`.
- **That number is load-bearing and hand-tuned.** From an operational ledger of real campaigns:
  "`max_concurrent_generations` silently sizes sglang CUDA-graph capture: 256 OOMs at startup on
  1×80GB with a big model (sigquit ~90s after memory-pool init, before any generation). 64 is
  safe; 128 verified on a100x4; test one step up at a time. Concurrency bump 64→128 bought only
  1.4× (prefill-bound)."
  > Receipt: datatrove-jobs operational notes **(internal)**.
  That paragraph is the product case in one quote: a number that OOMs the job if too high,
  under-delivers if too low, must be re-derived per model and per GPU, and is only discoverable
  by hand.
- **The response shape is fixed by the framework.** `InferenceResult` carries exactly `text`,
  `finish_reason`, `usage`. To capture reasoning content, a field the API returns and the
  dataclass drops, we had to fork and add `InferenceResult.reasoning`.
  > Receipt: `datatrove/pipeline/inference/types.py:8-20` @ `9f044f1` (three fields, no
  > `reasoning`); fork commit `f7df218` **(internal fork, not in this repo)**.
  saturate's `parse(row, resp)` sees the raw response body; nothing to fork.

**Complementary, not competitive.** The innermost saturate layer is an adaptive *semaphore*, not
a client — `AdaptiveLimiter` with `slot()` / `observe()`. Dropping it into datatrove is a ~5-line
diff in `_send_request` plus a constructor swap, keeping their transport, retries, checkpointing,
caching and metrics untouched.

> Receipt: [history/functional-core-proposal.md](history/functional-core-proposal.md) §Plumb-check (verified against a local datatrove
> checkout); the seam itself is `saturate/core.py` (`AdaptiveLimiter`).

**Honest counter.** datatrove wins whenever you need what it is: multi-stage pipelines, rollout
functions (a *program* per row, not one request), server crash-isolation and health-restart
inside a rank, its executor family, and its checkpointing. saturate is deliberately one request
per row and one stage; the moment your row needs a loop, datatrove or your own script is correct.

> Receipt: `CONTRACT.md` §8 non-guarantees — "one request per row (rollouts/trajectories are a
> different primitive)".

---

## 4. Why not Curator, lm-deluge, or the OpenAI cookbook script?

These are the credible small-client incumbents. They share one assumption: **rate limits are a
published quota, announced in response headers.** That assumption is exactly false against a
self-hosted engine.

**Curator** probes `x-ratelimit-*` headers once, at construction. Against vLLM those headers do
not exist, so detection fails and it falls back silently to hard-coded defaults: 200 requests per
minute, 200 concurrent.

> Receipt: probe at `src/bespokelabs/curator/request_processor/online/openai_online_request_processor.py:104`
> (inside `__init__`); fallback path and its log line "headers based detection failed, using
> default value of…" at `.../online/base_online_request_processor.py:193-200`; the values at
> `src/bespokelabs/curator/request_processor/_default_rate_limits.json` →
> `max_requests_per_minute: 200`, `max_concurrent_requests: 200`.

200 concurrent is not a measurement of your endpoint. Our own runs settled at final windows of
2, 6, 16, 32, 58, 64, 72, 80, 112, 128, and 257 across different engines and task shapes — a
static 200 is wrong in both directions depending on which run you are in.

> Receipt: RESULTS.md §Embeddings (final_limit 2), §TEI (6), §B sglang (16), §Fan-out
> (80/80/72/32), §Soak (58), §A (64); shape matrix job `6a6867146026358f64019236`
> (128/257/32/112).

**The OpenAI cookbook script** — the one everybody copies — has no concurrency cap at all. It has
zero `Semaphore` uses; admission is purely a requests-per-minute and tokens-per-minute token
bucket, and on a rate-limit error the whole loop pauses for a fixed 15 seconds.

> Receipt: `openai/openai-cookbook` → `examples/api_request_parallel_processor.py`, zero
> `Semaphore` occurrences; `seconds_to_pause_after_rate_limit_error = 15` at line 123;
> capacity model at lines 150-151, 200-206.

With rate-limit pacing but no in-flight cap, in-flight count is rate × latency — an unbounded
number when latency rises, which is precisely when you want it to fall.

**lm-deluge / BatchLLM / NeMo Curator** all take user-set limits (NeMo Curator defaults to 5
in-flight, and its docs tell you to reduce it by hand on 429).

> Receipt: prior-art deep dive **(internal)**, 2026-07-27, §5 source-verified sweep. BatchLLM is
> named as prior art extended — it ships fingerprinted checkpoints and per-row token columns.

**Honest counter.** Against a commercial API with a published quota, Curator's model is the
*correct* one and saturate's is worse: when the ceiling is a contract rather than a capacity,
discovering it by probing burns 429s for no information. That is why rate-limit pacing is a
separate concept (a Pacer over the transport), explicitly deferred rather than folded into the
controller.

> Receipt: [history/decisions.md](history/decisions.md) #14 — "Pacer: DEFER post-v1; seams only".

Curator also has real things we do not: a transparent fingerprinted request cache, structured
outputs, provider batch-API support, and a much larger user base.

---

## 5. Why not Ray Data LLM or Daft?

Both are serious distributed batch-inference systems with real streaming machinery. Neither
adapts to endpoint state.

- **Ray Data LLM**: `HttpRequestProcessorConfig` takes `concurrency` ("the number of
  concurrent requests to send… a fixed pool of `n` workers", or an autoscaling `(m, n)`
  worker pool) and an optional static `qps`. The autoscaling is of Ray's *actor pool*, on
  Ray's own scheduling signals — nothing observes the endpoint's delivered throughput or
  queue. Scaling workers on internal backpressure means scaling *into* a saturated
  endpoint, because a saturated endpoint's queue absorbs requests silently rather than
  pushing back.
  ([docs.ray.io — ray.data.llm.HttpRequestProcessorConfig](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.HttpRequestProcessorConfig.html), read 2026-07-29)
- **Daft**: their engineering blog's future-work section names the missing signal
  themselves — the router balances "using the number of prompts sent to each serving
  engine replica", and "the router should monitor the actual number of unfinished
  requests on each replica to better load balance". They have identified in-flight
  awareness as the gap and, as of that post, not shipped it.
  ([eventual.ai — Cutting LLM Batch Inference Time in Half: Dynamic Prefix Bucketing at Scale](https://www.eventual.ai/blog/cutting-llm-batch-inference-time-in-half-dynamic-prefix-bucketing-at-scale), 2025-11-04, read 2026-07-29)

**Honest counter, and it is a real one.** Ray Data wins the workload a single-loop client cannot
saturate: heavy CPU-side stages streaming into several GPU replicas concurrently, with
backpressure between heterogeneous stages and multi-replica management. If that is your shape,
use Ray. The cost is the Ray universe — multi-GB images, cluster cold-start time —
and you would still be adding endpoint-adaptive concurrency yourself, since Ray does not give it
to you.

> Receipt: the design survey **(internal)** §4 Ray-on-Jobs subsection.

Also honest: Daft's prefix-bucketing result (50.7% speedup, cache hits 29.2%→~54%; same
blog post as above, numbers verified against it 2026-07-29) is a real
batch-only optimization that saturate does not ship. Prefix-grouped admission is on the table,
not in v1.

> Receipt: the design survey **(internal)** §1 batch-only optimizations.

---

## 6. Why not NVIDIA DataDesigner? It genuinely does adapt.

It does. DataDesigner ships a
runtime-adaptive AIMD admission controller, on by default. Anything that says "no datagen
framework adapts at runtime" is retired.

What survives is a precise and, I think, sufficient distinction: **their controller throttles
down from a static cap on rate-limit responses; ours discovers capacity from delivered
throughput and server gauges.**

The mechanism, read at source:

- The only branch that *decreases* the limit is `outcome.kind == "rate_limited"`. Provider
  failures, provider timeouts, local timeouts and unexpected exceptions are all in the outcome
  taxonomy and none of them move the limit.
  > Receipt: `NVIDIA-NeMo/DataDesigner` →
  > `packages/data-designer-engine/src/data_designer/engine/models/request_admission/controller.py:577-603`
  > (`_apply_outcome`); taxonomy at `.../request_admission/outcomes.py:12-20`.
- Decrease is multiplicative by 0.75; increase is +1 after 25 consecutive successes; the limit is
  clamped at `effective_max`.
  > Receipt: `.../request_admission/config.py` — `multiplicative_decrease_factor: float = 0.75`,
  > `additive_increase_step: int = 1`, `successes_until_increase: int = 25`.
- `effective_max` derives from the user-set static cap `max_parallel_requests`, whose default
  is 4.
  > Receipt: `packages/data-designer-config/src/data_designer/config/models.py:435` —
  > `max_parallel_requests: int = Field(default=4, ge=1)`.

So the ceiling is always the user's guess. The controller's job is to back off below it, never to
find out that the endpoint would happily serve 257.

**Why that matters here specifically: self-hosted engines do not emit the signal it listens for.**
vLLM's waiting queue is unbounded — it accepts requests and queues them rather than rejecting.
The RFC to add `--max-waiting-queue-length` (returning 503) has been open since 2025-05-28 and is
still open, last touched 2026-07-22.

> Receipt: https://github.com/vllm-project/vllm/issues/18826 — state `open`, verified 2026-07-28
> via GitHub API. `--max-num-seqs` caps the running batch, not admission.

A 429-driven controller pointed at vLLM never receives a decrease signal, so it sits at its
static cap forever. That cap is either too low (you leave throughput on the floor) or too high
(you queue the engine toward OOM — the failure we actually hit; see §8).

**Honest counter, twice over.** Against hosted APIs that *do* 429, DataDesigner's design is
correct and simpler than ours. And their architecture independently validates ours: they split
admission (adaptive window, leases, outcome-classified release) from transport (HTTP plus
non-429 retries), with the explicit invariant that 429 is never retried at transport so the
signal reaches the admission loop. That is structurally `AdaptiveLimiter.slot()/observe()`. We
adopted their outcome-classification lesson — `observe()` takes a classified outcome, not a bare
bool.

> Receipt: [history/functional-core-proposal.md](history/functional-core-proposal.md) §NVIDIA DataDesigner; [history/decisions.md](history/decisions.md)
> §Prior-art correction 2026-07-28.

---

## 7. Why not llm-d / the k8s inference gateways?

Because they are the right idea at a different layer, in a different language, with a different
deployment story — and because their existence is evidence *for* this design, not against it.

**They read the same gauges.** The k8s Gateway API Inference Extension model-server protocol
requires model servers to expose `TotalQueuedRequests`, `TotalRunningRequests` and
`KVCacheUtilization`, and maps them to the exact per-engine metric names our dialect registry
uses — `vllm:num_requests_waiting` / `sglang:num_queue_reqs`, `vllm:num_requests_running` /
`sglang:num_running_reqs`, `vllm:kv_cache_usage_perc` / `sglang:token_usage`, plus Triton
TensorRT-LLM and trtllm-serve.

> Receipt: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/proposals/003-model-server-protocol/README.md
> §Metrics Reporting, read in full 2026-07-28.

**State this precisely.** The document is marked *Partially implemented*, it notes the protocol
"can, by definition, not be as strict" since the pluggable-architecture proposal, and it says
explicitly that metric **names** need not match — only the metric **types and semantics** MUST
follow. So: the semantics are contractual, the spellings are not. That is exactly why the dialect
registry matches both prefix spellings and falls back to blind mode on a 404, rather than
assuming names are stable.

> Receipt: [history/decisions.md](history/decisions.md) #6; the live demonstration that
> spellings drift is https://github.com/sgl-project/sglang/issues/12618 — SGLang users saw
> `sglang_` where the exporter emits `sglang:` (collector-side normalization, not an
> exporter rename; the fix touched only a Grafana dashboard). Matching both spellings
> covers whichever form reaches the client.

**Where llm-d actually is.** Two Go/Apache-2.0 repos, both actively developed:

- `llm-d-async` has the gauge-driven piece: an `endpoint-scrape` gate that scrapes raw
  `/metrics` with no Prometheus server, computes saturation as
  `vllm:num_requests_waiting / max_count_per_pod`, and returns an available budget.
  > Receipt: `llm-d/llm-d-async` README §gate types; integration tests at
  > `test/integration/endpoint_scrape_gate_factory_test.go`. Its `LocalConcurrencyGate`
  > (`pkg/async/inference/flowcontrol/local_concurrency_gate.go`) is a **fixed** limit.
- `llm-d-batch-gateway` has the AIMD piece, and its signal is **response status**, not gauges:
  decrease on 429, on 5xx, and on capacity-retry; increase on success.
  > Receipt: `internal/processor/pipeline/dispatcher_aimd.go`, `recordAIMDSignal` — the three
  > decrease branches and the success branch, read 2026-07-28.

**So the two halves are in different components.** Gauges feed a gate (admit / hold); status
codes feed the ramp. Nobody feeds the gauges *into* the ramp. "llm-d does AIMD on scraped
metrics" conflates the two components.

**And it is not something you can `uv run`.** Prerequisites: PostgreSQL 12+, Redis 6+ or Valkey
8+, and S3-compatible object storage or a filesystem, deployed as separate API server, batch
processor and dispatcher components.

> Receipt: `llm-d/llm-d-batch-gateway` README §Prerequisites, lines 166-168.

**Honest counter.** At platform scale — many tenants, many pods, a K8s cluster you already run —
llm-d and the inference gateways are the right answer and saturate is not competing. saturate is
for one person with a dataset, an endpoint and a Job. Routing across replicas is genuinely out of
scope: through a load balancer, per-endpoint gauges are wrong, and the honest behaviour is to
require a direct endpoint or degrade to blind mode.

> Receipt: prior-art deep dive **(internal)**, 2026-07-27, engine-side risk list — "Multi-replica LB:
> genuinely wrong through an LB".

---

## 8. Why a controller at all — why not just pick a number?

Because the right number is not knowable in advance, changes with task shape, and is dangerous
in both directions. Five receipts.

**(a) The optimum moves with task shape, on the same GPU and model.** Four arms — short and long
input crossed with short and long output — run on one A10G job. The window settled at 128, 257,
32 and 112 respectively.

> Receipt: job `6a6867146026358f64019236`, $0.09, 4,000/4,000 rows, 0 failed.
> **Caveat, and it matters: arms a and b ended mid-ramp after 7-8 controller ticks, so 128 and
> 257 are upper observations, not equilibria.** The load-bearing number is arm c: pinned at 32 by
> the throughput-plateau gate against a flat 36k tok/s prefill roof — the plateau signal
> correctly refusing to grow into a compute-bound regime.

A single fixed concurrency covering both 32 and 257 does not exist.

**(b) Kernel selection depends on the concurrency the client picks.** A vLLM maintainer's own
benchmark on a residual-add + RMSNorm fusion PR: **+18.31% output tok/s and −15.66% mean TPOT at
`--max-concurrency 32`**, and at 1,000 prompts with no concurrency cap, −1.7% ±34% — pure noise.
The cause is a size gate on FlashInfer's allreduce fusion on SM90/TP8.

> Receipt: https://github.com/vllm-project/vllm/pull/48757#issuecomment-5093704323
> (Qwen3-32B-FP8, 8×H100, TP8, 3 trials/arm), read in full 2026-07-28.

The same PR benchmarked on 8×MI355X at the same concurrency 32 gives +2.9%, not +18.3%.

> Receipt: https://github.com/vllm-project/vllm/pull/48757#issuecomment-5100074377.

So the efficiency-versus-concurrency landscape is non-monotonic, and its thresholds are specific
to hardware, tensor-parallel degree and engine version. Nobody can precompute your number.
Watching delivered throughput at runtime is the only general answer.

**Bench rule banked from the same comment**: `--dataset-name random` regenerates identical
prompts, so prefix-cache hit rate climbed 20% → 98% across trials. Any saturate A/B uses unique
prompts or disables prefix caching on both arms.

**(c) Too high kills the job.** A 5,000-page vision soak was OOMKilled (exit 137, host RAM) about
a minute in. With ~30s generations the early ticks had zero completions, so there was no
throughput signal for the plateau gate; multimodal preprocessing queued ahead of the scheduler's
gauges, so `waiting` stayed low. Slow-start doubled unopposed from 8 to 512 in about 12 seconds
and hundreds of in-flight 1540px images killed the container.

> Receipt: RESULTS.md §The OOM lesson.

The fix is the TCP rule — never widen a window nothing has ever been acknowledged through:

> Receipt: `saturate/controller.py:72` (`self._seen_ok = False  # ACK-clock: no growth before the
> first completion ever`); enforcement at `:96-98`.

This is also the honest admission that an adaptive controller has its own failure mode. It cost
us a job to find. The mitigation is in the code and in the guidance to cap `max_limit` for vision
work.

**(d) Too low leaves the endpoint idle, and you cannot tell by looking.** The published
uv-scripts OCR recipes use a fixed concurrency of 32; the same class of workload self-ranged to
376 in POC testing — roughly 10× headroom that a static recipe simply never claims.

> Receipt: sprint notes **(internal)**, item 11 — "the conc-32 vs window-376 gap (~10×
> headroom)".

**(e) It also finds ceilings downward, which is the part people don't expect.** Against vLLM
pooling, which serves one 64-sequence batch at a time, the window *shrank* to a final limit of 2
rather than queueing into it.

> Receipt: RESULTS.md §Embeddings — 500/500, 7.85M tokens, 33.5k tok/s, `final_limit=2`.

**Honest counter.** If you have already measured your exact model, hardware, engine version and
task shape, and none of them will change, a fixed number is free and carries no controller risk.
`window=Fixed(64)` is a supported first-class option for exactly that reason. The controller
earns its place when any of those variables move — which, across a fleet of Jobs, is always.

---

## 9. Why parquet plus a manifest, instead of JSONL or a database?

**Typed arrays and compression.** Embedding output is 1024-dimensional float arrays; those land
as typed array columns in parquet with zstd, and are read back without parsing.

> Receipt: RESULTS.md §Embeddings — "1024-dim array columns landed in parquet on hf://",
> 32k texts, 7.85M tokens.

**Resume is exact, and proven across process and job boundaries.** Twice:

- Small: cancelled at 20 of 50 durable rows; identical relaunch reported `rows_done_prior: 20,
  rows_processed: 30, rows_failed: 0`. Verified 5 parts, 5 manifests, 50 records, 50 unique ids,
  0 duplicates across two jobs.
  > Receipt: RESULTS.md §A — jobs `6a67c52b` → `6a67c606`.
- Large, and unplanned: a 5,000-page soak was SIGTERM'd platform-side at 4,450 durable rows.
  Identical relaunch: `rows_done_prior: 4450, processed: 550, failed: 0`. Verified 100 parts,
  5,000 records, 5,000 unique, 0 duplicates.
  > Receipt: RESULTS.md §Soak.

And across concurrent writers with no coordinator: 4 Jobs, 4 engines, one output directory —
80 parts, 4,000 records, 4,000 unique, 0 duplicates, with each shard's controller finding its own
equilibrium (80/80/72/32).

> Receipt: RESULTS.md §Fan-out.

**Why the manifest sidecar rather than scanning the parts.** The done-set is assembled
manifest-first from `[id, error]` sidecars; only a part whose sidecar is missing is scanned
individually. A crash between part-write and manifest-write therefore costs one part of rework,
never duplicates, and never the whole output.

> Receipt: `CONTRACT.md` §3.

**Why not a database.** The output has to be readable by anything that can glob a directory and
read parquet, without importing saturate, from a laptop, a Job, or DuckDB. A database is a
service to run, a schema to migrate, and a credential to pass into every Job.

**Sharp edge, stated rather than hidden.** Error rows are sparse — `{id, error}` with no user
columns — so a directory containing both success and error parts has two schemas in it, and a
naive `pq.read_table(dir)` will fail to union them. Readers doing strict schema unions must
expect nullable user columns. This is a real cost of "error rows, not lost rows" and it is in
the contract, not a surprise, but it does qualify the "any parquet reader" promise above.

> Receipt: `CONTRACT.md` §1 required-columns note; observed live on the shape-matrix run
> (job `6a6867146026358f64019236`), partially mitigated since by pinning the `error` column to
> string with a mixed-parts test (commit `4044793`).

**The escape hatch is a protocol, not a fork.** Resume is one invariant — *an id in
`existing_ids()` means its record is durable; an id absent means re-processing it is safe* — and
any sink satisfying it gets resume. `FileSink` is the second blessed implementation: one file per
row named by id, the filesystem is the manifest, overwrites are idempotent. Non-resumable IO is
the same protocol with an empty `existing_ids()`.

> Receipt: [history/functional-core-proposal.md](history/functional-core-proposal.md) open choice #0 (settled); `saturate/sink.py:105`
> (`FileSink`), `:53` and `:116` (the two `existing_ids` implementations), `:156`
> (`read_output`).

**Honest trade, stated in the code's own docs.** `FileSink` writes no file for a failed row, so
there is no error record and failures simply retry next run. That is a real loss of the
"error rows, not lost rows" invariant, and it is documented as the trade rather than papered
over.

> Receipt: `README.md` §Bring your own IO; `CONTRACT.md` §4.

**Honest counter.** For a few thousand rows, JSONL plus `sort -u` is fine and needs no library.
The parquet contract earns its keep at the point where you have array columns, multiple concurrent
writers, or a run long enough that a crash costs real money.

---

## 10. Why not build a pipeline framework or a DAG?

Because the DAG shape was inherited from a different problem, and the market has already said so.

**The exemplar froze.** distilabel — the purest DAG design in synthetic data — has had no release
since 1.5.3 on 2025-01-28. Since then the repository has only taken maintenance commits (the most
recent is a docs typo fix).

> Receipt: GitHub API, `argilla-io/distilabel`, verified 2026-07-28 — latest release `1.5.3`,
> published 2025-01-28; most recent commit 2025-12-15, "Fixed a pipeline docs typo".

**The preconditions that make DAGs work do not hold here.** dbt's DAG works because nodes are
cheap, deterministic, durable and inspectable relations on one substrate. LLM calls are
expensive, non-deterministic and billed, so "recompute the node" is the wrong failure story and
the useful recompute granularity is the *row*, which is exactly what an id-keyed append-only
output gives you.

> Receipt: the design survey **(internal)** §2 and §5.

**So composition is plain Python.** `pump()` is a composition of four small stages, and chaining
two stages is calling `through()` twice, or piping through storage with `read_output()`. No
scheduler, no graph, no `.compute()`. Persistence points are the composition boundaries: each
`drain()` buys crash-safety and resume for its stage; in-memory chaining is allowed and
documented as trading away crash granularity.

> Receipt: [history/functional-core-proposal.md](history/functional-core-proposal.md) §Rules — "itertools, not dask.bag"; `README.md`
> §The composable layer.

**Honest counter.** If you have a genuine DAG — heterogeneous CPU and GPU stages, fan-in,
conditional branches, cross-stage retries — you want an orchestrator, and saturate is not one.
It is designed to sit *inside* one, which is why the output convention (`completions/shard-{n}.done`)
is deliberately datatrove's, so a coordinator can watch the directory without knowing anything
about saturate.

> Receipt: `CONTRACT.md` §5.

---

## Appendix: claims from earlier drafts, retired or refuted

Kept public for honesty: if you have seen one of these claims made for this library
elsewhere, here is its current status.

| Claim | Status | Why |
|---|---|---|
| "datatrove's client costs 2.1× / 2.8×" | **REFUTED** | Controlled GPU A/B: 1.00×, all three arms. Cause was node contention. Receipt: datatrove-overhead probe notes **(internal)**, 2026-07-17 §GPU A/B, job `6a59f921d216bd6f3a1fad87`. |
| "No datagen framework adapts at runtime" / "nobody adapts" | **RETIRED** | DataDesigner ships AIMD admission, default-on. Use the narrowed claim in §6. Receipt: [history/decisions.md](history/decisions.md) §Prior-art correction. |
| "Nobody owns cost tracking" | **RETIRED** | batchata and BatchLLM ship budget-as-run-parameter; LiteLLM owns gateway budgets. Defensible remainder: per-row dollar *provenance* in the output data, self-hosted cost modeling, HF-native resumable parquet. Receipt: the design survey **(internal)** §1 finding 3 (corrected). |
| "The keep-alive HTTP client is a performance fix" | **RETIRED** | raw == pooled == bare at 1.00× on a real vLLM. Pooled only helps at ~1000+ req/s with very short outputs. Receipt: datatrove-overhead probe notes **(internal)**, 2026-07-17 §What this means. |
| "llm-d-async does AIMD on scraped /metrics" | **IMPRECISE** | The gauge-scrape gate is in `llm-d-async`; the AIMD dispatcher is in `llm-d-batch-gateway` and its signal is response status. See §7. |
| "The gauges are contractual" (unqualified) | **NARROW IT** | GAIE requires the metric *types and semantics*, explicitly not the names, and the proposal is marked *Partially implemented*. See §7. |
| "Serving beats offline" (unqualified) | **NARROW IT** | 1.19× is vision, one model, one GPU class. Text short-run showed offline winning 1.76× before the slow-start fix. See §2. |
