# Tier 1 round 2 (2026-07-27 late — fan-out, embeddings, TEI, the OOM lesson)

## Fan-out — 4 Jobs, 4 engines, ONE output: FLAWLESS
4×1000 dolly rows, strided assignment. Verified: 80 parts, **4,000 records / 4,000 unique /
0 dupes**, markers shard-0..3.done, 4 telemetry files. Each shard's controller found its own
equilibrium independently (final windows 80/80/72/32) — per-shard adaptivity with zero
coordination is the fan-out-to-storage argument, demonstrated.

## Embeddings (vLLM pooling) — zero library changes: PASS
`route="/embeddings"` + micro-batch-as-row (500 rows × 64 texts = 32k fineweb-edu texts).
500/500, 0 failed, 7.85M tokens at 33.5k tok/s; 1024-dim array columns landed in parquet on
hf://. **Controller discovered the server ceiling downward**: vLLM pooling served one 64-seq
batch at a time → window shrank to final_limit=2 instead of queueing into it, and the advisor
fired its first real hint. Advisor nuance found: it counts requests, not items — normalize by
items-per-request for micro-batch workloads (decision 12 note). Flag churn: `--task embed` is
gone; current = `--runner pooling --convert embed`.

## TEI — third serving stack, Job-to-Job, BLIND MODE: PASS
TEI (Qwen3-Embedding-0.6B, `:86-latest`, `--max-batch-tokens 32768`) in an exposed GPU Job;
pump clients hit `/v1/embeddings` through the jobs proxy with auth headers, **no gauges at
all** (TEI metrics port not proxied). CPU client Job: **200/200 batch-rows (6,400 texts,
1.1M tokens) at 23k tok/s in 48s, 0 failed**, blind AIMD settled at 6. Laptop client ran the
same shape from outside. Gotchas banked: `hf jobs run IMAGE args...` replaces the entrypoint
(name `text-embeddings-router` explicitly); `python:3.12-slim` has no curl.

## The OOM lesson → ACK-clocked slow start (the night's most valuable failure)
First 5k-page vision soak OOMKilled (exit 137, host RAM) ~1 min in: with ~30s generations,
early ticks have ZERO completions → no tok_s for the plateau gate, and multimodal
preprocessing queues ahead of the scheduler's gauges (waiting stayed low — the vLLM twin of
the SGLang undercount) → slow-start doubled unopposed 8→512 in ~12s → hundreds of in-flight
1540px images killed the container. **Fix (TCP rule): never widen a window nothing has ever
been acknowledged through** — `Auto` now holds until the first completion. Plus multimodal
guidance: cap `max_limit` (~128) as a host-RAM guard. Oracle after fix: 9/9.

## Soak v2 (5k MOH pages, ACK-clocked, max_limit=128)
Running at consolidation time; result appended when it lands.

---

# Tier 1 results (2026-07-27, evening fleet — total spend ≈ $2)

All against the CLEAN package (wheel 0.1.0) on real Jobs, a10g-small.

## A — LightOnOCR-2 / vLLM pilot + cross-job kill/resume (jobs 6a67c52b → cancelled → 6a67c606)

**PASS, exact.** Run 1 cancelled at 2 durable parts (20 rows of 50). Identical relaunch:
`rows_done_prior: 20, rows_processed: 30, rows_failed: 0` — paid precisely the remainder.
Output verified from laptop: 5 parts, 5 manifests, **50 records / 50 unique ids / 0 dupes**
across the two jobs. First-ever cross-job manifest resume over `hf://datasets`. Window
settled 64, 1,440 tok/s on vision. (Post-run `EngineDeadError` in logs = killpg teardown
noise, after stats.)

## B — SGLang v0.5.10.post1 boot + live dialect check (job 6a67c4eb)

**PASS, with a finding.** 400/400, 0 failed. Boot template + readiness gate clean.
Gauges parsed on 12/13 ticks (one scrape hiccup → graceful None). **Dialect quirk found:**
SGLang's `num_running_reqs` stayed 3–21 and `num_queue_reqs` stayed 0 while the client had
64 in flight — requests pending in the HTTP layer are invisible to both gauges. The
controller grew on that phantom headroom, hit backpressure, halved, and settled at 16 —
i.e. **the non-gauge signals (bp + throughput) carried it**, which is the signal-priority
decision doing its job on a real engine. Note for the dialect table: sglang gauges
undercount pending work; weight them lower than vllm's.
Also: `lmsysorg/sglang:latest` is currently a broken dev build (`sglang.srt.server_args`
missing) — boot templates should carry known-good pinned tags.

## C — streaming synth-data shape, 5k rows (job 6a67c43d)

**PASS, with a live catch.** 5,000/5,000 processed, 0 failed, 5,011 tok/s, window 8→136,
never input-bound (streaming dolly kept up). True generator source (nothing materialized);
content-hash ids' first live run — which surfaced **38 identical rows in dolly's first 5k**
(4,962 unique ids). Reader rule already dedupes; fixed properly same evening: within-run
exactly-once admission (`rows_deduped` Stats field) — content-hash ids now dedupe input for
free. Oracle re-run after fix: 9/9.

## Fleet bloopers (cheap, all mine)

`python` vs `python3` in the vllm image ($0.004) · wheel saved without its canonical
filename (pip validates the *name*) · `--limit 100` on a 50-row dataset. Total wasted ≈ $0.05.

---

# AsyncLLM spike results (2026-07-27)

Job: `davanstrien/6a67be2d6026358f64018db6` (a10g-small, vllm/vllm-openai:latest, ~$0.15).
Qwen/Qwen2.5-0.5B-Instruct · 2,000 dolly prompts · 200 forced completion tokens (ignore_eos)
· `max_num_seqs=256` · each arm a fresh subprocess (clean VRAM).

| arm | wall s | tok/s | Δ vs offline | peak RSS* |
|---|---|---|---|---|
| offline `LLM.generate` | 12.9 | 31,095 | — | 1.7 |
| AsyncLLM **naive** (all 2k tasks at once) | 13.5 | 29,620 | −4.7% | 1.4 |
| AsyncLLM **windowed** (semaphore 512) | 13.2 | 30,202 | −2.9% | 1.4 |

*`peak_rss_mb` field is mislabeled — Linux `ru_maxrss` is KB, so values are ~GB. Consistent
across arms, which is what the comparison needs.

## Read

1. **Harry is right at this scale**: the engine's scheduler keeps itself fed however requests
   arrive — all three arms within ~5% (≈ run-to-run noise). Client feed pattern is
   throughput-neutral in-process.
2. **The window is free**: windowed ≈ naive on throughput, so a bounded feed-ahead window is
   costless insurance — validating the design call that in-process control degenerates to
   `Fixed(~2× max_num_seqs)`, no AIMD needed when you own the engine.
3. **What this does NOT test** (deliberately): the naive-RSS blowup case (1M rows /
   multimodal payloads held in-process), remote endpoints, resume. The library's value was
   never "feed the engine better" — it's kill-safe resume + incremental parquet + one client
   for endpoints that aren't your local vLLM. This spike is the receipt that in-process
   transport is *easy to add* (post-v1), not a reason to exist.

## Caveats

Single run per arm; 0.5B text-only; short prompts; A10G. Fusion-cliff effects (vLLM #48757)
won't show at this model size — don't over-extrapolate the flatness to big models.
