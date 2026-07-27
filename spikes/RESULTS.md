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
