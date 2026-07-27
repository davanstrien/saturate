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
