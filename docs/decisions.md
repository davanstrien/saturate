# DECISIONS.md — build decisions (kickoff 2026-07-27)

Confirmed at kickoff (Daniel, 2026-07-27, Slack-thread-day). Source lineage: the 14 staged
sprint decisions (vault: `pumpjack-clean-poc-proposal-2026-07-17` §9 + project note items
11–14), amended by the prior-art deep dive and the validation-thread replies. POC = oracle,
not substrate: `pumpjack-poc` is frozen; this repo is judged by `pumpjack-oracle` (9 tests).

| # | Decision | Resolution |
|---|---|---|
| 1 | Layout | 9 modules; LOC ceiling CI-checked (see `tests/test_loc_ceiling.py`) |
| 2 | Request | Typed union json⊕multipart, `.kind` discriminator; `_files` hack dead |
| 3 | parse | `parse(row, resp)` row passthrough; single-arg `parse(resp)` also accepted (introspected) — keeps oracle + POC recipes working |
| 4 | Engine | Boot templates (vllm/sglang/sgl-omni/llamacpp) + ceiling-flag table; readiness = health×N + trial completion; killpg lifecycle; GGUF = download weights |
| 5 | Controller | Sans-IO `decide(obs, limit)`; **throughput-primary, gauges secondary, blind AIMD floor**; debounced slow-start exit (the traces' priority fix); two-condition cut (kv high AND hits low); fixed band pending calibration grid; 429+Retry-After honored, never cut; `breaker_max_open_s` so a dead server exits |
| 6 | Signals | `SignalSource` seam: `http-scrape` (MVP) / `in-process` (post-v1) / `none`. Scrape table keyed on GAIE model-server-protocol semantics, **dual prefix spellings** (`vllm:`/`vllm_`, `sglang:`/`sglang_`) — SGLang broke once (#12618), vLLM flip is roadmapped |
| 7 | Sink | `_manifest/` sidecar 1:1 with parts; manifest-first **exact** resume (unmatched parts scanned individually — kill-safe, no dupes); minimal-effort implementation per Daniel's steer |
| 8 | CONTRACT | Frozen 2026-07-27 (this repo). id = content-hash default (`content_id`), simplest correct form |
| 9 | Name/repo | `pumpjack` working name; private `davanstrien/pumpjack`, GitHub before PyPI; name not precious. Layer split (Daniel): "Inference Pipelines"-style official layer sits ON TOP of this client; HF names the wrapper, this library keeps its own identity |
| 10 | Schemas | Telemetry v1 = 8 core + `tok_s`/`kv`/`hits`/`preempts`; Stats v1 = POC 11 keys + `breaker_opens`. Frozen in CONTRACT §6–7 |
| 11 | Recipes | uv-scripts stay static mini-pumps; pilot = lighton-ocr2 transport swap, gate: name + CONTRACT + pilot green. Not this build |
| 12 | Micro-batch-as-row | **PROVEN LIVE for embeddings** (64/req vLLM, 32/req TEI — array input is OpenAI-spec and effectively universal on `/v1/embeddings`). Route matrix (2026-07-28): `/embeddings` universal · `/completions` accepts `prompt: [list]` on vLLM+SGLang → a cheap high-rate escape for base-model workloads, worth trying BEFORE the in-process transport · `/chat/completions` never batches (spec: one conversation/request; `n`=samples) → chat keeps the in-process case alive |
| 13 | Embeddings/TEI | DEFER. Alvaro (2026-07-27): TEI router is **token-based** (`--max-batch-tokens`, opt. `--max-batch-requests`); `/embed` takes string or list → batch-per-request natively supported; right admission unit is likely tokens. TEI dialect gauge ready when needed: `te_queue_size` (canonical KEDA signal). Ref: his FineWiki/Cloud Run example (hf.co/docs/google-cloud) |
| 14 | Pacer | DEFER post-v1; seams only (header-dialect slot, 429 taxonomy in controller) |
| 15 | In-process async transports | Harry (2026-07-27): "use AsyncLLM, manage concurrency with asyncio, skip /metrics." SGLang parallel: `sgl.Engine.async_generate`. Verdict: **transport option, not redesign** — neither has persistence/resume/admission/remote; `Transport` is a protocol from day 0 (HTTP-only impl in MVP); the adaptive controller is a remote/shared-endpoint thesis, in-process degenerates to a Fixed feed-ahead window |

## Prior-art correction (2026-07-28, DataDesigner recon)

**NVIDIA DataDesigner ships runtime-adaptive AIMD admission, default-on** — the exception to
round-1's "nothing adapts" sweep (which had it as "fixed, user-set": wrong; the sweep saw the
config knob, not the `request_admission` controller behind it). Precise surviving claim for
all external writing: *pumpjack is the only standalone client whose controller **discovers**
capacity (throughput-primary + gauges) — DataDesigner's controller **throttles down** from a
static cap (default 4) on 429s only, which self-hosted engines never send.* Full analysis:
PROPOSAL-functional-core.md §plumb-check. Their admission/transport split independently
validates the AdaptiveLimiter layering.

## Thread receipts informing the design (2026-07-27)

- **vLLM PR #48757, Harry's own benchmark comment (read in full 2026-07-27)**: residual-add+RMSNorm
  fusion gives **+18.3% tok/s / −15.7% TPOT at concurrency 32, and no measurable gain in an
  UNCAPPED 1000-prompt run** (noise ±34%) — FlashInfer's allreduce fusion has a 51-token size gate
  on SM90/TP8, so *which kernels run depends on the concurrency the client picks*. Cross-hardware
  kicker (same PR, comment 5100074377): on 8×MI355X the identical `--max-concurrency 32` gives only
  **+2.9%** — thresholds are hardware/TP/version-specific. The efficiency-vs-concurrency landscape
  is non-monotonic and unknowable in advance → runtime delivered-throughput watching is the only
  general answer. Decision 5's receipt, from a vLLM maintainer's bench.
  **Bench rule banked from the same comment**: `--dataset-name random` regenerates identical
  prompts (his prefix-cache hit rate climbed 20%→98% across trials) — Tier 1/2 runs use unique
  prompts or disable prefix caching on both arms.
- Julien: "find a good catchy name and then we'll build an official UI for it" → the wrapper
  seam in CONTRACT ("The wrapper seam") is the interface that UI builds on.
- Quentin: datatrove-rebranding/home suggestion → position held: lightweight primitive datatrove
  can *build on*, not built inside (interop via the storage contract + `completions/` convention).

## Deviations from staged defaults

- LOC ceiling set to **800** (was ~700): the clean split carries module boundaries, the typed
  Request union, manifest sidecars, SignalSource seam, and the observable breaker the POC
  lacked. Enforced in CI; renegotiate downward after M3 trims, not by deleting docstrings.
- **Renegotiated to 1100 (2026-07-28 PM)**: +10 for the `completions/stats-{n}.json` sidecar
  (console-gap findings G2/G5 from the UI POC — exact counts + shard geometry for
  storage-only readers). Prior step was 1050 for probe-and-revert.
- **Renegotiated to 1000 (2026-07-28)** with the approved functional core: `core.py`
  (AdaptiveLimiter/AdaptiveClient/through, 187) + FileSink/read_output/drain are new public
  surface with named consumers, not bloat — the facade itself *shrank* 203→123. Current
  total 971 **non-blank lines** (the CI measure: `grep -cve '^\s*$' pumpjack/*.py`; raw
  `wc -l` reads ~1190). Same rule: trim honestly later, never by stripping docs.
- Markers: **advisory** (CONTRACT §5). Sparse error rows kept (§1). Per-row token columns
  standardized-when-present, not required (§1). Telemetry: all four proposed keys enter v1
  (the controller consumes them; a v2 bump later would be sillier).

## Verification bars (this build)

Tier 0: oracle `ORACLE_ADAPTER=clean` → 9 passed / 0 xfail; ruff clean; LOC check green.
Tier 1 (≤$20, signal-first): throughput parity vs bare-httpx/POC on one real model →
kill/resume on a real Job → 50-page OCR parity → engine boots if budget remains.
Tier 2 (~$30–50) is sprint-week material, flagged before spend.

## Pre-public checklist (added 2026-07-28 — none of these block internal sharing)

- [ ] Docstring de-storying: strip design-history references from code (keep constraints,
      move stories here / RESULTS). After the Codex fix-diff re-review closes.
- [ ] Scrub `docs/history/` + this file for internal thread quotes/names before flipping public.
- [ ] why.md: replace the two second-hand rows (Ray Data, Daft) with directly-fetched sources.
- [ ] Daniel voice pass on README + why.md (writing-review).
- [ ] PyPI name registration; decide public-repo timing vs the embeddings showcase.

## Console-gap findings (2026-07-28, from the UI POC — storage-only dashboard build)

The POC's job was to find where the CONTRACT makes a console awkward. It found five (G1–G5);
dispositions:

| gap | disposition |
|---|---|
| G5 exact counts need parquet · G2 world undiscoverable | **FIXED in pumpjack**: `completions/stats-{n}.json` (full Stats + rank/world, CONTRACT §5). Telemetry Σok stays approximate mid-run — documented |
| G1 no progress denominator · G4 items-per-row not in storage | **Wrapper layer** (`run.json` launch-spec sidecar: expected rows, world, items_per_row, written by the pipelines layer at launch — it's intent, not output; pumpjack stays a recorder of what happened). Spec belongs in the inference-pipelines design |
| G2 dead-on-arrival shard (live case) | run.json's `world` covers it at launch time; stats-{n}.json covers it at completion |
| G3 liveness (stalled vs slow) | **DEFERRED with tension named**: a heartbeat needs the CONTRACT to promise a telemetry flush cadence, which costs commits on dataset-repo sinks (free on buckets). Candidate: periodic telemetry flush (default ~60s, configurable), cadence advisory. Decide alongside the staged-output (bucket-hot) work — heartbeats and buckets want each other |
