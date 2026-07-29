# DECISIONS.md — build decisions (kickoff 2026-07-27)

Confirmed at kickoff (2026-07-27). Source lineage: the 14 staged sprint decisions
(internal planning notes), amended by the prior-art deep dive and the
validation-thread replies. POC = oracle,
not substrate: `pumpjack-poc` is frozen; this repo is judged by `pumpjack-oracle` (9 tests).

| # | Decision | Resolution |
|---|---|---|
| 1 | Layout | 9 modules (+`sources`). Scope-creep rule (final form, 2026-07-28): surface grows only by deliberate, recorded decision — a decisions.md entry naming the new surface and why. No line counting (the numeric ceiling/note is retired; its history below) |
| 2 | Request | Typed union json⊕multipart, `.kind` discriminator; `_files` hack dead |
| 3 | parse | `parse(row, resp)` row passthrough; single-arg `parse(resp)` also accepted (introspected) — keeps oracle + POC recipes working |
| 4 | Engine | Boot templates (vllm/sglang/sgl-omni/llamacpp) + ceiling-flag table; readiness = health×N + trial response (**alive-only by default**: <500 incl. 404; workload-strict via `ready_accept=`); killpg group lifecycle; GGUF = download weights |
| 5 | Controller | Sans-IO `decide(obs, limit)`; **throughput-primary, gauges secondary, blind AIMD floor**; debounced slow-start exit (the traces' priority fix); two-condition cut (kv high AND hits low); fixed band pending calibration grid; 429+Retry-After honored, never cut; `breaker_max_open_s` so a dead server exits |
| 6 | Signals | `SignalSource` seam: `http-scrape` (MVP) / `in-process` (post-v1) / `none`. Scrape table keyed on GAIE model-server-protocol semantics, **dual prefix spellings** (`vllm:`/`vllm_`, `sglang:`/`sglang_`) — SGLang broke once (#12618), vLLM flip is roadmapped |
| 7 | Sink | `_manifest/` sidecar 1:1 with parts; manifest-first **exact** resume (unmatched parts scanned individually — kill-safe, no dupes); minimal-effort implementation per Daniel's steer |
| 8 | CONTRACT | Frozen 2026-07-27 (this repo). id = content-hash default (`content_id`), simplest correct form |
| 9 | Name/repo | `pumpjack` working name; private `davanstrien/pumpjack`, GitHub before PyPI; name not precious. Layer split (Daniel): "Inference Pipelines"-style official layer sits ON TOP of this client; HF names the wrapper, this library keeps its own identity |
| 10 | Schemas | Telemetry v1 = 8 core + `tok_s`/`kv`/`hits`/`preempts`; Stats v1 = POC 11 keys + `breaker_opens`. Frozen in CONTRACT §6–7 |
| 11 | Recipes | uv-scripts stay static mini-pumps; pilot = lighton-ocr2 transport swap, gate: name + CONTRACT + pilot green. Not this build |
| 12 | Micro-batch-as-row | **PROVEN LIVE for embeddings** (64/req vLLM, 32/req TEI — array input is OpenAI-spec and effectively universal on `/v1/embeddings`). Route matrix (2026-07-28): `/embeddings` universal · `/completions` accepts `prompt: [list]` on vLLM+SGLang → a cheap high-rate escape for base-model workloads, worth trying BEFORE the in-process transport · `/chat/completions` never batches (spec: one conversation/request; `n`=samples) → chat keeps the in-process case alive |
| 13 | Embeddings/TEI | DEFER. TEI-side steer (2026-07-27): TEI router is **token-based** (`--max-batch-tokens`, opt. `--max-batch-requests`); `/embed` takes string or list → batch-per-request natively supported; right admission unit is likely tokens. TEI dialect gauge ready when needed: `te_queue_size` (canonical KEDA signal). Ref: the TEI FineWiki/Cloud Run example (hf.co/docs/google-cloud) |
| 14 | Pacer | DEFER post-v1; seams only (header-dialect slot, 429 taxonomy in controller) |
| 15 | In-process async transports | Engine-side steer (2026-07-27): use AsyncLLM, manage concurrency with asyncio, skip /metrics. SGLang parallel: `sgl.Engine.async_generate`. Verdict: **transport option, not redesign** — neither has persistence/resume/admission/remote; `Transport` is a protocol from day 0 (HTTP-only impl in MVP); the adaptive controller is a remote/shared-endpoint thesis, in-process degenerates to a Fixed feed-ahead window |

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

- **vLLM PR #48757, the author's benchmark comment (read in full 2026-07-27)**: residual-add+RMSNorm
  fusion gives **+18.3% tok/s / −15.7% TPOT at concurrency 32, and no measurable gain in an
  UNCAPPED 1000-prompt run** (noise ±34%) — FlashInfer's allreduce fusion has a 51-token size gate
  on SM90/TP8, so *which kernels run depends on the concurrency the client picks*. Cross-hardware
  kicker (same PR, comment 5100074377): on 8×MI355X the identical `--max-concurrency 32` gives only
  **+2.9%** — thresholds are hardware/TP/version-specific. The efficiency-vs-concurrency landscape
  is non-monotonic and unknowable in advance → runtime delivered-throughput watching is the only
  general answer. Decision 5's receipt, from a vLLM maintainer's bench.
  **Bench rule banked from the same comment**: `--dataset-name random` regenerates identical
  prompts (the reported prefix-cache hit rate climbed 20%→98% across trials) — Tier 1/2 runs use
  unique prompts or disable prefix caching on both arms.
- Product steer: name first, then an official UI built on it → the wrapper seam in CONTRACT
  ("The wrapper seam") is the interface that UI builds on.
- A datatrove-home suggestion was raised → position held: lightweight primitive datatrove
  can *build on*, not built inside (interop via the storage contract + `completions/` convention).

## Deviations from staged defaults

- **Ceiling retired (2026-07-28, owner steer, truly final form)**: the number is gone
  (`tests/test_loc_ceiling.py` removed). The mission it stood for stays as the scope-creep
  rule in decision 1: new surface area requires a deliberate decision recorded here —
  correctness fixes, validation, and comments are never scope and never negotiate for
  space. History of the numeric form kept below for the record.
- LOC ceiling set to **800** (was ~700): the clean split carries module boundaries, the typed
  Request union, manifest sidecars, SignalSource seam, and the observable breaker the POC
  lacked. Enforced in CI; renegotiate downward after M3 trims, not by deleting docstrings.
- **Ceiling → NOTE (2026-07-28, owner steer, final form)**: the LOC check no longer fails CI.
  It *warns* past the noted figure so that new surface area gets a deliberate decision here —
  a feature-creep note, not a budget. Five renegotiations in one day showed the hard gate was
  taxing correctness work, which was never the point.
- **Renegotiated to 1200 (2026-07-28, 5th)**: codex round-5 robustness batch — declared-schema
  mode (the one genuinely new, deliberate surface: `schema=` on pump/ParquetSink, the reviewer's
  correct answer to schema stability), full process-group teardown verification, per-attempt
  retry-timeout capping, prompt controller-death detection, serializability probing. Applied
  the clarified intent below: correctness/robustness raises the ceiling, it doesn't golf.
- **Renegotiated to 1120 (2026-07-28, 4th)**: codex round-3/4 correctness fixes (fatal-abort
  path, schema pinning, retry-budget deadline, teardown/leak guards, input validation).
  **Intent clarified with this bump (owner steer)**: the ceiling is a *feature-creep
  tripwire* — it forces a documented decision before new surface area lands (a fourth
  building block, DAG authoring, live-cluster features). It is NOT a line budget:
  correctness fixes, input validation, and honest comments never need to golf against it —
  just raise it and record why. Trips should be read as "is this new scope?", not "find
  lines to delete".
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
- [x] Scrub `docs/history/` + this file for internal thread quotes/names (done 2026-07-28,
      codex-r3 branch; originals preserved in internal notes + git history — squash/rewrite
      history before flipping public).
- [x] why.md: second-hand rows replaced with directly-fetched sources (2026-07-29: Ray docs + eventual.ai blog, quotes verified).
- [ ] Daniel voice pass on README + why.md (writing-review).
- [ ] PyPI name registration; decide public-repo timing vs the embeddings showcase.
- [ ] **Naming (2026-07-29)**: library → `saturate` (6-agent debate + Codex second opinion:
      CONFIRM-with-cautions; pumpjack killed on live Pumpjack® data-platform trademark).
      Before the official layer publicly depends on it: trademark counsel pass (note the
      Newfangled/Eventide "Saturate" audio-plugin asserted mark — distant category, check
      anyway; that diligence belongs to the product team at absorption time). Docs rule from
      the review: frame the mechanism as congestion-aware backoff on endpoints YOU control,
      never load-generation; attribution line: "powered by the open-source davanstrien/saturate
      library". Layer proposal: "Inference Jobs" (NOT "Inference Pipelines" — transformers
      pipeline() + AWS SageMaker collision).

## Sources module (2026-07-28, owner-approved — the input half of "dataset in")

Owner steer: pumpjack is **transport-agnostic but IO-native to HF** — the output side
already was (hf:// sink, buckets); `sources.dataset_rows` completes the input side.
Evidence it belongs here and not a layer up: every consumer was rewriting the same loop
(spike drivers, console codegen, uv-scripts recipes), and the input-side failure modes are
this library's concerns (streaming-default after the EBDC disk death; the
resume-rematerialization wart's fix — id-first streaming — is only reachable inside the
source). Streaming default backed by hf.co/blog/streaming-datasets (persistent file cache,
bundled resolution, prefetching — local-SSD speed for the one-sequential-pass pattern);
`streaming=False` is a flag for small datasets. Accepts a repo id or an already-loaded
(Iterable)Dataset. **Ids are a trust contract, not an enforced policy** (owner steer):
the strategy — index (default: cheap, image-safe, stable per dataset+revision+order),
content (strict JSON-only hash, dedups), a key column, or any callable — is the caller's
assertion of uniqueness + resume-stability; the pump trusts what it is handed (CONTRACT
§2 posture). Index as default is a deliberate divergence from `normalize()`'s
content-hash default: it works on image/audio columns and costs nothing. Lives under the
[hf] extra (now `huggingface_hub>=1.20` + `datasets>=5.0`, pinned deliberately recent —
hf:// URIs landed in hub 1.19; datasets 5.0 carries the streaming overhaul), lazy-imported
so the core pays nothing.

**bucket_rows (2026-07-29, follow-up PR)**: raw objects by fsspec glob ->
(id, {path, bytes}); id = path relative to the glob's static prefix (natural,
stable). datasets/imagefolder was probed and REJECTED for this: measured 1.6x
slower on a real bucket and, decisively, streaming imagefolder drops the file
path — no stable id to hang resume on. `skip=` filters paths BEFORE reading
(id-first resume for buckets — partially answers #9: a re-run with
skip=existing_ids pays listing only); `prefetch` is a BOUNDED rolling
read-ahead window (unbounded prefetch = bucket-sized RAM; the vision-OOM
lesson), measured 3.5x over sequential on 20 real pages. **PDF-to-pages (and
any decode-to-N-rows transform) deliberately excluded**: one object becoming N
rows silently changes id semantics (resume/dedup keyed on the object would
lose pages; keyed on pages needs a paging scheme the source cannot invent) —
that is caller/task-layer territory; the driver-side pattern is a 5-line
wrapper over (id, bytes).

Boundary held: sources yield rows, **never construct content** — prompt building, batching
opinions, multi-stage pipelines stay above (datatrove is that layer). Deliberate new
surface, recorded per the decision-1 scope rule (the numeric ceiling was retired the same
day). Deferred to issues: `bucket_rows` (raw objects + parquet manifest, the production
image shape from the 2026-07-16 input-side notes) and id-first streaming for cheap resume.

## Console-gap findings (2026-07-28, from the UI POC — storage-only dashboard build)

The POC's job was to find where the CONTRACT makes a console awkward. It found five (G1–G5);
dispositions:

| gap | disposition |
|---|---|
| G5 exact counts need parquet · G2 world undiscoverable | **FIXED in pumpjack**: `completions/stats-{n}.json` (full Stats + rank/world, CONTRACT §5). Telemetry Σok stays approximate mid-run — documented |
| G1 no progress denominator · G4 items-per-row not in storage | **Wrapper layer** (`run.json` launch-spec sidecar: expected rows, world, items_per_row, written by the pipelines layer at launch — it's intent, not output; pumpjack stays a recorder of what happened). Spec belongs in the inference-pipelines design |
| G2 dead-on-arrival shard (live case) | run.json's `world` covers it at launch time; stats-{n}.json covers it at completion |
| G3 liveness (stalled vs slow) | **DEFERRED with tension named**: a heartbeat needs the CONTRACT to promise a telemetry flush cadence, which costs commits on dataset-repo sinks (free on buckets). Candidate: periodic telemetry flush (default ~60s, configurable), cadence advisory. Decide alongside the staged-output (bucket-hot) work — heartbeats and buckets want each other |
