# spikes/ — live-run receipts ledger (newest first; "pumpjack" = the pre-rename codename)

Drivers: `endpoints_qwen.py` (Inference Endpoints arms, current) · `verify.py` (CONTRACT
checker for any output dir, active utility) · `tei_local.py`/`tei_jobs.py` (TEI third
stack) · `embed_1job.py`/`embed_4job.py` (embeddings, single vs fan-out) · `tier1_*.py`
(first-night validation: OCR, sglang, synth, fan-out, big-OCR, embeddings — historical)
· `tier2_bucket_parity.py`/`tier2b_parity10k.py` (bare-httpx parity + bucket sink)
· `asyncllm_spike.py` (in-process AsyncLLM arms, decision 15 — historical).

# TEI gauge dialect — the queue-only engine (2026-07-29)

Two `cpu-upgrade` Jobs on `ghcr.io/huggingface/text-embeddings-inference:cpu-latest`
(bge-small-en-v1.5, ONNX CPU backend — it forces `max_batch_requests=8`).

## 1. Signal-surface probe: `/metrics` is on the MAIN port — job `6a69f7ad4497041dbfc387a4`
`GET :8080/metrics` → **200**, 394 lines. `/proc/net/tcp` shows **one** listening socket
(0x1F90 = 8080); 3000/8000/9000/9090/80 all refuse. The older spike note claiming TEI
metrics live on a port the Jobs proxy doesn't carry is **wrong for this image** — an
exposed TEI Job proxies its metrics along with its API.

Rendered names use the underscore spelling the metrics-crate convention predicted:
```
# TYPE te_queue_size gauge
te_queue_size 0
te_request_success{method="single"} 1   te_request_count{method="single"} 1
te_embed_count 1                        te_embed_success 1
te_batch_next_size_bucket{le="1"} 2     te_request_input_length_bucket{le="4"} 1
```
`te_queue_size` is the **only** gauge — everything else is a counter or a histogram.
No running/in-flight gauge, no KV (there is no KV cache to report). So TEI is the first
**partial** dialect: `waiting` only, `running`/`kv`/`hits` stay `None`.

Backpressure model differs too: `--max-concurrent-requests` (default 512) **rejects with
429** rather than queueing, which is what `CEILING_FLAG["tei"]` now says.

## 2. Live gauge mode end-to-end: PASS — job `6a69f87c4497041dbfc387ae`
Branch wheel (`signals/tei-dialect`) pulled from `davanstrien/pumpjack-spike`, installed
in-Job. *(Image gotcha: the TEI image ships **no python** — only `curl`. Bootstrap `uv`
from `astral.sh/uv/install.sh` and let it fetch a CPython.)* 50 batch-rows × 8 inline
texts against `localhost:8080/v1` `/embeddings`, `Auto(target_waiting=4, initial=2,
max_limit=8)`:

**50/50 ok, 0 failed, 2,036 tok/s in 2.75s**, and the telemetry tick is the receipt —
gauge mode, not blind:
```json
{"t": 2.0, "limit": 2, "inflight": 2, "waiting": 6, "running": null,
 "bp": 0, "ok": 40, "tok_s": 2216.4, "kv": null, "hits": null, "preempts": null}
```
`waiting: 6` with `running`/`kv`/`hits` null is exactly the queue-only shape. (Telemetry
tick records carry no `dialect` field — the dialect is asserted in `tests/test_signals.py`
against a verbatim excerpt of the probe body above.)

# v0.1.1 — fresh-output auto-create, proven by the shared snippet verbatim (2026-07-29)

The OCR example shared internally was run byte-identical as a Job, installing saturate
from PyPI via its PEP 723 header. First attempt on 0.1.0 crashed exactly where a first
user would: `existing_ids()` → RepositoryNotFoundError on the never-created output repo
(job 6a6a03ed). Fix: ParquetSink now ensures hf:// dataset repos / buckets exist
(private, exist_ok) at construction — released as 0.1.1. Re-run, unchanged snippet:
**20/20 pages, 0 failed, 1,507.9 tok/s in 37.2s, window settled 32** (LightOnOCR-2-1B,
a10g-small, job 6a6a06bc), output auto-created at davanstrien/saturate-ocr-demo. Also
the first end-to-end receipts of the public install path: `dependencies = ["saturate[hf]"]`
resolving from PyPI inside `hf jobs uv run` (earlier same-day: quickstart.py, job
6a6a00c0, which demonstrated cross-machine exact resume by skipping all 100 rows written
by a different machine's run).

# Inference Endpoints — the fourth serving arrangement (2026-07-29): warm · autoscale · cold start

Dedicated IE endpoint `saturate-ie-spike` (Qwen2.5-0.5B-Instruct on `vllm/vllm-openai:latest`
custom image, L4 x1, aws us-east-1, `min_replica=0`, scale-to-zero 15 min; IE proxies no
engine metrics → all arms BLIND). Driver: `endpoints_qwen.py`. Endpoint DELETED after
(verified absent from `hf endpoints list`); total spend ≈ $1.5.

## Warm — 1k dolly rows from the laptop: PASS
**1000/1000, 0 failed, 3,466 tok/s in 49.8s, window 8→256** discovered blind. The breaker
rode an early warm-up 5xx burst from the managed proxy (opened, probed, closed — every
retried row healed). **`wait_for_health` finding (documented, not patched)**: the trial
request returned 404 and readiness passed — vLLM 404s a trial payload whose model name it
doesn't serve, and the alive-only default (<500 = alive, by design) accepts that. Same
shape as bare `vllm serve`; on IE "ready" therefore means alive, not workload-routable —
use `ready_accept=` to gate on the workload.

## Autoscale — does the window re-discover capacity as replicas arrive? (max 2, pendingRequests>64)
- 6k rows, 1 replica: **5,950 ok (50 real dolly dupes deduped), 0 failed, 9,556 tok/s,
  window 104, 106s.** Scale-up TRIGGERED ~35s in (target→2) but replica boot is
  minutes-scale — the burst finished on one replica. Managed autoscaling reacts on
  replica-boot timescales; the window adapts on tick timescales.
- Same 6k with 2 replicas READY: **11,002 tok/s (+15%), window 134, 92.2s, 0 failed**
  (2 breaker opens riding replica warm-up 5xxs). Blind AIMD found extra capacity through
  the LB but nowhere near 2×: behind one URL the per-replica equilibria blur. One client
  per shard (the Tier-1 fan-out receipt: 4 independent controllers, 80/80/72/32) remains
  the scaling shape; an LB is a capacity smear, not a second engine.

## Cold start — pump straight at a scaledToZero endpoint, NO wait_for_health: PASS
Timeline (poll at 10s): T+0 first requests hit the sleeping endpoint → **the request itself
triggers the wake** (T+13s state=initializing) → breaker OPEN after 9 consecutive
failures, 1s probes → replica ready ≈T+93s → **probe catches it, breaker closes, admission
resumes** → done at T+111s: **93 ok + 7 durable error rows** (rows admitted mid-wake got the
proxy's `http 409 "workload is not stopped"` — a client error, not retried, so they landed
as durable error rows immediately), **0 lost**. Healing re-run (`retry_errors=True`):
`rows_done_prior: 93, rows_processed: 7, rows_failed: 0` in 4.9s — **100/100.** The ladder
+ breaker ride the managed wake with zero special-casing; scale-to-zero endpoints are
usable as-is (rows admitted mid-wake surface as durable 409 error rows on the first pass —
the error-rows-not-lost-rows contract working as designed — and heal on the re-run).

Caveats: single run per arm; 0.5B model, 150-token outputs; the LB observation is n=1 on
2 replicas.

# HEAD regression + first field firing of the reworked breaker (2026-07-28 PM)

Fresh 1k-row generation on the fully-fixed wheel (both Codex rounds; job 6a68ad82):
**1000/1000, 0 failed, window 8→128, 4.4k tok/s.** Bonus receipt: a post-readiness warm-up
burst of 5xx opened the breaker (**its first real firing since the rework**) — probe got a
404 (<500 = server parsing), closed immediately, every retried row healed. Working as
designed; also a live demonstration of why the breaker exists even after a readiness gate
passes. (The "144 consecutive failures" in the log = counter racing ahead of the first gate
observation across ~128 in-flight workers — cosmetic.) A prior run same day also verified
the pure-resume path on HEAD: 5,000 skips via Hub manifest in 19.8s, zero requests sent.

# Extensibility probe: TTS bucket-to-bucket (2026-07-28) — PASS, seams held

An agent playing END USER (package modification forbidden) shipped an unsupported
modality — Qwen3-TTS on sglang-omni, **binary audio out** — bucket(.txt) → bucket(.mp3):
**300/300 clips, filenames = ids, telemetry landed in the bucket too** (~16 job attempts,
mostly fighting the sgl-omni image bootstrap, not pumpjack).

Verdict on the three questions the probe was designed to answer:
1. **Every helper plugged into a documented seam**: a bucket `(id, row)` source; an
   `AudioSink` on the Sink protocol (binary + fsspec, resumable by the one-invariant
   contract); an `AudioClient` = **AdaptiveLimiter + its own transport** — because it found
   the known binary gap precisely (`transport.py` ends in `r.json()`; `/audio/speech`
   returns bytes) and took the designed escape hatch. `through()` accepted the replacement
   client because it only touches `client.post()` + `client.limiter` — the duck-type
   surface held exactly as documented.
2. **The docs were sufficient**: it cited README/CONTRACT sections in its comments, applied
   CONTRACT §4 unprompted (an HTTP-200 under 2KB of audio = error row, never success), and
   — the standout — **implemented the id-first-streaming design note straight from this
   file** (done-set consulted before reading text bodies, with skip_done still authoritative
   downstream). The banked design notes are apparently actionable specs.
3. It built its own kill-test (`TTS_DIE_AFTER`) to verify mp3 resume under the Sink
   invariant.

Library gaps confirmed (already on the roadmap, now user-validated): binary responses in
the built-in transport; a blessed bytes-capable FileSink variant would have saved ~30 lines.

# Workload campaign day 2 (2026-07-28) — OCR + transcription lanes

## OCR at scale, second model (OvisOCR2/vLLM, 2,000 MOH pages)
Completed across THREE jobs via resume (a cancelled first job's durable rows + 1,150 +
700; jobs 6a687190, 6a68848a): **2,000/2,000, zero failures**, ~0.745 pages/s per a10g,
3.3–4.9k tok/s, window settled at 32 in both runs (consistent vision equilibrium).
Another unplanned cross-job resume proof in the wild.

## Transcription — the multipart route's first live contact: PASS ×2
`make_multipart_request("/audio/transcriptions", ...)` unmodified against a real server:
**300 clips then 2,403 clips, zero failures** (jobs 6a6866f2, 6a68847f), windows 56/40.
Contract nicety proven: transcription responses carry no `usage` → `tokens_per_sec: 0.0`
(blank-never-guessed), controller ran on gauges+latency without the throughput signal.
Every Request arm (json + multipart) and every route family is now field-tested.

# Tier 2 — bucket sink + bare-httpx parity (2026-07-28, the M1 bar)

## Bucket sink: VALIDATED both directions (jobs 6a686f15)
1,000 rows → `hf://buckets/davanstrien/pumpjack-scratch/tier2-synth`; identical re-run read
the manifest FROM THE BUCKET and skipped all 1,000 in 4.5s (`rows_done_prior: 1000`). The
staged-output pattern (pump→bucket hot, publish→dataset later) needs zero write-side code.

## Parity vs bare-httpx: a three-act story ending at **1.209** (bar ≥0.95)
Same model/rows/settings, prefix caching off, bare = expert-tuned fixed-64 httpx client.

| act | run | ratio | what it meant |
|---|---|---|---|
| 1 | 2k rows (6a686f15) | 0.515 | unfair test: Auto's discovery ramp dominates a 44s run |
| 2 | 10k rows (6a68704a) | 0.566 | **real bug**: controller parked at 32 — the plateau gate's `grew` reading is stale on real engines (generations lag the 2s tick) and `best_tok` ratchets it stuck |
| 3 | 10k rows, probe-and-revert (6a6881b4) | **1.209** | pump 18,434 tok/s in 91s vs bare 15,243 in 110s; **final_limit 272** — the expert's 64 was also a wrong guess; discovery beat it by 21% |

Fix: Vegas-style probe-and-revert (plateau-blocked growth probes +step on exponential-backoff
cooldown; confirms on improvement, reverts after a 3-tick settle window). Two lag-simulating
tests added; oracle 9/9 throughout. Caveats: single run per arm; 272 in-flight trades
per-request latency for throughput (correct for batch); 150-token outputs on a 0.5B.

# Tier 1 round 2 (2026-07-27 late — fan-out, embeddings, TEI, the OOM lesson)

## Fan-out — 4 Jobs, 4 engines, ONE output: FLAWLESS
4×1000 dolly rows, strided assignment. Verified: 80 parts, **4,000 records / 4,000 unique /
0 dupes**, markers shard-0..3.done, 4 telemetry files. Each shard's controller settled
independently (final windows 80/80/72/32 over ~6–8 min runs — settled values, not claimed as
equilibria; same caveat as the shape-matrix short arms) — per-shard adaptivity with zero
coordination is the fan-out-to-storage argument, demonstrated.

## Embeddings (vLLM pooling) — zero library changes: PASS
`route="/embeddings"` + micro-batch-as-row (500 rows × 64 texts = 32k fineweb-edu texts).
500/500, 0 failed, 7.85M tokens at 33.5k tok/s; 1024-dim array columns landed in parquet on
hf://. **Controller shrank to the serving pattern**: vLLM pooling served one 64-seq batch at a
time → window settled at final_limit=2 *requests* — which is ~128 texts in flight, so mind the
units: neither the window value nor the advisor's `--max-num-seqs 2` suggestion is
items-normalized (decision 12 note: normalize BOTH the advisor arithmetic and the narrative by
items-per-request for micro-batch workloads). The advisor fired its first real hint here;
correct diagnosis, request-unit arithmetic. Flag churn: `--task embed` is
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

## Soak (5k MOH pages, ACK-clocked, max_limit=128): PASS — including an unplanned resume
v2 ran healthy for ~68 min (no OOM — the ACK-clock + RAM cap held) and was SIGTERM'd
platform-side at 4,450/5,000 durable. Identical relaunch: `rows_done_prior: 4450,
processed: 550, failed: 0`, window settled 58. **Verified: 100 parts, 5,000 records,
5,000 unique, 0 dupes** — a second cross-job resume proof at 10× Job A's scale, unplanned.

**Design note found by the resume**: the tail run took ~25 min for 550 pages because the
source generator downloads + base64-encodes every image BEFORE the anti-join skips it —
resume re-pays source-side materialization for already-done rows. Fix direction (post-v1):
id-first streaming (derive id before expensive payload work, or push the done-set down into
the source). Banked for the source module.

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

# Sources validation — dataset_rows across three workload types (2026-07-28/29, feat/sources)

All four jobs on a10g-small, wheel built from this branch (staged at
pumpjack-spike/feat-sources/). Drivers: src_embed.py, src_gen.py,
lighton_ocr2_pumpjack.py (staged alongside; the OCR port is the decision-11
pilot, private — no public PR).

## 1. Embeddings (TEI bge-m3) + resume — dataset_rows feeding caller-side batching

- fineweb-edu train, columns=["text"], limit=6400 → 64-text batches, batch id =
  first member's index id (deterministic across runs).
- Run 1 (job 6a691deb15e81eca66a8d664): **100/100 batch-rows (6,400 texts), 0 failed,
  57,725 tok/s, 30.0s inference**, window settled at 7.
- Run 2, same command, separate Job (6a691ee4a9f4e0ab00b2bf04): **rows_done_prior=100,
  rows_processed=0, 6.8s** — fresh streaming pass, anti-join skipped everything.
  dataset_rows index ids (and batch ids derived from them) are stable across Jobs: the
  id-stability caveat holds in practice, not just in the docstring.

## 2. Generation (vLLM Qwen/Qwen3.5-4B) — prompt-in-column, ids="content"

- fka/awesome-chatgpt-prompts, row-per-request chat, content ids.
- Job 6a691deca9f4e0ab00b2befa: the dataset turned out to hold 2,066 prompts (not the
  ~200 expected — full-size receipt for free): **2,052 processed, 10 durable error
  rows, rows_deduped=4** — content ids caught 4 real duplicate prompts in the wild
  dataset, exactly the strategy's pitch. 1.08M prompt + 0.82M completion tokens,
  702.8s, 2,699 tok/s, window settled at 51, 0 breaker opens.

## 3. OCR recipe port (LightOnOCR-2-1B) — the "how much does it help" A/B

- uv-scripts lighton-ocr2-server.py vs pumpjack port, same 100 pages of
  davanstrien/bhl-impact-groundtruth (ids = gt_page_id column: natural key on
  an image dataset, where content-hash refuses by design), same model flags,
  sampling, image (vllm/vllm-openai:v0.22.1), flavor.
- Static: 651 → 88 lines (574 → 68 non-blank), −87%. Dropped: CLI args, card
  generation, inference_info column (publish concerns, live elsewhere).
  Gained: adaptive window (vs --concurrency 32), exact resume, durable per-row
  error records (vs "[OCR ERROR]" strings + "results are lost" on failed
  push), incremental parquet (vs all-in-RAM + end-of-run push), telemetry.
- **Round 1 (accidental error-path A/B)**: the first-100 slice of the input turned out
  to carry 62 rows with `image=None` — junk for throughput, gold for error semantics:
  - Old recipe (job 6a691df215e81eca66a8d668): 62 instant failures **pushed into the
    output dataset as `"[OCR ERROR]"` strings** mixed with real results; 1.96 img/s
    reported incl. the instant errors.
  - Port (job 6a691ded15e81eca66a8d666): **38 processed + 62 durable error rows**
    (to_request exceptions recorded per-id in `_manifest`, separable/retryable), run
    completed normally, stats honest (rows_failed=62). Exactly the CONTRACT behavior.
  - Neither side's throughput number is meaningful at n=38-real-pages.
- **Round 2 (clean A/B)**: 100 non-null pages → `davanstrien/sources-ab-pages`
  (built by CPU job 6a6921baa9f4e0ab00b2bf3a), each arm run TWICE:
  - Old recipe (jobs 6a69220615e81eca66a8d69f, 6a692366a9f4e0ab00b2bf5d): **0.80 img/s
    both runs** (its own inference-only metric — dataset fully materialized BEFORE the
    timer; total processing time 4.4 min).
  - Port (jobs 6a692202a9f4e0ab00b2bf48, 6a692362a9f4e0ab00b2bf5b): **138.6s / 138.1s
    for the whole pump = 0.72 img/s INCLUDING streaming the images from the Hub inline**
    (input_bound=false, window at the 48 cap, 100/100, 0 failed, ~2,080 tok/s both runs).
  - Honest read: pure-inference throughput is a ~10% edge to fixed conc-32 when input
    cost is excluded from its timer; end-to-end productive time favors the port
    (~138s vs ~264s) because streaming overlaps input with inference instead of paying
    materialization up front. Neither gap is the story — the port's case is the
    capability delta (resume, durable separable errors, incremental output, −87% code)
    at comparable throughput.
  - **Correctness (migration faithful)**: outputs joined on gt_page_id, 100/100 pairs,
    0 errors either side, length ratio mean 0.998 / median 1.000, difflib similarity
    mean 0.986 / median 1.000 (most pages byte-identical at temp 0.2); one page at 0.66
    = ordinary sampling divergence.

## 4. bucket_rows — raw-object input (feat/bucket-sources, 2026-07-29)

- Probe (20 real page images, 30MB, hf://buckets, home connection):
  datasets/imagefolder 24.6s vs fsspec-direct 15.6s — and imagefolder drops
  the path (no stable id), which decided the design more than the 1.6x.
- Library receipts, same 20 images: prefetch=8 read 30MB in 3.9s vs 13.6s
  sequential (3.5x, bounded window); skip-all re-list 0 rows in ~0s (id-first
  resume: listing only, zero transfer).
- OCR-from-bucket job (glob -> LightOnOCR-2, a10g-small): job
  6a69296aa9f4e0ab00b2bfe3 — **20/20 pages, 0 failed, 33.8s pump**, contract
  output with path-ids.
- Re-run, same command (job 6a692b95a9f4e0ab00b2c013): driver printed
  `skip-before-read: 20 already durable`, then **rows_total=0, elapsed 0.61s** —
  the source filtered by id before reading a single byte. Id-first resume for
  buckets (issue #9's mechanism), end-to-end on Jobs.

## 5. Scale A/B — 1,000 pages (ramp-amortization test, 2026-07-29)

Hypothesis (Daniel): the adaptive window's slow-start handicap amortizes over
longer runs. Confirmed:

| metric | 100 pages | 1,000 pages |
|---|---|---|
| old recipe, inference-only (input pre-materialized) | 0.80 img/s | 0.95 img/s (job 6a69985d) |
| port, whole pump INCL. inline streaming | 0.72 img/s | **0.955 img/s** (job 6a699859) |

At 10x scale the port matches fixed conc-32 on the old recipe's own metric
while carrying its input cost inside the number, and finishes faster
wall-to-wall (17.4 min pump vs 20.1 min processing). 999/1000 + 1 duplicate
gt_page_id caught by id admission; 0 failures both arms. Port throughput
3,438 p/h on a10g-small.

Context vs ocrscout's tuned claims (same GPU class, A10G; their numbers from
the published benchmarks Sebastian's EBDC budget uses): GLM-OCR 2,432 p/h,
PaddleOCR-VL-1.5 3,789 p/h — our untuned full-page LightOnOCR-2 lands at
3,438 p/h between them (different models: labeled context, not a claim).
Same-model GLM-OCR run: job 6a699d86, pending.

## 6. Same-model throughput vs ocrscout's tuned claims (2026-07-29)

Question (Daniel): ocrscout claims heavily-tuned per-model/GPU inference — how
far off are we? Same GPU class throughout (their A10G = Jobs a10g-small, the
mapping their own EBDC budget table uses).

| model | mode | p/h | source |
|---|---|---|---|
| GLM-OCR, ours | full-page, untuned, incl. inline streaming | **4,776** | job 6a69a31615e81eca66a8da75: 999/1000, 0 failed, 753s, 3,562 tok/s |
| GLM-OCR, ocrscout | layout pipeline, tuned | 2,432 | their published benchmarks (EBDC table) |
| PaddleOCR-VL-1.5, ocrscout | tuned | 3,789 | same |
| LightOnOCR-2, ours | full-page | 3,438 | job 6a699859 (section 5) |

Same-model, same-GPU: **1.96x their tuned GLM number**, with input streaming
inside our measurement. Caveats: their GLM path is the layout pipeline (region
detect + region OCR — more work/page, but it IS their tuned path for GLM);
corpora both BHL-family scans, not the identical sample; their vLLM version
unknown. Window again pinned at the 48 cap (final_limit=48) — likely further
headroom; an agent reading stats would raise the cap.

Getting there took two instructive failures, both diagnosed in one read:
(a) max_tokens=8192 + --max-model-len 8192 -> 400 on every request, 999
durable error rows carrying the exact server message (job 6a699d86);
(b) dropping the cap entirely -> GLM declares 131k native context, KV profile
killed the boot on 24GB (job 6a69a16c). The fix (explicit 16384) is the
argument for recipe-level [tool.serving] starting values: the invariant needs
both bounds — above input+max_tokens, below what the card's KV affords.
