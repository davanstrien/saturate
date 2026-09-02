# Design

saturate exists for one loop: **data → model → nicer data** — run every row of a
dataset (or every file in a bucket) through a model and get a dataset back. This page
gives the high-level shape first, then how it is currently implemented. (Comparisons: [why.md](why.md) · on-disk
format: [../CONTRACT.md](../CONTRACT.md) · benchmark numbers:
[../spikes/RESULTS.md](../spikes/RESULTS.md).)

## The shape

Three pieces:

- **a row source** — any iterable of `(id, row)`; helpers exist for Hub datasets and
  buckets, but nothing requires them
- **an adaptive client** — keeps the endpoint busy without overloading it, by adjusting
  the number of in-flight requests at runtime
- **a storage contract** — the output is append-only parquet plus a manifest, so it is
  itself a dataset, and resume works by reading it back

```mermaid
flowchart LR
    D[(dataset / bucket /\nany iterable)] --> S[source:\nids + skip done]
    S --> W[adaptive\nwindow]
    W --> E[any OpenAI-compatible\nendpoint]
    E --> P[parse]
    P --> K[(parquet parts\n+ manifest)]
    K -. done ids, next run .-> S
```

The endpoint is anything that speaks the OpenAI HTTP API — vLLM, SGLang, llama.cpp,
TEI, an Inference Endpoint, a colleague's server. The library can boot a server for you
(`Engine`, a convenience) but never requires it: launch whatever you want, pass its URL.

Progress and resume live entirely in the output directory. A coordinator or UI can
watch a run by reading storage; it never needs to talk to the running process.

## The life of one row

```
source (any iterable / dataset_rows / bucket_rows)
  → id derivation (yours, or content-hash)          source.py
  → skip_done: anti-join against existing_ids()      source.py   ← resume happens HERE
  → admission: wait for a window slot                window.py
  → to_request(row) → HTTP POST (+retry ladder,      transport.py
      circuit breaker)
  → parse(row, response) → success or error record   core.py
  → buffered append → parquet part + manifest        sink.py
      sidecar (flush_every rows)
```

Failures become durable *error rows* (never gaps); a re-run skips done ids exactly and
can retry errored ones (`retry_errors=True`). A later success record for a previously
errored id is called healing; readers let the success win.

## How it is currently implemented

### Modules

| module | role |
|---|---|
| `controller.py` | pure `decide(obs, limit) -> new_limit`: `Fixed(n)` or `Auto` (below) |
| `window.py` | asyncio admission gate whose limit the controller adjusts at runtime |
| `signals.py` | `SignalSource`: engine `/metrics` scrape (vLLM/SGLang/llama.cpp/TRT-LLM dialects, TEI queue-depth only) or none |
| `transport.py` | typed `Request` (json ⊕ multipart), retry ladder, circuit breaker |
| `core.py` | `AdaptiveLimiter` (slot/observe) → `AdaptiveClient` (+HTTP) → `through()` (stream of `Done` results — id, row, output, error, token usage) |
| `source.py` | `stream` (lazy normalize, content-hash ids), `skip_done` (anti-join + dedup), `shard_select` |
| `sources.py` | HF-native inputs (`[hf]` extra, lazy): `dataset_rows` (streaming Hub datasets), `bucket_rows` (raw objects by fsspec glob, bounded prefetch) |
| `sink.py` | Sink protocol: `ParquetSink` (full contract), `FileSink`; `drain`, `read_output` |
| `engine.py` | optional server lifecycle: boot templates, readiness gate, process-group kill |
| `telemetry.py` | per-tick records (CONTRACT §6) + run-end advisor |
| `__init__.py` | `pump()` = the composition of the above; `Stats`; agent-mode stdout contract |

### The controller (`Auto`)

Terms used below:

- **tick** — the controller runs on a ~2-second cadence; each tick it sees an `Obs`
  snapshot and returns the new window limit.
- **gauges** — engine-reported queue metrics scraped from `/metrics`: `waiting`
  (queued requests), `kv` (KV-cache utilization, 0–1), `hits` (prefix-cache hit
  rate). No gauges = **blind mode** (successes, errors/timeouts and delivered throughput only).
- **band** — the target range for the engine's `waiting` queue:
  `[max(1, target_waiting//4), target_waiting*2]` (default target 8 → band [2, 16]). Small
  and positive: the engine always has work, never a runaway backlog.
- **delivered throughput** — tokens/sec actually completed this tick; the primary
  signal. Gauges speed decisions up; throughput decides them.

Per tick, first match wins; the winner names itself as the tick's `reason` (telemetry
§6) and `Stats.cut_reasons` counts the `cut:*` ones:

```mermaid
flowchart TD
    T[tick] --> CD{cooling down after\na reduction?}
    CD -- yes --> HOLD[hold]
    CD -- no --> BP{backpressure?\n429 / timeout / 5xx}
    BP -- yes --> CUT[halve]
    BP -- no --> KV{KV high AND\nhit rate low?}
    KV -- yes --> CUT
    KV -- no --> ST{window full, nothing\ncompleting past patience?}
    ST -- yes --> CUT
    ST -- no --> IB{source slower\nthan engine?}
    IB -- yes --> HOLD
    IB -- no --> HI{queue above band?}
    HI -- yes --> DOWN[−step]
    HI -- no --> LO{queue below band,\nwindow ≥80% used?}
    LO -- no --> HOLD
    LO -- yes --> ACK{limit completions\nsince last evaluation?}
    ACK -- no --> HOLD
    ACK -- yes --> G{throughput over that\nwindow improved?}
    G -- yes --> GROW[slow-start: double\nafter: +step]
    G -- no --> PROBE[hold; probe +step on a\nbackoff schedule, revert\nif no gain]
```

| reason | when | action |
|---|---|---|
| `hold:cooldown` | a reduction is draining: at least max(2 ticks, p50 latency) (3 ticks when unknown), extended while the oldest request in flight predates the reduction, capped at the stall patience | hold; the old window's completions are not evidence |
| `cut:bp` | saturation-shaped 429 / timeout / 5xx this tick | halve |
| `cut:kv` | KV ≥ 0.9 AND prefix-hit rate < 0.5 or absent | halve — high KV with healthy hits is cache reuse |
| `cut:stall` | window full and nothing completed for max(3 ticks, 3× p50 latency); once per episode, and never before the first completion | halve — a wedged engine reads as an empty queue because nothing new is admitted |
| `hold:input_bound` | the source, not the engine, starves admission | hold; starved ticks are not evidence |
| `cut:queue` | queue above band | −step, then the same cooldown as a halving |
| `hold` | window under 80% used, or queue in band | hold |
| `hold:ack` | fewer than `limit` completions since the last evaluation | hold — growth is clocked by evidence: one tick for cheap text, one window of completions for long generations |
| `grow` | headroom, and throughput averaged over the evidence window beat the best seen | slow-start: double (until a queue durably forms, a plateau, or a reduction); after: +step |
| `probe` / `hold:plateau` / `revert` | throughput flat | hold; probe +step on an exponential-backoff schedule (4→32 evaluations); keep it if the next window improves, else revert |
| `hold:floor` | a reduction could not go below `min_limit` | hold; arms nothing |

Every reduction scales the throughput baseline with it (throughput proportional to a smaller
window is a plateau, not growth). Blind mode (no gauges) creeps +1 per evidence window instead
of doubling. Latency is the p50 of successful attempts only — retries, backoff, breaker waits
and poison rows are excluded — and waits are integrated in seconds from the measured tick.

`Fixed(n)` bypasses all of it. Defaults: `Auto(target_waiting=8, initial=16,
max_limit=512, step=8)`; cap `max_limit` (~128) for vision workloads — in-flight
images live in host RAM.

### Seams

- **Transport** — a protocol; HTTP is the shipped implementation. An in-process
  vLLM/SGLang transport is the planned second one (high-rate, short-output regime).
  Proven bring-your-own: a binary-audio TTS transport was written against the
  documented surface without modifying the package (RESULTS.md, TTS probe).
- **SignalSource** — `/metrics` scrape or none; the controller only ever sees a plain
  `Obs` dict. The dialect table matches both metric-name spellings per engine because
  the k8s Gateway API Inference Extension (GAIE) protocol fixes gauge *semantics* but
  not their names (see [why.md §7](why.md)). Dialects may be partial: TEI publishes
  queue depth (`te_queue_size`) and nothing else — no running gauge, no KV — so the
  controller runs the queue-band branches and skips the KV ones.
- **Sink** — one invariant: *an id returned by `existing_ids()` implies its record is
  durable; an id absent implies re-processing is safe.* `ParquetSink` carries the full
  CONTRACT; `FileSink` (one file per row, filesystem as manifest) is the second
  implementation; bring your own.
- **AdaptiveLimiter** — `slot()/observe(outcome)`: a drop-in replacement for a fixed
  semaphore in hosts that keep their own transport and IO (datatrove-shaped stacks).

### Invariants (tested)

Error rows, not lost rows · exactly-once admission per id (across runs via
manifest-first exact resume; within runs via dedup) · reserved columns `id`, `error`
always win over parse output · `error` is always string-typed · schema-invalid
responses are never marked success · the breaker can pause but never strand (waiters
always released; a server that stays down ends the run with an error) · stdout in
agent mode is exactly one JSON line.

### Known bounds (documented, not hidden)

Resume/dedup id-sets are in-memory (~10M rows comfortable; shard-scoped done-sets are
the planned fix beyond). Binary response routes (TTS audio) are not supported by the
built-in transport — use `AdaptiveLimiter` with your own transport. vLLM
pooling-model serving (embeddings) processes one batch per request, so embeddings
scale by fan-out, not by window. Frequent flushes to `hf://datasets/…` sinks mean
frequent commits — prefer buckets while a fan-out run is hot, publish to a dataset
repo at the end.
