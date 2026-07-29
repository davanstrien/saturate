# Design — current state (2026-07-28)

One page for reviewers and layer-builders. History and rationale live in
[decisions.md](history/decisions.md) (the log) and [history/](history/) (proposals);
receipts in [../spikes/RESULTS.md](../spikes/RESULTS.md); the "why not X" FAQ in
[why.md](why.md). This file describes what IS.

## The stack

```
console / product UI          forms + run views          (not this repo)
inference-pipelines layer     TaskSpec -> driver compiler (not this repo; seam below)
saturate                      THIS: engine + combinators + storage contract
Workloads / Jobs              image + command
```

The UI never sees saturate; it speaks TaskSpecs to a compiler layer that emits
pump() drivers. A run's progress is readable from storage alone (part counts,
markers, telemetry jsonl) — the CONTRACT doubles as the UI protocol.

## Modules (~1410 non-blank LOC; new surface requires a recorded decision, no numeric ceiling)

| module | role |
|---|---|
| `controller.py` | pure `decide(obs, limit)`: Fixed / Auto |
| `window.py` | asyncio admission gate with runtime-adjustable limit |
| `signals.py` | SignalSource: `/metrics` scrape (dual-prefix dialect table, GAIE semantics) or blind |
| `transport.py` | typed Request (json ⊕ multipart), retry ladder, circuit breaker |
| `core.py` | `AdaptiveLimiter` (slot/observe) → `AdaptiveClient` (+HTTP) → `through()` (Done stream) |
| `source.py` | `stream` (lazy normalize, content-hash ids), `skip_done` (anti-join + dedup), `shard_select` |
| `sources.py` | HF-native inputs (`[hf]` extra, lazy): `dataset_rows` (streaming-default Hub datasets), `bucket_rows` (raw objects by fsspec glob, id-first resume, bounded prefetch) |
| `sink.py` | Sink protocol: `ParquetSink` (full contract), `FileSink`; `drain`, `read_output` |
| `engine.py` | optional server lifecycle: boot templates, readiness gate, killpg |
| `telemetry.py` | tick records (frozen v1) + run-end advisor |
| `__init__.py` | `pump()` = composition of the above; Stats; agent-mode stdout contract |

## The controller, in one paragraph

Signal priority: **delivered throughput primary, engine gauges secondary
accelerator, blind AIMD floor**. Growth is ACK-clocked (never widen before the
first completion), slow-start doubles until a queue durably forms, a plateau in
delivered throughput blocks growth — but plateau-blocked growth **probes**
(+step on exponential-backoff cooldown, confirm-or-revert within a settle
window), because throughput readings lag real generations. Cuts: backpressure
(429-no-Retry-After, timeouts, 5xx) halves; high KV with unhealthy-or-absent
cache-hit signal halves; a standing queue above band steps down. A cut voids
in-flight probes. Input-bound (source slower than engine) freezes rather than
grows. Fixed(n) bypasses all of it.

## Seams (each with a named consumer)

- **Transport** — protocol; HTTP is the only impl today. In-process
  vLLM `AsyncLLM` / `sgl.Engine` planned for the high-rate short-output regime.
- **SignalSource** — `http-scrape | none`; in-process source arrives with the
  transport above. The controller only ever sees a plain Obs dict.
- **Sink** — one invariant: *an id returned by `existing_ids()` implies its
  record is durable; an id absent implies re-processing is safe.* ParquetSink
  carries the full CONTRACT; FileSink shows the protocol is real; bring your own.
- **AdaptiveLimiter** — `slot()/observe(outcome)`: the drop-in for a fixed
  semaphore in datatrove-shaped hosts.

## Invariants (tested)

Error rows, not lost rows · exactly-once admission per id (across runs via
manifest-first exact resume; within runs via dedup) · reserved columns `id`,
`error` always win over parse output · `error` column always string-typed ·
never mark schema-invalid as success · breaker can pause but never strand
(waiters always released; a dead server ends the run) · stdout in agent mode is
exactly one JSON line.

## Known bounds (documented, not hidden)

Resume/dedup id-sets are in-memory (~10M rows comfortable; shard-scoped
done-sets are the planned fix beyond). Binary response routes (TTS) are not
supported by the built-in transport (`r.json()` on 200) — use AdaptiveLimiter
with your own transport. vLLM pooling serializes per request: embeddings
scale by fan-out, not window. Commit-rate on `hf://datasets` sinks: prefer
bucket-hot + publish for high-frequency flushes.

## Provenance note

Docstrings currently carry design-history references ("the 10k-parity lesson");
they will be trimmed to constraints-only before any public release — the
stories live in [decisions.md](history/decisions.md) and RESULTS, not in the code.
