# CONTRACT.md — saturate storage protocol (v1 — stable, additive changes only)

saturate writes batch-inference results as an **append-only set of parquet parts under one
output directory**, keyed by a stable `id`. This file is the promise that makes the output
more durable than the tool: anything that can glob a directory and read parquet can consume,
monitor, or resume a run **without importing saturate** — and the same promise is the seam a
product layer builds on.

What the contract buys, in four properties:

1. **The output is the interface.** Plain parquet + an `[id, error]` manifest sidecar; no
   lock-in, no live process to query.
2. **Resume is a property of the data.** No run state exists outside the output directory;
   an identical relaunch pays only what isn't durable, exactly-once per id.
3. **Coordination without a coordinator.** Append-only parts with collision-free names let
   K writers share one output; progress is readable from storage alone.
4. **Error rows, not lost rows.** Every admitted row leaves exactly one durable record —
   success or error, never a gap. Failures are queryable data, and healable.

Plus one discipline: **record, never guess** — every written value is an observed fact
(tokens from the API, latency we timed); dollar cost is a reader-side computation.

Stability: additive changes are allowed within v1 (new columns, sidecar files, keys);
renames, retypes and removals are not. The controller, transport and engine lifecycle are
implementation — free to change.

## 1. Output layout

```
out/
├── part-{ms}-{seq}-{uuid8}.parquet        # append-only data parts, zstd
├── _manifest/
│   └── ids-part-….parquet                 # [id, error] sidecar, 1:1 with its part
├── completions/
│   ├── shard-{rank}.done                  # advisory completion marker per shard
│   └── stats-{rank}.json                  # exact run summary (Stats + rank/world)
└── telemetry-shard{rank}-….jsonl          # per-tick controller trajectory
```

`out` is a single fsspec URI (local path, `hf://datasets/...`, `hf://buckets/...`).
Part names are globally unique across shards and re-runs without coordination and carry no
semantics beyond uniqueness (the `{seq}` counter additionally makes same-writer names sort
in write order). Parts are never mutated or deleted. Each flush writes the part **first**,
then its manifest sidecar — the manifest is an index; **the parts are the truth**.

### Columns

| column | type | meaning |
|---|---|---|
| `id` | string | globally-unique, stable row key (§2) |
| `error` | string/null | null on success; diagnostic string on failure |

Success rows: `{id, <parse columns…>, error: null}`. Error rows: `{id, error}` with user
columns null. Schema stability: once a column's type is first seen, **every subsequent part
carries the full schema** (absent values as typed nulls); a column whose values are all null
before its type is ever seen is absent from those early parts, so readers doing strict
unions should expect nullable user columns. For a fixed schema from part one, declare it —
`pump(schema=...)` — every part is cast to it, and a row carrying fields outside the
declared schema becomes an error row (§4; a direct sink append raises instead). Without a
declared schema, a row whose type conflicts with an already-pinned column likewise becomes
an error row under `pump()` (§8). The library itself writes no token/latency columns — they
appear when your `parse` emits them. Standard names when written: `prompt_tokens`,
`completion_tokens`, `latency_s` — copied from the response `usage` or your own
measurement, blank never guessed.

## 2. The id scheme

`id` must be globally unique across all shards writing to one output and stable across
re-runs. The default is a content hash of the row via `saturate.source.content_id` —
order-independent and re-shard-safe; a caller-designated key column or a global-index id is
equally conformant. Width caveat: the default hash is 16 hex chars (~64 bits) — collision
odds are negligible at the ~10M-row scale v1 targets but reach coin-flip around 5×10⁹
rows; beyond that, widen the hash or supply your own ids.

Fan-out: each of K shards selects its slice by strided assignment
(`(idx - skip) % world == rank`); ids derive from row content or global position, never a
per-shard counter.

## 3. Resume semantics

Resume is an **anti-join on `id`**: rows whose id already has a durable record are skipped.
Re-running the same command is always safe. Admission is also exactly-once per id *within*
a run (first wins; counted in `rows_deduped` — content-hash ids make identical input rows
dedupe for free).

The done-set is assembled **manifest-first, exactly**: read every manifest sidecar; scan any
part whose sidecar is missing (a crash between part- and manifest-write costs one part of
scanning, never duplicates); skip-and-re-pay any unreadable file (≤ one flush — `flush_every` rows — of rework,
never a hard error). A manifest entry without a surviving part still counts as done —
resume trusts manifests; recovering rows from an out-of-band-deleted part means a fresh
output dir, or deleting its orphaned sidecar.

One exception, in resume's favor: rows in flight when a run-fatal abort fires (the circuit
breaker gave up — the server is gone) produce **no** record and are re-admitted next run;
writing them as errors would make resume skip them forever.

## 4. Error rows and healing

- If `to_request`/`parse` raises or the endpoint fails unrecoverably, saturate writes
  `{id, error: "<diagnostic>"}` — never drops the row.
- **Reader rule**: a retried error can leave two records for one id. The record with
  `error IS NULL` wins (that's a **healed** row); otherwise the row is errored.
- `retry_errors=True` re-admits only ids whose sole record is an error; append-only, the
  reader rule resolves the pair.
- **Never mark schema-invalid as success**: an HTTP-200 whose body fails the workload's
  structural expectation is an error row.

## 5. Completion markers and stats

`completions/shard-{rank}.done` (payload: the fixed sentinel `done`) is written last, when
a shard's run returns normally. Markers are **advisory** (datatrove's convention — a
coordinator can watch the directory); resume never depends on them, they don't certify
row-level success, and they are sticky across re-runs of one output dir. Coordinators
needing current-run state should read `stats-{rank}.json` (the exact final Stats (§7) plus
`rank`/`world`; telemetry tick sums are approximate, stats are exact) — noting that stats
and telemetry writes are best-effort sidecars: failures are non-fatal, and sinks without
`write_stats` produce no stats file. The marker guarantees ordering, not sidecar presence.

## 6. Telemetry (frozen keys)

One `telemetry-…jsonl` per run; one object per controller tick (~2s):

| key | meaning |
|---|---|
| `t` | seconds since run start |
| `limit` | the window limit this tick |
| `inflight` | requests currently in flight |
| `waiting` / `running` | engine queue gauges (null when blind) |
| `bp` | backpressure events this tick (saturation-shaped 429/timeout/5xx) |
| `ok` | requests that succeeded this tick (failures surface in `bp`) |
| `input_bound` | true when the source, not the endpoint, is the bottleneck |
| `tok_s` | delivered tokens/sec this tick (the controller's plateau signal) |
| `kv` / `hits` / `preempts` | KV-cache utilization · prefix-cache hit rate · scheduler preemptions (when exposed) |
| `reason` | the controller's decision at the end of this tick: `hold`, `grow`, `probe`, `revert`, `cut:bp`, `cut:kv`, `cut:stall`, `cut:queue`, or `hold:<why>` (`hold` for `Fixed` and custom controllers) |
| `latency_s` | p50 admission-to-completion time of recent requests (null before the first completes) |

## 7. Agent contract (stdout / stderr)

In agent mode (env `CLAUDECODE`, `CODEX_SANDBOX`, or `AI_AGENT` — set `AI_AGENT=1` to force it;
detection is truthiness of any of the three, so there is no off-switch once one is set),
**stdout carries exactly one line**: the run's Stats JSON. Everything human-facing
(progress, advisor hints, the resume hint) goes to stderr. Stats keys (frozen, same additive
rule): `rows_total`, `rows_done_prior`, `rows_processed`, `rows_failed`, `rows_deduped`, `prompt_tokens`,
`completion_tokens`, `elapsed_s`, `final_limit`, `input_bound`, `breaker_opens`, `hints`,
`tokens_per_sec`, `cut_reasons` (window reductions counted by telemetry `reason`, e.g.
`{"cut:bp": 1, "cut:stall": 2}`; empty when the window never shrank).

## 8. Non-guarantees

No output ordering · no dedup beyond `id` · no schema migration (keep `parse` types stable
per output dir; under `pump()` a row whose type conflicts with the pinned schema becomes an
error row, a direct sink append raises at flush; a declared schema is the fully stable
option) · one request per row (rollouts/trajectories are a different primitive) ·
no dollar figures in the data.
