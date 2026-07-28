# CONTRACT.md — pumpjack storage protocol (v1, frozen 2026-07-27)

pumpjack writes batch-inference results as an **append-only set of parquet parts under one
output directory**, keyed by a stable `id`. Any process that can read parquet and glob a
directory can consume, resume, or fan out over that output without importing pumpjack.

Frozen here: the interface between a run and its durable output. Not frozen: the controller,
transport retry ladder, engine lifecycle — implementation, free to change.

Two governing invariants:

- **Own measurement, never own prices.** Every written value is an observed fact (a token
  count from the API, a latency we timed, an error string we caught) — never a guess.
  Dollar cost is a reader-side computation from measured tokens × a user-supplied price.
- **Error rows, not lost rows.** Every admitted row produces exactly one output record —
  success or error — never nothing. This is what makes resume sound. One exception, in
  resume's favor: rows in flight when a run-fatal abort fires (circuit breaker gave up —
  the server is gone) produce **no** record and are re-admitted on the next run; writing
  them as errors would make resume skip them forever.

## 1. Output layout

```
out/
├── part-{ms}-{uuid8}.parquet              # append-only data parts, zstd
├── _manifest/
│   └── ids-part-{ms}-{uuid8}.parquet      # [id, error] sidecar, 1:1 with its part
├── completions/
│   └── shard-{n}.done                     # per-shard completion markers
└── telemetry-shard{n}-{ts}-{uuid6}.jsonl  # per-run controller trajectory (v1 schema)
```

`out` is a single fsspec URI (local path, `hf://datasets/...`, `hf://buckets/...`).

- Part names are globally unique across shards and re-runs without coordination; they carry
  no semantics beyond uniqueness. Parts are never mutated or deleted.
- Each flush writes the data part **first**, then its manifest sidecar `ids-{partname}` with
  only the `[id, error]` columns. The manifest is an index; **the parts are the truth**.

### Required columns

| column | type | meaning |
|---|---|---|
| `id` | string | globally-unique, stable row key (§2) |
| `error` | string/null | null on success; diagnostic string on failure |

Success rows: `{id, <parse columns…>, error: null}`. Error rows: `{id, error}` (sparse — no
user columns; readers doing strict schema unions should expect nullable user columns).
Per-row token/latency columns, when present, are copied verbatim from the response `usage` /
measured by the client — blank, never guessed. Standard names when written:
`prompt_tokens`, `completion_tokens`, `latency_s`. Not required in v1.

## 2. The id scheme

`id` must be **globally unique** across all shards writing to one output and **stable across
re-runs**. The v1 default is a **content hash**: `sha1(canonical-JSON of the row's id-relevant
fields)[:16]`, via `pumpjack.source.content_id(row)` — order-independent, re-shard-safe. A
caller-designated key column or a global-index id is equally conformant (the contract requires
the two invariants, not the derivation). Scale caveat: 16 hex chars ≈ 64 bits — collision odds
are negligible at the ~10M-row scale v1 targets (~3×10⁻⁶) but reach coin-flip around 5×10⁹
rows; beyond that, widen the hash or supply your own ids.

Fan-out: K shards write to one output; each selects its slice by strided assignment
`keep(idx) = (idx - skip) % world == rank`. Ids derive from row content or global position —
never a per-shard counter.

## 3. Resume semantics

Resume is an **anti-join on `id`**: rows whose id already has a durable record are skipped.
Re-running the same command is always safe. Admission is exactly-once per id *within* a run
too: a duplicate id later in the same stream is skipped (first admission wins; counted in
`rows_deduped`) — with content-hash ids this makes identical input rows dedupe for free.

The done-set is assembled **manifest-first, exactly**:

1. Read every `_manifest/ids-*.parquet`.
2. Any `part-*.parquet` whose manifest sidecar is missing is scanned individually
   (`[id, error]` columns only) — so a crash between part-write and manifest-write never
   produces duplicates, and the fallback cost is one part, not the whole output.
3. A part or manifest file that cannot be read is skipped and its rows re-paid (cost:
   ≤ `flush_every` rows of rework — never a hard error, never blocks resume).

A manifest entry without a surviving part still counts as done: **resume trusts manifests**.
Consequence: if a part file is deleted or corrupted out-of-band, its rows are neither
reprocessed (the manifest says done) nor returned by readers (`read_output` skips
absent/unreadable parts, logging unreadable ones to stderr). To recover such rows, run
against a fresh output directory — or delete the orphaned `_manifest/ids-*` sidecars,
accepting up to `flush_every` rows of re-spend per part. (Parts-absent is also the oracle's
probe for manifest-based resume.)

## 4. Error rows and healing

- If `to_request`/`parse` raises or the endpoint fails unrecoverably, pumpjack writes
  `{id, error: "<diagnostic>"}` — never drops the row.
- **Reader rule**: healing can produce two records for one id (an old error, a new success).
  For each id, the record with `error IS NULL` wins; otherwise the row is errored.
- `retry_errors=True` re-admits only ids whose sole record is an error. Append-only: the old
  error record remains; the reader rule resolves it.
- **Never mark schema-invalid as success**: an HTTP-200 whose body fails the workload's
  structural expectation is an error row (`error IS NULL` means *structurally valid for this
  workload*, not *the call returned 200*).

## 5. Completion markers

`completions/shard-{n}.done` is written when a shard's run function returns normally
(`{n}` = rank, or `rank * chunks + c` for chunked runs). Markers are **advisory** — a
coordination convenience (datatrove's convention); resume correctness never depends on them.
A marker does not certify row-level success. v1 payload: the fixed sentinel `done`. The
marker is written **last** — after the stats and telemetry writes have been *attempted*.
Sidecar failures are non-fatal (logged to stderr), and short runs produce no telemetry
ticks, so a marker guarantees ordering, not sidecar presence.

**`completions/stats-{n}.json`** (v1 addition, 2026-07-28): written beside the marker — the
run's full Stats object plus `rank`/`world`. This is the console-facing exact summary: final
`rows_processed`/`rows_failed` (telemetry tick sums are approximate — the tail between the
last tick and completion goes uncounted), and shard geometry so a reader can distinguish
"all shards finished" from "a shard never started". Advisory like the marker; absent on
sinks that don't implement `write_stats`.

## 6. Telemetry v1 (frozen keys)

One `telemetry-shard{n}-{ts}-{uuid6}.jsonl` per run; one object per controller tick (~2s).
The uuid suffix keeps two runs of the same shard within one second from overwriting.

Core (always present): `t` float · `limit` int · `inflight` int · `waiting` int|null ·
`running` int|null · `bp` int · `ok` int · `input_bound` bool.

v1 additions (present when the signal exists; null/absent otherwise): `tok_s` float
(delivered tokens/sec this tick — the plateau signal) · `kv` float|null · `hits` float|null ·
`preempts` int|null.

Additions are allowed within v1; renames/retypes/removals are not.

## 7. Agent contract (stdout / stderr)

- **stdout**: agent mode (env `CLAUDECODE` / `CODEX_SANDBOX` / `AI_AGENT`) emits exactly one
  line — the run's Stats JSON. Human mode: stdout empty.
- **stderr**: all human-facing output (progress, advisor hints, the resume hint).

Stats v1 keys (frozen): `rows_total`, `rows_done_prior`, `rows_processed`, `rows_failed`,
`prompt_tokens`, `completion_tokens`, `elapsed_s`, `final_limit`, `input_bound`,
`breaker_opens`, `hints`, `tokens_per_sec`. Same addition/no-rename rule as telemetry.

**Idempotent re-run guarantee**: re-running the exact same command is always safe and resumes
(§3 enforces it); agent mode prints this hint to stderr.

## 8. Non-guarantees

No output ordering · no dedup beyond `id` · no cross-run schema migration (keep `parse`
stable per output dir; *within* a run parts are cast to one unified schema, so an
incompatible mid-run type change raises at flush instead of writing inconsistent parts) ·
one request per row (rollouts/trajectories are a different primitive) · no dollar figures
in the data.

## The wrapper seam

`pump()` is the facade a thin product layer (UI, "Inference Pipelines"-style interface,
datatrove stage) builds on: rows + two lambdas in, this storage contract out. Transports and
signal sources are pluggable behind it; the contract above is what stays fixed.
