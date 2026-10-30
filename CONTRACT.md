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
implementation and can change.

Compatibility with 0.2 outputs (the one exception, made before 0.3.0): fan-out uses the
id-hash rule of §2, and the `dataset_rows` and `bucket_rows` id formats differ from 0.2's, so
0.3 does not resume an output that 0.2 wrote with those ids (its rows run again). Sidecars
live under `__saturate/` (§1); on resume, 0.3 also reads manifests from 0.2's `_manifest/`.

## 1. Output layout

```
out/
├── part-{ms}-{seq}-{uuid8}.parquet        # append-only data parts, zstd
└── __saturate/
    ├── manifest/
    │   └── ids-part-….parquet             # [id, error] sidecar, 1:1 with its part
    ├── completions/
    │   ├── shard-{rank}.done              # advisory completion marker per shard
    │   └── stats-{rank}.json              # exact run summary (Stats + rank/world)
    └── telemetry/
        └── telemetry-shard{rank}-….jsonl  # per-tick controller trajectory
```

The data parts are the only files outside `__saturate/`. `datasets` and pyarrow skip paths
that start with `__`, so their loaders see the parts alone. What they can load depends on
the schema:

- **Declared schema** (`schema=`): every part has the same columns, so `load_dataset`,
  `pq.read_table` and `pyarrow.dataset` load the whole output.
- **Inferred schema** (the default): a column can first appear in a later part (§1
  Columns). `load_dataset` then fails, and `pq.read_table` takes the first part's columns and
  drops the later ones without an error. `read_output` reads every column.

Either way the loaders read the parts raw: error rows are included, and a row that failed
and later succeeded appears twice. `read_output` applies the reader rule (§4). For an
`hf://datasets` output in a subdirectory, load the repo with `data_dir=`
(`load_dataset("org/repo", data_dir="data")`). An output written before 0.3 keeps its
`_manifest/`, `completions/` and telemetry at the root, also after a 0.3 resume; move them
under `__saturate/` before loading it with these tools.

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
unions should expect nullable user columns. Struct columns follow the same rule one level
down: a struct field seen only as null (or as empty lists) is left out of the part, and a
field first seen in a later part — typed for the first time, or a new key — is appended to
the struct from that part on. **No part ever carries a null-typed column or field**, and a
pinned type, top level or nested, never changes, so every part reads beside every other:
`pq.read_table(dir)` and `pyarrow.dataset` never fail on a type mismatch. They take the
schema of the first part they open, though, so columns and struct fields added by later
parts are dropped from that read; read every field with the parts' unified schema
(`ds.dataset(dir, schema=pa.unify_schemas([pq.read_schema(p) for p in parts]))`),
`read_output`, or DuckDB `union_by_name`. For a fixed schema from part one, declare it —
`pump(schema=...)` — every part is cast to it, and a row carrying fields outside the
declared schema (at any depth, inside lists too) becomes an error row (§4; a direct sink append raises instead). Without a
declared schema, a row whose value does not widen losslessly into an already-pinned column
type (§8) likewise becomes an error row — at flush, whichever way it was appended. The library itself writes no token/latency columns — they
appear when your `parse` emits them. Standard names when written: `prompt_tokens`,
`completion_tokens`, `latency_s` — copied from the response `usage` or your own
measurement, blank never guessed.

## 2. The id scheme

`id` must be globally unique across all shards writing to one output and stable across
re-runs. The default is a content hash of the row via `saturate.source.content_id` —
order-independent and re-shard-safe; a caller-designated key column or a global-index id is
equally conformant. The hash covers JSON values and raw bytes: each `bytes`, `bytearray`
or one-dimensional single-byte `memoryview` value, at any depth, enters as the SHA-1 of its
content, and the hashed text of such a row starts with `bytes-v1:` and the paths of its
bytes values, so a string that spells out a bytes hash never gives the id of the bytes.
Other values follow Python's `json.dumps` (a tuple hashes as a list, an int key as its
string), and an object `json.dumps` cannot encode is refused, because its text form is not
stable across runs. Width caveat: the default hash is 16 hex chars (~64 bits) — collision
odds are negligible at the ~10M-row scale v1 targets but reach coin-flip around 5×10⁹
rows; beyond that, widen the hash or supply your own ids.

An id names the data, not the query that found it. The built-in sources make ids this way:

- `dataset_rows` index ids are `{repo}[@{revision}]:{config}:{split}:{i:09d}`. The config
  is the one the loaded dataset reports, so leaving `config` out and naming the default
  config give the same ids. The revision is part of the id only when the caller pins one:
  a position names the same row only within one snapshot, so a new pin re-runs every row,
  while an unpinned run follows the branch and trusts that rows were not reordered. For an
  already-loaded dataset object the prefix is `{config}:{split}` from the arguments given,
  or `{split}`.
- `bucket_rows` path ids are the paths fsspec lists, without the scheme (the absolute path
  for a local file), or relative to a caller-given `root`. The glob plays no part, so sibling globs never
  share an id and widening a glob keeps every existing id.

Fan-out: K shards share one output. A row belongs to shard

```
shard_of(id, K) = int.from_bytes(sha1(id.encode("utf-8")).digest()[:8], "big") % K
```

and the shard of rank `r` processes only rows with `shard_of(id, K) == r`
(`pump(shard=(r, K))`). The rule reads only the id, so it is stable across re-runs,
input order, resume filtering and a growing input; the shards are disjoint and cover the
input; a source can apply it before it reads or decodes a row; and applying it twice
with the same `(r, K)` keeps the same rows. Changing K between runs moves rows between
shards, which is safe: resume is per id over the whole output (§3). Ids never come from a
per-shard counter. pump applies the rule to whatever input it is given, so input already
split by the same rule passes through whole; input split by any other rule does not, and
belongs in its own output without `shard`. A source given `shard` applies the same rule
before it fetches or decodes; given a different `(r, K)` from pump's, the run has nothing
to process. A source's `limit` counts input rows before sharding.

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
output dir, or deleting its orphaned sidecar. Sidecars are read concurrently (on a remote
store the cost is per-file round-trips, not parsing); the done-set is independent of read
order. Past 1000 sidecars the reader logs the count at INFO. A larger `flush_every` (fast runs) or a
longer `flush_interval_s` on `ParquetSink` (slow runs) writes fewer files.

While the circuit is open the breaker probes the run's real route about once a second for
up to `max_open_s` (600 s by default) with a one-token request; against a shared hosted API
those probes may count toward a requests-per-minute quota.

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

`__saturate/completions/shard-{rank}.done` (payload: the fixed sentinel `done`) is written last, when
a shard's run returns normally. Markers are **advisory** (datatrove's convention — a
coordinator can watch the directory); resume never depends on them, they don't certify
row-level success, and they are sticky across re-runs of one output dir. Coordinators
needing current-run state should read `stats-{rank}.json` (the exact final Stats (§7) plus
`rank`/`world`; telemetry tick sums are approximate, stats are exact) — noting that stats
and telemetry writes are best-effort sidecars: failures are non-fatal, and sinks without
`write_stats` produce no stats file. The marker guarantees ordering, not sidecar presence.

Before the marker, a run that ends normally checks that every row it read was skipped or
handed to the sink (counted as it is handed over):
`rows_total == rows_done_prior + rows_deduped + rows_processed + rows_failed`. If not, it raises
`RowsLostError` and writes no marker; the rows already written stay, and a re-run resumes the
rest. The check covers the pipeline, not the sink's own storage.

## 6. Telemetry (frozen keys)

One `__saturate/telemetry/telemetry-shard{rank}-….jsonl` per run and shard; one object per
controller tick (~2 s). The file is rewritten during the run and once more at exit, so a reader
can see a partial trajectory mid-run. Each rewrite sends the whole trajectory, so the interval
grows with the run: it is the larger of 30 ticks (about a minute) locally, 150 ticks (about
five minutes) on a remote store, and a quarter of the ticks so far.

| key | meaning |
|---|---|
| `t` | seconds since run start |
| `limit` | the window limit this tick |
| `inflight` | requests currently in flight |
| `waiting` / `running` | engine queue gauges (null when blind) |
| `bp` | backpressure events this tick (saturation-shaped 429/timeout/5xx) |
| `ok` | requests that succeeded this tick (failures surface in `bp`) |
| `retries` | retry attempts started this tick (`bp` counts each row once; this counts every re-send) |
| `input_bound` | true when the client side (the source iterator or `to_request`), not the endpoint, starved admission this tick: `bound_by` is `source`, `prep` or `sink` with the window under half used |
| `tok_s` | delivered tokens/sec this tick (the controller's plateau signal) |
| `kv` / `hits` / `preempts` | KV-cache utilization · prefix-cache hit rate · scheduler preemptions (when exposed) |
| `reason` | the controller's decision at the end of this tick: `hold`, `trial:up`, `trial:down`, `revert`, `cut:bp`, `cut:kv`, `cut:stall`, `cut:queue`, or `hold:<why>` (`hold` for `Fixed` and custom controllers). The set of values is not frozen: a reader should treat an unknown value as a hold |
| `latency_s` | p50 duration of recent successful attempts: the attempt alone, without earlier failed attempts, backoff or breaker waits (null before the first success) |
| `bound_by` | what limited this tick; the first that applies wins: (1) `source`, `prep` or `sink`, whichever of `source_s`, `prep_s / max(1, prep_workers)` and `sink_s` is largest (ties in that order), when it is at least a quarter of the tick and the window is under half used (`inflight < 0.5 * limit`); `sink` means new rows waited for room because the output is behind; (2) `prep` when `prep_workers` is 0 and `prep_s` is at least half the tick; (3) `loop` when `loop_lag_s` is at least half the tick (something else blocked the event loop, such as `parse`); (4) `engine` when the window was full or admission waits exceeded `source_s + prep_s`; null when nothing was conclusive (idle) |
| `source_s` / `prep_s` | seconds this tick spent in the source iterator / in `to_request` (`prep_s` is the raw sum over prepare workers) |
| `sink_s` | seconds new rows waited this tick for room in the backlog: the output was behind |
| `prep_n` / `prep_workers` | `to_request` calls measured this tick / threads running it (0 = on the event loop) |
| `loop_lag_s` | seconds of this tick beyond the tick period and the gauge scrape: time the event loop was blocked |

## 7. Agent contract (stdout / stderr)

In agent mode (env `CLAUDECODE`, `CODEX_SANDBOX`, or `AI_AGENT` — set `AI_AGENT=1` to force it;
detection is truthiness of any of the three, so there is no off-switch once one is set),
**stdout carries exactly one line**: the run's Stats JSON (unless the caller points a logging
handler at stdout). Everything human-facing
(progress, advisor hints, the resume hint) is logged under the `saturate` logger and, while the
caller has configured no logging handler, printed to stderr; a handler the caller points at
stdout is the caller's choice. Stats keys (frozen, same additive
rule): `rows_total`, `rows_done_prior`, `rows_errored_prior`, `rows_processed`, `rows_failed`, `rows_deduped`, `prompt_tokens`,
`completion_tokens`, `elapsed_s`, `final_limit`, `input_bound`, `breaker_opens`, `retries`, `hints`,
`tokens_per_sec`, `cut_reasons` (window reductions counted by telemetry `reason`, e.g.
`{"cut:bp": 1, "cut:stall": 2}`; empty when the window never shrank), `bound_by` (ticks per
telemetry `bound_by` verdict, e.g. `{"engine": 40, "prep": 3}`; inconclusive ticks are not
counted). `input_bound` is true when any tick's `input_bound` was true.

## 8. Non-guarantees

No output ordering · no dedup beyond `id` · no schema migration (keep `parse` types stable
per output dir; an output dir written before null-typed fields were left out of parts can
hold such fields on disk, and a whole-folder read of it can still fail; a pinned column type never moves and is read back from the newest readable
existing part on resume — a value of the same kind that fits is accepted and cast (an int
into a double or int32 column, `[1]` into a `list<double>` column, a struct missing a pinned
field), a struct field the pin lacks extends the pinned struct (§1), any other type change (a
float into an int column, an int into a string column or struct field, an int that
overflows) becomes an error row at flush, never a raise; a column or struct field seen only
as nulls or empty lists is not pinned until a typed value arrives, and only rows that are
written shape the pin — a row demoted to an error row never does;
K shards writing one output dir concurrently each pin from their own first rows, so fan-out
with a dynamic schema can leave parts that disagree — a declared schema is the fully stable
option and the right one for fan-out) · one request per row (rollouts/trajectories are a different primitive) ·
no dollar figures in the data.
