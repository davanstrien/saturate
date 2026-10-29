# Changelog

All notable changes to saturate are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - unreleased

### Breaking changes and migration

Read this section before you upgrade a running pipeline. Each item says what changed and
what to do.

- **Finish or restart in-flight 0.2 runs that use `dataset_rows` or `bucket_rows` ids.**
  The id formats changed, so a 0.3 resume of a 0.2 output does not recognise those rows and
  runs them again (#61). Content ids, column ids and your own `id_key`/`id_fn` ids do not
  change.
  - `dataset_rows` index ids were `{split}-{i:09d}`. They are now
    `{repo}[@{revision}]:{config}:{split}:{i:09d}`, for example
    `org/books:en:train:000000012`. The revision is in the id only when you pin one.
  - `bucket_rows` ids were relative to the glob's static prefix. They are now the full path
    that fsspec lists, without the scheme (`buckets/org/b/scans/p1.png`). To keep the 0.2
    ids, pass `root=` as the directory before the first glob character:

    ```python
    # 0.2 ids, under 0.3
    bucket_rows("hf://buckets/org/b/scans/**/*.png", root="hf://buckets/org/b/scans")
    ```

- **Sidecars moved under `__saturate/`** (#62). `datasets` and pyarrow skip paths that start
  with `__`, so `load_dataset(out)` and `pq.read_table(out)` now read the data parts alone.

  | 0.2 | 0.3 |
  |---|---|
  | `_manifest/ids-part-….parquet` | `__saturate/manifest/ids-part-….parquet` |
  | `completions/shard-{rank}.done`, `stats-{rank}.json` | `__saturate/completions/…` |
  | `telemetry-shard{rank}-….jsonl` | `__saturate/telemetry/telemetry-shard{rank}-….jsonl` |

  - Resume reads manifests from both places, so a 0.2 output still resumes. 0.3 writes only
    the new layout and leaves the old files where they are.
  - To load a 0.2 output with `load_dataset` or `pq.read_table`, move `_manifest/`,
    `completions/` and the `telemetry-*.jsonl` files to the paths in the table. Resume
    still works after the move.
  - A coordinator that watches `completions/` must watch `__saturate/completions/`.
  - A 0.2 output can also hold null-typed struct fields on disk (fixed for new parts in
    #59), so a whole-folder read of it can still fail. Use `read_output` for those.

- **`pump(shard=(k, n))` now selects input** (#61, fixes #19). In 0.2 it only labelled the
  output, so each shard ran every row unless you split the input yourself. Now each shard
  processes only rows with `shard_of(id, n) == k`, a hash of the id.
  - Give every Job the same `rows` and its own `shard`. You no longer need `shard_select`:

    ```python
    # 0.2
    pump(shard_select(stream(rows), rank, world), ..., shard=(rank, world))
    # 0.3
    pump(rows, ..., shard=(rank, world))
    ```

    The 0.2 form still works in 0.3, because `shard_select` now uses the same hash rule.
  - `shard_select` no longer does strided assignment, and its `skip` parameter is removed.
  - Input that you split by another rule (for example `split_dataset_by_node`) must not get
    `shard=`. Without `shard=`, give each Job its own output. With it, pump drops the rows
    that hash to other shards.
  - For large objects, also pass `shard=` to `dataset_rows` or `bucket_rows`. The source
    then skips other shards' rows before it reads or decodes them.
  - `shard` must be a pair of ints with `0 <= rank < world`. Other values raise.
  - The `limit` of `dataset_rows` and `bucket_rows` counts input rows before sharding and
    `skip`. Both sources now load or list when you call them, not at the first row.

- **The HTTP client is aiohttp, not httpx2** (#57). httpx2 is no longer a dependency.
  - `proxy=` on `pump()` and `AdaptiveClient` sets the proxy. `None` (the default) uses the
    environment's `HTTPS_PROXY`/`NO_PROXY` for the endpoint's host, and never proxies
    localhost. `False` uses no proxy. A URL uses that proxy.
  - Redirects are not followed. A 3xx is an error row, as before.
  - The `"pool"` timeout is now the whole wait for a connection, connect included. Each
    request has a total limit of pool + write + `read_timeout`.
  - `Engine(ready_accept=...)` receives a reply object with `status_code`, not an
    `httpx.Response`.

- **Human output goes through `logging`** (#71). All messages use loggers under
  `saturate`. If your program configures no handler, saturate prints the same `[pump] …`
  lines to stderr as before. When you add a handler (for example `logging.basicConfig()`),
  the lines go to your handlers instead. Set the level on `logging.getLogger("saturate")`.
  Code that parsed stderr should read the log records instead.

- **The default retry budget scales with `read_timeout`** (#48). In 0.2 each row had a
  fixed 300 s for all its attempts, so a slow request never got a retry. The default is now
  `max(300, 2 * read_timeout)`, which is 3600 s at the default `read_timeout`. A failing
  slow row can now take up to an hour. Set `pump(retry_budget_s=...)` to limit it.

- **`pump()` checks its settings before the run** (#76). It raises `ValueError` for values
  that cannot work: a negative `flush_every`, `read_timeout`, `retry_budget_s`,
  `prepare_workers` or `fail_fast`; an infinite `read_timeout` or phase timeout; an unknown
  `timeouts` phase; and any `signal_source` other than `"auto"` or `"none"`. In 0.2,
  `signal_source="metrics"` scraped anyway and a negative `fail_fast` meant 0. `Auto`
  checks its settings the same way (#70, #74).

- **A run stops when its first 32 rows all fail** (#38). It raises `FatalTransportError`
  and stores none of those rows, so a fixed re-run retries them. Set `fail_fast=0` to keep
  the 0.2 behaviour. The check is off with `retry_errors=True`.

- **A run checks that every row it read reached the sink** (#58). If the counts do not
  match, it raises the new `RowsLostError` and writes no completion marker.

- **Ids and outputs that were silently wrong now raise** (#43):
  - A `None` or empty id from `id_key` or `id_fn`. In 0.2 these all became `"None"`, and
    all but the first row were deduped.
  - `id_key` or `id_fn` with `(id, row)` input. In 0.2 they were ignored.
  - `FileSink("hf://…")`. In 0.2 it wrote to a local directory named `hf:`.

- **Embedder API** (#50, #54):
  - `through()` is an async context manager:

    ```python
    # 0.2
    results = through(client, rows, to_request, parse)
    stats = await drain(results, sink)
    # 0.3
    async with through(client, rows, to_request, parse) as results:
        stats = await drain(results, sink)
    ```

  - `AdaptiveLimiter.observe(ok=..., tokens=..., rate_limited=...)` is now
    `observe(Attempt(...))`. Report each attempt as it ends, and set `pressure` at most
    once per row.
  - `AdaptiveLimiter.limit` and `.inflight` are public properties. `limiter.window` is
    removed.

- **Telemetry and stats** (CONTRACT §6, §7):
  - The telemetry `reason` values `grow` and `probe` are replaced by `trial:up`,
    `trial:down` and `revert` (#65). New `hold:<why>` values appear, for example
    `hold:jitter`. The set of values is no longer frozen: treat an unknown value as a hold.
  - `bp` counts each row once, not each failed attempt (#37). The new `retries` key counts
    every re-send (#42).
  - `latency_s` is the p50 of successful attempts only, without earlier failed attempts,
    backoff or breaker waits (#50).
  - `bound_by` has a new value, `sink`, and `input_bound` is also true when the sink is
    behind (#54). The new `sink_s` key gives the wait.
  - `Stats` gains `rows_errored_prior` (#38), `retries` (#42) and the property
    `rows_unaccounted` (#58). No key is removed.

- **Dependency floors** (#36, #51, #57, #59): `anyio>=4.15.1` and `aiohttp>=3.14.4` are new
  direct dependencies. `pyarrow>=21` (was 15) and `fsspec>=2024.3` (was 2024.2).
  Python stays at 3.10 or later.

### Added

- `shard_of(id, world)`: the shard that owns an id (#61).
- `shard=` on `dataset_rows` and `bucket_rows`, and `root=` on `bucket_rows` (#61).
- `pump(retry_budget_s=, timeouts=)` and the same on `AdaptiveClient`. `timeouts` sets the
  `"connect"`, `"write"` and `"pool"` limits, 30 s each by default (#48).
- `pump(proxy=)` and `AdaptiveClient(proxy=)` (#57).
- `pump(fail_fast=)` and `through(fail_fast=)` (#38).
- `ParquetSink(flush_interval_s=, create=)` (#28, #48).
- `Auto(bp_frac=0.05)`: the share of a tick's rows under backpressure that cuts the
  window (#37). `Auto(seed=)` fixes the trial jitter (#70).
- `RowsLostError`, `Attempt`, the `Sink` protocol and `sink_hooks()` in the public API
  (#50, #54, #58).
- Stats: `rows_errored_prior`, `retries`, `rows_unaccounted`. Telemetry: `retries`,
  `sink_s` (#38, #42, #54, #58).
- A resume that skips error rows logs how many, and the run adds a `RETRY-ERRORS` hint
  (#38).
- The run logs how many rows it deduped and how to keep repeats (#43).
- The README shows how to start a new batch at the last window:
  `Auto(initial=stats.final_limit)` (#69).

### Changed

- **Auto finds the best window faster and holds it more steadily** (#41, #64, #65, #67,
  #70, #77, #78). What you see:
  - Auto starts at the engine's reported capacity on SGLang, llama.cpp, TEI and
    TensorRT-LLM. Elsewhere it starts at 64 (was 16). Your own `initial` still wins.
  - On TEI, Auto never goes above the engine's `max_concurrent_requests`.
  - Auto tries larger and smaller windows (×2, ×4, ×1.5, ×1.25, ×⅞), keeps a move only
    when throughput improves, and waits longer after each failed move.
  - Blind mode (no engine gauges, for example SGLang without `--enable-metrics`) no longer
    stops near a window of 20. On SGLang it reaches the best fixed window's throughput.
  - Auto remembers the window where a hosted API returned 429s and stays below it. After
    5 minutes it tries again, at most an eighth above.
  - Several clients on one endpoint no longer change their windows on the same tick.
    The known gap: blind clients that share one engine all grow to `max_limit`.
  - A full KV cache cuts the window only on a tick with a preemption, when the engine
    counts preemptions (vLLM, SGLang). Large models with a full cache and no preemptions
    are no longer held back.
  - One 429, timeout or 5xx no longer cuts the window. A cut needs `bp_frac` of the tick's
    rows under backpressure, so a few rows that always fail do not hold the window down.
- **Engine probe at startup** (#67, #78). Every run, `Fixed` included, sends a few GETs
  (2 s at most) to name the engine and read its capacity. `signal_source="none"` skips
  them. TensorRT-LLM is scraped at `/prometheus/metrics`. llama.cpp without `--metrics` is
  not scraped. A blind run logs why it is blind and the flag that turns gauges on.
- vLLM's prefix-cache hit rate comes from its counters, because vLLM V1 has no hit-rate
  gauge. This needs vLLM 0.12 or later (#79).
- **Flushing** (#28, #68). `flush_every=None` is the default. Every output writes a part at
  1000 rows, or when the oldest buffered row is 10 s old (60 s for remote outputs). In 0.2
  a local output wrote a part every 10 rows.
- Part and manifest writes retry a transient I/O error 3 times under the same name (#28).
- Local parts are written to a temporary name and then renamed, so a kill leaves no
  truncated part (#43).
- Any 2xx response with a JSON object body is a success (#47).
- `Engine` refuses a port that is already in use before it starts a server (#48).
- `existing_ids()` and `skip_done()` no longer create the output repo, bucket or directory.
  A missing output reads as empty (#48).
- A struct field seen only as `None` or `[]` is not written until it has a type. A new
  nested key extends the struct, like a new column (#59).
- `Retry-After` waits get up to 20% random extra, so rows told the same delay do not
  return together (#42).
- A row whose attempts fail in different ways keeps the first failure in its error text
  (#50).
- The pipeline runs on anyio task groups, with one writer thread for the sink (#51, #54).
  The aiohttp client uses 53–85% less CPU per row and doubles throughput at large fixed
  windows (#57).

### Fixed

- `read_output` keeps only successful records; a later error no longer replaces a success
  (#27).
- A rejected row no longer changes the pinned schema. Values Arrow cannot convert (for
  example `uuid.UUID`) become error rows and do not stop the run (#40, #59).
- Parts with optional nested fields can be read together; no part carries a null-typed
  field (#59).
- A parse function with a default second parameter gets the response, not the row (#38).
- 5xx and 429 error rows keep the server's message (#42, #44).
- A non-JSON 2xx body is retried and reported as such, not as a `JSONDecodeError` (#47).
- A flush error after a failed run no longer hides the run's own error (#47).
- `drain()` stops upstream requests when the sink fails (#43). On cancel, every finished
  row is written (#54).
- `prepare_ahead` no longer stops at a `None` item (#43).
- Cancelling a task no longer leaks a window slot (#51).
- A controller loop cancelled from outside fails admissions instead of freezing the
  window. The breaker gives up on time at `max_open_s` (#58).
- `load_dataset(out)` and `pq.read_table(out)` no longer fail on sidecar files (#62).
- Multipart uploads send the file name only, and send a file from its start (#57).
- A Prometheus value of `+Inf` or `NaN` no longer breaks the metrics scrape (#80).
