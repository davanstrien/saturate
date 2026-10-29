# Changelog

All notable changes to saturate are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - unreleased

### Breaking changes and migration

Read this section before you upgrade a running pipeline. Each item says what changed and
what to do.

- **`pump(shard=(k, n))` now selects input** (#61, fixes #19). In 0.2 it only labelled the
  output, so each shard ran every row unless you split the input yourself. Now each shard
  processes only rows with `shard_of(id, n) == k`, a hash of the id. This is the one change
  that can lose rows without an error.
  - **If you split the input yourself by another rule** (for example `split_dataset_by_node`)
    and passed `shard=` as a label, remove `shard=`: with it, pump drops the rows that hash
    to other shards and still writes a completion marker. Without `shard=`, give each Job
    its own output.
  - Give every Job the same `rows` and its own `shard`. You no longer need `shard_select`:

    ```python
    # 0.2
    pump(shard_select(stream(rows), rank, world), ..., shard=(rank, world))
    # 0.3
    pump(rows, ..., shard=(rank, world))
    ```

    The 0.2 form still works in 0.3, because `shard_select` now uses the same hash rule.
  - `shard_select` no longer does strided assignment, and its `skip` parameter is removed.
  - For large objects, also pass `shard=` to the source. `bucket_rows` then reads only its
    own objects. `dataset_rows` decodes only its own rows when it loads a repo id itself
    and uses index or column ids. Otherwise it decodes every row and then drops the other
    shards' rows.
  - `shard` must be a pair of ints with `0 <= rank < world`. Other values raise.
  - The `limit` of `dataset_rows` and `bucket_rows` counts input rows before sharding and
    `skip`. Both sources now load or list when you call them, not at the first row.

- **Finish or restart in-flight 0.2 runs that use `dataset_rows` or `bucket_rows` ids.**
  The id formats changed, so a 0.3 resume of a 0.2 output does not recognise those rows and
  runs them again (#61). Content ids, column ids and your own `id_key`/`id_fn` ids do not
  change.
  - `dataset_rows` index ids were `{split}-{i:09d}`. For a repo id they are now
    `{repo}[@{revision}]:{config}:{split}:{i:09d}`, for example
    `org/books:en:train:000000012`. For an already-loaded dataset object they are
    `{config}:{split}:{i:09d}`. The revision is in the id only when you pin one, and the
    config only when you pass one, in both forms.
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
  - A whole-folder read of a 0.2 output can still fail on its null-typed fields (see Fixed,
    #59). Use `read_output` for those.

- **The HTTP client is aiohttp, not httpx2** (#57). httpx2 is no longer a dependency.
  - `proxy=` on `pump()` and `AdaptiveClient` sets the proxy. `None` (the default) uses the
    environment's `HTTPS_PROXY`/`NO_PROXY` for the endpoint's host, and never proxies
    localhost. `False` uses no proxy. A URL uses that proxy.
  - Redirects are not followed. A 3xx is an error row, as before.
  - The `"pool"` timeout is now the whole wait for a connection, connect included. When
    pool, write and `read_timeout` all have a limit, a request also has a total limit of
    their sum.
  - `Engine(ready_accept=...)` receives a reply object with `status_code`, not an
    `httpx.Response`.

- **Progress messages go through `logging`** (#71). All messages use loggers under
  `saturate`. If your program configures no handler, saturate prints the same `[pump] …`
  lines to stderr as before. When you add a handler (for example `logging.basicConfig()`),
  the lines go to your handlers instead. Set the level on `logging.getLogger("saturate")`.
  Code that parsed stderr should read the log records instead.

- **The default retry budget scales with `read_timeout`** (#48). The budget is the time
  after which a row starts no new retry, and it caps each retry's timeouts. The first
  attempt always gets the full `read_timeout`, and time paused on an open breaker does not
  count. In 0.2 the budget was a fixed 300 s, so a slow request never got a retry. The
  default is now `max(300, 2 * read_timeout)`, which is 3600 s at the default
  `read_timeout`. Set `pump(retry_budget_s=...)` to change it.

- **`pump()` checks its settings before the run** (#76). It raises `ValueError` for values
  that cannot work: a negative `flush_every`, `read_timeout`, `retry_budget_s` or
  `prepare_workers`; an infinite `read_timeout` or phase timeout; an unknown `timeouts`
  phase; and any `signal_source` other than `"auto"` or `"none"`. In 0.2,
  `signal_source="metrics"` scraped anyway. `read_timeout=None` (no read limit) now needs
  an explicit `retry_budget_s`, because the default budget is derived from `read_timeout`. `Auto` checks its settings the same way
  (#70, #74).

- **A run stops when its first 32 completed rows all fail** (#38). It raises `FatalTransportError`
  and stores none of those rows, so a fixed re-run retries them. Set `fail_fast=0` to keep
  the 0.2 behaviour. The check is off with `retry_errors=True`.

- **A run checks that every row it read reached the sink** (#58). If the counts do not
  match, it raises the new `RowsLostError` and writes no completion marker.

- **Ids and outputs that were silently wrong now raise** (#43):
  - A `None` or empty id from `id_key` or `id_fn`. In 0.2 a `None` id became `"None"`,
    and all but the first such row were deduped.
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
  - `AdaptiveLimiter.limit` and `.inflight` are public properties. `limiter.window` and
    `limiter.events` are removed.
  - `through` moved to `saturate.pipeline` and `drain` to `saturate.writer`. Import them
    from `saturate`: `from saturate.core import through` no longer works.
  - `Stats` has new fields between existing ones. Construct it with keywords, not
    positional arguments.

- **Telemetry and stats** (CONTRACT §6, §7):
  - The telemetry `reason` values `grow` and `probe` are replaced by `trial:up`,
    `trial:down` and `revert` (#65). New `hold:<why>` values appear, for example
    `hold:jitter`. The set of values is no longer frozen: treat an unknown value as a hold.
  - `bp` counts each row once, not each failed attempt (#37). The new `retries` key counts
    every re-send (#42).
  - `bound_by` has a new value, `sink`, and `input_bound` is also true when the sink is
    behind (#54). The new `sink_s` key gives the wait.
  - `Stats` gains fields (see Added); no key is removed.

- **A custom sink runs on a writer thread** (#54). `append`, `flush` and the other sink
  calls run on one thread off the event loop, in order. A 0.2 sink that touches the event
  loop, or state shared with other threads, must be made thread-safe.

- **Dependency floors** (#36, #51, #57, #59): `anyio>=4.15.1` and `aiohttp>=3.14.4` are new
  direct dependencies. `pyarrow>=21` (was 15) and `fsspec>=2024.3` (was 2024.2).
  Python stays at 3.10 or later.

### Added

- `shard_of(row_id, world)`: the shard that owns an id (#61).
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
  - Without your own `initial`, Auto starts from the capacity that SGLang, llama.cpp,
    TEI or TensorRT-LLM reports: a little above it on engines that queue, at it on TEI.
    Elsewhere it starts at 64 (was 16).
  - On TEI, Auto never goes above the engine's `max_concurrent_requests`, even with your
    own `initial` or `max_limit`.
  - Auto tries larger and smaller windows (×2, ×4, ×1.5, ×1.25, ×⅞). It keeps a larger
    window only when throughput improves, and a smaller one when throughput holds. It
    waits longer after each move it reverts.
  - Blind mode (no engine gauges, for example SGLang without `--enable-metrics`) no longer
    stops near a window of 20.
  - Auto remembers the window where a hosted API returned 429s and stays below it. After
    5 minutes it tries again, at most an eighth above.
  - Random delays make clients on one endpoint less likely to try new windows on the
    same tick. The known gap: blind clients that share one engine all grow to `max_limit`.
  - A full KV cache cuts the window only on a tick with a preemption, when the engine
    counts preemptions (vLLM, SGLang). Large models with a full cache and no preemptions
    are no longer held back.
  - A backpressure cut needs at least `bp_frac` (5%) of the tick's outcomes to be
    backpressure, counted once per row. In 0.2 any single 429, timeout or 5xx cut the
    window. A few rows that always fail no longer hold the window down.
- **Engine probe at startup** (#67, #78). Every run, `Fixed` included, sends a few GETs
  with short timeouts to name the engine and read its capacity. `signal_source="none"` skips
  them. TensorRT-LLM is scraped at `/prometheus/metrics`. llama.cpp without `--metrics` is
  not scraped. A blind run logs why it is blind and the flag that turns gauges on.
- vLLM's prefix-cache hit rate comes from its counters, because vLLM V1 has no hit-rate
  gauge. This needs vLLM 0.12 or later (#79).
- **Flushing** (#28, #68). `flush_every=None` is the default. A `ParquetSink` output writes
  a part at 1000 rows, or when the oldest buffered row is 10 s old (60 s for remote
  outputs). In 0.2 a local output wrote a part every 10 rows.
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
  The aiohttp client uses less CPU per row, and is faster at large windows with short
  outputs (#57; numbers in `bench/README.md`).

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
- `rolling_map` (used by `prepare_ahead`) no longer stops at a `None` item (#43).
- Cancelling a task no longer leaks a window slot (#51).
- A controller loop cancelled from outside fails admissions instead of freezing the
  window. The breaker gives up on time at `max_open_s` (#58).
- `load_dataset(out)` and `pq.read_table(out)` no longer fail on sidecar files (#62).
- Multipart uploads send the file name only, and send a file from its start (#57).
- A Prometheus value of `+Inf` or `NaN` no longer breaks the metrics scrape (#80).
