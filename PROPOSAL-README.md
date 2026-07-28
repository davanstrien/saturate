# pumpjack

Batch inference for datasets: rows in, any OpenAI-compatible endpoint, resumable parquet out.

You point it at an endpoint — a vLLM server you just started, a SGLang Job, TEI, a hosted
API — and give it two functions: one that turns a row into a request, one that turns a
response into output columns. It handles everything between: how many requests to keep in
flight (adaptively — you never pick a concurrency number), retries, crash-safe output, and
resume. Killing it at any point is fine. Re-running the same command is always safe.

```bash
uv pip install pumpjack   # not yet on PyPI — pip install git+https://github.com/davanstrien/pumpjack
```

## Quickstart: one model, one Job, one dataset

The most common shape — boot the model and pump a dataset through it, all in one process
(e.g. a single GPU Job on HF Jobs):

```python
from pumpjack import pump, Engine

with Engine("lightonai/LightOnOCR-2-1B", engine="vllm") as endpoint:   # vllm | sglang | llamacpp
    stats = pump(
        rows,                              # any iterable of dicts — streams, never materializes
        to_request=lambda row: {...},      # row -> OpenAI-style request body
        parse=lambda row, resp: {...},     # response -> your output columns
        endpoint=endpoint,
        output="hf://datasets/you/results/data",   # or a local path, or hf://buckets/...
    )
print(stats.rows_processed, stats.tokens_per_sec)
```

Already have an endpoint (a colleague's server, an exposed Job, a hosted API)? Skip
`Engine` and pass its URL as `endpoint=` — everything else is identical.

Notes on what you didn't have to do:

- **No concurrency number.** The in-flight window tunes itself: it backs off when the
  server shows pressure (errors, timeouts, a growing queue) and creeps up while delivered
  throughput keeps improving — the same idea TCP uses for network congestion. When the
  engine's `/metrics` gauges are reachable (vLLM, SGLang, llama.cpp, TEI) they sharpen the
  decisions; against opaque endpoints it works from latency and errors alone. It grows only
  after the first completion arrives, and it freezes (and tells you) when your *source* is
  the bottleneck rather than the server.
  Want control anyway? `window=Fixed(64)` pins it; `window=Auto(initial=32, max_limit=128)`
  sets the starting point and ceiling (do cap it for vision workloads — in-flight images
  live in RAM).
- **Ids are yours if you want them.** Pass `(id, row)` tuples, or `id_key="my_column"`, or
  an `id_fn=` callable. Default: a stable content-hash of the row — identical input rows
  then dedupe for free.
- **`kill -9` it, re-run the same command.** Output is append-only parquet with a manifest
  sidecar; resume is an exact anti-join on id — it re-pays at most one flush buffer, never
  duplicates a row. This holds across separate Jobs writing at different times.

## Task wrappers live above this library

`pump()` is deliberately the highest level here: two lambdas, no task opinions. Nicer
task-shaped wrappers ("OCR this dataset with model X") belong a layer up — recipe scripts
and product surfaces build them out of `pump()`; this library stays small underneath them.
```

(`Engine` boots the server in its own process group, health-gates it properly — health
checks lie during warm-up, so readiness requires a real completion — and kills it on exit.)

## The composable layer (for building on top — datatrove-shaped stacks, power users)

`pump()` is a composition of four small stages, and the middle one — `through()` — is the
real product: **a stream of completed results**. Parquet is just the default place that
stream lands.

```python
from pumpjack import AdaptiveClient, Auto, stream, skip_done, through, drain

rows = stream(load_dataset("...", streaming=True))     # (id, row) pairs, lazy
rows = skip_done(rows, sink)                            # exact resume filter

async with AdaptiveClient(endpoint, window=Auto()) as client:
    results = through(client, rows, to_request, parse)  # unordered async map, adaptive
    stats = await drain(results, sink)                  # parquet + manifest + markers

# chaining is just more piping — e.g. OCR then judge:
#   pages -> through(ocr_client, ...) -> drain(stage1_out)
#   stream_output(stage1_out) -> through(judge_client, ...) -> drain(stage2_out)
```

Two rules keep this honest:

1. **Persistence is the composition boundary.** Each `drain()` gives you crash-safety and
   resume for that stage. You *can* chain `through()` calls in memory; a crash then re-pays
   both stages. Pipe through storage when the work is expensive.
2. **This is itertools, not a pipeline framework.** No scheduler, no DAG, no `.compute()`.
   If you want orchestration, datatrove and friends sit naturally on top.

### Bring your own IO — resumably

Resume isn't tied to parquet; it's a tiny contract any sink can satisfy: *an id in
`existing_ids()` means its record is durable; an id absent means re-processing it is safe.*
Two sinks ship with that contract:

- **`ParquetSink`** (what `output="..."` gives you) — the full storage contract: exact
  resume, durable error rows, healing.
- **`FileSink(outdir, ext=".md", key="markdown")`** — one file per row, named by id. The
  filesystem is the manifest, overwrites are idempotent, so OCR→markdown files or
  audio→transcripts get exact resume too. (Failed rows leave no file, so they simply retry
  next run — there's no error record. That's the trade.)

```python
stats = pump(pages, to_request, parse, endpoint,
             output=FileSink("ocr-out/", ext=".md", key="markdown"))
```

Or skip sinks entirely and consume the stream yourself — no resume, full freedom:

```python
async for done in through(client, rows, to_request, parse):
    my_database.insert(done.id, done.out)
```

Embedding just the adaptive part in your own stack (your IO, your loop, your transport —
this is the datatrove-shaped seam):

```python
limiter = AdaptiveLimiter(window=Auto())          # a drop-in for your fixed semaphore
async with limiter.slot():
    result = await your_send(payload)             # your client, unchanged
limiter.observe(ok=True, tokens=n)
```

## Scaling: fan out to storage

K Jobs, one output directory, no coordinator:

```python
rows = shard_select(stream(source), rank=RANK, world=4)   # strided, disjoint by construction
stats = pump(rows, to_request, parse, endpoint, output, shard=(RANK, 4))
```

Each shard adapts to its own node independently and writes `completions/shard-{n}.done`
when finished (datatrove's marker convention — a coordinator can watch the directory).

## Embeddings

Same client, different route — a "row" can be a pre-grouped batch:

```python
stats = pump(batches, to_request=lambda b: {"model": "m", "input": b["texts"]},
             parse=parse_embeddings, endpoint=endpoint, output=output,
             route="/embeddings")
```

## Running under an agent

In agent mode (detected via env) stdout carries exactly one line — the run's stats as JSON —
and everything human goes to stderr, including the run-end advisor ("server ceiling: running
pinned at N — relaunch with --max-num-seqs 2N") and the standing hint that re-running the
same command is always safe.

## The output contract

Everything on disk is specified in [CONTRACT.md](CONTRACT.md): append-only parts, the
manifest sidecar, error rows (a failed row is a durable record, never a gap), the healing
reader rule, telemetry. Any process that can read parquet and glob a directory can consume
or resume pumpjack output without importing pumpjack.

## What it deliberately isn't

One request per row (rollouts/agent loops are a different tool). No DAG authoring. No
provider price tables — it records measured tokens and latency and leaves dollars to you.
No live clusters: scaling is shards writing to storage.
