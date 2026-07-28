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

## Quickstart

```python
from pumpjack import pump, Auto

stats = pump(
    rows,                                  # any iterable of dicts — streams, never materializes
    to_request=lambda row: {"model": "m", "messages": [{"role": "user", "content": row["text"]}]},
    parse=lambda row, resp: {"answer": resp["choices"][0]["message"]["content"]},
    endpoint="http://127.0.0.1:8000/v1",
    output="hf://datasets/you/results/data", # or a local path, or hf://buckets/...
)
print(stats.rows_processed, stats.tokens_per_sec)
```

That's the whole API for most jobs. Notes on what you didn't have to do:

- **No concurrency number.** The window starts small and adapts: delivered throughput is the
  primary signal, the engine's own `/metrics` gauges (vLLM, SGLang, llama.cpp, TEI) accelerate
  it when reachable, and against opaque endpoints it falls back to latency/error-driven AIMD.
  It grows only after the first completion arrives, cuts on real pressure, and freezes when
  your *source* is the bottleneck instead of the server (it tells you which).
- **No id column needed.** Rows get a stable content-hash id (pass `id_key=` to use your own).
  Identical input rows dedupe for free.
- **`kill -9` it, re-run the same command.** Output is append-only parquet with a manifest
  sidecar; resume is an exact anti-join on id — it re-pays at most one flush buffer, never
  duplicates a row. This holds across separate Jobs writing at different times.

## Serving the model yourself (e.g. inside one HF Job)

```python
from pumpjack import pump, Engine

with Engine("lightonai/LightOnOCR-2-1B", engine="vllm") as endpoint:   # vllm | sglang | llamacpp
    stats = pump(rows, to_request, parse, endpoint, output)
```

`Engine` boots the server in its own process group, health-gates it properly (health checks
lie during warm-up; readiness requires a real completion), and kills it on exit. On HF Jobs
this is the whole one-GPU recipe.

## The composable layer

`pump()` is a composition of four small stages. Use them directly when you want to pipe:

```python
from pumpjack import AdaptiveClient, Auto, stream, skip_done, through, drain

rows = stream(load_dataset("...", streaming=True))     # (id, row) pairs, lazy
rows = skip_done(rows, output)                          # exact resume filter

async with AdaptiveClient(endpoint, window=Auto()) as client:
    results = through(client, rows, to_request, parse)  # unordered async map, adaptive
    stats = await drain(results, output)                # parquet + manifest + markers

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

Embedding just the adaptive part in your own stack (your IO, your loop):

```python
async with AdaptiveClient(endpoint, window=Auto()) as client:
    body, err = await client.post(make_json_request("/chat/completions", payload))
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
