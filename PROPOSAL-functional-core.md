# Proposal: functional core — combinators + AdaptiveClient (2026-07-28)

Motivated by two steers (Daniel, overnight): decouple IO from the inference middle so
pumpjack plugs into datatrove and the "Inference Pipelines" layer cleanly; and make the
interface compositional — itertools/toolz-shaped piping, not a framework.

## The decomposition

`pump()` today fuses four things. Unfused, they are the public functional surface:

```python
from pumpjack import AdaptiveClient, Auto, stream, skip_done, through, drain

rows    = stream(source, id_key=None)        # normalize: dicts/tuples -> (id, row); lazy
rows    = skip_done(rows, output)            # anti-join, manifest-first, within-run dedup
async with AdaptiveClient(endpoint, window=Auto()) as client:
    results = through(client, rows, to_request, parse)   # adaptive unordered async map
    stats   = await drain(results, output, shard=(0, 1)) # parts+manifest+markers+telemetry
```

- **`AdaptiveClient`** — window + controller + retry ladder + breaker + SignalSource behind
  one object: `await client.post(request) -> (body, err)`. No source, no storage. This is
  the piece datatrove embeds (their fixed semaphore -> this), and where the in-process
  vLLM/SGLang transports slot in post-v1.
- **`through(client, rows, to_request, parse)`** — async generator: yields
  `(id, row, out_dict_or_error)` in completion order. Composable: a second `through()` is a
  two-stage pipeline in plain Python.
- **`drain(results, output)`** — the sink loop; owns the storage CONTRACT and Stats.
- **`pump(...)` stays exactly as-is** — the one-call composition of the above; the front
  door for recipes/agents/UI; the only thing the oracle sees.

## Rules (hold these hard)

1. **Persistence points are composition boundaries.** "Error rows not lost rows" and
   exactly-once-per-id are stage-with-sink properties. Multi-stage = pipe through storage
   (each stage its own output dir + anti-join = free multi-stage resume). In-memory chaining
   allowed, documented as trading away crash granularity.
2. **itertools, not dask.bag.** No scheduler, no graph, no partitioning, no `.compute()`.
   Plain functions; no `|` operator sugar in v1 (can be earned later).

## Migration

Mechanical: extract the worker/window/poller wiring from `__init__._pump` into
`core.py` (AdaptiveClient + through), `sink.py` gains `drain`, facade becomes ~40 lines of
composition. Oracle is the gate — it touches only `pump()` + storage layout, so **9/9
before and after is the entire review**. LOC estimate: +60 net, still under the 800 ceiling.

## Plumb-check: how this embeds elsewhere (Daniel's annotation #5)

### datatrove (verified against local checkout, 2026-07-28)

Their loop: `semaphore = asyncio.Semaphore(config.max_concurrent_generations)` created once
in `run_async` (`run_inference.py:560`); `_send_request(server, payload, semaphore)` does
`async with semaphore:` around **their own transport** (`InferenceServer.send_request` —
not raw HTTP; they keep their retries, their checkpointing, their metrics).

Consequence: the innermost primitive should be transport-agnostic — an **adaptive
semaphore**, not a client:

```python
# pumpjack's innermost layer:
limiter = AdaptiveLimiter(window=Auto(), signals=HttpScrape(endpoint))  # signals optional
async with limiter.slot():
    result = await server.send_request(payload)     # THEIR transport, unchanged
limiter.observe(ok=True, tokens=n)                   # feedback drives the controller
```

The datatrove integration is then a ~5-line diff in `_send_request` + one constructor swap,
touching none of their checkpointing/caching/metrics. Layering becomes:

```
AdaptiveLimiter   window + controller + signals; slot()/observe()   <- datatrove, DataDesigner
AdaptiveClient    = AdaptiveLimiter + HTTP transport + retry + breaker  <- our own through()
through/drain/... = combinators over AdaptiveClient                  <- pump(), wrappers
```

### NVIDIA DataDesigner (recon 2026-07-28, source-verified)

**Prior-art correction: DataDesigner ships runtime-adaptive AIMD admission, on by default**
(`request_admission/controller.py` — "AIMD-backed request admission controller with exact
request leases"). "No datagen framework adapts at runtime" is retired from our narrative.
What survives, precisely:
- Their signal is **429s only** — no latency, no throughput gradient, no server metrics.
  It cannot find the optimum on endpoints that saturate *without* 429ing — i.e. self-hosted
  vLLM/SGLang, pumpjack's core case (and vLLM never 429s; its queue is unbounded).
- It ramps **down from a static cap** (`max_parallel_requests`, default **4**; +1 per 25
  consecutive successes) — throttling a user guess, not discovering capacity.
- Engine-embedded, not a reusable client; per-request admission, no batch/source/sink shape.

**Design validation**: they independently converged on the same split the datatrove check
gave us — **admission (adaptive window, leases, outcome-classified release) separated from
transport (HTTP + non-429 retries)**, with the explicit invariant that 429 is never retried
at transport so the signal reaches the admission loop. Their admission seam is a Protocol:
`try_acquire / acquire_async / release(lease, outcome)` — structurally `AdaptiveLimiter`'s
`slot()/observe()`. Consequences adopted:
- `observe()` takes a classified outcome (`ok | rate_limited(retry_after) | timeout |
  failure`), not a bare bool — fits both host shapes (datatrove's dumb-semaphore hosts embed
  the limiter; DataDesigner-style engines could mount it at their admission seam via a thin
  Protocol shim).
- Their pressure-snapshot/event-sink observability pattern endorses open choice #4
  (`on_tick` callback): yes.

Refs: NVIDIA-NeMo/DataDesigner — `engine/models/request_admission/controller.py`,
`clients/base.py` (ModelClient Protocol), `clients/model_request_executor.py`,
`clients/retry.py` (the 429-passthrough invariant), `clients/factory.py`.

## Open choices (confirm/veto)

0. **(settled with Daniel 2026-07-28, stream-first steer) Sink protocol — resumability as a
   contract any IO can satisfy.** The invariant, one sentence: *an id returned by
   `existing_ids()` implies its record is durable; an id not returned implies re-processing
   it is safe (idempotent write or reader-resolved duplicate).* Protocol (duck-typed):
   `existing_ids() -> set[str]` · `append(done)` · `flush()`.
   - `ParquetSink` = today's code = the full CONTRACT (exact resume + error rows/healing +
     reader rule). `pump(output="...")` string → ParquetSink, unchanged.
   - `FileSink(outdir, ext, key)` = second blessed impl (~25 LOC): id → `{outdir}/{id}.md`;
     the filesystem is the manifest; naturally idempotent; failed = absent = retried next
     run (no error records — documented trade). Covers OCR→md, audio→transcript.
   - Non-resumable IO = same protocol, empty `existing_ids()` — degradation, not a second
     interface. `skip_done(rows, sink)` always just calls `sink.existing_ids()`.
   Two blessed sinks is the v1 line; a protocol with one impl is just a class.
1. Names: `stream / skip_done / through / drain` — happy to bikeshed (`amap`? `pump_through`?).
2. `through()` yields `(id, row, result)` triples with `result["error"]` convention, or a
   small `Done(id, row, out, error)` dataclass? (Lean: dataclass — self-documenting.)
3. Stats: attach to `drain()` return (proposed) — means bare `through()` users compute their
   own; acceptable since they own IO anyway?
4. Does `AdaptiveClient` expose `breaker_events`/telemetry hooks for embedders (datatrove
   would want its own logging)? (Lean: yes, a small `on_tick` callback.)
