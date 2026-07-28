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

## Open choices (confirm/veto)

1. Names: `stream / skip_done / through / drain` — happy to bikeshed (`amap`? `pump_through`?).
2. `through()` yields `(id, row, result)` triples with `result["error"]` convention, or a
   small `Done(id, row, out, error)` dataclass? (Lean: dataclass — self-documenting.)
3. Stats: attach to `drain()` return (proposed) — means bare `through()` users compute their
   own; acceptable since they own IO anyway?
4. Does `AdaptiveClient` expose `breaker_events`/telemetry hooks for embedders (datatrove
   would want its own logging)? (Lean: yes, a small `on_tick` callback.)
