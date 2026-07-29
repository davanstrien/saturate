# UI POC brief — "Inference Pipelines" console page (spec-by-demo)

A private HF Space demonstrating the console page that would sit on top of the
inference-pipelines layer. **Disposable seed for the product team, not a
product**: one task template, no real launch, label it as a spec-by-demo in the
Space card. Do NOT modify the pumpjack package.

## Stack

**Static-first: plain HTML + CSS + vanilla JS (htmx/alpine only if state truly
demands it). NOT Gradio** — this must look like a console, not a demo widget.
A static Space is enough: all data comes from client-side `fetch` against the
Hub API with a user-pasted READ token (memory/localStorage only, never sent
anywhere else, read-only calls only). Dark, dense console aesthetic — cards,
tabular numerals, restrained color (see the dataviz skill for chart rules).

## Scope — exactly four things

1. **Embeddings wizard** (the one template): dataset id → split → text column
   → embedding model → GPU flavor (auto-suggested, overridable) → jobs count
   (1/2/4) → output (bucket path default) → "compile to dataset" toggle.
2. **Dry-run compile**: the form generates (a) the exact driver `.py` (template
   it from `spikes/tier1_embed.py` + the fan-out variant in
   `spikes/tier1_fanout.py`) and (b) the exact `hf jobs run ...` command(s),
   with copy buttons. No execution in v1 — the compile IS the demo.
3. **Run dashboard** (the wow-piece): given an output dataset repo id, render a
   live view purely from Hub reads — progress from `data/part-*.parquet`
   counts + rows-done summed from telemetry `ok` ticks; per-shard status
   lights from `completions/shard-N.done`; a window/throughput chart from
   `telemetry-shard*-*.jsonl` (keys: t, limit, inflight, waiting, running, bp,
   ok, tok_s — schema in CONTRACT.md §6). Prefill a picker with real finished
   runs: `davanstrien/pumpjack-embed-4job` (4 shards), `pumpjack-embed-1job`,
   `pumpjack-shapes`, `pumpjack-tier1-moh5k`. This demonstrates the claim in
   docs/design.md: the storage CONTRACT doubles as the UI protocol — no backend.
4. **Ship it**: private Space under davanstrien (e.g. `pumpjack-console-poc`),
   Space card stating: spec-by-demo, disposable, the layer diagram, and a link
   note that the engine underneath is pumpjack.

## Guardrails

One template only (no second task, no settings drawer). Read-only token use.
No pumpjack package changes. If something in the CONTRACT makes the dashboard
awkward, DOCUMENT the gap (that finding is half the point) rather than working
around it silently.

## Context to read first

pumpjack `README.md`, `CONTRACT.md` (§1 layout, §6 telemetry), `docs/design.md`
(the stack diagram + "storage as UI protocol"), `spikes/tier1_embed.py`,
`spikes/tier1_fanout.py`. Daniel's original flow sketch: embed dataset →
select column(s) → select embedding model → select GPU → output to bucket →
compile to dataset.
