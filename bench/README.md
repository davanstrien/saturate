# Benchmarks

Measures saturate from the outside: `pump()` against an endpoint, with nothing but the public API
and the documented outputs (Stats, telemetry). The same scripts therefore measure any version,
which is what an A/B across a rewrite needs.

| file | what it does |
|---|---|
| `workload.py` | one scenario, one process; prints one JSON line of metrics |
| `serve.py` | the test stub server in its own process (no GIL shared with the client) |
| `ab.py` | two versions, interleaved runs, head/base ratio with a 95% bootstrap interval |
| `jobs_ab.sh` | the same A/B against a real vLLM engine inside one HF Job |
| `results/` | JSON from runs worth keeping (each names its base and head commits) |

## Run

```bash
UV_NO_CONFIG=1 python bench/ab.py --base origin/main --head origin/main   # A/A: the noise floor
UV_NO_CONFIG=1 python bench/ab.py --base origin/main --head .             # your working tree vs main
UV_NO_CONFIG=1 python bench/ab.py --base origin/main --head . --suite smoke --reps 2  # harness check
```

`UV_NO_CONFIG=1` keeps a user-level uv config (a private index, `exclude-newer`) out of the
installs. Each ref is installed, not editable, into its own virtualenv; `.` is the working tree,
uncommitted changes included.

## Metrics

- `rows_per_s` — successful rows per second as the caller sees it (`rows_failed` is reported too;
  a run with failures is not comparable). `tokens_per_s` from Stats.
- `cpu_ms_per_row` — client CPU (all threads) per row. At zero endpoint latency this is the
  library's own cost, whatever the server's speed.
- `loop_lag_max_s`, `loop_lag_mean_s` — from the per-tick `loop_lag_s` telemetry: how long the
  event loop was blocked beyond a tick. A flush or a parse on the loop shows here. Ticks are 2 s,
  so a run has few (2–7 in the quick suite); `loop_lag_p99_s` is reported only from 20 ticks on.
- `first_durable_s` — time until the first part file exists in the output.
- `peak_rss_mib` — the client process's peak resident memory.

## Reading a result

Every rep runs both arms back to back, in a random order, after one discarded warmup run per
scenario and arm. The headline is the median of the per-rep **paired** ratios (head/base within a
rep, so drift between reps cancels), with their min–max range. The bootstrap interval of the ratio
of medians is a secondary number: with a handful of reps it takes few distinct values and
under-covers. Run an A/A on the same machine first: the spread of its paired ratios is the
smallest change that machine can detect.

The local stub answers from one asyncio thread, so at zero latency it can be the bottleneck for
`rows_per_s`; `cpu_ms_per_row` is the overhead number that does not depend on it.

These are informational, not a CI gate: shared CI runners vary by 20–30%.

## Noise floor

`results/aa-noise-floor-20261027.json`: main against main, `quick` suite, 5 reps, M1 Pro. It was
taken before the paired ratios, the warmup and the metric renames (its `loop_lag_p99_s` is in
effect the max, its `peak_rss_mb` is MiB on macOS), so it is the first-cut floor; the per-run JSON
lets the paired ratios be recomputed. The 95% bootstrap intervals of the head/base ratio:

| metric | interval width across scenarios | read |
|---|---|---|
| `rows_per_s` | about ±5–15% | a ~10% change is detectable |
| `cpu_ms_per_row` | about ±5–13% | a ~10% change is detectable |
| `peak_rss_mb` (MiB) | under ±1% (±9% on overhead-w64) | small changes are detectable |
| `first_durable_s` | ±5% to ±70% | only large changes |
| `loop_lag_p99_s` (≈ max) | up to −80% / +50% | too noisy for a verdict; use it to spot large blocks |

Two baseline facts from the same run, worth knowing before the rewrite:
- `overhead-w512`: the event loop is blocked up to ~0.45 s per tick at a 512 window with zero
  latency; at window 64 it is ~0.02 s.
- `default-flush-w64` against `overhead-w64`: the default local flush (10 rows) costs ~40% of
  zero-latency throughput (546 vs 928 rows/s) and ~50% more CPU per row.

### On a real engine

`results/aa-noise-floor-jobs-vllm-a10g-20261027.json`: Job `6ae0905741e90535df76060a`, vLLM
(`vllm/vllm-openai:latest`) serving Qwen2.5-0.5B-Instruct on a10g-large, both arms main
(`932e8f9`), `vllm` suite, 3 reps, taken before the warmup and the metric renames. Per-rep paired
head/base ratios, min–max across reps and scenarios:

| metric | paired range | read |
|---|---|---|
| `tokens_per_s` | 0.989–1.011 | a ~3% change is detectable |
| `rows_per_s` | 0.983–1.026 | a ~3–5% change is detectable |
| `cpu_ms_per_row` | 0.856–1.059 | ~10–15%; long generations are the noisiest |
| `peak_rss_mb` (MiB) | 0.988–1.012 | small changes are detectable |
| `first_durable_s` | 0.822–1.077 | only large changes |

The engine, not the laptop's single-thread stub, sets the pace here, so a Job run detects
throughput changes several times smaller than a local run. Cost: about $0.12 (a10g-large,
~5 min of running time, model load included).
