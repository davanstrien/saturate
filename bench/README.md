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

- `rows_per_s` — throughput as the caller sees it.
- `cpu_ms_per_row` — client CPU (all threads) per row. At zero endpoint latency this is the
  library's own cost, whatever the server's speed.
- `loop_lag_p99_s` — from the per-tick `loop_lag_s` telemetry: how long the event loop was blocked
  beyond a tick. A flush or a parse on the loop shows here. Ticks are 2 s, so short runs have few.
- `first_durable_s` — time until the first part file exists in the output.
- `peak_rss_mb` — the client process's peak resident memory.

## Reading a result

A ratio is head/base of the per-arm medians. An interval that contains 1.0 is no measurable
change. Run an A/A on the same machine first: its interval width is the smallest change that
machine can detect. One Job run reliably detects a ~10% change, not a 5% one.

The local stub answers from one asyncio thread, so at zero latency it can be the bottleneck for
`rows_per_s`; `cpu_ms_per_row` is the overhead number that does not depend on it.

These are informational, not a CI gate: shared CI runners vary by 20–30%.

## Noise floor

`results/aa-noise-floor-20261027.json`: main against main, `quick` suite, 5 reps, on an Apple
Silicon laptop (`arm64`). The 95% intervals of the head/base ratio:

| metric | interval width across scenarios | read |
|---|---|---|
| `rows_per_s` | about ±5–15% | a ~10% change is detectable |
| `cpu_ms_per_row` | about ±5–13% | a ~10% change is detectable |
| `peak_rss_mb` | under ±1% (±9% on overhead-w64) | small changes are detectable |
| `first_durable_s` | ±5% to ±70% | only large changes |
| `loop_lag_p99_s` | up to −80% / +50% | too noisy at 5 reps for a verdict; use it to spot large blocks |

Two baseline facts from the same run, worth knowing before the rewrite:
- `overhead-w512`: the event loop is blocked ~0.45 s per tick (p99) at a 512 window with zero
  latency; at window 64 it is ~0.02 s.
- `default-flush-w64` against `overhead-w64`: the default local flush (10 rows) costs ~40% of
  zero-latency throughput (546 vs 928 rows/s) and ~50% more CPU per row.
