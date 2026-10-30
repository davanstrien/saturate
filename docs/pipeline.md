# How the pipeline works

This page describes how rows get from your input, through the model endpoint, into the output:
`through()` (`saturate/pipeline.py`) and `drain()` (`saturate/writer.py`). The first half explains
the design in plain terms; the appendix holds the details reviewers need.

## In one paragraph

The work is split into a few small, separate jobs that hand rows to each other through queues.
Reading the input and writing the output run on their own threads, so a slow input or a slow
remote write never blocks the requests in flight. Every way a run can stop is handled so that
completed rows are kept.

## The jobs

```
  input ──▶ reader ──▶ feeder ──▶ one task per row ──▶ drain ──▶ writer ──▶ output
          (thread)                  (request + parse)             (thread)
```

| job | does | runs on |
|---|---|---|
| **reader** | reads rows from your input and hands each over as soon as it is read | its own thread, so a slow input never blocks requests |
| **feeder** | starts a task for each row, when there is room | the event loop |
| **row task** | builds the request, waits for a slot, sends it, parses the reply | the event loop (`to_request` can run in threads) |
| **drain** | takes finished rows and hands them to the writer | the event loop |
| **writer** | the only code that writes to the output | its own thread, so a slow write never blocks requests |

## A row's life

1. The **reader** reads the row from your input and hands it over straight away (it reads at most
   64 rows ahead of the feeder).
2. The **feeder** waits until there is room (see the limits below), then starts the row's task.
3. The **row task** calls your `to_request` to build the request body.
4. It waits for a free **slot** in the window, the number of requests allowed in flight at once,
   which the controller adjusts as the run goes.
5. It sends the request (with retries and the circuit breaker), then gives the slot back straight
   away, so the next row can go.
6. It calls your `parse` on the reply and puts the finished row in the results queue. From this
   moment the row counts as **finished**: whatever happens next, it will be written.
7. **drain** takes it from the queue and hands it to the **writer**, which adds it to the output
   and writes a file when `flush_every` rows have built up, or when the oldest buffered row reaches
   the output's flush interval (10 s local, 60 s remote).

## What keeps memory and concurrency in check

| limit | what it caps |
|---|---|
| the window | requests in flight; the controller tunes it |
| the prepare limit | request bodies built but not yet sent (bodies can be large: images) |
| the backlog limit | rows started but not yet taken by drain, so a slow output cannot fill memory; when it bites, the run reports "the sink is the bottleneck" |
| the writer's queue | rows waiting to be written; a full one makes drain wait, which in turn holds the backlog |

## When something goes wrong

Finished rows (step 6) are always written. What happens to the others depends on what failed:

| what failed | rows in flight | finished rows | then |
|---|---|---|---|
| your input raised an error | finish normally | written | the input's error is raised |
| the endpoint is gone (breaker gave up) | cancelled | written | that error is raised |
| the first N rows all failed (`fail_fast`) | cancelled | nothing stored, so a re-run retries them | that error is raised |
| writing the output failed | cancelled | as many as were written | the write error is raised |
| you cancelled (Ctrl-C) | cancelled | written | the cancellation continues |

If two things fail, the first one is the error you see; later ones are logged. A second Ctrl-C
stops waiting: finished rows not yet handed to the writer are logged as lost, and a write already
in progress finishes in the background (never half a file). Rows cancelled in flight were still
work for the engine; the policy trades that for stopping promptly.

## How it is checked

- The contract tests (`tests/test_contract.py`, `pytest -m 'not internals'`) pin the table above
  through public entry points.
- `bench/` measures throughput, CPU per row and event-loop blocking from the outside, locally and
  on a real engine ([bench/README.md](../bench/README.md)).

---

## Appendix: details for reviewers

**Usage.**
```python
async with through(client, rows, to_request, parse) as results:  # a stream of finished rows
    stats = await drain(results, sink)
```
`through()` is an async context manager: anyio does not allow yielding from inside a task group
in an async generator. `through` and `drain` stay separate so they can be used apart;
`pump()` is the one-liner. `drain` still accepts any async iterable of finished rows.

**Where a row becomes "finished".** Step 6: `parse` and the put into the results queue run with no
`await` between them, and the queue is unbounded (the backlog limit is the bound), so the put
never waits. No row can be parsed and then lost before it is queued.

**Slots never move between tasks.** A row task acquires and releases its own slot (`_Slot` in
`saturate/core.py`), including the anyio workaround tracked in #52.

**Prepare and backlog limits.** The feeder takes one of each per row before starting the row's
task. The prepare limit is returned once the row has a slot. The backlog limit is returned when
drain takes the finished row, so any consumer of `through()` works, not only `drain`. Default
backlog size: the controller's maximum window + `fail_fast` + 2048 rows for the sink to buffer
(`through(backlog=)` sets it). It is always larger than what can be in flight plus the failed rows
`fail_fast` holds back, so it cannot deadlock. Time spent waiting on it is reported as `sink` in the per-tick bottleneck verdict. A row
task cancelled before it starts cannot return its limits; both belong to the run and are thrown
away with it.

**Fatal errors without cancelling drain.** The feeder and the row tasks run inside their own task
group, in one task. A fatal error cancels that group only, closes the results queue and is
recorded; drain keeps going until it has taken everything queued; `through()` raises the recorded
error when the caller leaves the `async with`.

**Cancellation.** saturate runs under `asyncio.run`, so a Ctrl-C arrives as a native
`CancelledError`, which anyio's shield does not stop. drain catches it and runs its cleanup as a
separate task under `asyncio.shield`: stop the producers, take every row left in the queue, hand
them to the writer, wait for it to finish, then re-raise.

**Error precedence.** The run records the first failure; later ones are logged. The recorded
error is raised outside the task groups, so the caller gets the error itself, not an exception
group.

**The writer thread.** One `ThreadPoolExecutor(1)`: every call that touches the output's state
(`probe`, `append`, `flush`) runs there, in order. Each time, it takes every row already waiting
(so batches form on their own under load, and a lone row is not held back). When no row arrives
before the oldest buffered row reaches the output's flush interval, it calls `flush()`; with the
age check inside `append`, that bounds how long any row waits. The final stats and marker are
written after the writer has finished, on a worker thread. The thread is never abandoned:
abandoning a write could leave half a file, so shutdown waits for the call in progress. Outputs
must bound their own I/O (fsspec and HfFileSystem have transport timeouts). Telemetry rewrites run
separately, on their own worker thread (CONTRACT §6).

**The reader thread.** It hands each row to the event loop as soon as `next()` returns it
(`loop.call_soon_threadsafe`, a few microseconds), and a semaphore keeps it at most 64 rows ahead
of the feeder. It hands over each row, not batches: with batches, the rows read before an input
stalls in `next()` would be trapped in the unfinished batch. If the input raises, the rows before the error are already handed over. The thread is a daemon
thread: a blocking `next()` cannot be interrupted, so a stopped run leaves it behind and it never
holds up exit; an input left behind like that must not be reused (a resume uses a fresh input and
the stored ids).

**Output protocol.** A `typing.Protocol` with `existing_ids`, `append` and `flush`; the optional
features (`probe`, `flush_interval_s`, `begin_run`, `write_telemetry`, `write_stats`,
`write_marker`, `telemetry_every_ticks`) are read in one function, `sink_hooks()`. Two attributes the
sink keeps, `rows_demoted` (rows a flush turned into error rows) and `error_only_ids` (set by
`existing_ids()`), are read where they are used. No base class.

**Known costs.** The reader and writer threads cost about 0.15 ms of CPU per row (thread
wake-ups in both directions; about 4% of a core at 250 rows/s; throughput is unaffected). And on
Linux, while a process has more than one thread, growing its file-descriptor table waits for an RCU
grace period: opening the 64th, 128th, 256th, 512th and 1024th descriptor takes 60-100 ms, and that
call is a new connection's `socket()` on the event loop. A run that ramps its window to 512 pays this
a few times during the ramp, under half a second in all. Both are measured, and both are left as they
are for now (#55): the second would need a Linux-specific workaround in the HTTP
client setup.

**Leak checks.** The reader is a daemon thread, so the contract harness's check for leftover
non-daemon threads does not count a reader left behind on a stuck input; the test for that case
releases the input before the check anyway.
