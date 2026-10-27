# How the pipeline works

This is the design for saturate 0.3's pipeline: how rows get from your input, through the model
endpoint, into the output. It replaces the hand-written version in 0.2. The first half explains the
design in plain terms; the appendix holds the details reviewers need.

## What changes, in one paragraph

In 0.2 one big function (`through()`) did everything on the event loop: it read the input, started
a task per row, polled a queue, and the output was written on the loop too. A slow input or a slow
remote write froze everything else, and two edge cases lost rows that had already finished: on
cancel, and rows left unwritten while the input went quiet.
In 0.3 the work is split into a few small, separate jobs that hand rows to each other through
queues. Reading the input and writing the output move to their own threads, so they never block
the requests in flight, and every way a run can stop is handled so completed rows are kept.

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
| **drain** | takes finished rows and counts them | the event loop |
| **writer** | the only code that writes to the output | its own thread, so a slow write never blocks requests |

## A row's life

1. The **reader** reads the row from your input and hands it over straight away (it reads at most
   64 rows ahead of the feeder).
2. The **feeder** waits until there is room (see the limits below), then starts the row's task.
3. The **row task** calls your `to_request` to build the request body.
4. It waits for a free **slot** in the window, the number of requests allowed in flight at once,
   which the controller adjusts as the run goes.
5. It sends the request (with retries and the circuit breaker, as before), then gives the slot back
   straight away, so the next row can go.
6. It calls your `parse` on the reply and puts the finished row in the results queue. From this
   moment the row counts as **finished**: whatever happens next, it will be written.
7. **drain** takes it from the queue and hands it to the **writer**, which adds it to the output
   and writes a file when enough rows have built up, or when none have arrived for a while.

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
stops waiting for the final write.

## What gets simpler

Gone: the task bookkeeping, counters and polling loops in `through()`, sending exceptions through
the result queue, the separate thread-pool shutdown, the compatibility paths for limiters nobody
uses, writing on the event loop, and the `hasattr` checks for optional output features (they are
read in one place now).

## How it is checked

- The contract tests (`pytest -m 'not internals'`) must pass unchanged, apart from one timing fix
  described in the appendix. The two tests that document the edge cases that lost rows must now
  pass; they are strict, so CI fails until they do.
- A performance A/B against 0.2 runs locally and on a real engine (HF Jobs) before merge.
- One PR, two commits: the pipeline jobs first, then the writer thread, so a performance change can
  be traced to one of them.

---

## Appendix: details for reviewers

**Usage.**
```python
async with through(client, rows, to_request, parse) as results:  # a stream of finished rows
    stats = await drain(results, sink)
```
`through()` becomes an async context manager: anyio does not allow yielding from inside a task
group in an async generator. `through` and `drain` stay separate so they can be used apart;
`pump()` is the one-liner. `drain` still accepts any async iterable of finished rows.

**Where a row becomes "finished".** Step 6: `parse` and the put into the results queue run with no
`await` between them, and the queue is unbounded (the backlog limit is the bound), so the put
never waits. No row can be parsed and then lost before it is queued.

**Slots never move between tasks.** A row task acquires and releases its own slot (PR 4's
`_Slot`), including the anyio workaround tracked in #52.

**Prepare and backlog limits.** The feeder takes one of each per row before starting the row's
task. The prepare limit is returned once the row has a slot. The backlog limit is returned when
drain takes the finished row, so any consumer of `through()` works, not only `drain`. Default
backlog size: the controller's maximum window + `fail_fast` + twice the flush size; it is always
larger than what can be in flight plus the failed rows `fail_fast` holds back, so it cannot
deadlock. Time spent waiting on it is reported as `sink` in the per-tick bottleneck verdict. A row
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

**Error precedence.** The run records the first failure; later ones are logged. A group is
unwrapped only when it holds a single error.

**The writer thread.** One `ThreadPoolExecutor(1)`: every call that touches the output's state
(`probe`, `append`, `flush`) runs there, in order. Each time, it takes every row already waiting
(so batches form on their own under load, and a lone row is not held back) and calls `flush()` when
no row has arrived for the output's flush interval; with the age check inside `append`, that bounds
how long any row waits. The final stats and marker are written after the writer has finished, on a
worker thread. The thread is
never abandoned: abandoning a write could leave half a file, so shutdown waits for the call in
progress, as the on-loop flush does today. Outputs must bound their own I/O (fsspec and
HfFileSystem have transport timeouts). Telemetry rewrites stay separate, as the output contract
already allows.

**The reader thread.** It hands each row to the event loop as soon as `next()` returns it
(`loop.call_soon_threadsafe`, a few microseconds), and a semaphore keeps it at most 64 rows ahead
of the feeder. An earlier design batched rows in the thread; a test with an input stuck in `next()`
showed the rows read before the stall were trapped in the unfinished batch, so it hands over per
row. If the input raises, the rows before the error are already handed over. The thread is a daemon
thread: a blocking `next()` cannot be interrupted, so a stopped run leaves it behind and it never
holds up exit; an input left behind like that must not be reused (a resume uses a fresh input and
the stored ids, as today).

**Output protocol.** A `typing.Protocol` with `existing_ids`, `append` and `flush`; the optional
features (`probe`, `flush_interval_s`, `rows_demoted`, `begin_run`, `write_telemetry`,
`write_stats`, `write_marker`, `error_only_ids`, `telemetry_every_ticks`) are read in one function.
No base class.

**The one contract-test change.** `test_fatal_lands_every_row_that_completed_before_it` let a row
finish *after* the fatal error and expected it to be written, but the policy cancels rows in flight
at a fatal error, so the test contradicted the table. It is re-timed so the row finishes before the
fatal error. That re-timed test also passes on 0.2: results come out in finishing order, so a row
that finished before the fatal error was never behind it. It now pins the policy rather than
marking a bug; the rows 0.2 lost at a fatal error were the ones still in flight, which the policy
cancels.

**Leak checks.** The reader is a daemon thread, so the contract harness's check for leftover
non-daemon threads does not count a reader left behind on a stuck input; the test for that case
releases the input before the check anyway.
