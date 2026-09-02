"""Every tick names what limited it: `source` or `prep` (the client side starving an
under-used window — the only input-bound verdicts), `prep` or `loop` when on-loop work
blocked the loop with a full window, `engine` when the window was full or admission waits
dominated, nothing when the limiter was idle. The run-end advisor turns a run of
client-bound ticks into a hint with what was measured.

Ticks are driven directly (`limiter._tick()`) with the tick clock set by hand."""

import asyncio
import threading
import time

from stub_server import StubLimiter

from saturate import AdaptiveLimiter, Auto, Fixed, Stats, through
from saturate.telemetry import advise_input, bound_by_counts, tick_record


def tick(lim: AdaptiveLimiter, dt: float = 1.0) -> dict:
    """One tick whose wall-clock is `dt` seconds."""
    lim._t_last = time.monotonic() - dt
    asyncio.run(lim._tick())
    return lim.ticks[-1]


def test_idle_limiter_with_a_trace_of_source_wait_is_not_input_bound():
    lim = AdaptiveLimiter(window=Fixed(4))
    lim.note_source_wait(1e-6)
    rec = tick(lim)
    assert rec["bound_by"] is None and rec["input_bound"] is False
    assert bound_by_counts(lim.ticks) == {}


def test_prep_time_is_measured_and_named_when_the_window_sits_empty():
    lim = AdaptiveLimiter(window=Fixed(8))
    lim.note_prep(0.3)
    lim.note_prep(0.3)
    rec = tick(lim, dt=1.0)
    assert (rec["bound_by"], rec["input_bound"]) == ("prep", True)
    assert (rec["prep_s"], rec["prep_n"], rec["prep_workers"], rec["source_s"]) == (0.6, 2, 0, 0.0)
    assert bound_by_counts(lim.ticks) == {"prep": 1}
    nxt = tick(lim)
    assert (nxt["prep_s"], nxt["prep_n"]) == (0.0, 0)  # the accumulators reset every tick


def test_source_wait_names_the_source_when_it_takes_a_quarter_of_the_tick():
    lim = AdaptiveLimiter(window=Fixed(8))
    lim.note_source_wait(0.2)
    assert tick(lim, dt=1.0)["bound_by"] is None  # below the floor: not conclusive
    lim.note_source_wait(0.26)
    lim.note_prep(0.25)
    rec = tick(lim, dt=1.0)
    assert (rec["bound_by"], rec["source_s"], rec["prep_s"]) == ("source", 0.26, 0.25)


def test_full_window_or_dominant_admission_wait_names_the_engine():
    lim = AdaptiveLimiter(window=Fixed(2))

    async def full():
        async with lim.slot(), lim.slot():
            lim.note_prep(0.4)  # prep took real time too, but the window is full: the engine is the limit
            lim._t_last -= 1.0
            await lim._tick()
        return lim.ticks[-1]

    rec = asyncio.run(full())
    assert (rec["bound_by"], rec["input_bound"]) == ("engine", False)
    lim._wait["acquire"] = 3.0  # many tasks queued on the window, source and prep negligible
    lim.note_source_wait(0.01)
    assert tick(lim)["bound_by"] == "engine"
    assert bound_by_counts(lim.ticks) == {"engine": 2}


def test_on_loop_prep_that_eats_half_the_tick_is_prep_bound_but_not_input_bound():
    """to_request on the event loop blocks it: slots stay held because responses are not
    read, so the window looks full while the engine idles. Loop time names the culprit —
    but the window IS full, so the controller must not read the tick as starved."""
    lim = AdaptiveLimiter(window=Fixed(2))

    async def full():
        async with lim.slot(), lim.slot():
            lim.note_prep(0.55)
            lim._t_last -= 1.0
            await lim._tick()
            lim.prep_workers = 1  # one real thread doing the same work does not block the loop
            lim.note_prep(0.6)
            lim._t_last -= 1.0
            await lim._tick()
            lim.prep_workers = 4
            lim.note_prep(0.55)  # four threads: under a quarter each
            lim._t_last -= 1.0
            await lim._tick()
        return lim.ticks[-3:]

    on_loop, one_thread, four = asyncio.run(full())
    assert (on_loop["bound_by"], on_loop["input_bound"]) == ("prep", False)
    assert (one_thread["bound_by"], one_thread["prep_workers"]) == ("engine", 1)
    assert four["bound_by"] == "engine"


def test_blocked_loop_with_a_full_window_reads_loop():
    """A slow parse or a synchronous sink flush blocks the loop exactly like on-loop
    to_request; the tick is late by that much, and the verdict says so."""
    lim = AdaptiveLimiter(window=Fixed(2))

    async def full():
        async with lim.slot(), lim.slot():
            lim._t_last -= 6.0  # a 2 s tick that took 6 s: 4 s of lag
            await lim._tick()
            lim._t_last -= 2.5  # lag 0.5 s of 2.5: under half, the window decides
            await lim._tick()
        return lim.ticks[-2:]

    late, on_time = asyncio.run(full())
    assert (late["bound_by"], late["input_bound"]) == ("loop", False) and late["loop_lag_s"] >= 3.9
    assert on_time["bound_by"] == "engine"


def test_full_window_with_a_high_engine_queue_still_steps_down_under_on_loop_prep():
    """The `prep` verdict with a full window is not input-bound evidence: `Auto` must still
    see the queue and cut, not answer hold:input_bound."""

    class Gauges:
        waiting = 5  # in band

        async def read(self):
            return {"waiting": self.waiting, "running": 4}

    gauges = Gauges()
    lim = AdaptiveLimiter(window=Auto(initial=4, target_waiting=8), signals=gauges)

    async def go():
        async with lim.slot(), lim.slot(), lim.slot(), lim.slot():
            lim.observe(ok=True, tokens=10)  # the first completion ever: gauges now count
            lim._t_last -= 1.0
            await lim._tick()
            gauges.waiting = 40  # above hi
            lim.note_prep(0.6)
            lim._t_last -= 1.0
            await lim._tick()
        return lim

    lim = asyncio.run(go())
    rec = lim.ticks[-1]
    assert (rec["bound_by"], rec["input_bound"]) == ("prep", False)
    assert rec["reason"] == "cut:queue" and lim.window.limit < 4


def test_stats_carry_ticks_per_verdict():
    stats = Stats(bound_by={"engine": 5, "prep": 2})
    assert '"bound_by": {"engine": 5, "prep": 2}' in stats.to_json()
    assert Stats().bound_by == {}
    ticks = [{"bound_by": b} for b in ("engine", None, "prep", "loop", "engine")]
    assert bound_by_counts(ticks) == {"engine": 2, "prep": 1, "loop": 1}


def test_tick_record_carries_the_verdict_and_stage_times():
    rec = tick_record(1.0, 8, 2, None, 0, 3, True, 10.0, "hold", bound_by="prep",
                      source_s=0.0123456, prep_s=0.5, prep_n=7, prep_workers=2, loop_lag_s=0.25)
    assert (rec["bound_by"], rec["source_s"], rec["prep_s"]) == ("prep", 0.012, 0.5)
    assert (rec["prep_n"], rec["prep_workers"], rec["loop_lag_s"]) == (7, 2, 0.25)
    rec = tick_record(1.0, 8, 2, None, 0, 3, False, 10.0, "hold")
    assert (rec["bound_by"], rec["source_s"], rec["prep_s"], rec["prep_n"]) == (None, 0.0, 0.0, 0)


def ticks(verdict, n, stage_s=0.0, ok=0, bp=0, prep_n=0, t0=0.0, lag=0.0):
    return [{"t": t0 + 2.0 * (i + 1), "bound_by": verdict, "ok": ok, "bp": bp, "prep_n": prep_n,
             "prep_s": stage_s if verdict == "prep" else 0.0,
             "source_s": stage_s if verdict == "source" else 0.0, "loop_lag_s": lag}
            for i in range(n)]


def test_advisor_quotes_prep_cost_per_measured_call_not_per_success():
    # 4 prep ticks of 1 s over 20 calls each (50 ms/call); the poison rows never count as ok
    run = ticks("prep", 4, 1.0, ok=2, bp=0, prep_n=20) + ticks("engine", 6, ok=20)
    (hint,) = advise_input(run)
    assert hint == ("PREP-BOUND: to_request took 50 ms/row on average; run it ahead with "
                    "pump(prepare_workers=4)")
    (hint,) = advise_input(run, prepare_workers=4)
    assert hint == ("PREP-BOUND: to_request took 50 ms/row on average; prep is still the bottleneck "
                    "at 4 workers; raise prepare_workers or make to_request cheaper")
    assert advise_input(ticks("prep", 2, 1.0, prep_n=5) + ticks("engine", 8, ok=20)) == []  # 20% < 30%
    assert advise_input(ticks("prep", 2, 1.0, prep_n=5)) == []  # under three ticks: too short to diagnose


def test_advisor_names_the_source_and_a_blocked_loop():
    (hint,) = advise_input(ticks("source", 3, 0.6, ok=2, bp=1) + ticks(None, 7))
    assert hint == "SOURCE-BOUND: the source iterator took 200 ms/row; prefetch or shard the input"
    run = ticks("loop", 4, lag=1.5) + ticks("engine", 4, t0=8.0)  # 6 s of lag over a 16 s run
    (hint,) = advise_input(run)
    assert hint == ("LOOP-BOUND: the event loop was blocked 38% of the run outside to_request "
                    "(parse or sink writes); keep parse cheap and raise flush_every")


class _EchoClient:
    def __init__(self, limiter):
        self.limiter = limiter

    async def post(self, request, route="/chat/completions", on_admit=None):
        if on_admit is not None:
            on_admit()
        return {"req": request}, None


def test_prepare_workers_run_to_request_in_threads_and_errors_stay_row_errors():
    main = threading.get_ident()
    seen_threads = set()

    def to_request(row):
        seen_threads.add(threading.get_ident())
        if row["n"] == 3:
            raise ValueError("bad row")
        return {"n": row["n"]}

    async def go():
        client = _EchoClient(StubLimiter())
        rows = iter([(str(i), {"n": i}) for i in range(8)])
        done = [d async for d in through(client, rows, to_request, lambda r, b: b["req"],
                                         prepare_workers=2)]
        return client, done

    client, done = asyncio.run(go())
    assert main not in seen_threads and seen_threads  # prepared off the loop
    assert client.limiter.prep_workers == 2
    assert len(client.limiter.preps) == 7 and all(s >= 0 for s in client.limiter.preps)  # raise: unmeasured
    by_id = {d.id: d for d in done}
    assert by_id["3"].out is None and by_id["3"].error == "client: ValueError: bad row"
    assert {d.out["n"] for d in done if d.out} == {0, 1, 2, 4, 5, 6, 7}


def test_a_limiter_without_prep_accounting_still_gets_every_row():
    """The embedder seam: a limiter that predates note_prep/prep_workers is not an error row."""

    class Bare:
        window = type("W", (), {"limit": 4})()

        def note_source_wait(self, s):
            pass

    async def go():
        client = _EchoClient(Bare())
        rows = iter([(str(i), {"n": i}) for i in range(5)])
        return [d async for d in through(client, rows, lambda r: {"n": r["n"]}, lambda r, b: b["req"])]

    done = asyncio.run(go())
    assert all(d.error is None for d in done) and len(done) == 5
