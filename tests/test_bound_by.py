"""Every tick names what limited it: `source` or `prep` (the client side) only when that
stage ate a real share of the tick and the window sat under-used, `engine` when the
window was full or admission waits dominated, nothing when the limiter was idle.
`input_bound` is derived from that verdict, and the run-end advisor turns a run of
client-bound ticks into a hint with the measured cost per row.

Ticks are driven directly (`limiter._tick()`) with the tick clock set by hand."""

import asyncio

from saturate import AdaptiveLimiter, Fixed, Stats, through
from saturate.telemetry import advise_input, tick_record


def tick(lim: AdaptiveLimiter, dt: float = 1.0) -> dict:
    """One tick whose wall-clock is `dt` seconds."""
    import time

    lim._t_last = time.monotonic() - dt
    asyncio.run(lim._tick())
    return lim.ticks[-1]


def test_idle_limiter_with_a_trace_of_source_wait_is_not_input_bound():
    lim = AdaptiveLimiter(window=Fixed(4))
    lim.note_source_wait(1e-6)
    rec = tick(lim)
    assert rec["bound_by"] is None and rec["input_bound"] is False
    assert lim.input_bound_ever is False and lim.bound_by == {}


def test_prep_time_is_measured_and_named_when_the_window_sits_empty():
    lim = AdaptiveLimiter(window=Fixed(8))
    lim.note_prep(0.3)
    lim.note_prep(0.3)
    rec = tick(lim, dt=1.0)
    assert (rec["bound_by"], rec["input_bound"]) == ("prep", True)
    assert rec["prep_s"] == 0.6 and rec["source_s"] == 0.0
    assert lim.input_bound_ever and lim.bound_by == {"prep": 1}
    assert tick(lim)["prep_s"] == 0.0  # the accumulator resets every tick


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
    lim.note_prep(0.9, workers=4)  # 0.9 s of thread time over four workers: under a quarter each
    assert tick(lim)["bound_by"] is None
    lim._wait["acquire"] = 3.0  # many tasks queued on the window, source and prep negligible
    lim.note_source_wait(0.01)
    assert tick(lim)["bound_by"] == "engine"
    assert lim.bound_by == {"engine": 2}


def test_on_loop_prep_that_eats_half_the_tick_is_prep_bound_even_with_a_full_window():
    """to_request on the event loop blocks it: slots stay held because responses are not
    read, so the window looks full while the engine idles. Loop time names the culprit."""
    lim = AdaptiveLimiter(window=Fixed(2))

    async def full():
        async with lim.slot(), lim.slot():
            lim.note_prep(0.55)
            lim._t_last -= 1.0
            await lim._tick()
            lim.note_prep(0.55 / 4, workers=4)  # the same share off the loop: not blocking anything
            lim._t_last -= 1.0
            await lim._tick()
        return lim.ticks[-2:]

    on_loop, threaded = asyncio.run(full())
    assert (on_loop["bound_by"], on_loop["input_bound"]) == ("prep", True)
    assert threaded["bound_by"] == "engine"


def test_stats_carry_ticks_per_verdict():
    stats = Stats(bound_by={"engine": 5, "prep": 2})
    assert '"bound_by": {"engine": 5, "prep": 2}' in stats.to_json()
    assert Stats().bound_by == {}


def test_tick_record_carries_the_verdict_and_stage_times():
    rec = tick_record(1.0, 8, 2, None, 0, 3, True, 10.0, "hold", bound_by="prep",
                      source_s=0.0123456, prep_s=0.5)
    assert (rec["bound_by"], rec["source_s"], rec["prep_s"]) == ("prep", 0.012, 0.5)
    rec = tick_record(1.0, 8, 2, None, 0, 3, False, 10.0, "hold")
    assert (rec["bound_by"], rec["source_s"], rec["prep_s"]) == (None, 0.0, 0.0)


def test_advisor_names_the_client_stage_with_its_cost_per_row():
    def ticks(verdict, n, stage_s, ok):
        return [{"bound_by": verdict, "ok": ok, "prep_s": stage_s if verdict == "prep" else 0.0,
                 "source_s": stage_s if verdict == "source" else 0.0} for _ in range(n)]

    prep_run = ticks("prep", 4, 1.0, 5) + ticks("engine", 6, 0.0, 20)  # 40% prep, 20 rows/s of prep
    (hint,) = advise_input(prep_run)
    assert hint == ("PREP-BOUND: to_request took 29 ms/row on average; run it ahead with "
                    "pump(prepare_workers=4)")
    (hint,) = advise_input(ticks("source", 3, 0.6, 3) + ticks(None, 7, 0.0, 0))
    assert hint == "SOURCE-BOUND: the source iterator took 200 ms/row; prefetch or shard the input"
    assert advise_input(ticks("prep", 2, 1.0, 5) + ticks("engine", 8, 0.0, 20)) == []  # 20% < 30%
    assert advise_input([]) == []


class _StubLimiter:
    class window:
        limit = 4

    def __init__(self):
        self.prep = []

    def note_source_wait(self, s):
        pass

    def note_prep(self, s, workers=1):
        assert workers == 2
        self.prep.append(s)


class _EchoClient:
    def __init__(self):
        self.limiter = _StubLimiter()

    async def post(self, request, route="/chat/completions"):
        return {"req": request}, None


def test_prepare_workers_run_to_request_in_threads_and_errors_stay_row_errors():
    import threading

    main = threading.get_ident()
    seen_threads = set()

    def to_request(row):
        seen_threads.add(threading.get_ident())
        if row["n"] == 3:
            raise ValueError("bad row")
        return {"n": row["n"]}

    async def go():
        client = _EchoClient()
        rows = iter([(str(i), {"n": i}) for i in range(8)])
        done = [d async for d in through(client, rows, to_request, lambda r, b: b["req"],
                                         prepare_workers=2)]
        return client, done

    client, done = asyncio.run(go())
    assert main not in seen_threads and seen_threads  # prepared off the loop
    assert len(client.limiter.prep) == 7 and all(s >= 0 for s in client.limiter.prep)  # raise: unmeasured
    by_id = {d.id: d for d in done}
    assert by_id["3"].out is None and by_id["3"].error == "client: ValueError: bad row"
    assert {d.out["n"] for d in done if d.out} == {0, 1, 2, 4, 5, 6, 7}
