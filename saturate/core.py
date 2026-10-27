"""The adaptive middle, transport-agnostic first.

AdaptiveLimiter — window + controller + signals behind slot()/observe(): a
drop-in for a fixed asyncio.Semaphore in someone else's stack (datatrove's
InferenceRunner, a DataDesigner-style admission seam). It never touches HTTP.

AdaptiveClient — AdaptiveLimiter + the HTTP transport (retry ladder, breaker):
`await client.post(request)` with everything handled.

through() — the pipeline itself — lives in saturate/pipeline.py (docs/pipeline.md).
"""

from __future__ import annotations

import asyncio
import dataclasses
import statistics
import sys
import time
from collections import deque
from collections.abc import Callable

import anyio

from saturate import transport
from saturate.controller import TICK_S, Auto, Fixed, Obs
from saturate.http import HttpClient, Timeouts
from saturate.signals import HttpScrape, Null
from saturate.telemetry import tick_record
from saturate.transport import Attempt, Breaker, FatalTransportError, Request, call_endpoint, coerce_request


@dataclasses.dataclass(slots=True)
class _TickCounts:
    """What the attempts reported since the last tick."""

    successes: int = 0
    backpressure: int = 0  # rows, not attempts: each row reports its pressure once
    retries: int = 0
    latencies: list[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class Done:
    """One completed row: out is the parse result on success, None on error."""

    id: str
    row: dict
    out: dict | None
    error: str | None
    usage: dict = dataclasses.field(default_factory=dict)


class _Slot:
    def __init__(self, limiter: AdaptiveLimiter):
        self._l = limiter
        self._entered = False  # a slot is entered once: the limiter keys its waiters by slot

    def _check(self):
        """A controller loop that has ended while the limiter is open fails admissions for the
        whole run (never as row errors): crashed, or cancelled by someone other than __aexit__."""
        task = self._l._task
        if task is None or not task.done() or self._l._closing:
            return
        if task.cancelled():
            raise FatalTransportError("controller loop was cancelled while the run was going")
        if task.exception() is not None:
            raise FatalTransportError(f"controller loop died: {task.exception()!r}")

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, *exc):
        self.release()

    async def acquire(self) -> None:
        """Wait for a free slot. `async with limiter.slot():` calls this and release() for you;
        call them directly to hand a slot from the task that acquired it to another task."""
        if self._entered:
            raise RuntimeError("a limiter slot is entered once; take a new one with limiter.slot()")
        self._check()  # before queueing on the window...
        self._entered = True
        t = time.monotonic()
        capacity = self._l._capacity
        # the slot itself is the borrower, not the task: one task may hold several slots
        try:
            # A free token is taken without awaiting. anyio <= 4.15.1 releases the wrong borrower
            # when a native asyncio cancellation lands right after an immediate acquisition
            # (fixed upstream in anyio 9e6b71e, not yet released; removal tracked in #52).
            capacity.acquire_on_behalf_of_nowait(self)
        except anyio.WouldBlock:
            try:
                await capacity.acquire_on_behalf_of(self)
            except BaseException:
                self._entered = False
                raise
        admitted = time.monotonic()
        self._l._wait["acquire"] += admitted - t
        try:
            self._check()  # ...and after (r6): the controller may have died while we were queued
        except BaseException:
            self._give_back()
            raise
        self._l._admitted[self] = admitted

    def release(self) -> None:
        """Give the slot back; any task may call it. Synchronous: no cancellation can interrupt it."""
        self._l._admitted.pop(self, None)  # never in the way of the release: a leaked count wedges admission
        self._give_back()

    def _give_back(self) -> None:
        self._l._capacity.release_on_behalf_of(self)
        self._entered = False


class AdaptiveLimiter:
    """Adaptive admission: `async with limiter.slot(): ...; limiter.observe(...)`.

    Signals are optional (an HttpScrape, or None for blind mode). The tick loop
    runs while the limiter's context is open and drives the controller from
    observed outcomes + delivered tokens + gauges.
    """

    def __init__(self, window: Fixed | Auto | None = None, signals=None,
                 on_tick: Callable[[dict], None] | None = None):
        self.controller = window or Auto()
        self.signals = signals or Null()
        self.on_tick = on_tick
        # the admission window: the controller resizes it each tick. Shrinking never evicts a
        # holder; it holds new admissions until fewer than the new limit are in flight
        self._capacity = anyio.CapacityLimiter(int(getattr(self.controller, "initial", 16)))
        self._counts = _TickCounts()
        self.retries_total = 0  # retry attempts over the run (backpressure counts rows, not attempts)
        self.ticks: list[dict] = []
        self.tokens_total = 0
        self._wait = {"source": 0.0, "acquire": 0.0, "prep": 0.0, "sink": 0.0}  # seconds since the last tick
        self._prep_n = 0  # to_request calls measured since the last tick
        self.prep_workers = 0  # threads running to_request; 0 = on the event loop itself
        self._admitted: dict[_Slot, float] = {}  # admission time of every request in flight
        self._latencies: deque[float] = deque(maxlen=128)  # successful attempts only; recent is what matters
        self._task: asyncio.Task | None = None
        self._closing = False  # __aexit__ has begun: the controller loop is being stopped
        self._t0 = self._t_last = time.monotonic()
        self._last_tokens = 0

    def slot(self) -> _Slot:
        return _Slot(self)

    @property
    def running(self) -> bool:
        """Whether the tick loop is on: the limiter is inside its `async with`."""
        return self._task is not None and not self._task.done()

    @property
    def limit(self) -> int:
        """The admission window: requests that may be in flight at once."""
        return int(self._capacity.total_tokens)

    @property
    def inflight(self) -> int:
        """Requests holding a slot now."""
        return self._capacity.borrowed_tokens

    def observe(self, attempt: Attempt) -> None:
        """The one way outcomes reach the controller: AdaptiveClient's transport calls it after
        every attempt, and an embedder driving its own transport does the same. Report each
        attempt as it ends, not when the row is done, and set `pressure` at most once per row,
        even across its retries: the controller cuts on the share of rows under pressure, and
        per-attempt pressure would let one row that always fails cut the window on every retry."""
        counts = self._counts
        if attempt.ok:
            counts.successes += 1
            self.tokens_total += attempt.tokens
            if attempt.latency_s is not None:
                counts.latencies.append(attempt.latency_s)
        if attempt.pressure:
            counts.backpressure += 1
        if attempt.retry:
            counts.retries += 1
            self.retries_total += 1

    def note_source_wait(self, seconds: float) -> None:
        """Time the source iterator took to yield a row."""
        self._wait["source"] += seconds

    def note_sink_wait(self, seconds: float) -> None:
        """Time a new row waited for room in the backlog: the output, not the engine, was behind."""
        self._wait["sink"] += seconds

    def note_prep(self, seconds: float) -> None:
        """Wall time one `to_request` call took (where it ran: the loop, or a prepare thread)."""
        self._wait["prep"] += seconds
        self._prep_n += 1

    def _bound_by(self, dt: float, loop_lag_s: float) -> tuple[str | None, bool]:
        """(verdict, input_bound) for this tick. `source` or `prep` when that stage ate at least
        a quarter of the tick (prep per worker) and the window sat under half used, and `sink`
        when new rows waited that long for the output — the only verdicts that are input-bound,
        i.e. evidence the controller must not read as the engine's. `prep` also when on-loop
        `to_request` ate half the tick outright, and `loop` when the loop was blocked that long
        by something else (parse): a blocked
        loop holds slots without serving them, so the window count says nothing — but the
        window IS full, so these are not input-bound. `engine` when the window was full or
        admission waits dominated; None when nothing is conclusive (idle).

        Per-worker occupancy is an average over the prepare threads: one saturated lane (a
        single huge video row) can read under the threshold while the others idle. A
        per-lane maximum would catch that; it is a possible refinement, not done here."""
        if dt <= 0:
            return None, False
        w = self._wait
        prep = w["prep"] / max(1, self.prep_workers)  # per-worker occupancy of the prepare stage
        # the stages outside the engine that can starve admission: the input, building requests,
        # and the output (new rows wait for room in the backlog when writing falls behind)
        outside = {"source": w["source"], "prep": prep, "sink": w["sink"]}
        stage = max(outside, key=outside.__getitem__)  # ties go to the first: source, prep, sink
        if self.inflight < 0.5 * self.limit and outside[stage] >= 0.25 * dt:
            return stage, True
        if self.prep_workers == 0 and w["prep"] >= 0.5 * dt:
            return "prep", False
        if loop_lag_s >= 0.5 * dt:
            return "loop", False
        if self.inflight >= self.limit or w["acquire"] > w["source"] + w["prep"]:
            return "engine", False
        return None, False

    @property
    def latency_s(self) -> float | None:
        """p50 duration of recent successful attempts (None before the first)."""
        return statistics.median(self._latencies) if self._latencies else None

    @property
    def oldest_s(self) -> float | None:
        """Age of the oldest request in flight (None when nothing is in flight)."""
        return time.monotonic() - min(self._admitted.values()) if self._admitted else None

    async def __aenter__(self):
        self._t0 = self._t_last = time.monotonic()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task is None:
            return
        self._closing = True
        self._task.cancel()
        # asyncio.wait returns when the loop ends and never raises the loop's own cancellation;
        # a cancel of the caller itself still propagates from here
        await asyncio.wait({self._task})
        if self._task.cancelled():
            return
        failure = self._task.exception()  # a crashed tick loop surfaces here, never dies silently
        if failure is None:
            return
        if not exc or exc[0] is None:
            raise failure
        print("[pump] controller loop also failed (primary exception wins)",
              file=sys.stderr, flush=True)  # never mask the body's exception

    async def _loop(self):
        while True:
            await asyncio.sleep(TICK_S)
            await self._tick()

    async def _tick(self):
        t_scrape = time.monotonic()
        gauges = await self.signals.read()
        # everything since the last tick, read together with no await in between: reports that
        # arrived during the scrape count in this decision, and a cancelled scrape loses none
        counts, self._counts = self._counts, _TickCounts()
        self._latencies.extend(counts.latencies)
        now = time.monotonic()  # actual elapsed, not TICK_S: scrape latency skews the rate
        dt = now - self._t_last
        tok_s = (self.tokens_total - self._last_tokens) / dt if dt > 0 else 0.0
        self._last_tokens, self._t_last = self.tokens_total, now
        # the tick slept TICK_S and scraped; anything beyond that is time the loop was blocked
        loop_lag_s = max(0.0, dt - TICK_S - (now - t_scrape))
        bound_by, input_bound = self._bound_by(dt, loop_lag_s)
        g = gauges or {}
        # zero tokens with requests in flight is a reading (a stalled or slow engine);
        # None means unobservable: nothing in flight, or an endpoint that never reports usage
        observable = self.inflight > 0 and self.tokens_total > 0
        latency_s = self.latency_s
        obs = Obs(waiting=g.get("waiting"), running=g.get("running"),
                  inflight=self.inflight, backpressure=counts.backpressure,
                  successes=counts.successes, input_bound=input_bound,
                  kv=g.get("kv"), hits=g.get("hits"), tok_s=tok_s if tok_s or observable else None,
                  preempts=g.get("preempts"), latency_s=latency_s, oldest_s=self.oldest_s,
                  tick_s=dt if dt > 0 else None)
        new_limit = self.controller.decide(obs, self.limit)
        rec = tick_record(now - self._t0, self.limit, self.inflight, gauges,
                          counts.backpressure, counts.successes, input_bound, tok_s,
                          self.controller.last_reason, latency_s, bound_by=bound_by,
                          source_s=self._wait["source"], prep_s=self._wait["prep"],
                          sink_s=self._wait["sink"],
                          prep_n=self._prep_n, prep_workers=self.prep_workers, loop_lag_s=loop_lag_s,
                          retries=counts.retries)
        self.ticks.append(rec)
        if self.on_tick:
            self.on_tick(rec)
        self._wait = dict.fromkeys(self._wait, 0.0)
        self._prep_n = 0
        self._capacity.total_tokens = int(new_limit)  # the setter takes an int, never a float


class AdaptiveClient:
    """AdaptiveLimiter + HTTP transport. `await client.post(request)` acquires a
    slot, runs the retry ladder + breaker, feeds the controller, and returns
    (body, error). Use as an async context manager."""

    def __init__(self, endpoint: str, window: Fixed | Auto | None = None,
                 headers: dict | None = None, read_timeout: float = 1800.0,
                 signal_source: str = "auto",
                 on_tick: Callable[[dict], None] | None = None,
                 retry_budget_s: float | None = None,
                 timeouts: dict[str, float] | None = None,
                 proxy: str | bool | None = None):
        self.base = endpoint.rstrip("/")
        # a row's attempts share one wall-clock budget; it must hold a retry of a long request,
        # so by default it grows with the read timeout (300 s floor: 3600 s at the default 1800)
        self.retry_budget_s = (retry_budget_s if retry_budget_s is not None
                               else max(transport.RETRY_BUDGET_S, 2 * read_timeout))
        unknown = set(timeouts or {}) - {"connect", "write", "pool"}
        if unknown:
            raise ValueError(f"timeouts: unknown phase(s) {sorted(unknown)}; the read timeout is "
                             "read_timeout=, the others are 'connect', 'write' and 'pool'")
        self._timeouts = Timeouts(read=read_timeout, **(timeouts or {}))
        self._headers, self._proxy = headers or {}, proxy
        self._signal_source, self._window, self._on_tick = signal_source, window, on_tick
        self.breaker = Breaker()
        self._client, self.limiter = self._build()

    def _build(self) -> tuple[HttpClient, AdaptiveLimiter]:
        """A fresh HTTP client and limiter. Built at construction, so neither is ever None; the
        client opens no connection and the limiter starts no tick loop until `async with`."""
        client = HttpClient(headers=self._headers, timeouts=self._timeouts, proxy=self._proxy)
        signals = Null() if self._signal_source == "none" else HttpScrape(client, self.base)
        return client, AdaptiveLimiter(self._window, signals=signals, on_tick=self._on_tick)

    async def __aenter__(self):
        if self._client.closed:  # entered again after an earlier `async with`: start fresh
            self._client, self.limiter = self._build()
        await self.limiter.__aenter__()
        return self

    async def __aexit__(self, *exc):
        try:
            await self.limiter.__aexit__(*exc)  # may re-raise a crashed tick loop
        finally:
            await self._client.aclose()  # ... which must never leak the HTTP client

    @property
    def dialect(self) -> str | None:
        return getattr(self.limiter.signals, "dialect", None)

    async def post(self, request: Request | dict, route: str = "/chat/completions",
                   on_admit: Callable[[], None] | None = None) -> tuple[dict | None, str | None]:
        """`on_admit` fires once the request holds a window slot (through() releases its
        prepared-body bound there: from then on the body is in flight, not waiting)."""
        req = coerce_request(request, route)
        await self.breaker.gate(self._client, f"{self.base}{req.route}", (req.json or {}).get("model"))
        async with self.limiter.slot():
            if on_admit is not None:
                on_admit()
            return await call_endpoint(self._client, self.base, req, self.breaker,
                                       self.limiter.observe, self.retry_budget_s)
