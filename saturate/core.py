"""The adaptive middle, transport-agnostic first.

AdaptiveLimiter — window + controller + signals behind slot()/observe(): a
drop-in for a fixed asyncio.Semaphore in someone else's stack (datatrove's
InferenceRunner, a DataDesigner-style admission seam). It never touches HTTP.

AdaptiveClient — AdaptiveLimiter + the HTTP transport (retry ladder, breaker):
`await client.post(request)` with everything handled.

through() — the pipeline's product: an async generator of completed Done
results, in completion order, adaptively concurrent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import statistics
import sys
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

import httpx2 as httpx

from saturate.controller import TICK_S, Auto, Fixed, Obs
from saturate.signals import HttpScrape, Null
from saturate.telemetry import tick_record
from saturate.transport import Breaker, FatalTransportError, Request, call_endpoint, coerce_request
from saturate.window import Window


@dataclasses.dataclass
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

    def _check(self):
        task = self._l._task  # r5: a dead controller fails admissions run-fatally, never row errors
        if task and task.done() and not task.cancelled() and task.exception() is not None:
            raise FatalTransportError(f"controller loop died: {task.exception()!r}")

    async def __aenter__(self):
        self._check()  # before queueing on the window...
        t = time.monotonic()
        await self._l.window.acquire()
        admitted = time.monotonic()
        self._l._wait["acquire"] += admitted - t
        try:
            self._check()  # ...and after (r6): the controller may have died while we were queued
        except BaseException:
            await self._l.window.release()
            raise
        self._l._admitted[self] = admitted

    async def __aexit__(self, *exc):
        self._l._admitted.pop(self, None)  # never in the way of the release: a leaked count wedges admission
        await self._l.window.release()


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
        self.window = Window(getattr(self.controller, "initial", 16))
        self.events = {"backpressure": 0, "successes": 0, "latencies": []}
        self.ticks: list[dict] = []
        self.tokens_total = 0
        self._wait = {"source": 0.0, "acquire": 0.0, "prep": 0.0}  # seconds since the last tick
        self._prep_n = 0  # to_request calls measured since the last tick
        self.prep_workers = 0  # threads running to_request; 0 = on the event loop itself
        self._admitted: dict[_Slot, float] = {}  # admission time of every request in flight
        self._latencies: deque[float] = deque(maxlen=128)  # successful attempts only; recent is what matters
        self._task: asyncio.Task | None = None
        self._t0 = self._t_last = time.monotonic()
        self._last_tokens = 0

    def slot(self) -> _Slot:
        return _Slot(self)

    def observe(self, ok: bool, tokens: int = 0, rate_limited: bool = False,
                latency_s: float | None = None) -> None:
        """Outcome feedback for embedders driving their own transport. `latency_s` is the
        duration of the successful attempt alone (no retries, backoff or breaker waits)."""
        if ok:
            self.events["successes"] += 1
            self.tokens_total += tokens
            if latency_s is not None:
                self.events["latencies"].append(latency_s)
        if rate_limited:
            self.events["backpressure"] += 1

    def note_source_wait(self, seconds: float) -> None:
        """Time the source iterator took to yield a row."""
        self._wait["source"] += seconds

    def note_prep(self, seconds: float) -> None:
        """Wall time one `to_request` call took (where it ran: the loop, or a prepare thread)."""
        self._wait["prep"] += seconds
        self._prep_n += 1

    def _bound_by(self, dt: float, loop_lag_s: float) -> tuple[str | None, bool]:
        """(verdict, input_bound) for this tick. `source` or `prep` when that stage ate at least
        a quarter of the tick (prep per worker) and the window sat under half used — the only
        two verdicts that are input-bound, i.e. evidence the controller must not read as the
        engine's. `prep` also when on-loop `to_request` ate half the tick outright, and `loop`
        when the loop was blocked that long by something else (parse, a sink flush): a blocked
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
        if self.window.inflight < 0.5 * self.window.limit and max(w["source"], prep) >= 0.25 * dt:
            return ("source" if w["source"] >= prep else "prep"), True
        if self.prep_workers == 0 and w["prep"] >= 0.5 * dt:
            return "prep", False
        if loop_lag_s >= 0.5 * dt:
            return "loop", False
        if self.window.inflight >= self.window.limit or w["acquire"] > w["source"] + w["prep"]:
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
        if self._task:
            self._task.cancel()
            try:
                await self._task  # a crashed tick loop surfaces here, never dies silently
            except asyncio.CancelledError:
                pass
            except Exception:
                if not exc or exc[0] is None:
                    raise
                print("[pump] controller loop also failed (primary exception wins)",
                      file=sys.stderr, flush=True)  # r5: never mask the body's exception

    async def _loop(self):
        while True:
            await asyncio.sleep(TICK_S)
            await self._tick()

    async def _tick(self):
        self._latencies.extend(self.events["latencies"])
        self.events["latencies"].clear()
        t_scrape = time.monotonic()
        gauges = await self.signals.read()
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
        observable = self.window.inflight > 0 and self.tokens_total > 0
        latency_s = self.latency_s
        obs = Obs(waiting=g.get("waiting"), running=g.get("running"),
                  inflight=self.window.inflight, backpressure=self.events["backpressure"],
                  successes=self.events["successes"], input_bound=input_bound,
                  kv=g.get("kv"), hits=g.get("hits"), tok_s=tok_s if tok_s or observable else None,
                  preempts=g.get("preempts"), latency_s=latency_s, oldest_s=self.oldest_s,
                  tick_s=dt if dt > 0 else None)
        new_limit = self.controller.decide(obs, self.window.limit)
        rec = tick_record(now - self._t0, self.window.limit, self.window.inflight, gauges,
                          self.events["backpressure"], self.events["successes"], input_bound, tok_s,
                          self.controller.last_reason, latency_s, bound_by=bound_by,
                          source_s=self._wait["source"], prep_s=self._wait["prep"],
                          prep_n=self._prep_n, prep_workers=self.prep_workers, loop_lag_s=loop_lag_s)
        self.ticks.append(rec)
        if self.on_tick:
            self.on_tick(rec)
        self.events["backpressure"] = self.events["successes"] = 0
        self._wait["source"] = self._wait["acquire"] = self._wait["prep"] = 0.0
        self._prep_n = 0
        await self.window.set_limit(new_limit)


class AdaptiveClient:
    """AdaptiveLimiter + HTTP transport. `await client.post(request)` acquires a
    slot, runs the retry ladder + breaker, feeds the controller, and returns
    (body, error). Use as an async context manager."""

    def __init__(self, endpoint: str, window: Fixed | Auto | None = None,
                 headers: dict | None = None, read_timeout: float = 1800.0,
                 signal_source: str = "auto",
                 on_tick: Callable[[dict], None] | None = None):
        self.base = endpoint.rstrip("/")
        self._headers = headers or {}
        self._read_timeout = read_timeout
        self._signal_source = signal_source
        self.breaker = Breaker()
        self._window_arg = window
        self._on_tick = on_tick
        self.limiter: AdaptiveLimiter | None = None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
        timeout = httpx.Timeout(30.0, read=self._read_timeout)
        self._client = httpx.AsyncClient(limits=limits, timeout=timeout, headers=self._headers)
        sig = Null() if self._signal_source == "none" else HttpScrape(self._client, self.base)
        self.limiter = AdaptiveLimiter(self._window_arg, signals=sig, on_tick=self._on_tick)
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
            body, err = await call_endpoint(self._client, self.base, req,
                                            self.limiter.events, self.breaker)
        if err is None:
            usage = body.get("usage") or {}
            self.limiter.tokens_total += (usage.get("prompt_tokens", 0)
                                          + usage.get("completion_tokens", 0))
        return body, err


async def through(client: AdaptiveClient, rows: Iterable[tuple[str, dict]],
                  to_request: Callable, parse: Callable[[dict, dict], dict],
                  route: str = "/chat/completions", prepare_workers: int = 0) -> AsyncIterator[Done]:
    """(id, row) stream -> Done stream, in completion order, adaptively concurrent.

    Row PAYLOADS never materialize beyond two bounds: request bodies in flight are
    capped by the window, and built-but-unadmitted bodies by the prepare bound,
    `max(2 * prepare_workers, 4)`, held from before `to_request` until the window
    admits the request — the same rule on the loop and in threads. The feeder itself
    runs ahead of completions by at most ~2x the window (rows, not bodies). Honest
    bound: the id SETS are in-memory — resume holds the done-set and dedup holds
    admitted ids (~60B/id: fine to ~10M rows, plan shard-scoped done-sets beyond
    that). Source wait and `to_request` time are reported to the limiter so the
    per-tick `bound_by` verdict can name the client side when it is the bottleneck.

    `prepare_workers` > 0 runs `to_request` in that many threads ahead of
    admission; `parse` stays on the loop. A `to_request` error is that row's
    error row either way. A `to_request` still running in a prepare thread when
    the pump aborts cannot be interrupted: the pool is shut down without waiting,
    but the interpreter waits for those calls at exit — so `to_request` must bound
    its own I/O with timeouts.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks: set[asyncio.Task] = set()  # strong refs: asyncio only weak-refs tasks
    fed = 0
    feed_error: list[BaseException] = []
    feeding_done = asyncio.Event()
    pool = ThreadPoolExecutor(prepare_workers, thread_name_prefix="saturate-prepare") \
        if prepare_workers > 0 else None
    prepared = asyncio.Semaphore(max(2 * prepare_workers, 4))  # built bodies awaiting a slot
    loop = asyncio.get_running_loop()
    limiter = client.limiter
    # the bound is released at admission when the client's post offers the hook; a client
    # that predates it (the embedder seam) releases just before posting — a weaker bound
    admit_hook = "on_admit" in inspect.signature(client.post).parameters
    note_prep = getattr(limiter, "note_prep", None)  # an embedder's own limiter may predate it
    if hasattr(limiter, "prep_workers"):
        limiter.prep_workers = prepare_workers

    def prepare(row: dict) -> tuple[Request | dict, float]:
        t = time.monotonic()  # wall time where it runs: on the loop, or in a prepare worker
        return to_request(row), time.monotonic() - t

    async def worker(id_: str, row: dict):
        holding = False  # the prepare bound, from the start of prepare until admission

        def admitted() -> None:
            nonlocal holding
            if holding:
                prepared.release()
                holding = False

        try:
            await prepared.acquire()
            holding = True
            req, took = await loop.run_in_executor(pool, prepare, row) if pool else prepare(row)
            if note_prep is not None:
                note_prep(took)
            if admit_hook:
                body, err = await client.post(req, route, on_admit=admitted)
            else:
                admitted()
                body, err = await client.post(req, route)
            if err is None:
                out = parse(row, body)
                if not isinstance(out, dict):  # storage contract: rows are dicts
                    raise TypeError(f"parse must return a dict, got {type(out).__name__}")
                await queue.put(Done(id_, row, out, None, body.get("usage") or {}))
            else:
                await queue.put(Done(id_, row, None, err))
        except FatalTransportError as e:  # run-fatal: surfaced to the consumer, never a row error
            await queue.put(e)
        except Exception as e:  # to_request/parse bugs become error results, not lost rows
            await queue.put(Done(id_, row, None, f"client: {type(e).__name__}: {e}"))
        finally:
            admitted()  # never strand the bound: a failed or fatal row releases it too

    async def feed():
        nonlocal fed
        t_prev = time.monotonic()
        try:
            for id_, row in rows:
                limiter.note_source_wait(time.monotonic() - t_prev)
                fed += 1
                t = asyncio.create_task(worker(id_, row))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                while len(tasks) > limiter.window.limit * 2 + 8:
                    await asyncio.sleep(0.01)  # feed-ahead bound (window-scaled)
                t_prev = time.monotonic()
        except BaseException as e:  # a crashing source must still release the consumer
            feed_error.append(e)
        finally:
            feeding_done.set()

    feeder = asyncio.create_task(feed())
    served = 0
    try:
        while not (feeding_done.is_set() and served >= fed):
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            served += 1
            if isinstance(item, BaseException):
                raise item  # fatal transport: abort the pump, no durable error rows
            yield item
        if feed_error:
            raise feed_error[0]
    finally:
        feeder.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(feeder, *tasks, return_exceptions=True)  # reap, don't abandon
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)  # a running to_request still finishes
