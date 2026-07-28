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
import contextlib
import dataclasses
import time
from collections.abc import AsyncIterator, Callable, Iterable

import httpx2 as httpx

from pumpjack.controller import Auto, Fixed, Obs
from pumpjack.signals import HttpScrape, Null
from pumpjack.telemetry import tick_record
from pumpjack.transport import Breaker, FatalTransportError, Request, call_endpoint, coerce_request
from pumpjack.window import Window

TICK_S = 2.0


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

    async def __aenter__(self):
        t = time.monotonic()
        await self._l.window.acquire()
        self._l._wait["acquire"] += time.monotonic() - t

    async def __aexit__(self, *exc):
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
        self.events = {"backpressure": 0, "successes": 0}
        self.ticks: list[dict] = []
        self.tokens_total = 0
        self._wait = {"source": 0.0, "acquire": 0.0}
        self.input_bound_ever = False
        self._task: asyncio.Task | None = None
        self._t0 = time.monotonic()

    def slot(self) -> _Slot:
        return _Slot(self)

    def observe(self, ok: bool, tokens: int = 0, rate_limited: bool = False) -> None:
        """Outcome feedback for embedders driving their own transport."""
        if ok:
            self.events["successes"] += 1
            self.tokens_total += tokens
        if rate_limited:
            self.events["backpressure"] += 1

    def note_source_wait(self, seconds: float) -> None:
        self._wait["source"] += seconds

    async def __aenter__(self):
        self._t0 = time.monotonic()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task  # a crashed tick loop surfaces here, never dies silently

    async def _loop(self):
        last_tokens = 0
        while True:
            await asyncio.sleep(TICK_S)
            gauges = await self.signals.read()
            tok_s = (self.tokens_total - last_tokens) / TICK_S
            last_tokens = self.tokens_total
            total_wait = self._wait["source"] + self._wait["acquire"]
            input_bound = (total_wait > 0 and self._wait["source"] / total_wait > 0.5
                           and self.window.inflight < int(self.window.limit * 0.5))
            self.input_bound_ever = self.input_bound_ever or input_bound
            g = gauges or {}
            obs = Obs(waiting=g.get("waiting"), running=g.get("running"),
                      inflight=self.window.inflight, backpressure=self.events["backpressure"],
                      successes=self.events["successes"], input_bound=input_bound,
                      kv=g.get("kv"), hits=g.get("hits"), tok_s=tok_s or None,
                      preempts=g.get("preempts"))
            rec = tick_record(time.monotonic() - self._t0, self.window.limit,
                              self.window.inflight, gauges, self.events["backpressure"],
                              self.events["successes"], input_bound, tok_s)
            self.ticks.append(rec)
            if self.on_tick:
                self.on_tick(rec)
            self.events["backpressure"] = self.events["successes"] = 0
            self._wait["source"] = self._wait["acquire"] = 0.0
            await self.window.set_limit(self.controller.decide(obs, self.window.limit))


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
        await self.limiter.__aexit__(*exc)
        await self._client.aclose()

    @property
    def dialect(self) -> str | None:
        return getattr(self.limiter.signals, "dialect", None)

    async def post(self, request: Request | dict, route: str = "/chat/completions"
                   ) -> tuple[dict | None, str | None]:
        req = coerce_request(request, route)
        await self.breaker.gate(self._client, f"{self.base}/chat/completions")
        async with self.limiter.slot():
            body, err = await call_endpoint(self._client, self.base, req,
                                            self.limiter.events, self.breaker)
        if err is None:
            usage = body.get("usage") or {}
            self.limiter.tokens_total += (usage.get("prompt_tokens", 0)
                                          + usage.get("completion_tokens", 0))
        return body, err


async def through(client: AdaptiveClient, rows: Iterable[tuple[str, dict]],
                  to_request: Callable, parse: Callable[[dict, dict], dict],
                  route: str = "/chat/completions") -> AsyncIterator[Done]:
    """(id, row) stream -> Done stream, in completion order, adaptively concurrent.

    The feeder runs ahead of completions by at most ~2x the current window, so
    row PAYLOADS never materialize. Honest bound: the id SETS are in-memory —
    resume holds the done-set and dedup holds admitted ids (~60B/id: fine to
    ~10M rows, plan shard-scoped done-sets beyond that). Source wait time is
    reported to the limiter so input-bound detection keeps working.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks: set[asyncio.Task] = set()  # strong refs: asyncio only weak-refs tasks
    fed = 0
    feed_error: list[BaseException] = []
    feeding_done = asyncio.Event()

    async def worker(id_: str, row: dict):
        try:
            body, err = await client.post(to_request(row), route)
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

    async def feed():
        nonlocal fed
        t_prev = time.monotonic()
        try:
            for id_, row in rows:
                client.limiter.note_source_wait(time.monotonic() - t_prev)
                fed += 1
                t = asyncio.create_task(worker(id_, row))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                while len(tasks) > client.limiter.window.limit * 2 + 8:
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
