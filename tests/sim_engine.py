"""A simulated inference engine for controller tests.

The engine has a throughput curve (rows/s by number of running requests), a cap on running
requests (the rest wait in its queue), and optionally a hosted API's concurrency limit (requests
past it are rejected with 429s). Latency follows Little's law: everything in flight, queued
requests included, shares the engine's throughput, so a request takes window / rate seconds,
and a request slower than the read timeout times out. Readings lag the window by that latency,
as a real engine's completions do, and carry multiplicative noise.

`run(ctrl, engine, ticks)` drives a controller tick by tick and records what each tick delivered
(rows completed, at the window they were admitted under), the latency they saw and the rows that
failed, so a test can compare Auto with the best fixed window. `then=(tick, engine)` switches to
another engine mid-run: a capacity or prompt-length change.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import random

from saturate.controller import TICK_S


@dataclasses.dataclass
class Engine:
    """`curve` maps running requests to rows/s at the points given, linear in between and flat
    past the last point (a falling last point models preemption past a batch size). `cap` is
    the most requests the engine runs at once. `rate_limit` is a hosted API's concurrency
    limit: past it, the excess requests are rejected with 429s. `gauges=False` hides the queue,
    as an engine without /metrics does. `noise` is the relative spread of each reading.
    `kv_cap` sets the KV gauge, running / kv_cap (None: a small model, the cache never near
    full). Past `preempt_past` running requests (kv_cap when None) the engine preempts, one
    request a tick per eighth over, counted when `counts_preempts` (vLLM and SGLang do; others
    do not). `hits` is the prefix-cache hit rate (unique prompts: low)."""

    curve: list[tuple[int, float]]
    cap: int = 256
    rate_limit: int | None = None
    gauges: bool = True
    noise: float = 0.0
    read_timeout_s: float = 1800.0
    tokens_per_row: float = 100.0
    refill_s: float = 0.0  # a slot's idle time between a completion and the client's next request
    kv_cap: int | None = None
    preempt_past: int | None = None  # running requests past which it preempts (None: kv_cap)
    counts_preempts: bool = True
    hits: float = 0.9
    seed: int = 0
    _memo: dict = dataclasses.field(default_factory=dict, repr=False, compare=False)

    def _curve_rate(self, running: float) -> float:
        xs = [x for x, _ in self.curve]
        if running <= xs[0]:
            return self.curve[0][1] * running / xs[0]
        if running >= xs[-1]:
            return self.curve[-1][1]
        i = bisect.bisect_right(xs, running)
        (x0, y0), (x1, y1) = self.curve[i - 1], self.curve[i]
        return y0 + (y1 - y0) * (running - x0) / (x1 - x0)

    def rows_per_s(self, window: int) -> float:
        """Throughput of a fixed window, past any rate limit and timeouts: what Auto competes with.
        With `refill_s`, each request leaves its slot idle for that long before the client's next
        one arrives, unless one already waits in the queue: a window of exactly the cap keeps
        fewer than `cap` requests running, and a small queue on top fills the gaps."""
        admitted = min(window, self.rate_limit or window)
        running = min(admitted, self.cap)
        if self.refill_s and running:
            service_s = running / self._curve_rate(running)  # a request's time in a slot
            running = min(self.cap, admitted * service_s / (service_s + self.refill_s))
        rate = self._curve_rate(running)
        if rate and admitted / rate > self.read_timeout_s:
            return 0.0  # every request waits past the timeout
        return rate

    def latency_s(self, window: int) -> float:
        """Little's law: what is admitted (up to a rate limit) shares the throughput."""
        rate = self.rows_per_s(window)
        admitted = min(window, self.rate_limit or window)
        return admitted / rate if rate else self.read_timeout_s

    def best(self, max_window: int) -> tuple[int, float]:
        """(smallest window reaching the best throughput, that throughput)."""
        if ("best", max_window) not in self._memo:
            rates = [self.rows_per_s(w) for w in range(1, max_window + 1)]
            top = max(rates)
            self._memo["best", max_window] = next(w for w, r in enumerate(rates, 1) if r >= top * 0.999), top
        return self._memo["best", max_window]

    def useful_window(self, max_window: int) -> int:
        """The smallest window within 2% of the best throughput: anything above it only queues."""
        if ("useful", max_window) not in self._memo:
            _, top = self.best(max_window)
            self._memo["useful", max_window] = next(
                w for w in range(1, max_window + 1) if self.rows_per_s(w) >= top * 0.98)
        return self._memo["useful", max_window]


@dataclasses.dataclass
class Tick:
    limit: int
    reason: str
    rows_per_s: float  # rows delivered this tick (from the window `latency` ago), per second
    latency_s: float  # what those rows waited, queue included
    failed: int  # rows rejected (429) or timed out this tick
    engine: Engine  # the engine this tick ran against


def run(ctrl, engine: Engine, ticks: int, then: tuple[int, Engine] | None = None) -> list[Tick]:
    """Drive `ctrl` for `ticks` ticks against `engine`, the source never starving the window.
    A tick's reading describes the window from about one request latency before. Completions
    are counted with their fractions carried, so a slow engine still reports some."""
    return run_shared([ctrl], engine, ticks, then=then)[0]


def run_shared(ctrls: list, engine: Engine, ticks: int, starts: list[int] | None = None,
               then: tuple[int, Engine] | None = None) -> list[list[Tick]]:
    """Drive several controllers against one engine, as sharded Jobs pumping one endpoint do.
    Client i joins at tick `starts[i]` (default: all at once). Everything admitted shares the
    engine: the running cap, the queue, the throughput (split by each client's share of what is
    in flight) and a hosted API's 429 limit (rejections split the same way). Each client sees
    its own completions, the shared queue gauges and the shared latency; each reading has its
    own noise. `then=(tick, engine)` switches engine mid-run. Returns each client's ticks; a tick
    before a client joins has limit 0."""
    starts = starts or [0] * len(ctrls)
    rngs = [random.Random(engine.seed + 1000 * i) for i in range(len(ctrls))]  # client 0 draws as run() does
    histories = [[0] for _ in ctrls]
    limits = [0] * len(ctrls)
    carries = [0.0] * len(ctrls)
    timeout_carries = [0.0] * len(ctrls)
    preempts = 0  # the engine's cumulative preemption counter
    out: list[list[Tick]] = [[] for _ in ctrls]
    for i in range(ticks):
        if then is not None and i == then[0]:
            engine = then[1]
        for c, ctrl in enumerate(ctrls):
            if i == starts[c]:
                limits[c] = ctrl.initial
                histories[c][-1] = limits[c]
        total_now = sum(limits)
        lag = max(1, math.ceil(engine.latency_s(total_now) / TICK_S)) if total_now else 1
        seen = [h[max(0, len(h) - 1 - lag)] for h in histories]
        total_seen = sum(seen)
        delivered_total = engine.rows_per_s(total_seen) if total_seen else 0.0
        latency = engine.latency_s(total_seen) if total_seen else 0.0
        running = min(total_seen, engine.cap)
        kv = 0.3 if engine.kv_cap is None else min(1.0, running / engine.kv_cap)
        past = engine.preempt_past or engine.kv_cap
        if past is not None and running > past:
            preempts += math.ceil((running - past) / max(1, past // 8))
        rejected_total = 0.0
        if engine.rate_limit is not None and total_seen > engine.rate_limit:
            rejected_total = (total_seen - engine.rate_limit) / latency * TICK_S
        for c, ctrl in enumerate(ctrls):
            if i < starts[c]:
                out[c].append(Tick(0, "absent", 0.0, 0.0, 0, engine))
                histories[c].append(0)
                continue
            share = seen[c] / total_seen if total_seen else 0.0
            delivered = delivered_total * share
            jitter = rngs[c].gauss(1.0, engine.noise) if engine.noise else 1.0
            rate = delivered * max(0.0, jitter)
            carries[c] += rate * TICK_S
            successes, carries[c] = int(carries[c]), carries[c] - int(carries[c])
            rejected = round(rejected_total * share) or (1 if rejected_total and seen[c] else 0)
            if delivered_total == 0.0 and seen[c] > 0:
                timeout_carries[c] += seen[c] * TICK_S / engine.read_timeout_s
            timed_out = int(timeout_carries[c])
            timeout_carries[c] -= timed_out
            queued = max(0, total_now - engine.cap)
            obs = dict(
                waiting=queued if engine.gauges else None,
                running=min(total_now, engine.cap) if engine.gauges else None,
                inflight=limits[c], backpressure=rejected + timed_out,
                successes=successes, input_bound=False,
                kv=kv if engine.gauges else None, hits=engine.hits if engine.gauges else None,
                preempts=preempts if engine.gauges and engine.counts_preempts else None,
                tok_s=rate * engine.tokens_per_row,
                latency_s=latency if rate else None,
            )
            limits[c] = ctrl.decide(obs, limits[c])
            histories[c].append(limits[c])
            out[c].append(Tick(limits[c], ctrl.last_reason, delivered, latency, rejected + timed_out, engine))
    return out


def shared_share_of_best(clients: list[list[Tick]], start: int = 0, max_window: int = 512) -> float:
    """The clients' summed throughput from tick `start` on, as a share of the best fixed total
    window's."""
    engine = clients[0][0].engine
    ticks = len(clients[0])
    total = sum(sum(c[i].rows_per_s for c in clients) for i in range(start, ticks))
    return total / (ticks - start) / engine.best(max_window)[1]


def cuts_per_minute(clients: list[list[Tick]], start: int = 0) -> float:
    """Cuts and reverts per client per minute: how much the windows saw-tooth."""
    moves = sum(t.reason.startswith(("cut", "revert")) for c in clients for t in c[start:])
    minutes = (len(clients[0]) - start) * TICK_S / 60
    return moves / len(clients) / minutes


def share_of_best(ticks: list[Tick], engine: Engine | None = None, max_window: int = 512) -> float:
    """Mean delivered throughput over `ticks` as a share of the best fixed window's, each tick
    against the engine it ran on (`engine` is accepted for symmetry and unused)."""
    return sum(t.rows_per_s / t.engine.best(max_window)[1] for t in ticks) / len(ticks)


def share_past_useful(ticks: list[Tick], engine: Engine | None = None, max_window: int = 512) -> float:
    """Share of `ticks` spent with the window more than an eighth past the useful window: the
    excess requests only wait (or are rejected)."""
    return sum(t.limit > t.engine.useful_window(max_window) * 9 / 8 + 1 for t in ticks) / len(ticks)


def latency_inflation(ticks: list[Tick], max_window: int = 512) -> float:
    """Mean latency over `ticks` as a multiple of the latency at the useful window."""
    return sum(t.latency_s / t.engine.latency_s(t.engine.useful_window(max_window))
               for t in ticks) / len(ticks)


def failed_share(ticks: list[Tick]) -> float:
    """Rows rejected or timed out, as a share of all rows completed or failed."""
    failed = sum(t.failed for t in ticks)
    done = sum(t.rows_per_s * TICK_S for t in ticks)
    return failed / (failed + done) if failed + done else 0.0


# SGLang on an A10G (2026-10-27 Jobs measurements, shape only): throughput rises to a window of
# 32, is flat from 32 to 64, rises again to about 192, and SGLang runs at most 256 requests.
SGLANG_LIKE = [(32, 64.0), (64, 68.0), (192, 150.0)]
LINEAR = [(256, 200.0)]  # throughput grows with the window up to the engine's cap
PREEMPTING = [(64, 100.0), (192, 160.0), (320, 110.0)]  # past 192, preemption costs throughput
