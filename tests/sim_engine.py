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
    as an engine without /metrics does. `noise` is the relative spread of each reading."""

    curve: list[tuple[int, float]]
    cap: int = 256
    rate_limit: int | None = None
    gauges: bool = True
    noise: float = 0.0
    read_timeout_s: float = 1800.0
    tokens_per_row: float = 100.0
    refill_s: float = 0.0  # a slot's idle time between a completion and the client's next request
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


def run(ctrl, engine: Engine, ticks: int, limit: int | None = None,
        then: tuple[int, Engine] | None = None) -> list[Tick]:
    """Drive `ctrl` for `ticks` ticks against `engine`, the source never starving the window.
    A tick's reading describes the window from about one request latency before. Completions
    are counted with their fractions carried, so a slow engine still reports some."""
    rng = random.Random(engine.seed)
    limit = ctrl.initial if limit is None else limit
    history = [limit]
    out = []
    carry = timeout_carry = 0.0
    for i in range(ticks):
        if then is not None and i == then[0]:
            engine = then[1]
        lag = max(1, math.ceil(engine.latency_s(history[-1]) / TICK_S))
        seen = history[max(0, len(history) - 1 - lag)]
        delivered = engine.rows_per_s(seen)
        jitter = rng.gauss(1.0, engine.noise) if engine.noise else 1.0
        rate = delivered * max(0.0, jitter)
        carry += rate * TICK_S
        successes, carry = int(carry), carry - int(carry)
        rejected = 0
        if engine.rate_limit is not None and seen > engine.rate_limit:
            rejected = round((seen - engine.rate_limit) / engine.latency_s(seen) * TICK_S) or 1
        # past the timeout each request fails once per read timeout, not once per tick
        if delivered == 0.0 and seen > 0:
            timeout_carry += seen * TICK_S / engine.read_timeout_s
        timed_out, timeout_carry = int(timeout_carry), timeout_carry - int(timeout_carry)
        queued = max(0, limit - engine.cap)
        obs = dict(
            waiting=queued if engine.gauges else None,
            running=min(limit, engine.cap) if engine.gauges else None,
            inflight=limit, backpressure=rejected + timed_out,
            successes=successes, input_bound=False,
            kv=0.3 if engine.gauges else None, hits=0.9 if engine.gauges else None,
            tok_s=rate * engine.tokens_per_row,
            latency_s=engine.latency_s(seen) if rate else None,
        )
        limit = ctrl.decide(obs, limit)
        history.append(limit)
        out.append(Tick(limit, ctrl.last_reason, delivered, engine.latency_s(seen), rejected + timed_out,
                        engine))
    return out


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
