"""A simulated inference engine for controller tests.

The engine has a throughput curve (rows/s by number of running requests), a cap on running
requests (the rest wait in its queue), and optionally a hosted API's concurrency limit (requests
past it are rejected with 429s). Latency follows Little's law: everything in flight, queued
requests included, shares the engine's throughput, so a request takes window / rate seconds,
and a request slower than the read timeout times out. Readings lag the window by that latency,
as a real engine's completions do, and carry multiplicative noise.

`run(ctrl, engine, ticks)` drives a controller tick by tick and records what happened, so a test
can compare Auto with the best fixed window and count the ticks it spent past the engine's cap.
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
    seed: int = 0

    def rows_per_s(self, window: int) -> float:
        """Throughput of a fixed window, past any rate limit and timeouts: what Auto competes with."""
        served = min(window, self.cap, self.rate_limit or window)
        xs = [x for x, _ in self.curve]
        if served <= xs[0]:
            rate = self.curve[0][1] * served / xs[0]
        elif served >= xs[-1]:
            rate = self.curve[-1][1]
        else:
            i = bisect.bisect_right(xs, served)
            (x0, y0), (x1, y1) = self.curve[i - 1], self.curve[i]
            rate = y0 + (y1 - y0) * (served - x0) / (x1 - x0)
        if rate and min(window, self.rate_limit or window) / rate > self.read_timeout_s:
            return 0.0  # every request waits past the timeout
        return rate

    def latency_s(self, window: int) -> float:
        """Little's law: what is admitted (up to a rate limit) shares the throughput."""
        rate = self.rows_per_s(window)
        admitted = min(window, self.rate_limit or window)
        return admitted / rate if rate else self.read_timeout_s

    def best(self, max_window: int) -> tuple[int, float]:
        """(smallest window reaching the best throughput, that throughput)."""
        rates = [self.rows_per_s(w) for w in range(1, max_window + 1)]
        top = max(rates)
        return next(w for w, r in enumerate(rates, 1) if r >= top * 0.999), top

    def useful_window(self, max_window: int) -> int:
        """The smallest window within 2% of the best throughput: anything above it only queues."""
        _, top = self.best(max_window)
        return next(w for w in range(1, max_window + 1) if self.rows_per_s(w) >= top * 0.98)


@dataclasses.dataclass
class Tick:
    limit: int
    reason: str
    rows_per_s: float


def run(ctrl, engine: Engine, ticks: int, limit: int | None = None) -> list[Tick]:
    """Drive `ctrl` for `ticks` ticks against `engine`, the source never starving the window.
    A tick's reading describes the window from about one request latency before."""
    rng = random.Random(engine.seed)
    limit = ctrl.initial if limit is None else limit
    history = [limit]
    out = []
    for _ in range(ticks):
        lag = max(1, math.ceil(engine.latency_s(history[-1]) / TICK_S))
        seen = history[max(0, len(history) - 1 - lag)]
        jitter = rng.gauss(1.0, engine.noise) if engine.noise else 1.0
        rate = engine.rows_per_s(seen) * max(0.0, jitter)
        rejected = 0
        if engine.rate_limit is not None and seen > engine.rate_limit:
            rejected = round((seen - engine.rate_limit) / engine.latency_s(seen) * TICK_S) or 1
        timed_out = engine.rows_per_s(seen) == 0.0 and seen > 0
        queued = max(0, limit - engine.cap)
        obs = dict(
            waiting=queued if engine.gauges else None,
            running=min(limit, engine.cap) if engine.gauges else None,
            inflight=limit, backpressure=rejected + (seen if timed_out else 0),
            successes=round(rate * TICK_S), input_bound=False,
            kv=0.3 if engine.gauges else None, hits=0.9 if engine.gauges else None,
            tok_s=rate * engine.tokens_per_row,
            latency_s=engine.latency_s(seen) if rate else None,
        )
        limit = ctrl.decide(obs, limit)
        history.append(limit)
        out.append(Tick(limit, ctrl.last_reason, engine.rows_per_s(limit)))
    return out


def share_of_best(ticks: list[Tick], engine: Engine, max_window: int = 512) -> float:
    """Mean throughput over `ticks` as a share of the best fixed window's."""
    _, top = engine.best(max_window)
    return sum(t.rows_per_s for t in ticks) / len(ticks) / top


def share_past_useful(ticks: list[Tick], engine: Engine, max_window: int = 512) -> float:
    """Share of `ticks` spent with the window more than an eighth past the useful window: the
    excess requests only wait (or are rejected)."""
    useful = engine.useful_window(max_window)
    return sum(t.limit > useful * 9 / 8 + 1 for t in ticks) / len(ticks)


# SGLang on an A10G (2026-10-27 Jobs measurements, shape only): throughput rises to a window of
# 32, is flat from 32 to 64, rises again to about 192, and SGLang runs at most 256 requests.
SGLANG_LIKE = [(32, 64.0), (64, 68.0), (192, 150.0)]
LINEAR = [(256, 200.0)]  # throughput grows with the window up to the engine's cap
PREEMPTING = [(64, 100.0), (192, 160.0), (320, 110.0)]  # past 192, preemption costs throughput
