"""A simulated inference engine for controller tests: a known throughput curve, a cap on running
requests, and readings that lag the window by a tick, as a real engine's do.

`run(ctrl, engine, ticks)` drives a controller tick by tick against the engine and returns what
happened, so a test can compare Auto's throughput with the best fixed window's.
"""

from __future__ import annotations

import bisect
import dataclasses

from saturate.controller import TICK_S


@dataclasses.dataclass
class Engine:
    """`curve` maps concurrency to rows/s at the points given, linear in between and flat past
    the last point. Requests beyond `cap` wait in the engine's queue and add no throughput.
    `gauges=False` hides the queue, as an engine with no /metrics does (blind mode)."""

    curve: list[tuple[int, float]]
    cap: int = 256
    tokens_per_row: float = 100.0
    gauges: bool = True
    lag_ticks: int = 1

    def rows_per_s(self, window: int) -> float:
        running = min(window, self.cap)
        xs = [x for x, _ in self.curve]
        if running <= xs[0]:
            return self.curve[0][1] * running / xs[0]
        if running >= xs[-1]:
            return self.curve[-1][1]
        i = bisect.bisect_right(xs, running)
        (x0, y0), (x1, y1) = self.curve[i - 1], self.curve[i]
        return y0 + (y1 - y0) * (running - x0) / (x1 - x0)

    def best(self, max_window: int) -> tuple[int, float]:
        """(smallest window reaching the best throughput, that throughput)."""
        top = max(self.rows_per_s(w) for w in range(1, max_window + 1))
        window = next(w for w in range(1, max_window + 1) if self.rows_per_s(w) >= top * 0.999)
        return window, top


@dataclasses.dataclass
class Tick:
    limit: int
    reason: str
    rows_per_s: float


def run(ctrl, engine: Engine, ticks: int, limit: int | None = None) -> list[Tick]:
    """Drive `ctrl` for `ticks` controller ticks against `engine`, the source never starving
    the window. Each tick's reading describes the window from `lag_ticks` ticks before."""
    limit = ctrl.initial if limit is None else limit
    history = [limit]
    out = []
    for _ in range(ticks):
        seen = history[max(0, len(history) - 1 - engine.lag_ticks)]
        rate = engine.rows_per_s(seen)
        waiting = max(0, limit - engine.cap) if engine.gauges else None
        obs = dict(
            waiting=waiting, running=min(limit, engine.cap) if engine.gauges else None,
            inflight=limit, backpressure=0, successes=round(rate * TICK_S), input_bound=False,
            kv=0.3 if engine.gauges else None, hits=0.9 if engine.gauges else None,
            tok_s=rate * engine.tokens_per_row,
            latency_s=min(seen, engine.cap) / rate if rate else None,
        )
        limit = ctrl.decide(obs, limit)
        history.append(limit)
        out.append(Tick(limit, ctrl.last_reason, engine.rows_per_s(limit)))
    return out


def share_of_best(ticks: list[Tick], engine: Engine, max_window: int) -> float:
    """Mean throughput over `ticks` as a share of the best fixed window's."""
    _, top = engine.best(max_window)
    return sum(t.rows_per_s for t in ticks) / len(ticks) / top


# SGLang on an A10G (2026-10-27 Jobs measurements, shape only): throughput rises to a window of
# 32, is flat from 32 to 64, rises again to about 192, and SGLang runs at most 256 requests.
SGLANG_LIKE = [(32, 64.0), (64, 68.0), (192, 150.0)]
