"""Controllers: pure sans-IO decide(obs, limit) -> new limit.

Signal priority (decision 5): delivered throughput primary, engine gauges a
secondary accelerator, blind AIMD the universal floor. The controller never
knows where an observation came from (HTTP scrape, in-process engine, nowhere)
— that's the SignalSource seam's job.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass
class Obs:
    waiting: int | None = None  # engine queue depth (None = no gauges)
    running: int | None = None
    inflight: int = 0  # our requests currently in flight
    backpressure: int = 0  # saturation-shaped 429/timeout events since last tick
    successes: int = 0  # completed requests since last tick
    input_bound: bool = False  # admission starved by the source, not the engine
    kv: float | None = None  # kv-cache utilisation gauge (0-1)
    hits: float | None = None  # prefix-cache hit rate gauge (0-1)
    tok_s: float | None = None  # delivered tokens/sec this tick; 0.0 = nothing delivered, None = unobservable
    preempts: int | None = None  # cumulative preemption count
    latency_s: float | None = None  # p50 admission-to-completion time of recent requests (None = unknown)


_FIELDS = {f.name for f in dataclasses.fields(Obs)}


def as_obs(obs: Obs | dict) -> Obs:
    if isinstance(obs, Obs):
        return obs
    return Obs(**{k: v for k, v in obs.items() if k in _FIELDS})


class Fixed:
    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"Fixed window must be >= 1, got {n}")  # 0 deadlocks admission
        self.n = n

    initial = property(lambda self: self.n)

    def decide(self, obs: Obs | dict, limit: int) -> int:
        return self.n


class Auto:
    """Hold the engine's queue small-and-positive; never outgrow delivered throughput.

    decide() evaluates these rules in order; the first that fires sets the
    limit and names itself in `last_reason`:

    1. cut:bp — backpressure (429/timeout) this tick, outside a cooldown: halve.
    2. hold:cooldown — after any halving, hold while the requests admitted under
       the old window drain: max(2, ceil(latency_s / tick_s)) ticks, 3 when the
       latency is unknown. Those requests keep failing after the cut; counting
       them again would halve once per tick instead of once per episode.
    3. cut:kv — KV high with a low (or absent) prefix-hit rate: halve. High KV
       with a healthy hit rate is the cache doing its job.
    4. cut:stall — window full (inflight >= limit) and nothing completing for
       `stall_ticks` consecutive ticks, or 2x latency_s when known, whichever is
       longer: halve. A wedged engine reads as an empty queue because nothing
       new is admitted; a slow engine on a small window looks the same for one
       request's worth of ticks, so patience follows the request latency.
    5. hold:input_bound — the source, not the engine, is starving admission.
    6. cut:queue — engine queue above target (waiting > hi): step down.
    7. revert — a probe (10) did not confirm within its settle window.
    8. hold — window under 80% used (idle engine, source-paced), or queue in band.
    9. hold:ack — fewer than `limit` completions since the last evaluation.
       Growth is clocked by evidence: one tick for cheap text, one window of
       completions for long generations. Throughput is judged over that same
       window, so a slow engine's zero-token ticks average instead of vetoing.
    10. grow — headroom and throughput over the evidence window beat the best
        seen: double in slow-start, +step after. Slow-start ends at the first
        standing queue, plateau, stall or cut.
    11. probe / hold:plateau — throughput flat: hold, and on a backoff schedule
        (4, 8, 16, 32 evaluations) try +step anyway, because a real engine's
        throughput lags the tick. Revert (7) unless it confirms.

    Cuts and the queue step-down scale the throughput baseline with the
    reduction, so throughput proportional to a smaller window is a plateau and
    throughput above it is growth. Blind mode (no gauges) creeps +1 at each
    evidence window instead of doubling, and cannot hold a probe.
    """

    def __init__(self, target_waiting: int = 8, initial: int = 16, min_limit: int = 2,
                 max_limit: int = 512, step: int = 8, kv_hi: float = 0.9,
                 hits_lo: float = 0.5, improve: float = 1.05, stall_ticks: int = 3,
                 tick_s: float = 2.0):
        if min_limit < 1 or max_limit < min_limit or step < 1:
            raise ValueError("Auto requires 1 <= min_limit <= max_limit and step >= 1")  # 0 deadlocks
        if stall_ticks < 1 or tick_s <= 0:
            raise ValueError("Auto requires stall_ticks >= 1 and tick_s > 0")
        self.lo, self.hi = max(1, target_waiting // 4), target_waiting * 2
        self.min, self.max, self.step = min_limit, max_limit, step
        self.initial = max(min_limit, min(initial, max_limit))  # clamp into [min, max]
        self.kv_hi, self.hits_lo, self.improve = kv_hi, hits_lo, improve
        self.stall_ticks, self.tick_s = stall_ticks, tick_s
        self.last_reason = "hold"
        self._slow_start = True
        self._cooldown = 0  # ticks left holding after a halving
        self._stalled = 0  # consecutive full-window, zero-completion ticks
        self._queue_ticks = 0  # consecutive ticks with a standing queue
        self._acked_ever = False  # stall patience is meaningless before anything has completed
        self._acks, self._tok_sum, self._tok_n = 0, 0.0, 0  # evidence since the last evaluation
        self._tok_first = 0.0  # the window's first sample still carries the previous limit's throughput
        self._best_tok = 0.0  # best evidence-window throughput, scaled down with every reduction
        self._probe_wait, self._probe_cooldown = 0, 4
        self._probe_from: int | None = None
        self._probe_age = 0
        self._rules = (self._backpressure, self._cooling, self._kv_pressure, self._stall,
                       self._input_bound, self._queue, self._widen)

    def decide(self, obs: Obs | dict, limit: int) -> int:
        obs = as_obs(obs)
        self._observe(obs, limit)
        for rule in self._rules:
            hit = rule(obs, limit)
            if hit is not None:
                new, self.last_reason = hit
                if new != limit:
                    self._reset_evidence()  # a new limit starts a new evidence window
                return new
        self.last_reason = "hold"
        return limit

    # --- per-tick bookkeeping -------------------------------------------------

    def _observe(self, obs: Obs, limit: int) -> None:
        self._acks += obs.successes
        self._acked_ever = self._acked_ever or obs.successes > 0
        if obs.tok_s is not None:
            self._tok_first = obs.tok_s if not self._tok_n else self._tok_first
            self._tok_sum, self._tok_n = self._tok_sum + obs.tok_s, self._tok_n + 1
        self._stalled = self._stalled + 1 if obs.inflight >= limit and obs.successes == 0 else 0
        if obs.waiting is None:
            self._void_probe()  # a probe begun in gauge mode cannot settle blind
        else:
            self._queue_ticks = self._queue_ticks + 1 if obs.waiting >= self.lo else 0
            if self._queue_ticks >= 2:  # a durable queue, not one cold-prefill spike
                self._slow_start = False
        if self._probe_from is not None:
            self._probe_age += 1

    def _reset_evidence(self) -> None:
        self._acks, self._tok_sum, self._tok_n = 0, 0.0, 0

    def _evaluate(self) -> bool | None:
        """Verdict on the evidence window: True = throughput improved, False = flat, None = no signal."""
        grew = None
        if self._tok_n:  # the first tick after a change still reflects the old window: leave it out
            n = self._tok_n
            avg = self._tok_sum / n if n == 1 else (self._tok_sum - self._tok_first) / (n - 1)
            grew = avg > self._best_tok * self.improve
            if grew:
                self._best_tok = avg
        self._reset_evidence()
        return grew

    def _void_probe(self) -> None:
        self._probe_from, self._probe_wait = None, 0

    def _shrink(self, limit: int, new: int) -> int:
        if new < limit:  # the baseline follows the reduction: flat throughput at the floor is not growth
            self._best_tok *= new / limit
        return new

    def _cut(self, obs: Obs, limit: int) -> int:
        self._slow_start = False
        self._void_probe()
        self._cooldown = 3 if obs.latency_s is None else max(2, math.ceil(obs.latency_s / self.tick_s))
        return self._shrink(limit, max(self.min, limit // 2))

    # --- rules, in priority order ----------------------------------------------

    def _backpressure(self, obs: Obs, limit: int):
        if obs.backpressure and not self._cooldown:
            return self._cut(obs, limit), "cut:bp"

    def _cooling(self, obs: Obs, limit: int):
        if self._cooldown:
            self._cooldown -= 1
            self._reset_evidence()  # completions of the old window say nothing about the new one
            return limit, "hold:cooldown"

    def _kv_pressure(self, obs: Obs, limit: int):
        # with no hits signal at all (prefix caching off), high KV is unverifiably benign -> cut
        if obs.kv is not None and obs.kv >= self.kv_hi and (obs.hits is None or obs.hits < self.hits_lo):
            return self._cut(obs, limit), "cut:kv"

    def _stall(self, obs: Obs, limit: int):
        patience = self.stall_ticks
        if obs.latency_s is not None:
            patience = max(patience, math.ceil(2 * obs.latency_s / self.tick_s))
        if self._acked_ever and self._stalled >= patience:
            return self._cut(obs, limit), "cut:stall"

    def _input_bound(self, obs: Obs, limit: int):
        if obs.input_bound:
            return limit, "hold:input_bound"

    def _queue(self, obs: Obs, limit: int):
        if obs.waiting is not None and obs.waiting > self.hi:
            self._slow_start = False
            self._void_probe()  # an independent reduction: a pending revert must not climb back over it
            return self._shrink(limit, max(self.min, limit - self.step)), "cut:queue"

    def _widen(self, obs: Obs, limit: int):
        due = self._acks >= limit
        grew = self._evaluate() if due else None
        if self._probe_from is not None:  # a probe is settling: confirm or revert
            if grew is True:
                self._probe_from, self._probe_cooldown = None, 4  # confirmed
            elif due and self._probe_age >= 3:
                back, self._probe_from = self._probe_from, None
                self._probe_cooldown = min(self._probe_cooldown * 2, 32)
                return min(self.max, max(self.min, back)), "revert"  # clamped both ends
        headroom = obs.waiting is None or obs.waiting < self.lo
        if not (headroom and obs.inflight >= int(limit * 0.8)):
            return limit, "hold"
        if not due:
            return limit, "hold:ack"
        if grew is not False:
            if obs.waiting is None:
                return min(self.max, limit + 1), "grow"  # blind floor: creep
            return min(self.max, limit * 2 if self._slow_start else limit + self.step), "grow"
        self._slow_start = False
        if obs.waiting is not None and self._probe_from is None:  # plateau-blocked: probe on backoff
            self._probe_wait += 1
            if self._probe_wait >= self._probe_cooldown:
                self._probe_wait, self._probe_from, self._probe_age = 0, limit, 0
                return min(self.max, limit + self.step), "probe"
        return limit, "hold:plateau"
