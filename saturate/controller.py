"""Controllers: pure sans-IO decide(obs, limit) -> new limit.

Signal priority (decision 5): delivered throughput primary, engine gauges a
secondary accelerator, blind AIMD the universal floor. The controller never
knows where an observation came from (HTTP scrape, in-process engine, nowhere)
— that's the SignalSource seam's job.
"""

from __future__ import annotations

import dataclasses

TICK_S = 2.0  # nominal tick period: the loop sleeps this long, and tick-count floors are multiples of it


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
    latency_s: float | None = None  # p50 duration of recent successful attempts (None = unknown)
    oldest_s: float | None = None  # age of the oldest request in flight (None = nothing in flight)
    tick_s: float | None = None  # seconds since the previous observation (None = TICK_S)


_FIELDS = {f.name for f in dataclasses.fields(Obs)}


def as_obs(obs: Obs | dict) -> Obs:
    if isinstance(obs, Obs):
        return obs
    return Obs(**{k: v for k, v in obs.items() if k in _FIELDS})


class Fixed:
    last_reason = "hold"

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

    1. hold:cooldown — after any reduction, hold while the requests admitted
       under the old window drain: at least max(2 ticks, latency_s) (3 ticks
       when the latency is unknown), extended while the oldest request in
       flight predates the reduction, never longer than the stall patience.
       Nothing the old window does meanwhile is evidence: its completions do
       not re-arm the stall rule and its silence does not count towards one.
    2. cut:bp — backpressure (429/timeout) this tick: halve.
    3. cut:kv — KV high with a low (or absent) prefix-hit rate: halve. High KV
       with a healthy hit rate is the cache doing its job.
    4. cut:stall — window full (inflight >= limit) and nothing completing for
       `stall_ticks` ticks, or 3x latency_s when known, whichever is longer:
       halve. A wedged engine reads as an empty queue because nothing new is
       admitted; a slow engine on a small window looks the same for a few
       requests' worth of ticks, so patience follows the request latency. One
       stall cut per episode: the next needs a completion after the last
       reduction has drained, and nothing before the first completion ever counts.
    5. hold:input_bound — the source, not the engine, is starving admission.
    6. cut:queue — engine queue above target (waiting > hi): step down. Not
       before the first completion ever: a cold engine queues the whole
       initial window while it warms up, and that is not a backlog.
    7. revert — a probe (10) did not confirm within its settle window.
    8. hold — window under 80% used (idle engine, source-paced), or queue in band.
    9. hold:ack — fewer than `limit` completions since the last evaluation.
       Growth is clocked by evidence: one tick for cheap text, one window of
       completions for long generations. Only ticks with the window at least
       80% used and not input-bound count; a starved window says nothing
       about the engine. Throughput is judged over that same window, so a
       slow engine's zero-token ticks average instead of vetoing.
    10. grow — headroom and throughput over the evidence window beat the best
        seen: double in slow-start, +step after. Slow-start ends at the first
        durable queue or reduction; a plateau does not end it, because with
        real generations the reading that follows a widening still describes
        the old window (the growth gate throttles slow-start on its own).
    11. probe / hold:plateau — throughput flat: hold, and on a backoff schedule
        (4, 8, 16, 32 evaluations) try +step anyway, because a real engine's
        throughput lags the tick. Revert (7) unless it confirms.

    Every reduction scales the throughput baseline with it, so throughput
    proportional to a smaller window is a plateau and throughput above it is
    growth. A reduction that cannot go below `min_limit` reports `hold:floor`
    and arms nothing; a widening that cannot go above `max_limit` reports
    `hold:ceiling` and opens no probe. Blind mode (no gauges) creeps +1 at each
    evidence window instead of doubling, and cannot hold a probe. Durations are
    integrated in seconds from the observed `tick_s` (TICK_S when absent), so
    one slow tick neither shortens nor recomputes a wait.
    """

    def __init__(self, target_waiting: int = 8, initial: int = 16, min_limit: int = 2,
                 max_limit: int = 512, step: int = 8, kv_hi: float = 0.9,
                 hits_lo: float = 0.5, improve: float = 1.05, stall_ticks: int = 3):
        if min_limit < 1 or max_limit < min_limit or step < 1:
            raise ValueError("Auto requires 1 <= min_limit <= max_limit and step >= 1")  # 0 deadlocks
        if stall_ticks < 1:
            raise ValueError("Auto requires stall_ticks >= 1")
        self.lo, self.hi = max(1, target_waiting // 4), target_waiting * 2
        self.min, self.max, self.step = min_limit, max_limit, step
        self.initial = self._clamp(initial)
        self.kv_hi, self.hits_lo, self.improve, self.stall_ticks = kv_hi, hits_lo, improve, stall_ticks
        self.last_reason = "hold"
        self._slow_start = True
        self._acked_ever = False  # queue gauges mean nothing before the first completion ever
        self._hold_s = 0.0  # cooldown armed by the last reduction (0 = none pending)
        self._since_cut_s = 0.0  # seconds since the last reduction
        self._settling = False  # inside the cooldown: the old window is still draining
        self._silent_s = 0.0  # seconds of a full window with nothing completing
        self._stall_armed = False  # a completion has arrived since the last reduction drained (or ever)
        self._queue_ticks = 0  # consecutive ticks with a standing queue
        self._acks, self._toks = 0, []  # evidence since the last evaluation: completions, tok_s samples
        self._best_tok = 0.0  # best evidence-window throughput, scaled down with every reduction
        self._probe_wait, self._probe_cooldown = 0, 4
        self._probe_from: int | None = None
        self._probe_age = 0
        self._rules = (self._cooling, self._backpressure, self._kv_pressure, self._stall,
                       self._input_bound, self._queue, self._widen)  # _widen always fires

    def decide(self, obs: Obs | dict, limit: int) -> int:
        obs = as_obs(obs)
        self._observe(obs, limit)
        for rule in self._rules:
            hit = rule(obs, limit)
            if hit is not None:
                break
        else:
            raise AssertionError("the rule list must end in a rule that always fires")
        new, self.last_reason = hit
        return new

    # --- per-tick bookkeeping -------------------------------------------------

    def _observe(self, obs: Obs, limit: int) -> None:
        tick = self._tick(obs)
        self._since_cut_s += tick
        draining = obs.oldest_s is not None and obs.oldest_s > self._since_cut_s  # oldest predates the cut
        self._settling = bool(self._hold_s) and (self._since_cut_s <= self._hold_s or (
            draining and self._since_cut_s <= self._patience_s(obs)))
        self._acked_ever = self._acked_ever or obs.successes > 0
        if self._settling:
            self._silent_s = 0.0  # the old window's completions and silence belong to the old window
        else:
            self._stall_armed = self._stall_armed or obs.successes > 0
            silent = obs.inflight >= limit and obs.successes == 0
            self._silent_s = self._silent_s + tick if silent else 0.0
        if not obs.input_bound and obs.inflight >= int(limit * 0.8):  # only a used window is evidence
            self._acks += obs.successes
            if obs.tok_s is not None:
                self._toks.append(obs.tok_s)
        if obs.waiting is None:
            self._void_probe()  # a probe begun in gauge mode cannot settle blind
        elif self._acked_ever:
            self._queue_ticks = self._queue_ticks + 1 if obs.waiting >= self.lo else 0
            if self._queue_ticks >= 2:  # a durable queue, not one cold-prefill spike
                self._slow_start = False
        if self._probe_from is not None:
            self._probe_age += 1

    @staticmethod
    def _tick(obs: Obs) -> float:
        return obs.tick_s or TICK_S

    def _patience_s(self, obs: Obs) -> float:
        """Seconds of a full, silent window that count as a stall."""
        floor = self.stall_ticks * self._tick(obs)
        return floor if obs.latency_s is None else max(floor, 3 * obs.latency_s)

    def _reset_evidence(self) -> None:
        self._acks, self._toks = 0, []

    def _evaluate(self, limit: int) -> tuple[bool, bool | None]:
        """(due, grew) once `limit` completions are in: True = throughput improved, False = flat,
        None = no throughput signal. The first sample after a change still reflects the old window,
        so one sample can confirm growth but cannot declare a plateau: that waits for a second."""
        if self._acks < limit:
            return False, None
        samples = self._toks[1:] or self._toks
        if not samples:  # tokens unobservable: completions alone are the evidence
            self._reset_evidence()
            return True, None
        avg = sum(samples) / len(samples)
        if avg > self._best_tok * self.improve:
            self._best_tok = avg
            self._reset_evidence()
            return True, True
        if len(self._toks) < 2:
            return False, None
        self._reset_evidence()
        return True, False

    def _void_probe(self) -> None:
        self._probe_from, self._probe_wait = None, 0

    def _clamp(self, n: int) -> int:
        return min(self.max, max(self.min, n))

    def _shrink(self, obs: Obs, limit: int, new: int, reason: str) -> tuple[int, str]:
        """The single way down: ends slow-start, voids any probe, and on an actual reduction
        scales the baseline with it, restarts the evidence window and arms the cooldown."""
        measured_at = limit if self._probe_from is None else self._probe_from  # where the baseline was set
        self._slow_start = False
        self._void_probe()  # an independent reduction: a pending revert must not climb back over it
        new = self._clamp(new)
        if new >= limit:  # at the floor: flat throughput there is not growth, so the baseline stays
            return limit, "hold:floor"
        if new < measured_at:
            self._best_tok *= new / measured_at
        self._reset_evidence()
        tick = self._tick(obs)
        self._hold_s = 3 * tick if obs.latency_s is None else max(2 * tick, obs.latency_s)
        self._since_cut_s = self._silent_s = 0.0
        self._stall_armed = False
        return new, reason

    def _cut(self, obs: Obs, limit: int, reason: str) -> tuple[int, str]:
        return self._shrink(obs, limit, limit // 2, reason)

    # --- rules, in priority order ----------------------------------------------

    def _cooling(self, obs: Obs, limit: int):
        if self._settling:
            self._reset_evidence()  # completions of the old window say nothing about the new one
            return limit, "hold:cooldown"
        self._hold_s = 0.0  # drained, or a wedged request that must not pin the window past the patience

    def _backpressure(self, obs: Obs, limit: int):
        if obs.backpressure:
            return self._cut(obs, limit, "cut:bp")

    def _kv_pressure(self, obs: Obs, limit: int):
        # with no hits signal at all (prefix caching off), high KV is unverifiably benign -> cut
        if obs.kv is not None and obs.kv >= self.kv_hi and (obs.hits is None or obs.hits < self.hits_lo):
            return self._cut(obs, limit, "cut:kv")

    def _stall(self, obs: Obs, limit: int):
        if self._stall_armed and self._silent_s >= self._patience_s(obs):
            return self._cut(obs, limit, "cut:stall")

    def _input_bound(self, obs: Obs, limit: int):
        if obs.input_bound:
            return limit, "hold:input_bound"

    def _queue(self, obs: Obs, limit: int):
        if self._acked_ever and obs.waiting is not None and obs.waiting > self.hi:
            return self._shrink(obs, limit, limit - self.step, "cut:queue")

    def _widen(self, obs: Obs, limit: int):
        due, grew = self._evaluate(limit)
        if self._probe_from is not None:  # a probe is settling: confirm or revert
            if grew is True:
                self._probe_from, self._probe_cooldown = None, 4  # confirmed
            elif due and self._probe_age >= 3:
                back, self._probe_from = self._probe_from, None
                self._probe_cooldown = min(self._probe_cooldown * 2, 32)
                return self._clamp(back), "revert"
        headroom = obs.waiting is None or obs.waiting < self.lo
        if not (headroom and obs.inflight >= int(limit * 0.8)):
            return limit, "hold"
        if not due:
            return limit, "hold:ack"
        if grew is not False:
            if obs.waiting is None:
                new = self._clamp(limit + 1)  # blind floor: creep
            else:
                new = self._clamp(limit * 2 if self._slow_start else limit + self.step)
            return (new, "grow") if new > limit else (limit, "hold:ceiling")
        if obs.waiting is not None and self._probe_from is None:  # plateau-blocked: probe on backoff
            self._probe_wait += 1
            if self._probe_wait >= self._probe_cooldown:
                new = self._clamp(limit + self.step)
                if new == limit:
                    return limit, "hold:ceiling"
                self._probe_wait, self._probe_from, self._probe_age = 0, limit, 0
                return new, "probe"
        return limit, "hold:plateau"
