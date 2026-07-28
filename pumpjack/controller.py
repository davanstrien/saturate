"""Controllers: pure sans-IO decide(obs, limit) -> new limit.

Signal priority (decision 5): delivered throughput primary, engine gauges a
secondary accelerator, blind AIMD the universal floor. The controller never
knows where an observation came from (HTTP scrape, in-process engine, nowhere)
— that's the SignalSource seam's job.
"""

from __future__ import annotations

import dataclasses


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
    tok_s: float | None = None  # delivered tokens/sec this tick (plateau signal)
    preempts: int | None = None  # cumulative preemption count


_FIELDS = {f.name for f in dataclasses.fields(Obs)}


def as_obs(obs: Obs | dict) -> Obs:
    if isinstance(obs, Obs):
        return obs
    return Obs(**{k: v for k, v in obs.items() if k in _FIELDS})


class Fixed:
    def __init__(self, n: int):
        self.n = n

    initial = property(lambda self: self.n)

    def decide(self, obs: Obs | dict, limit: int) -> int:
        return self.n


class Auto:
    """Hold the engine's queue small-and-positive; never outgrow delivered throughput.

    - slow-start doubles until a queue *durably* forms (debounced exit — one
      cold-prefill spike must not end the ramp; the 12-shard traces showed the
      POC quit ~5x too early, then crawled ~90s to equilibrium)
    - growth is gated by the throughput plateau: if delivered tok/s stopped
      improving, hold even when the queue reports headroom (KV != throughput)
    - two-condition cut: high KV with a *healthy* prefix-hit rate is benign;
      only high KV together with a low hit rate is real memory pressure
    - input-bound freeze: a starved engine is ambiguous between "window too
      small" and "source too slow" — never widen on the latter
    """

    def __init__(self, target_waiting: int = 8, initial: int = 16, min_limit: int = 2,
                 max_limit: int = 512, step: int = 8, kv_hi: float = 0.9,
                 hits_lo: float = 0.5, improve: float = 1.05):
        self.lo, self.hi = max(1, target_waiting // 4), target_waiting * 2
        self.initial = initial
        self.min, self.max, self.step = min_limit, max_limit, step
        self.kv_hi, self.hits_lo, self.improve = kv_hi, hits_lo, improve
        self._cooldown = 0
        self._slow_start = True
        self._queue_ticks = 0  # consecutive ticks with a standing queue (debounce)
        self._best_tok = 0.0
        self._seen_ok = False  # ACK-clock: no growth before the first completion ever
        # probe-and-revert (the 10k-parity lesson): real generations lag the tick,
        # so a plateau reading can be stale — the gate must not block exploration
        # forever. When growth is plateau-blocked, probe +step on a backoff
        # schedule; revert if throughput doesn't confirm within the settle window.
        self._probe_wait = 0
        self._probe_cooldown = 4
        self._probe_from: int | None = None
        self._probe_age = 0

    def _cut(self, limit: int) -> int:
        self._cooldown, self._slow_start = 2, False
        return max(self.min, limit // 2)

    def decide(self, obs: Obs | dict, limit: int) -> int:
        obs = as_obs(obs)
        if obs.backpressure:
            return self._cut(limit)
        if self._cooldown:
            self._cooldown -= 1
            return limit
        # two-condition cut; with no hits signal at all (prefix caching off), high
        # KV is unverifiably benign -> cut conservatively (shapes-run finding)
        if obs.kv is not None and obs.kv >= self.kv_hi and (obs.hits is None or obs.hits < self.hits_lo):
            return self._cut(limit)
        if obs.input_bound:
            return limit
        # ACK-clocked slow start (the 5k-vision OOM lesson): with long generations,
        # ticks pass with zero completions while gauges show phantom headroom
        # (multimodal preprocessing queues ahead of the scheduler's gauges) —
        # doubling on that evidence is how a client OOMs the box. TCP rule:
        # never widen a window nothing has ever been acknowledged through.
        self._seen_ok = self._seen_ok or obs.successes > 0
        if not self._seen_ok:
            return limit
        grew = None  # None = no throughput signal this tick
        if obs.tok_s:
            grew = obs.tok_s > self._best_tok * self.improve
            if grew:
                self._best_tok = obs.tok_s
        if obs.waiting is not None:  # gauge mode
            self._queue_ticks = self._queue_ticks + 1 if obs.waiting >= self.lo else 0
            if self._queue_ticks >= 2:  # debounced slow-start exit
                self._slow_start = False
            if self._probe_from is not None:  # a probe is settling: confirm or revert
                self._probe_age += 1
                if grew is True:
                    self._probe_from, self._probe_cooldown = None, 4  # confirmed
                elif self._probe_age >= 3:
                    back, self._probe_from = self._probe_from, None
                    self._probe_cooldown = min(self._probe_cooldown * 2, 32)
                    return max(self.min, back)  # revert: the plateau was real
            if obs.waiting < self.lo and obs.inflight >= int(limit * 0.8):
                if grew is not False:
                    return min(self.max, limit * 2 if self._slow_start else limit + self.step)
                if self._probe_from is None:  # plateau-blocked: probe on backoff
                    self._probe_wait += 1
                    if self._probe_wait >= self._probe_cooldown:
                        self._probe_wait, self._probe_from, self._probe_age = 0, limit, 0
                        return min(self.max, limit + self.step)
            if obs.waiting > self.hi:
                return max(self.min, limit - self.step)
            return limit
        # blind floor: creep on sustained success unless throughput plateaued
        if obs.successes > 0 and grew is not False:
            return min(self.max, limit + 1)
        return limit
