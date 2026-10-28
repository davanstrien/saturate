"""Controllers: pure sans-IO decide(obs, limit) -> new limit.

Signal priority (decision 5): delivered throughput primary, engine gauges a
secondary accelerator, blind AIMD the universal floor. The controller never
knows where an observation came from (HTTP scrape, in-process engine, nowhere)
— that's the SignalSource seam's job.
"""

from __future__ import annotations

import dataclasses

TICK_S = 2.0  # nominal tick period: the loop sleeps this long, and tick-count floors are multiples of it
BP_CEILING_RELEASE_S = 300.0  # backpressure-free seconds before the backpressure ceiling is released


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

    1. hold:cooldown — after any reduction or revert, hold while the requests
       admitted under the old window drain: at least max(2 ticks, the last
       latency seen) (3 ticks before any latency is known), extended while the oldest request in
       flight predates the reduction, never longer than the stall patience.
       Nothing the old window does meanwhile is evidence: its completions do
       not re-arm the stall rule and its silence does not count towards one.
    2. cut:bp — backpressure (429/timeout/5xx) on at least `bp_frac` of this tick's
       outcomes (backpressure + successes, both counted per row): halve, and remember the
       window as a ceiling: growth and probes stop an eighth below the lowest window that
       met backpressure, so a hosted API's rate limit is met once, not on every climb. After
       BP_CEILING_RELEASE_S without backpressure the ceiling is released, because a provider's
       limit can rise; meeting the old limit again steps just under it instead of halving,
       and doubles the wait before the next release. A share, not any
       event: one row that always fails (a 500 on one bad image) is not overload, and at a
       large window it would otherwise halve it every tick. Overload keeps failing NEW rows,
       so it keeps meeting the share; a fixed set of failing rows counts once and then only
       retries. The share is of the tick's traffic, not of the window: with requests much
       faster than a tick, set bp_frac lower (0 cuts on any event, the old behaviour).
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
    7. revert — a probe or trial (10, 11) did not confirm once its settle window
       has passed: three ticks and twice the latency expected at the probe's window
       (by Little's law, latency grows with the window at flat throughput), since
       until then the readings describe the old window. A blind trim that lost
       throughput reverts the same way.
    8. hold — window under 80% used (idle engine, source-paced), or queue in band.
    9. hold:ack — fewer than `limit` completions since the last evaluation.
       Growth is clocked by evidence: one tick for cheap text, one window of
       completions for long generations. Only ticks with the window at least
       80% used and not input-bound count; a starved window says nothing
       about the engine. Throughput is judged over that same window, so a
       slow engine's zero-token ticks average instead of vetoing.
    10. grow / grow:trial — headroom and throughput over the evidence window beat
        the best seen: double in slow-start, +step after (blind: see below).
        With gauges, slow-start ends at the first durable queue or reduction; a
        plateau does not end it, because with real generations the reading that
        follows a widening still describes the old window.
    11. probe / trim / hold:plateau — throughput flat: hold, and on a backoff
        schedule (4, 8, 16, 32 evaluations) try a wider window anyway, because a
        real engine's throughput lags the tick and can be flat over a stretch of
        windows before it rises again. Revert (7) unless it confirms. Blind, once
        the search for a wider window has ended, try a smaller one (trim).

    Every reduction scales the throughput baseline with it, so throughput
    proportional to a smaller window is a plateau and throughput above it is
    growth. A reduction that cannot go below `min_limit` reports `hold:floor`
    and arms nothing; a widening that cannot go above `max_limit` reports
    `hold:ceiling` and opens no probe.

    Blind mode (no gauges) cannot see a queue, so it judges every change on throughput:
    - Slow-start doubles on trial (grow:trial) while throughput keeps improving; a doubling
      that does not pay is reverted and ends slow-start. After that it grows by an eighth of
      the window (at least 1) at each evidence window: growth needs throughput `improve`
      times the best so far, and a +1 step from a window of L gains at most 1/L, too little
      to measure.
    - Plateau probes try 2x, then 4x, then 8x the window, because a flat stretch (SGLang's
      from 32 to 64) can be wider than any step. A probe is judged once settled (rule 7), on
      its own window's readings, against the reading just before it: the best ever seen is
      lifted by noise. Any measured gain starts the search again at 2x. A probe that meets
      backpressure goes back to where it began, and the backpressure ceiling then bounds
      the next ones.
    - Once an 8x probe has failed, or a probe would only repeat a failed window, it trims:
      it tries an eighth less, keeps it if throughput holds (within `improve` of the reading
      before the trim, judged on the smaller window's own readings), and otherwise reverts.
      A second failed trim to the same window ends trimming until the next gain. There are
      no trims at a backpressure ceiling: nothing waits in a queue there. Trims undo
      a doubling that confirmed a gain but overshot the engine's cap (64 -> 128 for a cap
      of 100); with noisy readings their verdicts are less reliable.
    Gauge mode probes by +step, and a queue above target ends any probe. Durations are
    integrated in seconds from the observed `tick_s` (TICK_S when absent), so one slow tick
    neither shortens nor recomputes a wait.
    """

    def __init__(self, target_waiting: int = 8, initial: int = 16, min_limit: int = 2,
                 max_limit: int = 512, step: int = 8, kv_hi: float = 0.9,
                 hits_lo: float = 0.5, improve: float = 1.05, stall_ticks: int = 3,
                 bp_frac: float = 0.05):
        if min_limit < 1 or max_limit < min_limit or step < 1:
            raise ValueError("Auto requires 1 <= min_limit <= max_limit and step >= 1")  # 0 deadlocks
        if stall_ticks < 1:
            raise ValueError("Auto requires stall_ticks >= 1")
        if not 0.0 <= bp_frac <= 1.0:
            raise ValueError("Auto requires 0 <= bp_frac <= 1")
        self.lo, self.hi = max(1, target_waiting // 4), target_waiting * 2
        self.min, self.max, self.step = min_limit, max_limit, step
        self._bp_ceiling: int | None = None  # the lowest window that met backpressure (None = none)
        self._bp_seen: int | None = None  # the last ceiling, kept after it is released
        self._clean_s = 0.0  # seconds since the last backpressure cut
        self._release_s = BP_CEILING_RELEASE_S  # clean seconds before the ceiling rises; doubles on a re-meet
        self.initial = self._clamp(initial)
        self.kv_hi, self.hits_lo, self.improve, self.stall_ticks = kv_hi, hits_lo, improve, stall_ticks
        self.bp_frac = bp_frac
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
        self._probe_blind = False  # the pending probe was opened without gauges
        self._blind_reach = 2  # a blind probe tries this many times the window: 2, 4, 8; 0 = done
        self._probe_age, self._probe_age_s = 0, 0.0  # ticks and seconds since the probe opened
        self._probe_latency_s = 0.0  # the longest latency seen since the probe opened
        self._probe_trial = False  # the pending probe is a slow-start doubling, not a plateau probe
        self._failed_probe_to: int | None = None  # where the last plateau probe went and failed
        self._probe_down = False  # the pending probe is a trim: a smaller window on trial
        self._trim_floor: int | None = None  # a trim to this window lost throughput: none at or below it
        self._last_avg = 0.0  # throughput of the last evidence window
        self._probe_ref = 0.0  # throughput just before the pending plateau probe or trim
        self._trim_failed_at: int | None = None  # a trim to this window lost throughput once
        self._latency_s: float | None = None  # the last request latency seen (timeouts report none)
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
        if obs.latency_s is not None:
            self._latency_s = obs.latency_s
        self._since_cut_s += tick
        self._clean_s += tick
        if self._bp_ceiling is not None and self._clean_s >= self._release_s:
            # a provider's limit can rise during a run: grow freely again, and if the old limit
            # is still there, meeting it steps under it rather than halving (_backpressure)
            self._bp_seen, self._bp_ceiling = self._bp_ceiling, None
            self._reopen_search()
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
            if self._probe_from is not None and not self._probe_blind:
                self._void_probe()  # a probe begun in gauge mode cannot settle blind
        elif self._acked_ever:
            self._queue_ticks = self._queue_ticks + 1 if obs.waiting >= self.lo else 0
            if self._queue_ticks >= 2:  # a durable queue, not one cold-prefill spike
                self._slow_start = False
        if self._probe_from is not None:
            self._probe_age += 1
            self._probe_age_s += tick
            self._probe_latency_s = max(self._probe_latency_s, obs.latency_s or 0.0)

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
        self._last_avg = avg
        if avg > self._best_tok * self.improve:
            self._best_tok = avg
            self._reset_evidence()
            return True, True
        if len(self._toks) < 2:
            return False, None
        self._reset_evidence()
        return True, False

    def _void_probe(self) -> None:
        self._probe_from, self._probe_wait, self._probe_blind, self._probe_trial = None, 0, False, False
        self._probe_down = False

    def _open_probe(self, limit: int, new: int, blind: bool, reason: str,
                    trial: bool = False) -> tuple[int, str]:
        """Widen to `new` on trial: kept if throughput confirms within the settle window,
        otherwise reverted to `limit` (rule 7)."""
        new = self._clamp(new)
        if new == limit:
            return limit, "hold:ceiling"
        self._probe_wait, self._probe_from, self._probe_age, self._probe_age_s = 0, limit, 0, 0.0
        # Little's law: at flat throughput, latency grows with the window, so a wider trial
        # window's own completions arrive that much later
        self._probe_latency_s = (self._latency_s or 0.0) * max(1.0, new / limit)
        self._probe_blind, self._probe_trial = blind, trial
        return new, reason

    def _reopen_search(self) -> None:
        """Blind: search wider windows again from 2x, and let trims go as low as before."""
        self._blind_reach, self._failed_probe_to = 2, None
        self._trim_floor = self._trim_failed_at = None

    def _ceiling_below(self, ceiling: int) -> int:
        return max(self.min, ceiling - max(1, ceiling // 8))

    def _clamp(self, n: int) -> int:
        top = self.max
        if self._bp_ceiling is not None:  # stay an eighth below where backpressure began
            top = min(top, self._ceiling_below(self._bp_ceiling))
        return min(top, max(self.min, n))

    def _arm_cooldown(self, obs: Obs) -> None:
        """Hold while the requests admitted under the old window drain (rule 1): their
        completions and silence are not evidence about the new one."""
        self._reset_evidence()
        tick = self._tick(obs)
        latency = obs.latency_s if obs.latency_s is not None else self._latency_s  # timeouts report none
        self._hold_s = 3 * tick if latency is None else max(2 * tick, latency)
        self._since_cut_s = self._silent_s = 0.0
        self._stall_armed = False

    def _shrink(self, obs: Obs, limit: int, new: int, reason: str) -> tuple[int, str]:
        """The single way down: ends slow-start, voids any probe, and on an actual reduction
        scales the baseline with it, restarts the evidence window and arms the cooldown."""
        measured_at = limit if self._probe_from is None else self._probe_from  # where the baseline was set
        if self._probe_from is not None and self._probe_blind:
            new = min(new, self._probe_from)  # a blind trial met a limit: back to where it began
        self._slow_start = False
        self._void_probe()  # an independent reduction: a pending revert must not climb back over it
        new = self._clamp(new)
        if new >= limit:  # at the floor: flat throughput there is not growth, so the baseline stays
            return limit, "hold:floor"
        if new < measured_at:
            self._best_tok *= new / measured_at
        self._arm_cooldown(obs)
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
        if obs.backpressure and obs.backpressure >= self.bp_frac * (obs.backpressure + obs.successes):
            # a hosted API's rate limit is a wall: remember it, so growth stops below it instead
            # of meeting it again and halving each time
            # back at a limit met before (the window climbed to just below it): step under it
            # again rather than halve; anywhere lower, this is fresh overload and halves
            known = self._bp_ceiling if self._bp_ceiling is not None else self._bp_seen
            at_known = known is not None and limit >= self._ceiling_below(known)
            self._bp_ceiling = limit if self._bp_ceiling is None else min(self._bp_ceiling, limit)
            self._clean_s = 0.0
            below = self._ceiling_below(self._bp_ceiling)
            if at_known:
                self._release_s *= 2  # the limit held: look for a higher one less often
            if at_known and below < limit:
                return self._shrink(obs, limit, below, "cut:bp")
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
        if grew is True:  # a gain, whatever was pending: there may be more above
            self._reopen_search()
        if self._probe_from is not None:  # a probe is settling: confirm or revert
            # the trial window's own completions take a latency to arrive, and readings until then
            # describe the old window: wait two latencies, the longest seen since the probe began
            settled = self._probe_age >= 3 and self._probe_age_s >= 2 * self._probe_latency_s
            if self._probe_down:  # a trim: kept unless throughput fell
                # against the reading just before the trim, not the best ever: noise lifts the best
                held = self._last_avg * self.improve >= self._probe_ref
                held = held and self._last_avg * self.improve ** 2 >= self._best_tok  # no creeping
                if due and settled and grew is not None and (grew is True or held):
                    self._void_probe()
                    self._probe_cooldown = 2  # trim again soon: the window may still be oversized
                    return limit, "hold:trimmed"
                if due and settled and grew is not None:  # one noisy reading is not proof:
                    back = self._probe_from  # block trims only on a second failure here
                    if self._trim_failed_at == limit:
                        self._trim_floor = limit
                    self._trim_failed_at = limit
                    self._probe_cooldown = min(self._probe_cooldown * 2, 32)
                    self._void_probe()
                    self._arm_cooldown(obs)
                    return self._clamp(back), "revert"
                return limit, "hold:ack"
            # a slow-start trial confirms on any gain at once (a fast ramp); a plateau probe is
            # judged once settled, against the reading just before it (noise lifts the best)
            fresh = due and settled and grew is not None
            paired_gain = fresh and self._last_avg >= self._probe_ref * self.improve
            if (self._probe_trial and grew is True) or (not self._probe_trial and paired_gain):
                self._probe_cooldown = 4  # confirmed
                self._void_probe()
                self._reopen_search()
            elif due and settled:  # the trial window's own completions have arrived: revert
                back = self._probe_from
                self._probe_cooldown = min(self._probe_cooldown * 2, 32)
                if self._probe_blind and self._probe_trial:
                    self._slow_start = False  # a blind doubling that did not pay: step from here
                elif self._probe_blind:  # the flat stretch may be wider: try further, up to 8x
                    self._failed_probe_to = limit
                    self._blind_reach = 0 if self._blind_reach >= 8 else self._blind_reach * 2
                self._void_probe()
                self._arm_cooldown(obs)  # the trial window's late completions are not evidence here
                return self._clamp(back), "revert"
        headroom = obs.waiting is None or obs.waiting < self.lo
        if not (headroom and obs.inflight >= int(limit * 0.8)):
            return limit, "hold"
        if not due:
            return limit, "hold:ack"
        blind = obs.waiting is None
        if grew is not False:
            if blind and self._slow_start and grew is True and self._probe_from is None:
                # no queue to see: double on trial, so a doubling into a full engine reverts
                return self._open_probe(limit, limit * 2, blind=True, reason="grow:trial", trial=True)
            if blind:  # grow in steps big enough to measure
                new = self._clamp(limit + max(1, limit // 8))
            else:
                new = self._clamp(limit * 2 if self._slow_start else limit + self.step)
            return (new, "grow") if new > limit else (limit, "hold:ceiling")
        if blind and self._probe_from is None and self._blind_reach == 0:  # searched up: trim down
            self._probe_wait += 1
            smaller = limit - max(1, limit // 8)
            floor = max(self.min, self._trim_floor or 0)
            # at a backpressure ceiling nothing waits in a queue: a trim could only lose rows
            at_ceiling = self._bp_ceiling is not None and limit >= self._ceiling_below(self._bp_ceiling)
            if self._probe_wait >= self._probe_cooldown and smaller > floor and not at_ceiling:
                result = self._open_probe(limit, smaller, blind=True, reason="trim")
                self._probe_down, self._probe_ref = True, self._last_avg
                self._reset_evidence()  # a trim is judged on the smaller window's readings only
                return result
        elif self._probe_from is None:  # probe on backoff
            self._probe_wait += 1
            if self._probe_wait >= self._probe_cooldown:
                # blind, a flat stretch can be wider than any step: try a multiple of the window
                ref = self._last_avg  # the reading the probe is judged against
                new = limit * self._blind_reach if blind else limit + self.step
                at_top = self._clamp(new) <= limit
                if blind and (at_top or (self._failed_probe_to is not None
                                         and self._clamp(new) <= self._failed_probe_to)):
                    self._blind_reach = 0  # nothing new to try above: search down (trims)
                    return limit, "hold:ceiling" if at_top else "hold:plateau"
                self._probe_ref = ref
                self._reset_evidence()  # judged on the probe window's own readings
                return self._open_probe(limit, new, blind, "probe")
        return limit, "hold:plateau"
