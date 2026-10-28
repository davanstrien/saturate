"""Controllers: pure sans-IO decide(obs, limit) -> new limit.

`Auto` searches for the window with the most delivered throughput (a hill climber) and cuts on
overload; engine gauges, when there are any, feed the cuts and hold trials while a queue
stands. The controller never knows where an observation came from (HTTP scrape, in-process
engine, nowhere) — that's the SignalSource seam's job.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the controller stays free of I/O imports at run time
    from saturate.signals import Capacity

TICK_S = 2.0  # nominal tick period: the loop sleeps this long, and tick-count floors are multiples of it
BP_CEILING_RELEASE_S = 300.0  # backpressure-free seconds before the backpressure ceiling is released
MAX_SAMPLES = 16  # throughput samples kept: the most recent ticks only


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


# --- Auto: phases and state, all frozen ----------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """One tick's throughput: tokens/s when the engine reports them, and completions/s."""

    clock: float
    tok_s: float | None
    rows_s: float
    completions: int


@dataclass(frozen=True)
class Reading:
    """Mean throughput over some samples. `tok_s` is None unless every sample had tokens."""

    tok_s: float | None
    rows_s: float

    def ratio(self, other: Reading) -> float:
        """self / other, in tokens when both have them, else in completions (like with like)."""
        if self.tok_s is not None and other.tok_s is not None:
            return self.tok_s / other.tok_s if other.tok_s else float("inf")
        return self.rows_s / other.rows_s if other.rows_s else float("inf")

    @property
    def nothing(self) -> bool:
        """Nothing delivered, by either measure: no basis to compare a trial with."""
        return self.rows_s == 0 or self.tok_s == 0


@dataclass(frozen=True)
class Steady:
    """Holding the window. The next trial opens once `backoff` evidence windows have passed
    (`waits` counts them) and moves the window by MOVES[move]. `anchor` is the reading where a
    run of kept moves down began: every move in the run is judged against it, so small losses
    cannot add up."""

    waits: int = 0
    backoff: int = 1
    move: int = 0
    anchor: Reading | None = None


@dataclass(frozen=True)
class Trial:
    """The window moved from `origin` by MOVES[move] (`widen` = new / origin). It is judged
    against `ref`, the reading just before the move, once three ticks and two latencies at the
    trial window have passed: the latency at the start (or the first one seen, if none was
    known), times `widen` (at flat throughput latency grows with the window, by Little's law)."""

    origin: int
    ref: Reading  # the reading before the move (for a move down: the run's anchor)
    opened: float
    widen: float
    move: int
    backoff: int
    latency_s: float = 0.0  # the latency the trial's wait is based on (0 until one is known)
    ticks: int = 0

    def settled_by(self) -> float:
        """The clock time from which the trial window's own readings have arrived."""
        return self.opened + 2 * self.latency_s * max(1.0, self.widen)


@dataclass(frozen=True)
class Settling:
    """After a cut or a revert at clock `since`, the old window's requests drain until `until`
    (longer while the oldest request in flight predates the change); then `then`."""

    since: float
    until: float
    then: Steady


@dataclass(frozen=True)
class Ceiling:
    """The lowest window that met backpressure, at clock `met`; released BP_CEILING_RELEASE_S
    later, because a provider's limit can rise."""

    limit: int
    met: float


@dataclass(frozen=True)
class State:
    clock: float = 0.0  # seconds observed so far
    phase: Steady | Trial | Settling = Steady()
    ceiling: Ceiling | None = None
    seen_ceiling: Ceiling | None = None  # the last ceiling, kept after it is released
    samples: tuple[Sample, ...] = ()  # evidence since the last change: one per used tick
    latency_s: float | None = None  # the last request latency seen (timeouts report none)
    acked_ever: bool = False  # queue gauges mean nothing before the first completion ever
    silent_s: float = 0.0  # seconds of a full window with nothing completing
    stall_armed: bool = False  # a completion has arrived since the last change drained


Decision = tuple[int, str, State]


def _tick(obs: Obs) -> float:
    return obs.tick_s or TICK_S


def _below(ceiling: int) -> int:
    """The window an eighth under a backpressure ceiling: where growth stops."""
    return ceiling - max(1, ceiling // 8)


class Auto:
    """Find the window with the most throughput by trial and error; cut on overload.

    The search is a hill climber. Holding a window, Auto measures its throughput (tokens/s, or
    completions/s when tokens are not reported). Then it tries one move, a *trial*: the window
    times the next factor of MOVES (x2, x4, x1.5, x1.25, x7/8, then over again). Once the trial
    window's own readings have arrived, its throughput is compared with the reading just before
    the move:
    - kept: up by `improve` (5%) for a wider window, or not down by `improve` for a narrower
      one. A kept move up starts MOVES over at x2, so while each doubling pays, Auto keeps
      doubling (slow-start). A kept move down tries down again, and every move in such a run
      is judged against the reading where the run began, so small losses cannot add up.
    - reverted: anything else. Auto goes back, moves on to the next factor, and waits twice as
      many evidence windows before the next trial (1, 2, 4 ... 32). A plateau is retried
      rarely but never given up on, so a later rise in capacity is found.
    The x4 move crosses a stretch where throughput is flat (SGLang's from 32 to 64) that a
    doubling cannot see past; x7/8 trims a window that overshot the engine's cap. The window
    starts at `initial` if given; else at the engine's reported capacity when the engine
    reports one (begin: SGLang, llama.cpp, TEI); else at 64, so most of a short run is not
    spent ramping.

    A trial's readings arrive one request latency late, and a wider window has a longer latency
    at flat throughput (Little's law), so a trial is judged only after two latencies at its
    window and three ticks. A gain is kept as soon as it shows; a loss waits until then.

    decide() tries these rules in order; the first that applies sets the window and names
    itself in `last_reason`:

    1. hold:cooldown — after a cut or revert, hold while the old window's requests drain: at
       least max(2 ticks, the last latency seen) (3 ticks before any latency), longer while
       the oldest request in flight predates the change, never longer than the stall patience.
       Nothing the old window does meanwhile is evidence.
    2. cut:bp — backpressure (429/timeout/5xx) on at least `bp_frac` of this tick's outcomes:
       halve, and remember the window as a ceiling; growth then stops an eighth under it. A
       share, not any event: one row that always fails is not overload. At a ceiling met
       before (this one, or one released earlier), step just under it instead of halving.
       A ceiling is released after BP_CEILING_RELEASE_S, because a provider's limit can rise.
    3. cut:kv — KV high with a low (or absent) prefix-hit rate: halve.
    4. cut:stall — window full and nothing completing for `stall_ticks` ticks, or 3x the
       latency when known, whichever is longer: halve. Once per episode: the next needs a
       completion after the last change has drained.
    5. hold:input_bound — the source, not the engine, is starving admission.
    6. cut:queue — engine queue above target (waiting > hi): step down by the excess, at least
       `step`, at most half the window (the queue may hold other clients' requests). Not before
       the first completion ever: a cold engine queues the whole initial window.
    7. hold:kept / revert / hold:ack — a trial is open: judge it once its evidence is in.
    8. hold — window under 80% used (the source sets the pace), or a queue in band.
    9. hold:ack / hold:plateau / trial:up / trial:down / hold:ceiling / hold:floor — steady:
       wait for an evidence window (`limit` completions and two throughput samples), wait out
       the backoff, then open the next trial. A move that the bounds or the ceiling make empty
       is skipped; there is no move down at a backpressure ceiling (nothing waits there).

    A cut during a trial counts as a failed trial: the window goes no higher than where the
    trial began (except a queue cut, which steps by what the gauge shows). Every rule is a pure
    function of (Auto, State, Obs, limit) in RULES; the state is frozen dataclasses (State, and
    its phase: Steady, Trial or Settling). Durations are integrated in seconds from the
    observed `tick_s` (TICK_S when absent).
    """

    MOVES: tuple[float, ...] = (2.0, 4.0, 1.5, 1.25, 0.875)
    MAX_BACKOFF = 32  # evidence windows between trials on a long plateau

    DEFAULT_INITIAL = 64  # the start when neither the caller nor the engine says otherwise

    def __init__(self, target_waiting: int = 8, initial: int | None = None, min_limit: int = 2,
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
        self.step = step
        self.kv_hi, self.hits_lo, self.improve = kv_hi, hits_lo, improve
        self.stall_ticks, self.bp_frac = stall_ticks, bp_frac
        # the caller's configuration, never changed: each run's bounds are worked out from it (begin)
        self.config = (initial, min_limit, max_limit)
        self.begin(None)

    def decide(self, obs: Obs | dict, limit: int) -> int:
        obs = as_obs(obs)
        state = _observe(self, self.state, obs, limit)
        for rule in RULES:
            decision = rule(self, state, obs, limit)
            if decision is not None:
                new, self.last_reason, self.state = decision
                return new
        raise AssertionError("RULES must end in a rule that always applies")

    def begin(self, capacity: Capacity | None) -> None:
        """Start a run: this run's bounds and start, from the caller's configuration and the
        engine's reported `capacity` (None: none reported), and a fresh search state. The same
        Auto can run again, against the same engine or another.

        An engine that queues past its capacity starts above it, by min(capacity, 2 x
        target_waiting): a few waiting requests refill a slot as soon as it frees, which a window
        of exactly the capacity does not (llama.cpp, 8 slots: a window of 16 beat 8 by 9%), and
        that queue sits inside the band the queue cut allows. An engine that rejects past its
        capacity (TEI: 429s) starts at it and never goes above it, the caller's own start and
        bounds included. The caller's `initial` otherwise wins; with neither, DEFAULT_INITIAL."""
        initial, min_limit, max_limit = self.config
        self.min, self.max = min_limit, max_limit
        start = initial
        if capacity is not None and capacity.requests > 0:
            n = capacity.requests
            if not capacity.queues:  # a hard limit: nothing above it is ever admitted
                self.max, self.min = min(self.max, n), min(self.min, n)
            if start is None:
                start = n + min(n, self.hi) if capacity.queues else n
        start = self.DEFAULT_INITIAL if start is None else start
        self.initial = min(self.max, max(self.min, start))
        self.state = State()
        self.last_reason = "hold"

    # --- helpers ----------------------------------------------------------------------------------

    def clamp(self, state: State, n: int) -> int:
        """`n` within [min_limit, max_limit] and under any backpressure ceiling; min_limit wins."""
        top = self.max if state.ceiling is None else min(self.max, _below(state.ceiling.limit))
        return max(self.min, min(top, n))

    def patience_s(self, obs: Obs) -> float:
        """Seconds of a full, silent window that count as a stall."""
        floor = self.stall_ticks * _tick(obs)
        return floor if obs.latency_s is None else max(floor, 3 * obs.latency_s)

    def settle(self, state: State, obs: Obs, then: Steady) -> State:
        """Enter Settling: hold for max(2 ticks, the latency), 3 ticks if it is unknown."""
        tick = _tick(obs)
        latency = obs.latency_s if obs.latency_s is not None else state.latency_s
        hold = 3 * tick if latency is None else max(2 * tick, latency)
        return replace(state, phase=Settling(state.clock, state.clock + hold, then), samples=(),
                       silent_s=0.0, stall_armed=False)

    def cut(self, state: State, obs: Obs, limit: int, new: int, reason: str,
            keep_target: bool = False) -> Decision:
        """The one way down that is not a trial. A trial that meets a cut failed: the next trial
        tries the next move, and the window goes no higher than where the trial began, unless
        `keep_target` (a queue gauge says exactly how far to step down)."""
        then = Steady()
        if isinstance(state.phase, Trial):
            new = new if keep_target else min(new, state.phase.origin)
            then = Steady(backoff=min(self.MAX_BACKOFF, state.phase.backoff * 2), move=state.phase.move + 1)
        new = self.clamp(state, new)
        if new >= limit:  # at the floor already
            return limit, "hold:floor", replace(state, phase=then)
        return new, reason, self.settle(state, obs, then)

    def evidence(self, state: State, limit: int) -> Reading | None:
        """Mean throughput once `limit` completions and two samples are in, over the samples
        kept (the most recent MAX_SAMPLES ticks) but the first: the first after a change still
        describes the old window. A settled trial's samples still include some from before it
        settled, which dilutes a small gain at short latencies; judging on post-settle samples
        alone measured worse (more latency, a slower ramp), so this is a trade-off."""
        if sum(x.completions for x in state.samples) < limit or len(state.samples) < 2:
            return None  # `limit` completions within the samples kept, not before them
        samples = state.samples[1:]
        rows = sum(x.rows_s for x in samples) / len(samples)
        has_tokens = all(x.tok_s is not None for x in samples)
        tok = sum(x.tok_s or 0.0 for x in samples) / len(samples) if has_tokens else None
        return Reading(tok, rows)


# --- per-tick bookkeeping ----------------------------------------------------------------------------

def _observe(auto: Auto, state: State, obs: Obs, limit: int) -> State:
    tick = _tick(obs)
    state = replace(state, clock=state.clock + tick, acked_ever=state.acked_ever or obs.successes > 0,
                    latency_s=obs.latency_s if obs.latency_s is not None else state.latency_s)
    if state.ceiling is not None and state.clock - state.ceiling.met >= BP_CEILING_RELEASE_S:
        state = replace(state, ceiling=None, seen_ceiling=state.ceiling)  # a provider's limit can rise
    if isinstance(state.phase, Trial):
        # the latency at the trial's start, or the first one seen if none was known then; not a
        # later one, which an overshooting trial inflates with its own queueing
        latency = state.phase.latency_s or obs.latency_s or 0.0
        state = replace(state, phase=replace(state.phase, ticks=state.phase.ticks + 1, latency_s=latency))
    if isinstance(state.phase, Settling):
        age = state.clock - state.phase.since
        draining = obs.oldest_s is not None and obs.oldest_s > age and age <= auto.patience_s(obs)
        if state.clock <= state.phase.until or draining:
            return replace(state, silent_s=0.0, samples=())  # the old window is not evidence
        state = replace(state, phase=state.phase.then)  # drained
    silent = obs.inflight >= limit and obs.successes == 0
    state = replace(state, stall_armed=state.stall_armed or obs.successes > 0,
                    silent_s=state.silent_s + tick if silent else 0.0)
    if not obs.input_bound and obs.inflight >= int(limit * 0.8):  # only a used window is evidence
        sample = Sample(state.clock, obs.tok_s, obs.successes / tick, obs.successes)
        state = replace(state, samples=(state.samples + (sample,))[-MAX_SAMPLES:])
    return state


# --- rules, in priority order ----------------------------------------------------------------------

def _cooling(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    if isinstance(state.phase, Settling):
        return limit, "hold:cooldown", state
    return None


def _backpressure(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    if not (obs.backpressure and obs.backpressure >= auto.bp_frac * (obs.backpressure + obs.successes)):
        return None
    known = state.ceiling or state.seen_ceiling
    at_known = known is not None and _below(known.limit) <= limit <= known.limit * 9 // 8
    lowest = min(limit, state.ceiling.limit) if state.ceiling is not None else limit
    state = replace(state, ceiling=Ceiling(lowest, state.clock))
    new = _below(limit) if at_known else limit // 2  # a known limit: step under; fresh overload: halve
    return auto.cut(state, obs, limit, new, "cut:bp")


def _kv_pressure(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    # with no hits signal at all (prefix caching off), high KV is unverifiably benign -> cut
    if obs.kv is not None and obs.kv >= auto.kv_hi and (obs.hits is None or obs.hits < auto.hits_lo):
        return auto.cut(state, obs, limit, limit // 2, "cut:kv")
    return None


def _stall(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    if state.stall_armed and state.silent_s >= auto.patience_s(obs):
        return auto.cut(state, obs, limit, limit // 2, "cut:stall")
    return None


def _input_bound(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    if obs.input_bound:
        return limit, "hold:input_bound", state
    return None


def _queue(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    if state.acked_ever and obs.waiting is not None and obs.waiting > auto.hi:
        # step down by the excess the gauge shows, at least `step` and at most half the window
        # (the queue may hold other clients' requests); during a trial too: the queue says where
        # the engine's limit is, which is finer than going back to where the trial began
        down = min(max(auto.step, obs.waiting - auto.hi), limit // 2)
        return auto.cut(state, obs, limit, limit - down, "cut:queue", keep_target=True)
    return None


def _judge(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    """A trial holds its window until its evidence is in, then is kept or reverted."""
    if not isinstance(state.phase, Trial):
        return None
    trial = state.phase
    up = limit > trial.origin
    reading = auto.evidence(state, limit)
    ratio = reading.ratio(trial.ref) if reading is not None else None
    settled = trial.ticks >= 3 and state.clock >= trial.settled_by()
    gain = up and ratio is not None and ratio >= auto.improve  # a gain shows even through old readings
    if ratio is None or not (settled or gain):  # wait for the trial window's own readings
        return limit, "hold:ack", state
    kept = ratio >= auto.improve if up else ratio * auto.improve >= 1
    if kept:  # a gain starts MOVES over; a kept move down tries down again, against the same anchor
        steady = Steady() if up else Steady(move=trial.move, anchor=trial.ref)
        return limit, "hold:kept", replace(state, phase=steady, samples=())
    then = Steady(backoff=min(auto.MAX_BACKOFF, trial.backoff * 2), move=trial.move + 1)
    return auto.clamp(state, trial.origin), "revert", auto.settle(state, obs, then)


def _idle(auto: Auto, state: State, obs: Obs, limit: int) -> Decision | None:
    headroom = obs.waiting is None or obs.waiting < auto.lo
    if not (headroom and obs.inflight >= int(limit * 0.8)):
        return limit, "hold", state
    return None


def _open_trial(auto: Auto, state: State, obs: Obs, limit: int) -> Decision:
    """With an evidence window in and the backoff waited out, open the next trial."""
    reading = auto.evidence(state, limit)
    if reading is None or reading.nothing or state.silent_s > 0:  # nothing delivered: no basis for a trial
        return limit, "hold:ack", state
    phase = state.phase
    assert isinstance(phase, Steady)
    if phase.waits + 1 < phase.backoff:
        waiting = replace(phase, waits=phase.waits + 1)
        return limit, "hold:plateau", replace(state, phase=waiting, samples=())
    factor = auto.MOVES[phase.move % len(auto.MOVES)]
    new = auto.clamp(state, round(limit * factor))
    at_ceiling = state.ceiling is not None and limit >= _below(state.ceiling.limit)
    if new == limit or (factor < 1 and at_ceiling):  # nothing to try this way: the next move
        skipped = replace(state, phase=replace(phase, waits=0, move=phase.move + 1), samples=())
        at_floor = factor < 1 and not at_ceiling
        return limit, "hold:floor" if at_floor else "hold:ceiling", skipped
    ref = phase.anchor if new < limit and phase.anchor is not None else reading
    trial = Trial(origin=limit, ref=ref, opened=state.clock, widen=new / limit, move=phase.move,
                  backoff=phase.backoff, latency_s=state.latency_s or 0.0)
    return new, "trial:up" if new > limit else "trial:down", replace(state, phase=trial, samples=())


Rule = Callable[[Auto, State, Obs, int], "Decision | None"]
RULES: tuple[Rule, ...] = (_cooling, _backpressure, _kv_pressure, _stall, _input_bound, _queue,
                           _judge, _idle, _open_trial)
