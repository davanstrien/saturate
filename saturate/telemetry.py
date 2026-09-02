"""Telemetry v1 (frozen keys, CONTRACT §6) + the run-end advisor."""

from __future__ import annotations

from collections import Counter


def tick_record(t: float, limit: int, inflight: int, gauges: dict | None,
                bp: int, ok: int, input_bound: bool, tok_s: float, reason: str,
                latency_s: float | None = None, bound_by: str | None = None,
                source_s: float = 0.0, prep_s: float = 0.0, prep_n: int = 0, prep_workers: int = 0,
                loop_lag_s: float = 0.0) -> dict:
    g = gauges or {}
    return {"t": round(t, 1), "limit": limit, "inflight": inflight,
            "waiting": g.get("waiting"), "running": g.get("running"),
            "bp": bp, "ok": ok, "input_bound": input_bound,
            "tok_s": round(tok_s, 1), "kv": g.get("kv"), "hits": g.get("hits"),
            "preempts": g.get("preempts"), "reason": reason,
            "latency_s": None if latency_s is None else round(latency_s, 3),
            "bound_by": bound_by, "source_s": round(source_s, 3), "prep_s": round(prep_s, 3),
            "prep_n": prep_n, "prep_workers": prep_workers, "loop_lag_s": round(loop_lag_s, 3)}


def cut_reasons(telemetry: list[dict]) -> dict[str, int]:
    """How many times the window was reduced, by the controller's stated reason (`cut:*`)."""
    return dict(Counter(t["reason"] for t in telemetry if str(t.get("reason", "")).startswith("cut:")))


def bound_by_counts(telemetry: list[dict]) -> dict[str, int]:
    """Ticks per `bound_by` verdict (engine / source / prep / loop); inconclusive ticks are not counted."""
    return dict(Counter(t["bound_by"] for t in telemetry if t.get("bound_by")))


def advise_input(telemetry: list[dict], prepare_workers: int = 0) -> list[str]:
    """Client-side bottlenecks: when at least 30% of ticks were bound by `to_request` (prep),
    by the source iterator, or by a blocked event loop, say so with what was measured and
    the remedy. A run under three ticks is too short to diagnose (start-up and the sink's
    first flushes dominate it)."""
    if len(telemetry) < 3:
        return []
    counts = bound_by_counts(telemetry)
    hints = []
    if counts.get("prep", 0) / len(telemetry) >= 0.3:
        calls = max(1, sum(t.get("prep_n", 0) for t in telemetry))
        ms = 1000 * sum(t.get("prep_s", 0.0) for t in telemetry) / calls
        if prepare_workers >= 4:
            hints.append(f"PREP-BOUND: prep is still the bottleneck at {prepare_workers} workers "
                         f"({ms:.0f} ms/row): raise prepare_workers, or move the work to prepare_ahead "
                         "with a process pool")
        else:
            hints.append(f"PREP-BOUND: to_request took {ms:.0f} ms/row on average; if it releases the "
                         "GIL (Pillow, ffmpeg, I/O) set pump(prepare_workers=4), otherwise prepare rows "
                         "ahead of the pump with prepare_ahead(rows, fn, executor=ProcessPoolExecutor())")
    if counts.get("source", 0) / len(telemetry) >= 0.3:
        rows = max(1, sum(t.get("ok", 0) + t.get("bp", 0) for t in telemetry))
        ms = 1000 * sum(t.get("source_s", 0.0) for t in telemetry) / rows
        hints.append(f"SOURCE-BOUND: the source took {ms:.0f} ms/row; prefetch it "
                     "(prepare_ahead / bucket_rows(prefetch=)), select fewer columns, or shard the input")
    if counts.get("loop", 0) / len(telemetry) >= 0.3:
        run_s = max(telemetry[-1].get("t", 0.0), 1e-9)
        pct = min(100, round(100 * sum(t.get("loop_lag_s", 0.0) for t in telemetry) / run_s))
        hints.append(f"LOOP-BOUND: the event loop was blocked {pct}% of the run (parse, sink writes, "
                     "and any to_request share under the prep threshold); keep parse cheap, raise "
                     "flush_every, and move to_request work off the loop")
    return hints


def advise(telemetry: list[dict], dialect: str | None, final_limit: int,
           ceiling_flags: dict) -> list[str]:
    """Server-ceiling detection: `running` modally pinned below our limit while
    a queue stands — the server, not the client, is the bottleneck; hand back
    the exact relaunch flag."""
    ticks = [t for t in telemetry if t.get("running") is not None]
    if len(ticks) < 5:
        return []
    queued = [t for t in ticks if (t.get("waiting") or 0) > 0]
    if len(queued) / len(ticks) < 0.5:
        return []  # no standing queue -> engine wasn't the ceiling
    mode, freq = Counter(t["running"] for t in queued).most_common(1)[0]
    if freq / len(queued) >= 0.7 and mode < final_limit:
        flag = ceiling_flags.get(dialect or "", "raise the server's max concurrent requests")
        return [f"server ceiling: running pinned at {mode} with a standing queue for "
                f"{100 * len(queued) // len(ticks)}% of the run — relaunch with "
                f"{flag.format(n=mode * 2)}"]
    return []
