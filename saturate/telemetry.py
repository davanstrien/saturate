"""Telemetry v1 (frozen keys, CONTRACT §6) + the run-end advisor."""

from __future__ import annotations

from collections import Counter


def tick_record(t: float, limit: int, inflight: int, gauges: dict | None,
                bp: int, ok: int, input_bound: bool, tok_s: float, reason: str,
                latency_s: float | None = None, bound_by: str | None = None,
                source_s: float = 0.0, prep_s: float = 0.0) -> dict:
    g = gauges or {}
    return {"t": round(t, 1), "limit": limit, "inflight": inflight,
            "waiting": g.get("waiting"), "running": g.get("running"),
            "bp": bp, "ok": ok, "input_bound": input_bound,
            "tok_s": round(tok_s, 1), "kv": g.get("kv"), "hits": g.get("hits"),
            "preempts": g.get("preempts"), "reason": reason,
            "latency_s": None if latency_s is None else round(latency_s, 3),
            "bound_by": bound_by, "source_s": round(source_s, 3), "prep_s": round(prep_s, 3)}


def cut_reasons(telemetry: list[dict]) -> dict[str, int]:
    """How many times the window was reduced, by the controller's stated reason (`cut:*`)."""
    return dict(Counter(t["reason"] for t in telemetry if str(t.get("reason", "")).startswith("cut:")))


def advise_input(telemetry: list[dict]) -> list[str]:
    """Client-side bottlenecks: when at least 30% of ticks were bound by `to_request` (prep)
    or by the source iterator, say so with the measured cost per row and the remedy."""
    if not telemetry:
        return []
    verdicts = Counter(t.get("bound_by") for t in telemetry)
    rows = max(1, sum(t.get("ok", 0) for t in telemetry))
    hints = []
    if verdicts["prep"] / len(telemetry) >= 0.3:
        ms = 1000 * sum(t.get("prep_s", 0.0) for t in telemetry) / rows
        hints.append(f"PREP-BOUND: to_request took {ms:.0f} ms/row on average; run it ahead with "
                     "pump(prepare_workers=4)")
    if verdicts["source"] / len(telemetry) >= 0.3:
        ms = 1000 * sum(t.get("source_s", 0.0) for t in telemetry) / rows
        hints.append(f"SOURCE-BOUND: the source iterator took {ms:.0f} ms/row; prefetch or shard the input")
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
