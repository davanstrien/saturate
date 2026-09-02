"""Telemetry v1 (frozen keys, CONTRACT §6) + the run-end advisor."""

from __future__ import annotations

from collections import Counter


def tick_record(t: float, limit: int, inflight: int, gauges: dict | None,
                bp: int, ok: int, input_bound: bool, tok_s: float,
                reason: str = "hold", latency_s: float | None = None) -> dict:
    g = gauges or {}
    return {"t": round(t, 1), "limit": limit, "inflight": inflight,
            "waiting": g.get("waiting"), "running": g.get("running"),
            "bp": bp, "ok": ok, "input_bound": input_bound,
            "tok_s": round(tok_s, 1), "kv": g.get("kv"), "hits": g.get("hits"),
            "preempts": g.get("preempts"), "reason": reason,
            "latency_s": None if latency_s is None else round(latency_s, 3)}


def cut_reasons(telemetry: list[dict]) -> dict[str, int]:
    """How many times the window was reduced, by the controller's stated reason (`cut:*`)."""
    return dict(Counter(t["reason"] for t in telemetry if str(t.get("reason", "")).startswith("cut:")))


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
