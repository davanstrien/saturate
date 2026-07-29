"""SignalSource seam: where controller observations come from.

Implementations: HttpScrape (poll the engine's /metrics — MVP), Null (blind
mode: hosted APIs, opaque LBs), and post-v1 an in-process source reading
engine internals directly. The controller never knows which one fed it.

The scrape table is keyed on the GAIE model-server-protocol semantics
(queued / running / kv-util) rather than exact strings, and matches BOTH
prefix spellings per engine (`sglang:` and `sglang_`). The exporters
themselves emit colons today; the dual spelling costs one character and
absorbs a collector that normalises them away (Prometheus relabelling,
OpenTelemetry) or a future rename. Neither engine has actually flipped:
sgl-project/sglang#12618 was a collector-side dashboard fix, and vLLM's
proposed flip was rejected (vllm-project/vllm#33824).

Not every engine exposes the full set. TEI publishes queue depth only
(`te_queue_size`) — no running/in-flight gauge, no KV — so `parse_gauges`
returns a partial dict and the controller runs on the one real signal.
"""

from __future__ import annotations

import re

import httpx2 as httpx

_PAT = re.compile(
    r"^(?:"
    r"vllm[:_](?:num_requests_(?P<v>waiting|running)"
    r"|(?:gpu_cache|kv_cache)_usage_perc(?P<vkv>)"
    r"|gpu_prefix_cache_hit_rate(?P<vh>)"
    r"|num_preemptions_total(?P<vp>))"
    r"|sglang[:_](?:num_(?P<s>queue|running)_reqs"
    r"|token_usage(?P<skv>)"
    r"|cache_hit_rate(?P<sh>))"
    r"|llamacpp[:_]requests_(?P<l>deferred|processing)"
    r"|trtllm_num_requests_(?P<t>waiting|running)"
    r"|te[:_]queue_size(?P<eq>)"
    r")(?:\{[^}]*\})?\s+(?P<val>[0-9.eE+-]+)",
    re.M,
)

_DIALECT = {"v": "vllm", "vkv": "vllm", "vh": "vllm", "vp": "vllm",
            "s": "sglang", "skv": "sglang", "sh": "sglang", "l": "llamacpp", "t": "trtllm",
            "eq": "tei"}

# boot-frozen server-side ceilings: the client can only diagnose them and hand
# back the exact relaunch flag (advisor, not tuner)
CEILING_FLAG = {
    "vllm": "--max-num-seqs {n}",
    "sglang": "--max-running-requests {n}",
    "llamacpp": "-np {n} (per-slot context halves unless -c is raised too)",
    "trtllm": "--max_batch_size {n}",
    "tei": "--max-concurrent-requests {n} (rejects with 429 above the cap — TEI does not queue)",
}


def parse_gauges(text: str) -> dict | None:
    """Extract waiting/running/kv/hits/preempts + dialect from a /metrics body."""
    out: dict = {"waiting": None, "running": None, "kv": None, "hits": None,
                 "preempts": None, "dialect": None}
    seen = False
    for m in _PAT.finditer(text):
        val = float(m.group("val"))
        for g in ("v", "s", "l", "t", "vkv", "vh", "vp", "skv", "sh", "eq"):
            if m.group(g) is not None:
                seen = True
                out["dialect"] = _DIALECT[g]
                if g in ("v", "s", "l", "t"):
                    kind = m.group(g)
                    key = "waiting" if kind in ("waiting", "queue", "deferred") else "running"
                    out[key] = (out[key] or 0) + int(val)
                elif g == "eq":
                    out["waiting"] = (out["waiting"] or 0) + int(val)
                elif g in ("vkv", "skv"):
                    out["kv"] = val
                elif g in ("vh", "sh"):
                    out["hits"] = val
                elif g == "vp":
                    out["preempts"] = int(val)
                break
    return out if seen else None


class Null:
    """Blind mode: no gauges — the controller runs on throughput + backpressure."""

    dialect: str | None = None

    async def read(self) -> dict | None:
        return None


class HttpScrape:
    """Poll the engine's /metrics; give up gracefully after `max_fails`
    consecutive failures (metrics are opt-in on SGLang/llama.cpp)."""

    def __init__(self, client: httpx.AsyncClient, endpoint: str, max_fails: int = 5):
        base = endpoint.rsplit("/v1", 1)[0]
        self.url = f"{base}/metrics"
        self.client = client
        self.max_fails = max_fails
        self._fails = 0
        self.dialect: str | None = None

    async def read(self) -> dict | None:
        if self._fails >= self.max_fails:
            return None
        try:
            r = await self.client.get(self.url, timeout=5)
            gauges = parse_gauges(r.text)
        except Exception:
            self._fails += 1
            return None
        if gauges is None:
            self._fails += 1
            return None
        self._fails = 0
        if gauges["dialect"]:
            self.dialect = gauges["dialect"]
        return gauges
