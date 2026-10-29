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

import dataclasses
import logging
import re
from collections.abc import Callable

from saturate.http import HttpClient, Timeouts, TransportError

log = logging.getLogger(__name__)

_PAT = re.compile(
    r"^(?:"
    r"vllm[:_](?:num_requests_(?P<v>waiting|running)"
    r"|(?:gpu_cache|kv_cache)_usage_perc(?P<vkv>)"
    r"|gpu_prefix_cache_hit_rate(?P<vh>)"
    r"|num_preemptions_total(?P<vp>))"
    r"|sglang[:_](?:num_(?P<s>queue|running)_reqs"
    r"|token_usage(?P<skv>)"
    r"|cache_hit_rate(?P<sh>)"
    r"|num_retracted_requests_total(?P<sp>))"
    r"|llamacpp[:_]requests_(?P<l>deferred|processing)"
    r"|trtllm_num_requests_(?P<t>waiting|running)"
    r"|te[:_]queue_size(?P<eq>)"
    r")(?:\{[^}]*\})?\s+(?P<val>[0-9.eE+-]+)",
    re.M,
)

_DIALECT = {"v": "vllm", "vkv": "vllm", "vh": "vllm", "vp": "vllm",
            "s": "sglang", "skv": "sglang", "sh": "sglang", "sp": "sglang", "l": "llamacpp", "t": "trtllm",
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
    """Extract waiting/running/kv/hits/preempts + dialect from a /metrics body. `preempts` is
    the engine's cumulative preemption count (SGLang: retractions), summed over its series."""
    out: dict = {"waiting": None, "running": None, "kv": None, "hits": None,
                 "preempts": None, "dialect": None}
    seen = False
    for m in _PAT.finditer(text):
        val = float(m.group("val"))
        for g in ("v", "s", "l", "t", "vkv", "vh", "vp", "skv", "sh", "sp", "eq"):
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
                elif g in ("vp", "sp"):  # a counter per engine or label set: their sum
                    out["preempts"] = (out["preempts"] or 0) + int(val)
                break
    return out if seen else None


class Null:
    """Blind mode: no gauges — the controller runs on throughput + backpressure."""

    dialect: str | None = None

    async def read(self) -> dict | None:
        return None



SCRAPE_TIMEOUTS = Timeouts(read=5.0, connect=5.0, pool=5.0)


class HttpScrape:
    """Poll the engine's /metrics. After `max_fails` consecutive failures the
    controller runs blind (metrics are opt-in on SGLang/llama.cpp, and an
    exporter can come up after /health does), but one probe every `retry_every`
    reads keeps the door open; the skipped reads cost no request."""

    def __init__(self, client: HttpClient, endpoint: str, max_fails: int = 5,
                 retry_every: int = 30):
        base = endpoint.rsplit("/v1", 1)[0]
        self.url = f"{base}/metrics"
        self.client = client
        self.max_fails = max_fails
        self.retry_every = retry_every
        self._fails = 0
        self._blind = False
        self._skipped = 0
        self.dialect: str | None = None

    async def read(self) -> dict | None:
        if self._fails >= self.max_fails:
            if not self._blind:
                self._blind = True
                self._skipped = 0
                log.warning(f"metrics scrape unavailable at {self.url} — running blind; "
                            "will retry periodically")
            self._skipped += 1
            if self._skipped < self.retry_every:
                return None
            self._skipped = 0
        try:
            r = await self.client.get(self.url, timeouts=SCRAPE_TIMEOUTS)
            gauges = parse_gauges(r.text)
        except TransportError:  # parse_gauges returns None for text it cannot read
            self._fails += 1
            return None
        if gauges is None:
            self._fails += 1
            return None
        self._fails = 0
        if gauges["dialect"]:
            self.dialect = gauges["dialect"]
        if self._blind:
            self._blind = False
            log.info(f"metrics scrape recovered ({self.dialect})")
        return gauges


@dataclasses.dataclass(frozen=True)
class Capacity:
    """How many requests an engine runs at once, and whether it queues the rest (else it
    rejects them, as TEI does with 429s)."""

    requests: int
    queues: bool
    engine: str


def _sglang_loads(body) -> int | None:
    """SGLang /v1/loads: one entry per data-parallel rank (under "loads", or the body itself);
    the engine's capacity is their sum."""
    entries = body.get("loads", [body]) if isinstance(body, dict) else body
    if not isinstance(entries, list) or not entries:
        return None
    caps = [e.get("max_running_requests") for e in entries if isinstance(e, dict) and "num_running_reqs" in e]
    return sum(caps) if caps and len(caps) == len(entries) and all(_count(c) for c in caps) else None


def _count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _top_level(key: str, *signature: str) -> Callable[[object], int | None]:
    """A count under `key` at the top level of a JSON object that also has every `signature` key:
    the shape that names the engine, so another server's /info or /props does not pass for it."""
    def read(body) -> int | None:
        if not isinstance(body, dict) or not all(k in body for k in signature):
            return None
        value = body.get(key)
        return value if _count(value) else None
    return read


# where each engine reports its capacity, in order of preference: (path from the server root,
# reader, engine, queues past it). Checked on live servers 2026-10-28; vLLM reports none.
CAPACITY_SOURCES: tuple[tuple[str, Callable[[object], int | None], str, bool], ...] = (
    ("/v1/loads", _sglang_loads, "sglang", True),
    ("/get_server_info", _top_level("max_running_requests", "max_prefill_tokens"), "sglang", True),
    ("/props", _top_level("total_slots", "default_generation_settings"), "llamacpp", True),
    ("/info", _top_level("max_concurrent_requests", "max_batch_tokens", "model_id"), "tei", False),
)
CAPACITY_TIMEOUTS = Timeouts(read=2.0, connect=2.0, pool=2.0, write=2.0)  # a one-off read: fail fast


async def _read_one(client: HttpClient, url: str, reader) -> int | None:
    try:
        reply = await client.get(url, timeouts=CAPACITY_TIMEOUTS)
        return reader(reply.json()) if reply.status_code == 200 else None
    except (TransportError, ValueError):  # unreachable, or not JSON
        return None


async def read_capacity(client: HttpClient, endpoint: str) -> Capacity | None:
    """The engine's capacity from the first source (in CAPACITY_SOURCES order) that reports one,
    or None (vLLM, a hosted API, a proxy). Every source is asked at once, with short timeouts, so
    the read costs one round trip, two seconds at most."""
    import asyncio

    base = endpoint.rsplit("/v1", 1)[0]
    found = await asyncio.gather(*(_read_one(client, f"{base}{path}", reader)
                                   for path, reader, _, _ in CAPACITY_SOURCES))
    for requests, (_, _, engine, queues) in zip(found, CAPACITY_SOURCES, strict=True):
        if requests is not None:
            return Capacity(requests, queues, engine)
    return None
