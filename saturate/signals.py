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

Not every engine publishes its gauges at /metrics either. trtllm-serve answers /metrics with
JSON iteration stats and clears them on each read, so polling it would take them from every
other reader; its Prometheus gauges are at /prometheus/metrics. The startup `probe` names the
engine where it can, so the scrape reads the right path from the first tick.
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


# where an engine publishes its Prometheus gauges, when not at /metrics
METRICS_PATH = {"trtllm": "/prometheus/metrics"}

# what turns an engine's gauges on: the blind-mode warning names it
METRICS_FLAG = {
    "vllm": "vLLM publishes them unless started with --disable-log-stats",
    "sglang": "start SGLang with --enable-metrics",
    "llamacpp": "start llama-server with --metrics",
    "trtllm": ("set return_perf_metrics: true and enable_iter_perf_stats: true in the "
               "--extra_llm_api_options YAML (the queue gauges need TensorRT-LLM 1.3 or later)"),
    "tei": "TEI publishes them on its main port",
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
    """Poll the engine's metrics path (/metrics, or its own: METRICS_PATH). After `max_fails`
    consecutive misses the controller runs blind (metrics are opt-in on SGLang, llama.cpp and
    TRT-LLM, and an exporter can come up after /health does), but one probe every `retry_every`
    reads keeps the door open; the skipped reads cost no request. The blind-mode warning says
    why the last read missed and, for a known engine, the flag that turns its gauges on."""

    def __init__(self, client: HttpClient, endpoint: str, max_fails: int = 5,
                 retry_every: int = 30):
        self.base = endpoint.rsplit("/v1", 1)[0]
        self.url = f"{self.base}/metrics"
        self.client = client
        self.max_fails = max_fails
        self.retry_every = retry_every
        self._fails = 0
        self._blind = False
        self._skipped = 0
        self._off = False  # the engine reported its metrics endpoint off: no read can succeed
        self.engine: str | None = None
        self.reason: str | None = None  # why the last read had no gauges
        self.dialect: str | None = None

    def identify(self, probe: Probe) -> None:
        """Scrape the probed engine's own metrics path. An engine that reports its metrics
        endpoint off (llama.cpp) is never scraped: the run is blind from the start."""
        self.engine = probe.engine
        self.dialect = self.dialect or probe.engine
        self.url = self.base + METRICS_PATH.get(probe.engine or "", "/metrics")
        if probe.metrics_on is False:
            self._off = True
            self.reason = "the engine reports its metrics endpoint off"

    async def read(self) -> dict | None:
        if self._off:
            if not self._blind:
                self._blind = True
                log.warning(f"{self.engine} reports its metrics endpoint off — running blind; "
                            f"{self._hint().removesuffix('; ')}")
            return None
        if self._fails >= self.max_fails:
            if not self._blind:
                self._blind = True
                self._skipped = 0
                log.warning(f"metrics scrape unavailable at {self.url} ({self.reason}) — "
                            f"running blind; {self._hint()}will retry periodically")
            self._skipped += 1
            if self._skipped < self.retry_every:
                return None
            self._skipped = 0
        try:
            r = await self.client.get(self.url, timeouts=SCRAPE_TIMEOUTS)
        except TransportError:
            return self._miss("unreachable")
        if r.status_code != 200:
            return self._miss(f"HTTP {r.status_code}")
        if self.dialect is None and r.text.lstrip()[:1] == "[":
            # a JSON list at /metrics from a server not yet named: trtllm-serve's iteration
            # stats (a release without /server_info for the probe), which each read clears.
            # Never read them again. A JSON object (a gateway's error body) is an ordinary miss,
            # and so is any JSON once the probe or the gauges have named the engine.
            self.engine = self.dialect = "trtllm"
            self.url = self.base + METRICS_PATH["trtllm"]
            log.info(f"JSON at {self.base}/metrics: TensorRT-LLM; scraping {self.url} instead")
            return self._miss("TensorRT-LLM's /metrics is not Prometheus")
        gauges = parse_gauges(r.text)
        if gauges is None:  # a 200 that is not one of the engines' gauges
            return self._miss("no engine gauges in the reply")
        self._fails = 0
        self.reason = None
        if gauges["dialect"]:
            self.dialect = gauges["dialect"]
        if self._blind:
            self._blind = False
            log.info(f"metrics scrape recovered ({self.dialect})")
        return gauges

    def _miss(self, reason: str) -> None:
        self._fails += 1
        self.reason = reason

    def _hint(self) -> str:
        flag = METRICS_FLAG.get(self.engine or self.dialect or "")  # probed, else named by gauges
        return f"{flag}; " if flag else ""


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


def _trtllm_server_info(body) -> int | None:
    """trtllm-serve /server_info (from 1.3; it always carries disaggregated_params): its
    max_batch_size, past which it queues."""
    if not isinstance(body, dict) or "disaggregated_params" not in body:
        return None
    value = body.get("max_batch_size")
    return value if _count(value) else None


# where each engine reports its capacity, in order of preference: (path from the server root,
# reader, engine, queues past it). Checked on live servers 2026-10-28 (trtllm-serve /server_info
# from its source); vLLM reports none.
CAPACITY_SOURCES: tuple[tuple[str, Callable[[object], int | None], str, bool], ...] = (
    ("/v1/loads", _sglang_loads, "sglang", True),
    ("/get_server_info", _top_level("max_running_requests", "max_prefill_tokens"), "sglang", True),
    ("/props", _top_level("total_slots", "default_generation_settings"), "llamacpp", True),
    ("/info", _top_level("max_concurrent_requests", "max_batch_tokens", "model_id"), "tei", False),
    ("/server_info", _trtllm_server_info, "trtllm", True),
)
PROBE_TIMEOUTS = Timeouts(read=2.0, connect=2.0, pool=2.0, write=2.0)  # a one-off read: fail fast


@dataclasses.dataclass(frozen=True)
class Probe:
    """What the engine says about itself before the first request: which engine it is, its
    capacity, and whether its metrics endpoint is on. None where it does not say."""

    engine: str | None = None
    capacity: Capacity | None = None
    metrics_on: bool | None = None


async def _get_json(client: HttpClient, url: str) -> object:
    try:
        reply = await client.get(url, timeouts=PROBE_TIMEOUTS)
        return reply.json() if reply.status_code == 200 else None
    except (TransportError, ValueError):  # unreachable, or not JSON
        return None


# the keys that name an engine in a probe body, whether or not it reports a capacity there
ENGINE_SIGNATURES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("/get_server_info", ("max_prefill_tokens",), "sglang"),
    ("/props", ("default_generation_settings",), "llamacpp"),
    ("/info", ("model_id", "max_batch_tokens"), "tei"),
    ("/server_info", ("disaggregated_params",), "trtllm"),
)


def _engine(bodies: dict[str, object]) -> str | None:
    """The engine a probe body's shape names, even when it reports no capacity."""
    for path, keys, engine in ENGINE_SIGNATURES:
        body = bodies[path]
        if isinstance(body, dict) and all(k in body for k in keys):
            return engine
    return None


async def probe(client: HttpClient, endpoint: str) -> Probe:
    """Ask the engine what it is, once, before the first request. The capacity comes from the
    first source (in CAPACITY_SOURCES order) that reports one; llama.cpp's /props also says
    whether its metrics endpoint is on. Every path is asked at once, with short timeouts, so the
    probe costs one round trip, two seconds at most. vLLM, a hosted API or a proxy: Probe()."""
    import asyncio

    base = endpoint.rsplit("/v1", 1)[0]
    paths = [path for path, _, _, _ in CAPACITY_SOURCES]
    bodies = dict(zip(paths, await asyncio.gather(*(_get_json(client, f"{base}{p}") for p in paths)),
                      strict=True))
    capacity = None
    for path, reader, engine, queues in CAPACITY_SOURCES:
        requests = reader(bodies[path])
        if requests is not None:
            capacity = Capacity(requests, queues, engine)
            break
    engine = capacity.engine if capacity is not None else _engine(bodies)
    props = bodies["/props"]
    metrics_on = props.get("endpoint_metrics") if engine == "llamacpp" and isinstance(props, dict) else None
    return Probe(engine, capacity, metrics_on if isinstance(metrics_on, bool) else None)


async def read_capacity(client: HttpClient, endpoint: str) -> Capacity | None:
    """The engine's capacity (see `probe`), or None."""
    return (await probe(client, endpoint)).capacity
