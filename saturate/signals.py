"""SignalSource seam: where controller observations come from.

Implementations: HttpScrape (poll the engine's /metrics — MVP), Null (blind
mode: hosted APIs, opaque LBs), and post-v1 an in-process source reading
engine internals directly. The controller never knows which one fed it.

The scrape table (METRICS: one row per engine metric) is keyed on the GAIE
model-server-protocol semantics (queued / running / kv-util) rather than exact
strings, and matches BOTH prefix spellings per engine (`sglang:` and `sglang_`). The exporters
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
import math
from collections.abc import Callable, Iterator
from typing import Literal, NamedTuple

from saturate.http import HttpClient, Timeouts, TransportError

log = logging.getLogger(__name__)

class Metric(NamedTuple):
    """One engine metric the scrape reads: which `field` of the gauges it feeds, and how the
    values of its series (one per engine or label set) combine. "sum": a count, summed. "last":
    a ratio gauge, one per engine. "series": kept per label set under `key`, for HttpScrape to
    difference series by series."""

    engine: str
    name: str  # without the engine prefix; matched as `prefix:name` and `prefix_name`
    field: str
    combine: Literal["sum", "last", "series"]
    key: str | None = None


METRICS: tuple[Metric, ...] = (
    Metric("vllm", "num_requests_waiting", "waiting", "sum"),
    Metric("vllm", "num_requests_running", "running", "sum"),
    Metric("vllm", "kv_cache_usage_perc", "kv", "last"),
    Metric("vllm", "gpu_cache_usage_perc", "kv", "last"),  # before 0.12
    Metric("vllm", "gpu_prefix_cache_hit_rate", "hits", "last"),  # V0; V1 has the counters below
    Metric("vllm", "num_preemptions_total", "preempts", "sum"),
    Metric("vllm", "prefix_cache_queries_total", "prefix_counts", "series", "queries"),
    Metric("vllm", "prefix_cache_hits_total", "prefix_counts", "series", "hits"),
    Metric("vllm", "gpu_prefix_cache_queries_total", "prefix_counts", "series", "queries"),  # 0.8-0.9
    Metric("vllm", "gpu_prefix_cache_hits_total", "prefix_counts", "series", "hits"),
    Metric("sglang", "num_queue_reqs", "waiting", "sum"),
    Metric("sglang", "num_running_reqs", "running", "sum"),
    Metric("sglang", "token_usage", "kv", "last"),
    Metric("sglang", "cache_hit_rate", "hits", "last"),
    Metric("sglang", "num_retracted_requests_total", "preempts", "sum"),
    Metric("llamacpp", "requests_deferred", "waiting", "sum"),
    Metric("llamacpp", "requests_processing", "running", "sum"),
    Metric("trtllm", "num_requests_waiting", "waiting", "sum"),
    Metric("trtllm", "num_requests_running", "running", "sum"),
    Metric("tei", "queue_size", "waiting", "sum"),
)

# the metric-name prefix of an engine whose prefix is not its name
_PREFIX = {"tei": "te"}

# every accepted spelling of every metric's full name -> its Metric
_BY_NAME = {f"{_PREFIX.get(m.engine, m.engine)}{sep}{m.name}": m for m in METRICS for sep in (":", "_")}


def _samples(text: str) -> Iterator[tuple[str, str, float]]:
    """(name, labels, value) for each sample line of a Prometheus text body. Comments, blank
    lines and a value that is not a finite number (NaN, +Inf) are skipped; an optional
    timestamp after the value is ignored."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        brace = line.find("{")
        space = line.find(" ")
        if brace != -1 and (space == -1 or brace < space):
            close = line.rfind("}")
            if close < brace:
                continue
            name, labels, rest = line[:brace], line[brace:close + 1], line[close + 1:]
        else:
            name, labels, rest = (line[:space], "", line[space:]) if space != -1 else (line, "", "")
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        if math.isfinite(value):
            yield name, labels, value


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
    """Extract waiting/running/kv/hits/preempts + dialect from a /metrics body, as METRICS
    says. `preempts` is the engine's cumulative preemption count (SGLang: retractions), summed
    over its series. `prefix_counts` is vLLM's cumulative prefix-cache token counts per series
    (label set -> {"queries": n, "hits": n}): vLLM publishes no hit-rate gauge, so HttpScrape
    derives `hits` from two readings, series by series, so that one engine's reset is not
    hidden in a sum. None when no line is a known metric."""
    out: dict = {"waiting": None, "running": None, "kv": None, "hits": None,
                 "preempts": None, "prefix_counts": None, "dialect": None}
    seen = False
    for name, labels, value in _samples(text):
        metric = _BY_NAME.get(name)
        if metric is None:
            continue
        seen = True
        out["dialect"] = metric.engine
        if metric.combine == "sum":
            out[metric.field] = (out[metric.field] or 0) + int(value)
        elif metric.combine == "last":
            out[metric.field] = value
        else:
            series = out[metric.field] = out[metric.field] or {}
            series.setdefault(labels, {})[metric.key] = value
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
        self._prefix: dict | None = None  # vLLM's prefix-cache counters at the last good read

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
        rate = self._hit_rate(gauges["prefix_counts"])  # every read, so the baseline never goes stale
        if gauges["hits"] is None:
            gauges["hits"] = rate
        if gauges["dialect"]:
            self.dialect = gauges["dialect"]
        if self._blind:
            self._blind = False
            log.info(f"metrics scrape recovered ({self.dialect})")
        return gauges

    def _hit_rate(self, counts: dict | None) -> float | None:
        """vLLM's prefix-cache hit rate since the last good read: new hit tokens over new query
        tokens, over every series. None (counted as no evidence of reuse) on the first reading,
        after a missed read, when no query arrived since the last one, and when the readings do
        not pair up: a series appeared, vanished or lacks one of its two counters, or a counter
        went down (an engine restarted). vLLM 0.12 or later: before it, the counters also count
        the recompute of preempted requests, which reads as reuse."""
        last, self._prefix = self._prefix, counts
        if counts is None or last is None or counts.keys() != last.keys():
            return None
        queries = hits = 0.0
        for labels, now in counts.items():
            then = last[labels]
            if len(now) != 2 or len(then) != 2:
                return None
            dq, dh = now["queries"] - then["queries"], now["hits"] - then["hits"]
            if dq < 0 or dh < 0:
                return None
            queries, hits = queries + dq, hits + dh
        return min(hits / queries, 1.0) if queries > 0 else None

    def _miss(self, reason: str) -> None:
        self._fails += 1
        self._prefix = None  # no rate across a gap: the next good read is a new baseline
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
