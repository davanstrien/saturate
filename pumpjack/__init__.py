"""pumpjack — adaptive batch-inference pump.

Rows in -> any OpenAI-compatible endpoint -> resumable parquet out.
The pump() facade is the wrapper seam (CONTRACT.md): storage protocol fixed,
transports and signal sources pluggable behind it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import sys
import time
from collections.abc import Callable, Iterable

import httpx

from pumpjack import source as source_mod
from pumpjack import telemetry as telemetry_mod
from pumpjack.controller import Auto, Fixed, Obs
from pumpjack.engine import Engine, wait_for_health
from pumpjack.signals import CEILING_FLAG, HttpScrape, Null
from pumpjack.sink import Sink
from pumpjack.transport import (
    Breaker,
    Request,
    call_endpoint,
    coerce_request,
    make_json_request,
    make_multipart_request,
)
from pumpjack.window import Window

__all__ = [
    "pump", "Stats", "Fixed", "Auto", "Obs", "Engine", "wait_for_health",
    "Request", "make_json_request", "make_multipart_request", "existing_ids",
]
__version__ = "0.1.0"

USER_AGENT = f"pumpjack/{__version__}"
AGENT_ENV_VARS = ("CLAUDECODE", "CODEX_SANDBOX", "AI_AGENT")
TICK_S = 2.0


def agent_mode() -> bool:
    return any(os.environ.get(v) for v in AGENT_ENV_VARS)


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)


def existing_ids(out_uri: str, retry_errors: bool = False) -> set[str]:
    """The resume anti-join set for an output directory (manifest-first, exact)."""
    return Sink(out_uri).existing_ids(retry_errors=retry_errors)


def _adapt_parse(parse: Callable) -> Callable[[dict, dict], dict]:
    """parse(row, resp) with row passthrough (decision 3); single-arg
    parse(resp) accepted for POC-era recipes and the oracle workload."""
    params = [p for p in inspect.signature(parse).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(params) >= 2:
        return parse
    return lambda row, resp: parse(resp)


@dataclasses.dataclass
class Stats:
    rows_total: int = 0
    rows_done_prior: int = 0
    rows_processed: int = 0
    rows_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    final_limit: int = 0
    input_bound: bool = False
    breaker_opens: int = 0
    hints: list = dataclasses.field(default_factory=list)

    @property
    def tokens_per_sec(self) -> float:
        total = self.prompt_tokens + self.completion_tokens
        return round(total / self.elapsed_s, 1) if self.elapsed_s else 0.0

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["tokens_per_sec"] = self.tokens_per_sec
        return json.dumps(d)


def pump(
    rows: Iterable,
    to_request: Callable[[dict], Request | dict],
    parse: Callable,
    endpoint: str,
    output: str,
    window: Fixed | Auto | None = None,
    shard: tuple[int, int] = (0, 1),
    flush_every: int = 10,
    read_timeout: float = 1800.0,
    route: str = "/chat/completions",
    headers: dict | None = None,
    retry_errors: bool = False,
    id_key: str | None = None,
    signal_source: str = "auto",  # "auto" (scrape, blind fallback) | "none"
) -> Stats:
    return asyncio.run(_pump(rows, to_request, parse, endpoint, output, window or Auto(),
                             shard, flush_every, read_timeout, route, headers,
                             retry_errors, id_key, signal_source))


async def _pump(rows, to_request, parse, endpoint, output, controller, shard, flush_every,
                read_timeout, route, extra_headers, retry_errors, id_key, signal_source) -> Stats:
    stats = Stats()
    parse2 = _adapt_parse(parse)
    sink = Sink(output, flush_every=flush_every)
    done = sink.existing_ids(retry_errors=retry_errors)
    stream = source_mod.normalize(rows, id_key=id_key)

    win = Window(getattr(controller, "initial", 16))
    breaker = Breaker()
    events = {"backpressure": 0, "successes": 0}
    ticks: list[dict] = []
    src_wait = {"source": 0.0, "acquire": 0.0}
    last_gauges: dict | None = None
    base = endpoint.rstrip("/")
    probe_url = f"{base}/chat/completions"
    hdrs = {"User-Agent": USER_AGENT + (" (agent)" if agent_mode() else ""), **(extra_headers or {})}
    limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
    timeout = httpx.Timeout(30.0, read=read_timeout)
    t0 = time.monotonic()

    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=hdrs) as client:
        sig = Null() if signal_source == "none" else HttpScrape(client, endpoint)

        async def poll():
            nonlocal last_gauges
            last_tokens = 0
            while True:
                await asyncio.sleep(TICK_S)
                gauges = await sig.read()
                last_gauges = gauges or last_gauges
                total_tokens = stats.prompt_tokens + stats.completion_tokens
                tok_s = (total_tokens - last_tokens) / TICK_S
                last_tokens = total_tokens
                total_wait = src_wait["source"] + src_wait["acquire"]
                input_bound = (total_wait > 0 and src_wait["source"] / total_wait > 0.5
                               and win.inflight < int(win.limit * 0.5))
                stats.input_bound = stats.input_bound or input_bound
                g = gauges or {}
                obs = Obs(waiting=g.get("waiting"), running=g.get("running"),
                          inflight=win.inflight, backpressure=events["backpressure"],
                          successes=events["successes"], input_bound=input_bound,
                          kv=g.get("kv"), hits=g.get("hits"), tok_s=tok_s or None,
                          preempts=g.get("preempts"))
                ticks.append(telemetry_mod.tick_record(
                    time.monotonic() - t0, win.limit, win.inflight, gauges,
                    events["backpressure"], events["successes"], input_bound, tok_s))
                events["backpressure"] = events["successes"] = 0
                src_wait["source"] = src_wait["acquire"] = 0.0
                await win.set_limit(controller.decide(obs, win.limit))

        async def worker(id_: str, row: dict):
            try:
                req = coerce_request(to_request(row), route)
                body, err = await call_endpoint(client, base, req, events, breaker)
                if err is None:
                    out = parse2(row, body)
                    usage = body.get("usage") or {}
                    stats.prompt_tokens += usage.get("prompt_tokens", 0)
                    stats.completion_tokens += usage.get("completion_tokens", 0)
                    stats.rows_processed += 1
                    sink.append({"id": id_, **out, "error": None})
                else:
                    stats.rows_failed += 1
                    sink.append({"id": id_, "error": err})
            except Exception as e:  # to_request/parse bugs become error rows, not lost rows
                stats.rows_failed += 1
                sink.append({"id": id_, "error": f"client: {type(e).__name__}: {e}"})
            finally:
                await win.release()

        poller = asyncio.create_task(poll())
        tasks: set[asyncio.Task] = set()
        try:
            t_prev = time.monotonic()
            for id_, row in stream:
                stats.rows_total += 1
                if id_ in done:
                    stats.rows_done_prior += 1
                    t_prev = time.monotonic()
                    continue
                src_wait["source"] += time.monotonic() - t_prev
                await breaker.gate(client, probe_url)
                t_acq = time.monotonic()
                await win.acquire()
                src_wait["acquire"] += time.monotonic() - t_acq
                t = asyncio.create_task(worker(id_, row))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                t_prev = time.monotonic()
            while tasks:
                await asyncio.gather(*list(tasks), return_exceptions=True)
        finally:
            poller.cancel()
            sink.flush()  # crash path: whatever completed gets persisted

    stats.elapsed_s = round(time.monotonic() - t0, 2)
    stats.final_limit = win.limit
    stats.breaker_opens = breaker.opens
    if ticks:
        try:
            sink.write_telemetry(shard, [json.dumps(x) for x in ticks])
        except Exception as e:
            _log(f"telemetry write failed (non-fatal): {e}")
    sink.write_marker(shard)
    stats.hints = telemetry_mod.advise(ticks, sig.dialect, win.limit, CEILING_FLAG)
    if stats.input_bound:
        stats.hints.append("run was INPUT-BOUND — the source, not the engine, was the bottleneck")
    for h in stats.hints:
        _log(f"advisor: {h}")
    _log(f"done: {stats.rows_processed} ok, {stats.rows_failed} failed, "
         f"{stats.tokens_per_sec} tok/s, window settled at {win.limit}")
    if agent_mode():
        print(stats.to_json(), flush=True)  # data on stdout, one line
        print("Hint: re-run the same command to resume — it is always safe.", file=sys.stderr)
    return stats
