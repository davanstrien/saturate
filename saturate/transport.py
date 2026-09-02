"""Transport: typed Request union, retry ladder, circuit breaker.

Transport is a protocol seam (decision 15): this module is the HTTP
implementation; an in-process engine transport (vLLM AsyncLLM, sgl.Engine)
slots in post-v1 behind the same call shape.

Retry discipline (patterns per hynek/stamina; implementation ours because
429/timeout events must feed the controller live): explicit allowlist only,
exponential backoff with full jitter, capped by attempts AND total time,
Retry-After honored. 3xx/4xx = poison row, never retried, never a breaker event.
429 with Retry-After = a paced quota, not saturation: wait, don't cut.
"""

from __future__ import annotations

import asyncio
import dataclasses
import random
import sys
import time

import httpx2 as httpx

RETRY_ACTIVE = True  # kill-switch: flip off in tests for determinism
RETRY_BUDGET_S = 300.0  # total retry wall-clock per row
PROBE_HEADERS = {"x-saturate-probe": "breaker"}  # lets a server (or a stub) tell probes from rows
PROBE_TIMEOUT_S = 30.0


def probe_body(model: str | None) -> dict:
    """A tiny request for any OpenAI-style route: a chat server answers it in milliseconds,
    an embeddings or other route rejects the extra fields with a 4xx — alive either way under
    the < 500 rule. It carries the caller's model so a multi-model server resolves the same
    backend. Never a caller's own body: a long generation cannot answer inside the probe
    timeout, and a poison body would hold the circuit open on a healthy server."""
    return {"model": model or "readiness-probe", "messages": [{"role": "user", "content": "hi"}],
            "input": "hi", "max_tokens": 1}


class FatalTransportError(RuntimeError):
    """Run-level failure (breaker gave up): aborts the run — never a durable row error."""


@dataclasses.dataclass
class Request:
    """Typed request union: json XOR multipart, discriminated by `kind`."""

    kind: str  # "json" | "multipart"
    route: str
    json: dict | None = None
    data: dict | None = None
    files: dict | None = None


def make_json_request(route: str, json: dict) -> Request:
    return Request(kind="json", route=route, json=json)


def make_multipart_request(route: str, data: dict, files: dict) -> Request:
    return Request(kind="multipart", route=route, data=data, files=files)


def coerce_request(payload: Request | dict, route: str) -> Request:
    """Accept a plain dict from to_request (POC recipes) as a json request."""
    if isinstance(payload, Request):
        return payload
    return Request(kind="json", route=route, json=payload)


class Breaker:
    """Sustained all-failure means the server is down, not busy — stop feeding
    rows into the dead zone. Counts consecutive 5xx/transport *attempts*
    (poison 4xx never count); opens at `threshold`, probes until any
    server-generated response (<500), and gives up after `max_open_s` so a
    dead server exits instead of burning Job money forever."""

    def __init__(self, threshold: int = 8, probe_interval: float = 1.0, max_open_s: float = 600.0):
        self.threshold, self.probe_interval, self.max_open_s = threshold, probe_interval, max_open_s
        self.consecutive = 0
        self.opens = 0
        self.dead = False  # set when max_open_s expires: every gate raises, run ends
        self._open = False
        self._closed = asyncio.Event()
        self._closed.set()

    def fail(self) -> None:
        self.consecutive += 1

    def ok(self) -> None:
        self.consecutive = 0

    async def gate(self, client: httpx.AsyncClient, probe_url: str, model: str | None = None) -> None:
        """Called at admission AND before every retry attempt — an open breaker
        pauses the whole pump, not just new rows. One caller probes; the rest
        wait on the close event (no double-counted opens). The probe is
        `probe_body(model)` posted to the caller's own route: any response below
        500, a 4xx included, proves the server is up."""
        while True:
            if self.dead:
                raise FatalTransportError(
                    f"circuit breaker gave up after {self.max_open_s}s — server is not coming back")
            if self._open:
                await self._closed.wait()
                continue  # re-check: the breaker may have re-opened (or died)
            if self.consecutive < self.threshold:
                return
            break  # we are the opener (no await between check and set: atomic)
        self._open = True
        self._closed.clear()
        self.opens += 1
        print(f"[pump] circuit OPEN after {self.consecutive} consecutive 5xx/transport "
              "failures — pausing admission, probing", file=sys.stderr, flush=True)
        opened = time.monotonic()
        try:
            while True:
                if time.monotonic() - opened > self.max_open_s:
                    self.dead = True  # waiters released below; they raise on re-check
                    raise FatalTransportError(
                        f"circuit breaker open for {self.max_open_s}s — server is not coming back")
                try:
                    r = await client.post(probe_url, json=probe_body(model), timeout=PROBE_TIMEOUT_S,
                                          headers=PROBE_HEADERS)
                    if r.status_code < 500:
                        break
                except (httpx.TimeoutException, httpx.TransportError):
                    pass
                await asyncio.sleep(self.probe_interval)
            self.consecutive = 0
        finally:
            self._open = False
            self._closed.set()  # NEVER strand waiters — even when the opener raises
        print("[pump] circuit CLOSED — server responding again, resuming admission",
              file=sys.stderr, flush=True)


async def call_endpoint(client: httpx.AsyncClient, base: str, req: Request,
                        events: dict, breaker: Breaker) -> tuple[dict | None, str | None]:
    """Returns (response_json, error). Poison rows (3xx/4xx) never retry; multipart is
    single-attempt (file objects are consumed by the wire — a re-send posts empty bodies).
    Budget semantics: the FIRST attempt gets the client's full read window (long generations
    are legitimate); retries get their timeout capped to the remaining budget. Time spent
    waiting on an open breaker deliberately does NOT consume row budgets — a paused pump
    that recovers must resume its rows, not fail them all."""
    url = f"{base.rstrip('/')}{req.route}"
    delay, t0 = 1.0, time.monotonic()

    def left() -> float:
        return RETRY_BUDGET_S - (time.monotonic() - t0)

    async def backoff(retry_after: float | None = None) -> None:
        nonlocal delay
        wait = retry_after if retry_after is not None else random.uniform(0, delay)
        await asyncio.sleep(min(wait, max(0.0, left())))
        delay = min(delay * 2, 60.0)

    last_err = "retry budget exhausted"
    for attempt in range(5):
        if attempt and (not RETRY_ACTIVE or left() <= 0):
            return None, last_err  # hard wall-clock deadline: no attempt starts past it (r6)
        g0 = time.monotonic()
        await breaker.gate(client, url, (req.json or {}).get("model"))  # an open breaker pauses retries too
        t0 += time.monotonic() - g0  # r6: breaker-open time never consumes the row budget (docstring)
        a0 = time.monotonic()
        try:
            if req.kind == "multipart":
                r = await client.post(url, data=req.data, files=req.files)
            elif attempt and RETRY_ACTIVE:  # retries: wall-clock capped to the exact remaining budget
                r = await asyncio.wait_for(client.post(url, json=req.json, timeout=left()), left())
            else:
                r = await client.post(url, json=req.json)
        except (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError) as e:
            events["backpressure"] += 1
            breaker.fail()
            last_err = f"transport: {type(e).__name__}: {e}"
            if attempt == 4 or req.files is not None or not RETRY_ACTIVE:
                return None, last_err
            await backoff()
            continue
        if r.status_code == 200:
            events["successes"] += 1
            # the successful attempt alone: retries, backoff, breaker waits and poison rows are not
            # what a request costs the engine, and the controller's waits are scaled to that
            events.setdefault("latencies", []).append(time.monotonic() - a0)
            breaker.ok()
            return r.json(), None
        retry_after = r.headers.get("retry-after")
        if r.status_code == 429:
            if retry_after is None:
                events["backpressure"] += 1  # saturation-shaped; a paced quota is not
        elif 300 <= r.status_code < 400:  # redirects are not followed: a config error, not pressure
            return None, f"http {r.status_code}: endpoint redirects — use the final URL"
        elif 400 <= r.status_code < 500:
            return None, f"http {r.status_code}: {r.text[:300]}"  # poison, no retry, no breaker
        else:
            events["backpressure"] += 1  # intermittent 5xx IS server pressure
            breaker.fail()
        last_err = f"http {r.status_code} after retries"
        if attempt == 4 or req.files is not None or not RETRY_ACTIVE:
            return None, last_err
        await backoff(_parse_retry_after(retry_after))
    return None, "unreachable"


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110)."""
    if not value:
        return None
    if value.replace(".", "", 1).isdigit():  # one dot at most: "1.5.3" is not a number
        return float(value)
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return None  # anything unparseable: fall back to jittered backoff, never crash the row
