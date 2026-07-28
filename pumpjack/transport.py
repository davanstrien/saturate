"""Transport: typed Request union, retry ladder, circuit breaker.

Transport is a protocol seam (decision 15): this module is the HTTP
implementation; an in-process engine transport (vLLM AsyncLLM, sgl.Engine)
slots in post-v1 behind the same call shape.

Retry discipline (patterns per hynek/stamina; implementation ours because
429/timeout events must feed the controller live): explicit allowlist only,
exponential backoff with full jitter, capped by attempts AND total time,
Retry-After honored. 4xx = poison row, never retried, never a breaker event.
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

    async def gate(self, client: httpx.AsyncClient, probe_url: str) -> None:
        """Called at admission AND before every retry attempt — an open breaker
        pauses the whole pump, not just new rows. One caller probes; the rest
        wait on the close event (no double-counted opens)."""
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
                    r = await client.post(probe_url, json={"model": "readiness-probe", "messages": [
                        {"role": "user", "content": "hi"}], "max_tokens": 1}, timeout=30)
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
    """Returns (response_json, error). Poison rows (4xx) never retry; multipart is
    single-attempt (file objects are consumed by the wire — a re-send posts empty bodies).
    Budget semantics: the FIRST attempt gets the client's full read window (long generations
    are legitimate); retries get their timeout capped to the remaining budget. Time spent
    waiting on an open breaker deliberately does NOT consume row budgets — a paused pump
    that recovers must resume its rows, not fail them all."""
    url = f"{base.rstrip('/')}{req.route}"
    delay, t0 = 1.0, time.monotonic()

    def budget_left() -> bool:
        return RETRY_ACTIVE and time.monotonic() - t0 < RETRY_BUDGET_S

    async def backoff(retry_after: float | None = None) -> None:
        nonlocal delay
        wait = retry_after if retry_after is not None else random.uniform(0, delay)
        await asyncio.sleep(min(wait, max(0.0, RETRY_BUDGET_S - (time.monotonic() - t0))))
        delay = min(delay * 2, 60.0)

    probe_url = f"{base.rstrip('/')}/chat/completions"
    for attempt in range(5):
        await breaker.gate(client, probe_url)  # an open breaker pauses retries too
        try:
            if req.kind == "multipart":
                r = await client.post(url, data=req.data, files=req.files)
            elif attempt and RETRY_ACTIVE:  # retries: read window capped to the remaining budget
                left = max(1.0, RETRY_BUDGET_S - (time.monotonic() - t0))
                r = await client.post(url, json=req.json, timeout=left)
            else:
                r = await client.post(url, json=req.json)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            events["backpressure"] += 1
            breaker.fail()
            if attempt == 4 or req.files is not None or not budget_left():
                return None, f"transport: {type(e).__name__}: {e}"
            await backoff()
            if not budget_left():  # the sleep itself may exhaust the budget: no extra request
                return None, f"transport: {type(e).__name__}: {e}"
            continue
        if r.status_code == 200:
            events["successes"] += 1
            breaker.ok()
            return r.json(), None
        retry_after = r.headers.get("retry-after")
        if r.status_code == 429:
            if retry_after is None:
                events["backpressure"] += 1  # saturation-shaped; a paced quota is not
        elif 400 <= r.status_code < 500:
            return None, f"http {r.status_code}: {r.text[:300]}"  # poison, no retry, no breaker
        else:
            events["backpressure"] += 1  # intermittent 5xx IS server pressure
            breaker.fail()
        if attempt == 4 or req.files is not None or not budget_left():
            return None, f"http {r.status_code} after retries"
        await backoff(_parse_retry_after(retry_after))
        if not budget_left():  # the sleep itself may exhaust the budget: no extra request
            return None, f"http {r.status_code} after retries"
    return None, "unreachable"


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date (RFC 9110)."""
    if not value:
        return None
    if value.replace(".", "").isdigit():
        return float(value)
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError):
        return None
