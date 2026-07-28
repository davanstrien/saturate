"""Codex r3 findings #11/#12: Retry-After sleeps are capped by the retry
budget (a 3600s header must not overrun RETRY_BUDGET_S); multipart requests
never retry (their file objects are consumed — a re-send posts empty bodies)."""

import asyncio
import time

import pumpjack.transport as transport
from pumpjack.transport import Breaker, call_endpoint, make_json_request, make_multipart_request


class _Resp:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = "nope"


class _Client:
    def __init__(self, resp: _Resp):
        self.resp = resp
        self.posts = 0
        self.timeouts: list = []

    async def post(self, url, data=None, files=None, json=None, timeout="absent"):
        self.posts += 1
        self.timeouts.append(timeout)
        return self.resp


def test_retry_after_capped_by_budget(monkeypatch):
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 0.2)
    client = _Client(_Resp(429, {"retry-after": "3600"}))
    t0 = time.monotonic()
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_json_request("/chat/completions", {}),
        {"backpressure": 0, "successes": 0}, Breaker()))
    assert body is None and "429" in err
    assert time.monotonic() - t0 < 2.0  # not the header's 3600s
    assert client.posts == 1  # r4: the budget-capped sleep must not buy another request


def test_retry_attempts_get_budget_capped_timeouts(monkeypatch):
    """Codex r5 blocker #2: a retry starting with little budget left must not
    inherit the client's full 1800s read window — its timeout is capped to the
    remaining budget. The first attempt keeps the full window (long generations
    are legitimate, the budget governs retrying)."""
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 1.0)
    client = _Client(_Resp(500))
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_json_request("/chat/completions", {}),
        {"backpressure": 0, "successes": 0}, Breaker()))
    assert body is None and "500" in err
    assert client.timeouts[0] == "absent"  # first attempt: client default window
    assert all(isinstance(t, float) and t <= 1.0 for t in client.timeouts[1:])
    assert len(client.timeouts) >= 2  # it did retry, with capped windows


def test_multipart_never_retries():
    client = _Client(_Resp(500))
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_multipart_request("/upload", {"a": "1"}, {"file": b"x"}),
        {"backpressure": 0, "successes": 0}, Breaker()))
    assert body is None and err == "http 500 after retries"
    assert client.posts == 1  # single attempt: the file stream is already consumed
