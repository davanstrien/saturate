"""Codex r3 findings #11/#12: Retry-After sleeps are capped by the retry
budget (a 3600s header must not overrun RETRY_BUDGET_S); multipart requests
never retry (their file objects are consumed — a re-send posts empty bodies)."""

import asyncio
import time

import saturate.transport as transport
from saturate.transport import Breaker, call_endpoint, make_json_request, make_multipart_request


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


def test_no_timeout_floor_below_budget(monkeypatch):
    """Codex r6 blocker #2: max(1.0, remaining) granted a full second when only
    milliseconds remained. Retry timeouts must be the exact remaining budget."""
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 0.05)
    client = _Client(_Resp(429, {"retry-after": "0"}))  # zero backoff: attempts spin freely
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_json_request("/chat/completions", {}),
        {"backpressure": 0, "successes": 0}, Breaker()))
    assert body is None
    assert all(isinstance(t, float) and t <= 0.05 for t in client.timeouts[1:])


def test_breaker_open_time_credited_back_to_budget(monkeypatch):
    """Codex r6 blocker #2: the docstring promises breaker-open waits don't
    consume row budgets — now the code makes it true (t0 is credited)."""
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 0.05)

    class SlowGate(Breaker):
        async def gate(self, client, probe_url, probe_json=None):
            await asyncio.sleep(0.1)  # longer than the entire budget

    class _SeqClient:
        def __init__(self):
            self.resps = [_Resp(429, {"retry-after": "0"}), _Resp(200)]

        async def post(self, url, data=None, files=None, json=None, timeout="absent"):
            r = self.resps.pop(0)
            r.json = lambda: {"ok": True}
            return r

    body, err = asyncio.run(call_endpoint(
        _SeqClient(), "http://x", make_json_request("/chat/completions", {}),
        {"backpressure": 0, "successes": 0}, SlowGate()))
    assert err is None and body == {"ok": True}  # the retry survived two 0.1s breaker waits


def test_multipart_never_retries():
    client = _Client(_Resp(500))
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_multipart_request("/upload", {"a": "1"}, {"file": b"x"}),
        {"backpressure": 0, "successes": 0}, Breaker()))
    assert body is None and err == "http 500 after retries: nope"  # the server message is kept
    assert client.posts == 1  # single attempt: the file stream is already consumed


def test_latency_is_the_successful_attempt_alone():
    """A slow failed attempt, its backoff and a fast success: the controller's
    latency signal is the fast success, not the row's whole retry ladder."""

    class _Ok(_Resp):
        def json(self):
            return {"ok": True}

    class _SlowThenFast:
        def __init__(self):
            self.posts = 0

        async def post(self, url, data=None, files=None, json=None, timeout="absent"):
            self.posts += 1
            if self.posts == 1:
                await asyncio.sleep(0.05)  # a slow attempt that fails...
                return _Resp(429, {"retry-after": "0.05"})  # ...then a backoff
            return _Ok(200)  # a fast success

    events = {"backpressure": 0, "successes": 0}
    t = time.monotonic()
    body, err = asyncio.run(call_endpoint(
        _SlowThenFast(), "http://x", make_json_request("/chat/completions", {}), events, Breaker()))
    assert body == {"ok": True} and err is None
    assert time.monotonic() - t >= 0.1  # the row took the slow attempt plus the backoff...
    assert len(events["latencies"]) == 1 and events["latencies"][0] < 0.05  # ...the sample did not


def test_parse_retry_after_never_raises():
    """A malformed Retry-After header must fall back to normal backoff, not crash the row."""
    from saturate.transport import _parse_retry_after

    assert _parse_retry_after("12") == 12.0
    assert _parse_retry_after("1.5") == 1.5
    assert _parse_retry_after("1.5.3") is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("abc") is None
    assert _parse_retry_after("inf") is None
    assert _parse_retry_after("nan") is None
    assert _parse_retry_after(None) is None


def test_redirect_is_poison_not_pressure():
    """A 3xx means the endpoint URL is wrong (redirects are not followed): fail the row
    once, without retrying, counting backpressure, or feeding the breaker."""
    client = _Client(_Resp(302, {"location": "https://x/v1/chat/completions"}))
    events = {"backpressure": 0, "successes": 0}
    breaker = Breaker()
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_json_request("/chat/completions", {}), events, breaker))
    assert body is None and "302" in err and "redirect" in err
    assert client.posts == 1
    assert events["backpressure"] == 0
    assert breaker.consecutive == 0


def test_a_row_that_always_fails_counts_as_backpressure_once(monkeypatch):
    """Every failed attempt used to add backpressure, so one row that always returns 500
    (an image the model's processor rejects) cut the window on each of its retries and a 1%
    poison rate held it at the floor. The row now counts once; the breaker sees every attempt."""
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 30.0)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(transport.asyncio, "sleep", no_sleep)
    fails = []
    breaker = Breaker()
    monkeypatch.setattr(breaker, "fail", lambda: fails.append(1))
    events = {"backpressure": 0, "successes": 0}
    client = _Client(_Resp(500))
    body, err = asyncio.run(call_endpoint(
        client, "http://x", make_json_request("/chat/completions", {}), events, breaker))
    assert body is None and "500" in err
    assert client.posts == 5  # it retried...
    assert events["backpressure"] == 1  # ...but pressured the controller once
    assert len(fails) == 5  # a dead server still trips the breaker attempt by attempt



def test_retry_after_is_jittered_so_rows_do_not_wake_together(monkeypatch):
    """Rows told the same Retry-After used to sleep exactly that long and hit the server in
    one burst. The wait is now the header's value times uniform(1, 1.2)."""
    monkeypatch.setattr(transport, "RETRY_BUDGET_S", 30.0)
    waits = []

    async def record(seconds):
        waits.append(seconds)

    monkeypatch.setattr(transport.asyncio, "sleep", record)
    client = _Client(_Resp(429, {"retry-after": "10"}))
    asyncio.run(call_endpoint(client, "http://x", make_json_request("/chat/completions", {}),
                              {"backpressure": 0, "successes": 0}, Breaker()))
    assert len(waits) == 4 and all(10.0 <= w <= 12.0 for w in waits), waits
    assert len(set(waits)) > 1  # jittered, not identical


def test_a_5xx_error_row_keeps_the_server_message():
    client = _Client(_Resp(500))
    client.resp.text = "CUDA out of memory"
    transport_active = transport.RETRY_ACTIVE
    transport.RETRY_ACTIVE = False
    try:
        body, err = asyncio.run(call_endpoint(client, "http://x", make_json_request("/chat/completions", {}),
                                              {"backpressure": 0, "successes": 0}, Breaker()))
    finally:
        transport.RETRY_ACTIVE = transport_active
    assert body is None and err == "http 500 after retries: CUDA out of memory"



# --- the pure units: what a failed attempt means, and how long to wait --------------------

import pytest  # noqa: E402

from saturate.transport import classify, next_wait, transport_failure  # noqa: E402


@pytest.mark.parametrize("status, retry_after, retry, pressure, breaker_fail", [
    (301, None, False, False, False),  # redirect: a config error
    (400, None, False, False, False),  # poison row
    (404, None, False, False, False),  # e.g. unknown model: poison, fail_fast catches a run of them
    (429, None, True, True, False),  # saturation-shaped
    (429, "5", True, False, False),  # a paced quota: wait, don't cut
    (500, None, True, True, True),
    (503, "2", True, True, True),  # 5xx is pressure even with a Retry-After
])
def test_classify(status, retry_after, retry, pressure, breaker_fail):
    out = classify(status, retry_after, "the server said why")
    assert (out.retry, out.pressure, out.breaker_fail) == (retry, pressure, breaker_fail)
    assert str(status) in out.error


def test_classify_keeps_the_server_message_and_parses_retry_after():
    assert classify(503, None, "CUDA out of memory").error == "http 503 after retries: CUDA out of memory"
    assert classify(400, None, "x" * 1000).error == "http 400: " + "x" * 300
    assert classify(429, "7").retry_after == 7.0
    assert classify(429, "not a date").retry_after is None  # unparseable: treated as no header


def test_transport_failure_is_pressure_and_a_breaker_event():
    out = transport_failure(TimeoutError("read timed out"))
    assert (out.retry, out.pressure, out.breaker_fail) == (True, True, True)
    assert out.error == "transport: TimeoutError: read timed out"


def test_next_wait_is_full_jitter_up_to_a_doubling_ceiling():
    top = lambda a, b: b  # noqa: E731 (the largest draw uniform could make)
    delays = [1.0]
    for _ in range(8):
        wait, nxt = next_wait(delays[-1], uniform=top)
        assert wait == delays[-1]
        delays.append(nxt)
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60, 60]


def test_next_wait_jitters_a_server_given_wait_upwards_only():
    assert next_wait(1.0, retry_after=10.0, uniform=lambda a, b: a)[0] == 10.0
    assert next_wait(1.0, retry_after=10.0, uniform=lambda a, b: b)[0] == 12.0


# --- properties: must hold for every input, not just the listed cases ---------------------

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

delays = st.floats(min_value=0.0, max_value=60.0, allow_nan=False)
draws = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)  # where uniform lands in its range


def at(fraction):
    """A deterministic stand-in for random.uniform: the point `fraction` of the way from a to b."""
    return lambda a, b: a + (b - a) * fraction


@given(delays, draws)
def test_next_wait_stays_within_the_jitter_ceiling(delay, fraction):
    wait, nxt = next_wait(delay, uniform=at(fraction))
    assert 0.0 <= wait <= delay
    assert nxt == min(delay * 2, 60.0) and nxt <= 60.0


@given(delays, st.floats(min_value=0.0, max_value=3600.0, allow_nan=False), draws)
def test_next_wait_never_undercuts_a_server_given_wait(delay, retry_after, fraction):
    wait, _ = next_wait(delay, retry_after, uniform=at(fraction))
    assert retry_after <= wait <= retry_after * 1.2 + 1e-9


@given(st.integers(min_value=300, max_value=599), st.one_of(st.none(), st.text(max_size=8)),
       st.text(max_size=2000))
def test_classify_rules_hold_for_every_status(status, retry_after, text):
    out = classify(status, retry_after, text)
    poison = status < 500 and status != 429
    assert out.retry is not poison  # 3xx/4xx never retry; 429 and 5xx always do
    assert out.breaker_fail is (status >= 500)  # only a server-side failure counts against the server
    if status >= 500:
        assert out.pressure
    assert str(status) in out.error and len(out.error) <= 400  # the server's text is capped
