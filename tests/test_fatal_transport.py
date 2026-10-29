"""A dead breaker aborts the run instead of writing
durable error rows (which resume would silently skip); a non-dict parse result
becomes a healable error row instead of crashing after the API spend."""

import asyncio
from typing import cast

import pytest
from stub_server import StubLimiter

from saturate import AdaptiveClient, FatalTransportError, ParquetSink
from saturate.http import HttpClient
from saturate.pipeline import through
from saturate.sink import drain
from saturate.transport import Breaker


def test_dead_breaker_gate_raises_fatal():
    b = Breaker()
    b.dead = True
    with pytest.raises(FatalTransportError):
        # a dead breaker raises before it touches the client, so no client is needed
        asyncio.run(b.gate(cast(HttpClient, None), "http://x/probe"))


@pytest.mark.internals
def test_fatal_aborts_run_without_error_rows(tmp_path):
    class _FatalClient:
        limiter = StubLimiter()

        async def post(self, request, route="/chat/completions", on_admit=None):
            raise FatalTransportError("circuit breaker gave up")

    sink = ParquetSink(str(tmp_path), flush_every=5)

    async def go():
        client = cast(AdaptiveClient, _FatalClient())  # a stand-in with the `limiter` and `post` through uses
        async with through(
            client, iter([(str(i), {"n": i}) for i in range(20)]), lambda r: {"p": r["n"]}, lambda r, b: b
        ) as results:
            await drain(results, sink)

    with pytest.raises(FatalTransportError):
        asyncio.run(go())
    assert sink.existing_ids() == set()  # nothing durable: every row retries next run


@pytest.mark.internals
def test_nondict_parse_is_error_row(tmp_path):
    class _OkClient:
        limiter = StubLimiter()

        async def post(self, request, route="/chat/completions", on_admit=None):
            return {"usage": {}}, None

    sink = ParquetSink(str(tmp_path), flush_every=1)

    async def go():
        client = cast(AdaptiveClient, _OkClient())  # a stand-in with the `limiter` and `post` through uses
        async with through(client, iter([("a", {})]), lambda r: {}, lambda r, b: None) as results:
            return await drain(results, sink)

    stats = asyncio.run(go())
    assert stats.rows_failed == 1
    assert sink.existing_ids() == {"a"}  # durable as an error row...
    assert sink.existing_ids(retry_errors=True) == set()  # ...healable via retry_errors
