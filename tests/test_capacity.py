"""Auto starts at the engine's reported capacity: SGLang, llama.cpp and TEI report one, read once
before the first request; the caller's own `initial` wins, and engines without one (vLLM, a
hosted API) keep the default start."""

import asyncio

import pytest
from stub_server import StubServer

from saturate import Auto, Fixed
from saturate.core import AdaptiveClient
from saturate.http import HttpClient
from saturate.signals import Capacity, read_capacity


@pytest.mark.parametrize("requests, queues, start, ceiling", [
    (256, True, 256 + 32, 512),  # SGLang: an eighth more, so a waiting request refills a slot
    (8, True, 16, 512),          # llama.cpp slots: at least 8 more (16 beat 8 on a live server)
    (300, False, 300, 300),      # TEI rejects past its limit: start there and grow no further
])
def test_auto_starts_at_the_reported_capacity(requests, queues, start, ceiling):
    auto = Auto()
    auto.start_at_capacity(requests, queues)
    assert (auto.initial, auto.max) == (start, ceiling)


def test_the_callers_initial_wins_over_the_engines_capacity():
    auto = Auto(initial=32)
    auto.start_at_capacity(256, True)
    assert auto.initial == 32


def test_with_no_capacity_reported_auto_starts_at_its_default():
    assert Auto().initial == Auto.DEFAULT_INITIAL


@pytest.mark.parametrize("path, payload, expected", [
    ("/v1/loads", {"loads": [{"num_running_reqs": 0, "max_running_requests": 256}]},
     Capacity(256, True, "sglang")),
    ("/get_server_info", {"max_running_requests": 128}, Capacity(128, True, "sglang")),
    ("/props", {"total_slots": 8, "n_ctx": 4096}, Capacity(8, True, "llamacpp")),
    ("/info", {"max_concurrent_requests": 300, "max_batch_tokens": 16384}, Capacity(300, False, "tei")),
    ("/info", {"model_id": "x"}, None),  # a 200 without the field is no capacity
])
def test_read_capacity_finds_each_engines_field(path, payload, expected):
    async def read(endpoint):
        client = HttpClient()
        try:
            return await read_capacity(client, endpoint)
        finally:
            await client.aclose()

    with StubServer() as stub:
        stub.json_routes[path] = payload
        assert asyncio.run(read(stub.endpoint)) == expected


def test_the_client_starts_the_window_at_the_capacity_before_the_first_request():
    async def enter(endpoint, **kw):
        async with AdaptiveClient(endpoint, **kw) as client:
            return client.limiter.limit

    with StubServer() as stub:
        stub.json_routes["/props"] = {"total_slots": 8}
        assert asyncio.run(enter(stub.endpoint)) == 16
        assert asyncio.run(enter(stub.endpoint, window=Auto(initial=40))) == 40  # the caller's wins
        assert asyncio.run(enter(stub.endpoint, signal_source="none")) == Auto.DEFAULT_INITIAL
        assert asyncio.run(enter(stub.endpoint, window=Fixed(5))) == 5
