"""Auto starts at the engine's reported capacity: SGLang, llama.cpp, TEI and TRT-LLM report one, read once
before the first request; the caller's own `initial` wins, a rejecting engine's limit is never
exceeded, and engines without one (vLLM, a hosted API) keep the default start. Each run starts
from the caller's configuration, so one Auto can run again, against any engine."""

import asyncio

import pytest
from stub_server import StubServer

from saturate import Auto, Fixed
from saturate.core import AdaptiveClient
from saturate.http import HttpClient
from saturate.signals import Capacity, read_capacity

SGLANG = Capacity(256, True, "sglang")
LLAMACPP = Capacity(8, True, "llamacpp")
TEI = Capacity(300, False, "tei")


@pytest.mark.parametrize(
    "capacity, start, ceiling",
    [
        (SGLANG, 256 + 16, 512),  # + 2 x target_waiting: a queue inside the band the queue cut allows
        (LLAMACPP, 16, 512),  # + capacity when smaller: 16 beat 8 slots on a live server
        (TEI, 300, 300),  # rejects past its limit: start there, never above
    ],
)
def test_auto_starts_at_the_reported_capacity(capacity, start, ceiling):
    auto = Auto()
    auto.begin(capacity)
    assert (auto.initial, auto.max) == (start, ceiling)


def test_the_callers_initial_wins_but_never_past_a_rejecting_engines_limit():
    auto = Auto(initial=32)
    auto.begin(SGLANG)
    assert auto.initial == 32
    auto = Auto(initial=600, max_limit=1000)
    auto.begin(TEI)
    assert (auto.initial, auto.max) == (300, 300)


def test_the_callers_max_limit_bounds_the_start():
    auto = Auto(max_limit=12)
    auto.begin(LLAMACPP)
    assert auto.initial == 12


def test_a_rejecting_engines_limit_below_min_limit_lowers_the_minimum():
    auto = Auto(min_limit=2)
    auto.begin(Capacity(1, False, "tei"))
    assert (auto.initial, auto.min, auto.max) == (1, 1, 1)


def test_one_auto_runs_again_against_another_engine_from_its_own_configuration():
    """TEI's limit applied to one run only: the next run, against an engine that reports none,
    starts from the caller's configuration with a fresh search."""
    auto = Auto()
    auto.begin(TEI)
    auto.decide(dict(inflight=300, successes=100, tok_s=1000.0), 300)
    auto.begin(None)
    assert (auto.initial, auto.max) == (Auto.DEFAULT_INITIAL, 512)
    assert auto.state.clock == 0.0


SGLANG_LOADS = {
    "loads": [
        {"dp_rank": 0, "num_running_reqs": 0, "max_running_requests": 128},
        {"dp_rank": 1, "num_running_reqs": 3, "max_running_requests": 128},
    ]
}


@pytest.mark.parametrize(
    "path, payload, expected",
    [
        ("/v1/loads", SGLANG_LOADS, Capacity(256, True, "sglang")),  # summed over data-parallel ranks
        ("/v1/loads", {"num_running_reqs": 0, "max_running_requests": 64}, Capacity(64, True, "sglang")),
        (
            "/get_server_info",
            {"max_running_requests": 128, "max_prefill_tokens": 16384},
            Capacity(128, True, "sglang"),
        ),
        ("/get_server_info", {"max_running_requests": None, "max_prefill_tokens": 16384}, None),
        ("/props", {"total_slots": 8, "default_generation_settings": {}}, Capacity(8, True, "llamacpp")),
        (
            "/info",
            {"model_id": "m", "max_concurrent_requests": 300, "max_batch_tokens": 16384},
            Capacity(300, False, "tei"),
        ),
        ("/info", {"model_id": "x"}, None),  # a 200 without the field is no capacity
        ("/info", {"gateway": {"max_concurrent_requests": 3}}, None),  # a nested field is not TEI's
        ("/props", {"total_slots": True, "default_generation_settings": {}}, None),  # not a count
        ("/server_info", {"disaggregated_params": None, "max_batch_size": 64}, Capacity(64, True, "trtllm")),
        ("/server_info", {"max_batch_size": 64}, None),  # not TensorRT-LLM's shape
    ],
)
def test_read_capacity_reads_each_engines_own_shape(path, payload, expected):
    with StubServer() as stub:
        stub.json_routes[path] = payload
        assert asyncio.run(_read(stub.endpoint)) == expected


async def _read(endpoint):
    client = HttpClient()
    try:
        return await read_capacity(client, endpoint)
    finally:
        await client.aclose()


def test_the_client_starts_the_window_at_the_capacity_before_the_first_request():
    async def enter(endpoint, **kw):
        async with AdaptiveClient(endpoint, **kw) as client:
            return client.limiter.limit

    with StubServer() as stub:
        stub.json_routes["/props"] = {"total_slots": 8, "default_generation_settings": {}}
        assert asyncio.run(enter(stub.endpoint)) == 16
        assert asyncio.run(enter(stub.endpoint, window=Auto(initial=40))) == 40  # the caller's wins
        assert asyncio.run(enter(stub.endpoint, signal_source="none")) == Auto.DEFAULT_INITIAL
        assert asyncio.run(enter(stub.endpoint, window=Fixed(5))) == 5


def test_the_client_hands_the_probe_to_the_scrape_whatever_the_window():
    async def enter(endpoint):
        async with AdaptiveClient(endpoint, window=Fixed(5)) as client:
            return client.limiter.signals

    with StubServer() as stub:
        stub.json_routes["/props"] = {
            "total_slots": 8,
            "default_generation_settings": {},
            "endpoint_metrics": False,
        }
        signals = asyncio.run(enter(stub.endpoint))
        assert (signals.engine, signals.reason) == ("llamacpp", "the engine reports its metrics endpoint off")


def test_a_reused_client_reads_the_capacity_again():
    """The second `async with` reads the engine afresh: a capacity that changed between runs
    takes effect."""

    async def twice(stub):
        client = AdaptiveClient(stub.endpoint)
        async with client:
            first = client.limiter.limit
        stub.json_routes["/props"] = {"total_slots": 32, "default_generation_settings": {}}
        async with client:
            return first, client.limiter.limit

    with StubServer() as stub:
        stub.json_routes["/props"] = {"total_slots": 8, "default_generation_settings": {}}
        assert asyncio.run(twice(stub)) == (16, 48)


def test_cancelling_the_entry_during_the_capacity_read_closes_the_client():
    """The read happens inside `async with` before the run starts: a cancel there must not leave
    the HTTP session open, since __aexit__ never runs."""

    async def scenario():
        async def silent(reader, writer):  # accepts, never answers
            await asyncio.sleep(30)

        server = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = AdaptiveClient(f"http://127.0.0.1:{port}/v1")
        entry = asyncio.create_task(client.__aenter__())
        await asyncio.sleep(0.2)
        entry.cancel()
        with pytest.raises(asyncio.CancelledError):
            await entry
        server.close()
        return client._client.closed

    assert asyncio.run(scenario())
