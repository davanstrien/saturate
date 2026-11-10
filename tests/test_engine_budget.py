"""TEI's `max_concurrent_requests` is a count of inputs, not of requests: it takes one permit per
text and rejects a request for which one is missing. The client keeps its own copy of that
count, so rows that are batches of texts are not rejected, under any window."""

import asyncio
import json
import threading

import pytest
from aiohttp import web

from saturate import Auto, Fixed, pump
from saturate.core import _EngineBudget
from saturate.signals import Capacity, tei_inputs
from saturate.transport import make_multipart_request

TEXTS = ["a", "b", "c"]


@pytest.mark.parametrize(
    "body, inputs",
    [
        ({"input": "one text"}, 1),
        ({"input": TEXTS}, 3),
        ({"input": [101, 2023, 102]}, 1),  # one text, as token ids
        ({"input": [[101, 102], [101, 103]]}, 2),  # two texts, as token ids
        ({"input": []}, 1),
        ({"inputs": TEXTS}, 3),  # /embed
        ({"inputs": [["query", "text"], ["query", "other"]]}, 2),  # /predict
        ({"query": "q", "texts": TEXTS}, 3),  # /rerank
        ({"messages": [{"role": "user", "content": "hi"}]}, 1),
        (None, 1),  # not a JSON body
    ],
)
def test_tei_inputs_counts_one_per_input(body, inputs):
    assert tei_inputs(body) == inputs


def _budget(total: int) -> _EngineBudget:
    return _EngineBudget(Capacity(total, False, "tei"))


async def _hold(budget: _EngineBudget, units: int, entered: list, release: asyncio.Event, name: str) -> None:
    async with budget.hold(units):
        entered.append(name)
        await release.wait()


def test_units_are_taken_all_or_none_in_arrival_order():
    async def scenario():
        budget, entered, release = _budget(10), [], asyncio.Event()
        tasks = [
            asyncio.create_task(_hold(budget, units, entered, release, name))
            for name, units in [("first", 8), ("large", 8), ("small", 1)]
        ]
        await asyncio.sleep(0.01)
        # 2 units are free: "small" would fit, but it arrived after "large" and waits behind it
        assert entered == ["first"] and budget.free == 2
        release.set()
        await asyncio.gather(*tasks)
        assert entered == ["first", "large", "small"] and budget.free == 10

    asyncio.run(scenario())


def test_a_cancelled_waiter_lets_those_behind_it_in():
    async def scenario():
        budget, entered, release = _budget(10), [], asyncio.Event()
        first = asyncio.create_task(_hold(budget, 8, entered, release, "first"))
        large = asyncio.create_task(_hold(budget, 8, entered, release, "large"))
        small = asyncio.create_task(_hold(budget, 2, entered, release, "small"))
        await asyncio.sleep(0.01)
        large.cancel()
        await asyncio.sleep(0.01)
        assert entered == ["first", "small"] and budget.free == 0
        release.set()
        await asyncio.gather(first, small)
        with pytest.raises(asyncio.CancelledError):
            await large
        assert budget.free == 10

    asyncio.run(scenario())


def test_a_waiter_cancelled_just_before_units_are_released_raises_its_cancellation():
    """The units come free in the same loop step as the cancel, before the waiting task runs
    again: the waiter is skipped, it raises CancelledError (not an error from the queue), and
    no unit is lost."""

    async def scenario():
        budget, entered, release = _budget(4), [], asyncio.Event()
        first = asyncio.create_task(_hold(budget, 4, entered, release, "first"))
        waiter = asyncio.create_task(_hold(budget, 4, entered, release, "waiter"))
        await asyncio.sleep(0.01)
        waiter.cancel()
        budget._give(0)  # a release in the same step: the queue is walked with the waiter cancelled
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await first
        assert entered == ["first"] and budget.free == 4

    asyncio.run(scenario())


def test_units_granted_to_a_task_that_is_then_cancelled_are_given_back():
    async def scenario():
        budget, entered, release = _budget(4), [], asyncio.Event()
        await budget._take(4)
        waiter = asyncio.create_task(_hold(budget, 4, entered, release, "waiter"))
        await asyncio.sleep(0.01)
        budget._give(4)  # grants the waiter its units; it has not run again yet
        assert budget.free == 0
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert entered == [] and budget.free == 4

    asyncio.run(scenario())


class FakeTEI:
    """TEI's admission rule: `permits` in all, one per input, all taken or the request is a 429."""

    def __init__(self, permits: int, service_s: float = 0.02):
        self.permits, self.free, self.service_s = permits, permits, service_s
        self.accepted = self.rejected = 0
        self.endpoint = ""
        self._loop = asyncio.new_event_loop()

    async def _info(self, request: web.Request) -> web.Response:
        body = {"model_id": "fake/model", "max_concurrent_requests": self.permits, "max_batch_tokens": 16384}
        return web.json_response(body)

    async def _embed(self, request: web.Request) -> web.Response:
        body = await request.json()
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        if len(texts) > self.free:
            self.rejected += 1
            return web.json_response({"message": "Model is overloaded", "type": "overloaded"}, status=429)
        self.free -= len(texts)
        self.accepted += 1
        try:
            await asyncio.sleep(self.service_s)
        finally:
            self.free += len(texts)
        data = [{"object": "embedding", "index": i, "embedding": [0.0]} for i in range(len(texts))]
        return web.json_response({"object": "list", "data": data, "usage": {"prompt_tokens": len(texts)}})

    def __enter__(self) -> "FakeTEI":
        ready = threading.Event()

        def serve() -> None:
            asyncio.set_event_loop(self._loop)
            app = web.Application()
            app.router.add_get("/info", self._info)
            app.router.add_post("/v1/embeddings", self._embed)
            runner = web.AppRunner(app)
            self._loop.run_until_complete(runner.setup())
            site = web.TCPSite(runner, "127.0.0.1", 0)
            self._loop.run_until_complete(site.start())
            port = runner.addresses[0][1]
            self.endpoint = f"http://127.0.0.1:{port}/v1"
            ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(runner.cleanup())
            self._loop.close()

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        assert ready.wait(5)
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)


def _batches(rows: int, texts: int) -> list[dict]:
    return [{"texts": [f"row {row} text {i}" for i in range(texts)]} for row in range(rows)]


def _embed(rows: list[dict], endpoint: str, output: str, window: Auto | Fixed | None):
    return pump(
        rows,
        to_request=lambda row: {"model": "m", "input": row["texts"]},
        parse=lambda row, body: {"count": len(body["data"])},
        endpoint=endpoint,
        output=output,
        route="/embeddings",
        window=window,
    )


@pytest.mark.parametrize(
    "window",
    [None, Fixed(16), Fixed(64), Auto(initial=32)],
    ids=["Auto()", "Fixed(16)", "Fixed(64)", "Auto(initial=32)"],
)
def test_batch_rows_are_never_rejected_by_tei(tmp_path, window):
    """32 permits admit four rows of 8 texts. Every window here is above four requests, so
    without the client's own count TEI rejects most attempts and rows fail after retries."""
    with FakeTEI(permits=32) as tei:
        stats = _embed(_batches(rows=60, texts=8), tei.endpoint, str(tmp_path / "out"), window)
    assert (stats.rows_processed, stats.rows_failed, stats.retries) == (60, 0, 0)
    assert (tei.accepted, tei.rejected) == (60, 0)


def test_rows_of_mixed_sizes_are_never_rejected(tmp_path):
    rows = [
        {"texts": [f"row {row} text {i}" for i in range(size)]}
        for row, size in enumerate([1, 4, 16, 32] * 15)
    ]
    with FakeTEI(permits=32) as tei:
        stats = _embed(rows, tei.endpoint, str(tmp_path / "out"), None)
    assert (stats.rows_processed, stats.rows_failed) == (60, 0)
    assert tei.rejected == 0


def test_a_row_with_more_inputs_than_tei_admits_fails_without_a_request(tmp_path):
    rows = _batches(rows=3, texts=8) + [{"texts": [f"text {i}" for i in range(40)]}]
    with FakeTEI(permits=32) as tei:
        stats = _embed(rows, tei.endpoint, str(tmp_path / "out"), None)
        errors = _row_errors(tmp_path / "out")
    assert (stats.rows_processed, stats.rows_failed) == (3, 1)
    assert (tei.accepted, tei.rejected) == (3, 0)  # the large row was never sent
    assert errors == ["request has 40 inputs, more than tei admits at once (32): send fewer inputs per row, "
                      "or raise the engine's limit"]  # fmt: skip


def test_a_multipart_request_takes_one_unit():
    request = make_multipart_request("/embeddings", {"model": "m"}, {"file": b"bytes"})
    assert _budget(8).units(request) == 1


def _row_errors(output) -> list[str]:
    import pyarrow.parquet as pq

    errors = []
    for path in sorted(output.rglob("ids-part-*.parquet")):
        errors += [error for error in pq.read_table(path).column("error").to_pylist() if error]
    return errors


def test_fake_tei_rejects_as_tei_does():
    """The stand-in itself: a ninth request of 4 texts against 32 permits is a 429."""

    async def scenario(endpoint: str) -> list[int]:
        import aiohttp

        async with aiohttp.ClientSession() as session:

            async def post() -> int:
                async with session.post(
                    f"{endpoint}/embeddings", data=json.dumps({"input": ["t"] * 4})
                ) as reply:
                    return reply.status

            return sorted(await asyncio.gather(*(post() for _ in range(9))))

    with FakeTEI(permits=32, service_s=0.2) as tei:
        statuses = asyncio.run(scenario(tei.endpoint))
    assert statuses == [200] * 8 + [429]
