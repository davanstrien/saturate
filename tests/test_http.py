"""saturate.http against servers that misbehave on purpose: what the retry ladder and the breaker
see when a connection drops, a body is cut off, a server never answers, or a proxy is set."""

import asyncio
import io

import pytest

import saturate.http as http
from saturate.engine import _get
from saturate.http import HttpClient, TimedOut, Timeouts, TransportError, multipart

OK = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\n{\"ok\": true}"


class RawServer:
    """A TCP server whose handler gets each connection's reader and writer; counts requests and
    the connections open at once."""

    def __init__(self, handle):
        self.handle = handle
        self.requests = 0
        self.open = self.peak = 0
        self.received: list[bytes] = []

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def _serve(self, reader, writer):
        self.open += 1
        self.peak = max(self.peak, self.open)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = next((int(line.split(b":")[1]) for line in head.split(b"\r\n")
                               if line.lower().startswith(b"content-length:")), 0)
                self.received.append(head + await reader.readexactly(length))
                self.requests += 1
                if await self.handle(writer) is False:
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.open -= 1
            writer.close()


def run(coro):
    return asyncio.run(coro)


def test_a_dropped_connection_is_a_transport_error_and_the_request_reached_the_server():
    async def drop(writer):
        return False  # close without answering

    async def go():
        async with RawServer(drop) as server, HttpClient() as client:
            with pytest.raises(TransportError):
                await client.post(f"{server.url}/v1/chat/completions", json={"a": 1})
            return server.requests

    assert run(go()) == 1


def test_a_cut_off_body_is_a_transport_error():
    async def short(writer):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{\"ok\"")
        await writer.drain()
        return False

    async def go():
        async with RawServer(short) as server, HttpClient() as client:
            await client.get(server.url)

    with pytest.raises(TransportError):
        run(go())


def test_a_server_that_never_answers_times_out_at_the_read_limit():
    async def silent(writer):
        await asyncio.sleep(1)  # longer than the read limit

    async def go():
        async with RawServer(silent) as server, HttpClient() as client:
            loop = asyncio.get_running_loop()
            t = loop.time()
            with pytest.raises(TimedOut):
                # the guard fails fast if the read limit is ignored (the server answers after 1 s)
                await asyncio.wait_for(client.get(server.url, timeouts=Timeouts(read=0.2)), 0.8)
            return loop.time() - t

    assert run(go()) < 2.0


def test_the_pool_never_opens_more_connections_than_its_size():
    async def slow_ok(writer):
        await asyncio.sleep(0.05)
        writer.write(OK)
        await writer.drain()

    async def go():
        async with RawServer(slow_ok) as server, HttpClient(max_connections=4) as client:
            replies = await asyncio.gather(*(client.get(server.url) for _ in range(20)))
            return server.peak, [r.json() for r in replies]

    peak, bodies = run(go())
    assert peak == 4
    assert bodies == [{"ok": True}] * 20


def test_reply_headers_ignore_case():
    async def limited(writer):
        writer.write(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()

    async def go():
        async with RawServer(limited) as server, HttpClient() as client:
            return await client.get(server.url)

    reply = run(go())
    assert (reply.status_code, reply.headers.get("retry-after")) == (429, "7")


def test_a_closed_client_refuses_requests():
    async def go():
        client = HttpClient()
        await client.aclose()
        await client.get("http://127.0.0.1:1")

    with pytest.raises(RuntimeError, match="closed"):
        run(go())


def test_the_environment_proxy_applies_to_remote_hosts_only(monkeypatch):
    for name in ("ALL_PROXY", "all_proxy", "https_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "internal.example")
    http.environment_proxy.cache_clear()
    try:
        client = HttpClient()
        assert client._proxy_for("https://api.example/v1") == "http://proxy.example:3128"
        assert client._proxy_for("https://localhost/v1") is None
        assert client._proxy_for("https://127.0.0.1:8000/v1") is None
        assert client._proxy_for("https://internal.example/v1") is None
        assert HttpClient(proxy=False)._proxy_for("https://api.example/v1") is None
        monkeypatch.setenv("NO_PROXY", "api.example:8443")
        http.environment_proxy.cache_clear()
        assert client._proxy_for("https://api.example:8443/v1") is None  # NO_PROXY with a port
        monkeypatch.delenv("HTTPS_PROXY")
        monkeypatch.setenv("ALL_PROXY", "http://all.example:3128")
        http.environment_proxy.cache_clear()
        assert client._proxy_for("https://api.example/v1") == "http://all.example:3128"
        assert HttpClient(proxy="http://mine:1")._proxy_for("http://localhost/v1") == "http://mine:1"
    finally:
        http.environment_proxy.cache_clear()


def test_a_multipart_form_carries_its_fields_and_files():
    async def ok(writer):
        writer.write(OK)
        await writer.drain()

    async def go():
        async with RawServer(ok) as server, HttpClient() as client:
            image = io.BytesIO()
            image.write(b"PNGDATA")  # left at its end, as after encoding an image into it
            form = multipart({"model": "m"}, {"file": ("page.png", image, "image/png")})
            await client.post(f"{server.url}/v1/upload", form=form)
            return server.received[0]

    body = run(go())
    assert b'name="model"' in body and b'filename="page.png"' in body and b"PNGDATA" in body


def test_a_boot_check_reads_an_error_status_as_a_reply():
    async def not_found(writer):
        writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()

    async def go():
        async with RawServer(not_found) as server:
            return await asyncio.to_thread(_get, f"{server.url}/health", 5)

    assert run(go()).status_code == 404


def test_a_redirect_is_the_answer_not_followed():
    """A 3xx reaches the retry ladder, which rejects it; following it could turn a POST into a
    GET whose reply would be taken as the result."""
    async def moved(writer):
        writer.write(b"HTTP/1.1 302 Found\r\nLocation: /elsewhere\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()

    async def go():
        async with RawServer(moved) as server, HttpClient() as client:
            reply = await client.post(f"{server.url}/v1/chat/completions", json={})
            return reply.status_code, server.requests

    assert run(go()) == (302, 1)


def test_a_server_that_stops_reading_an_upload_hits_the_total_limit():
    """aiohttp times reading the reply only after the body is sent; a server that never reads
    the body is stopped by the total limit (pool + write + read)."""
    async def never_reads(reader, writer):
        await asyncio.sleep(1.5)  # longer than the 0.6 s total
        writer.close()

    async def go():
        server = await asyncio.start_server(never_reads, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async with server, HttpClient() as client:
            limits = Timeouts(read=0.2, connect=1.0, pool=0.2, write=0.2)
            with pytest.raises(TimedOut):
                await asyncio.wait_for(client.post(f"http://127.0.0.1:{port}/v1/upload",
                                                   json={"x": "a" * 5_000_000}, timeouts=limits), 3.0)

    run(go())


def test_boot_checks_never_proxy_localhost_and_never_follow_redirects(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # a proxy that refuses everything
    http.environment_proxy.cache_clear()

    async def moved(writer):
        writer.write(b"HTTP/1.1 302 Found\r\nLocation: /elsewhere\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()

    async def go():
        async with RawServer(moved) as server:
            reply = await asyncio.to_thread(_get, f"{server.url}/health", 5)
            return reply.status_code, server.requests

    try:
        assert run(go()) == (302, 1)
    finally:
        http.environment_proxy.cache_clear()


def test_a_multipart_file_goes_by_its_base_name_and_a_stream_is_not_rewound(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"PNGDATA")

    async def ok(writer):
        writer.write(OK)
        await writer.drain()

    class Stream(io.RawIOBase):  # a pipe-like file: readable once, not seekable
        def __init__(self):
            self.data = io.BytesIO(b"STREAMED")

        def readable(self):
            return True

        def readinto(self, buffer):
            chunk = self.data.read(len(buffer))
            buffer[: len(chunk)] = chunk
            return len(chunk)

    async def go():
        async with RawServer(ok) as server, HttpClient() as client:
            with open(path, "rb") as f:
                await client.post(f"{server.url}/v1/upload", form=multipart(None, {"file": f}))
            return server.received[0]

    body = run(go())
    assert b'filename="page.png"' in body and str(tmp_path).encode() not in body
    multipart(None, {"file": ("s.bin", io.BufferedReader(Stream()))})  # no seek on a stream
