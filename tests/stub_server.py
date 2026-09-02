"""A local stand-in for an OpenAI-compatible inference server, stdlib only.

Speaks enough HTTP/1.1 (keep-alive, Content-Length bodies) for httpx to treat
it as a real endpoint: `POST /v1/chat/completions` answers OpenAI-shaped JSON
with `usage`, `POST /v1/embeddings` answers an embeddings list, `GET /health`
answers 200, and `GET /metrics` publishes vLLM-shaped gauges so the scrape path
and gauge-mode controller are exercised too.

Runs on its own event loop in a daemon thread, so a test can drive `pump()`
(which owns the main thread's loop via asyncio.run) against it. Behaviour is
set through plain attributes read on every request: `latency_s` and
`status_for(request_json) -> int` may be changed between runs. Counters
(`requests`, `probes`, `peak_inflight`, token totals) are what the endpoint
observed, for asserting against the client's own accounting; `probe_log` keeps
the (path, body) of every circuit-breaker probe (identified by the
`x-saturate-probe` header the breaker sends).

    python tests/stub_server.py [latency_ms]   # serve until Ctrl-C, for manual runs
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable

_HEAD_LIMIT = 1 << 20  # header bytes before we give up on a connection


def always_ok(request: dict) -> int:
    return 200


class StubServer:
    def __init__(self, latency_s: float = 0.0, status_for: Callable[[dict], int] = always_ok):
        self.latency_s = latency_s
        self.status_for = status_for
        self.port = 0
        self.requests = 0  # completion requests served (any status), probes excluded
        self.probes = 0  # circuit-breaker readiness probes seen
        self.probe_log: list[tuple[str, dict]] = []  # (path, body) of every probe
        self.paths: list[str] = []  # every POST path in arrival order, probes included
        self.inflight = 0
        self.peak_inflight = 0
        self.prompt_tokens = 0  # usage reported on 200 responses only
        self.completion_tokens = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> StubServer:
        self._thread = threading.Thread(target=self._run, name="stub-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("stub server did not start")
        return self

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(5)

    def __enter__(self) -> StubServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        server = await asyncio.start_server(self._handle, "127.0.0.1", 0, limit=_HEAD_LIMIT)
        self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        async with server:
            await self._stop.wait()

    # -- protocol ------------------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One connection: serve requests until the client closes it (keep-alive)."""
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                    return
                request_line, _, header_block = head.decode("latin-1").partition("\r\n")
                method, path, _ = request_line.split(" ", 2)
                headers = {k.strip().lower(): v.strip() for k, _, v in
                           (line.partition(":") for line in header_block.split("\r\n") if line)}
                body = await reader.readexactly(int(headers.get("content-length", 0)))
                status, ctype, payload = await self._respond(method, path.split("?", 1)[0], body,
                                                             headers)
                writer.write(f"HTTP/1.1 {status} {_REASON.get(status, 'OK')}\r\n"
                             f"Content-Type: {ctype}\r\nContent-Length: {len(payload)}\r\n"
                             "Connection: keep-alive\r\n\r\n".encode() + payload)
                await writer.drain()
                if headers.get("connection", "").lower() == "close":
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _respond(self, method: str, path: str, body: bytes, headers: dict
                       ) -> tuple[int, str, bytes]:
        if method == "GET" and path == "/health":
            return 200, "text/plain", b"ok"
        if method == "GET" and path == "/metrics":
            text = (f'vllm:num_requests_waiting{{model="stub"}} 0\n'
                    f'vllm:num_requests_running{{model="stub"}} {self.inflight}\n'
                    f'vllm:kv_cache_usage_perc{{model="stub"}} 0.1\n')
            return 200, "text/plain", text.encode()
        if method != "POST":
            return 404, "text/plain", b"not found"
        request = json.loads(body or b"{}")
        self.paths.append(path)
        if "x-saturate-probe" in headers:
            self.probes += 1
            self.probe_log.append((path, request))
        else:
            self.requests += 1
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            if self.latency_s:
                await asyncio.sleep(self.latency_s)
        finally:
            self.inflight -= 1
        status = self.status_for(request)
        if status != 200:
            err = json.dumps({"error": {"message": f"stub returned {status}", "type": "stub"}})
            return status, "application/json", err.encode()
        if path == "/v1/embeddings":
            texts = request.get("input") or []
            texts = [texts] if isinstance(texts, str) else texts
            usage = {"prompt_tokens": sum(len(t) for t in texts), "completion_tokens": 0}
            self.prompt_tokens += usage["prompt_tokens"]
            resp = {"object": "list", "model": request.get("model", "stub"),
                    "data": [{"object": "embedding", "index": i, "embedding": [float(len(t)), 0.0]}
                             for i, t in enumerate(texts)],
                    "usage": {**usage, "total_tokens": usage["prompt_tokens"]}}
            return 200, "application/json", json.dumps(resp).encode()
        content = str((request.get("messages") or [{}])[-1].get("content", ""))
        reply = f"echo: {content}"
        usage = {"prompt_tokens": len(content), "completion_tokens": len(reply)}
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        resp = {"id": "stub-cmpl", "object": "chat.completion", "model": request.get("model", "stub"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {**usage, "total_tokens": sum(usage.values())}}
        return 200, "application/json", json.dumps(resp).encode()


_REASON = {200: "OK", 400: "Bad Request", 404: "Not Found", 429: "Too Many Requests",
           500: "Internal Server Error", 503: "Service Unavailable"}


if __name__ == "__main__":
    latency = float(sys.argv[1]) / 1000 if len(sys.argv) > 1 else 0.0
    with StubServer(latency_s=latency) as srv:
        print(f"stub endpoint: {srv.endpoint}  (latency {latency * 1000:.0f} ms) — Ctrl-C to stop",
              file=sys.stderr, flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
