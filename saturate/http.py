"""HTTP for saturate: one pooled aiohttp session per client.

Everything else talks HTTP through this module: `HttpClient` sends a request and returns a
`Reply` read in full; a request that gets no complete answer raises `TransportError`. The
retry ladder, the breaker and the metrics scrape decide what a reply or a failure means.
Why aiohttp: bench/README.md, "The HTTP client".
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json as jsonlib
import os
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit

import aiohttp

# The pool's size. The window limits requests in flight; a window above this queues the rest on
# the pool, where each waits at most the `pool` limit (a timeout there counts as a failure).
MAX_CONNECTIONS = 1024
KEEPALIVE_S = 4.0  # below the usual server idle timeout (uvicorn: 5 s), so a reused connection is open


class TransportError(Exception):
    """The request got no complete answer: the connection was refused or dropped, the body was
    cut off, or a phase timed out."""


class TimedOut(TransportError):
    """A phase of the request took longer than its limit."""


@dataclasses.dataclass(frozen=True, slots=True)
class Timeouts:
    """Per-phase limits in seconds; None means no limit.

    `read` is the longest wait for the next part of the reply, so a long generation is one long
    wait for its first byte. `connect` is opening a connection. `pool` is the whole wait for a
    connection to send on: a free one from the pool, or a new one opened. `write` is sending the
    request body; aiohttp cannot time sending on its own, so it is enforced as part of the
    request's total limit, pool + write + read, which also means no request waits forever."""

    read: float | None = 1800.0
    connect: float | None = 30.0
    pool: float | None = 30.0
    write: float | None = 30.0

    def capped(self, cap: float) -> Timeouts:
        """Each phase limited to `cap` seconds, a phase without a limit included."""

        def phase(limit: float | None) -> float:
            return cap if limit is None else min(limit, cap)

        return Timeouts(read=phase(self.read), connect=phase(self.connect), pool=phase(self.pool),
                        write=phase(self.write))

    def for_aiohttp(self) -> aiohttp.ClientTimeout:
        limits = [limit for limit in (self.pool, self.write, self.read) if limit is not None]
        total = sum(limits) if len(limits) == 3 else None  # a phase without a limit: no total
        # aiohttp's names differ: its `connect` is our pool wait, its `sock_connect` our connect
        return aiohttp.ClientTimeout(total=total, connect=self.pool, sock_connect=self.connect,
                                     sock_read=self.read)


@dataclasses.dataclass(frozen=True, slots=True)
class Reply:
    """A response read in full; its connection is back in the pool. `headers` ignores case."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        """The body parsed as JSON; raises ValueError when it is not JSON."""
        return jsonlib.loads(self.body)


class HttpClient:
    """A pool of connections, and the defaults every request starts from.

    `proxy`: None uses the proxy the environment sets for each host (HTTPS_PROXY, NO_PROXY and
    the like, looked up once per host; localhost is never proxied); False uses none; a URL is
    used for every request. The session opens on the first request, inside the running loop."""

    def __init__(self, headers: Mapping[str, str] | None = None, timeouts: Timeouts | None = None,
                 proxy: str | bool | None = None, max_connections: int = MAX_CONNECTIONS):
        self.timeouts = timeouts or Timeouts()
        self._headers = dict(headers or {})
        self._proxy = proxy
        self._max_connections = max_connections
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def post(self, url: str, *, json=None, form: aiohttp.FormData | None = None,
                   timeouts: Timeouts | None = None, headers: Mapping[str, str] | None = None) -> Reply:
        """POST a JSON body or a multipart form."""
        return await self._request("POST", url, timeouts, headers, json=json, data=form)

    async def get(self, url: str, *, timeouts: Timeouts | None = None,
                  headers: Mapping[str, str] | None = None) -> Reply:
        return await self._request("GET", url, timeouts, headers)

    async def _request(self, method: str, url: str, timeouts: Timeouts | None,
                       headers: Mapping[str, str] | None, **body) -> Reply:
        session = self._open()
        limits = (timeouts or self.timeouts).for_aiohttp()
        try:
            # no redirects: a 3xx is the server's answer, which the retry ladder rejects as poison
            async with session.request(method, url, timeout=limits, headers=headers,
                                       proxy=self._proxy_for(url), allow_redirects=False, **body) as r:
                return Reply(r.status, r.headers, await r.read())
        except aiohttp.InvalidURL:
            raise  # a caller's mistake, not the server's
        except asyncio.TimeoutError as e:  # aiohttp's own timeout errors are TimeoutErrors too
            raise TimedOut(f"{method} {url} timed out ({type(e).__name__})") from e
        except aiohttp.ClientError as e:  # refused, dropped, cut off, or not HTTP
            raise TransportError(f"{type(e).__name__}: {e}") from e

    def _open(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("the HTTP client is closed")
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=self._max_connections, limit_per_host=0,
                                             keepalive_timeout=KEEPALIVE_S)
            # no trust_env: it reads the proxy settings and ~/.netrc on every request, which cost
            # more CPU than the rest of a request; _proxy_for looks the proxy up once per host
            self._session = aiohttp.ClientSession(connector=connector, headers=self._headers)
        return self._session

    def _proxy_for(self, url: str) -> str | None:
        if self._proxy is False:
            return None
        if isinstance(self._proxy, str):
            return self._proxy
        parts = urlsplit(url)
        return environment_proxy(parts.scheme, parts.hostname or "", parts.port)


@functools.cache
def environment_proxy(scheme: str, host: str, port: int | None = None) -> str | None:
    """The proxy the environment sets for this host and port (the scheme's proxy, else
    ALL_PROXY), or None: always None for localhost, and None when NO_PROXY names the host."""
    if host in ("localhost", "127.0.0.1", "::1"):
        return None
    if urllib.request.proxy_bypass(host if port is None else f"{host}:{port}"):
        return None
    proxies = urllib.request.getproxies()
    return proxies.get(scheme) or proxies.get("all")


def multipart(data: Mapping | None, files: Mapping) -> aiohttp.FormData:
    """Form fields and files as one multipart body. A file is a file object or a
    (filename, file[, content type]) tuple. A file object is sent from its start, wherever it was
    left (a BytesIO just written to sits at its end)."""
    form = aiohttp.FormData()
    for name, value in (data or {}).items():
        form.add_field(name, str(value))
    for name, value in files.items():
        if isinstance(value, tuple):
            filename, fileobj = value[0], value[1]
            content_type = value[2] if len(value) > 2 else None
        else:
            fileobj, content_type = value, None
            path = getattr(fileobj, "name", None)
            filename = os.path.basename(path) if isinstance(path, str) else name  # never the local path
        if getattr(fileobj, "seekable", lambda: False)():
            fileobj.seek(0)
        form.add_field(name, fileobj, filename=filename, content_type=content_type)
    return form
