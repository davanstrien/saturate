"""An aiohttp-backed HTTP client with the small part of httpx's AsyncClient that saturate uses.

`post`, `get`, `timeout`, `is_closed` and `aclose`; responses have `status_code`, `headers`,
`text` and `json()`. Failures raise httpx's exception types, so the retry, breaker and
classification code is the same for either client. Prototype for the transport A/B (#55).
"""

from __future__ import annotations

import asyncio
import json as jsonlib

import aiohttp
import httpx2 as httpx

KEEPALIVE_S = 4.0  # below the usual server idle timeout (uvicorn: 5 s), so a reused connection is open


class Response:
    """A response read in full: the body is in memory and the connection is back in the pool."""

    __slots__ = ("status_code", "headers", "_body")

    def __init__(self, status_code: int, headers, body: bytes):
        self.status_code, self.headers, self._body = status_code, headers, body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

    def json(self):
        return jsonlib.loads(self._body)  # a body that is not JSON raises ValueError, as httpx does


class AsyncClient:
    """One aiohttp session, created on the first request (a session belongs to a running loop)."""

    def __init__(self, max_connections: int, timeout: httpx.Timeout, headers: dict):
        self.timeout = timeout
        self._max_connections, self._headers = max_connections, headers
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def post(self, url: str, *, json=None, data=None, files=None, timeout=None, headers=None):
        body = _form(data, files) if files is not None else data
        return await self._request("POST", url, json=json, data=body, timeout=timeout, headers=headers)

    async def get(self, url: str, *, timeout=None, headers=None):
        return await self._request("GET", url, timeout=timeout, headers=headers)

    async def _request(self, method: str, url: str, *, timeout, headers, **kwargs) -> Response:
        try:
            async with self._open().request(method, url, timeout=_timeout(timeout or self.timeout),
                                            headers=headers, **kwargs) as r:
                return Response(r.status, r.headers, await r.read())
        except asyncio.TimeoutError as e:  # aiohttp's own timeouts are TimeoutErrors too
            raise httpx.ReadTimeout(f"timed out: {e!r}") from e
        except aiohttp.ClientConnectorError as e:
            raise httpx.ConnectError(str(e)) from e
        except aiohttp.ClientError as e:  # a dropped connection, a cut-off body, a protocol error
            raise httpx.RemoteProtocolError(f"{type(e).__name__}: {e}") from e

    def _open(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("the HTTP client is closed")
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=self._max_connections, limit_per_host=0,
                                             keepalive_timeout=KEEPALIVE_S)
            self._session = aiohttp.ClientSession(connector=connector, headers=self._headers,
                                                  trust_env=True)
        return self._session


def _timeout(t) -> aiohttp.ClientTimeout:
    """httpx's per-phase timeouts in aiohttp's terms: no overall limit (a long generation is
    legitimate), the pool wait as `connect`, and the socket phases as they are."""
    if not isinstance(t, httpx.Timeout):
        t = httpx.Timeout(t)
    return aiohttp.ClientTimeout(total=None, connect=t.pool, sock_connect=t.connect, sock_read=t.read)


def _form(data: dict | None, files: dict) -> aiohttp.FormData:
    """httpx's `data=` and `files=` as one multipart form. A file is a file object or a
    (filename, file[, content type]) tuple."""
    form = aiohttp.FormData()
    for name, value in (data or {}).items():
        form.add_field(name, str(value))
    for name, value in files.items():
        if isinstance(value, tuple):
            filename, fileobj, *rest = value
            form.add_field(name, fileobj, filename=filename, content_type=rest[0] if rest else None)
        else:
            form.add_field(name, value, filename=getattr(value, "name", name))
    return form
