"""Engine: optional in-job server lifecycle (boot templates, readiness gate,
process-group kill). Only used when saturate launches the server; pointing at
an already-running endpoint bypasses this module entirely.

GGUF rule: download weights to local disk first — mmap-over-FUSE-mount stalls.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from multidict import CIMultiDict  # aiohttp's case-insensitive headers (installed with it)

from saturate.http import Reply, environment_proxy
from saturate.transport import PROBE_HEADERS, probe_body

log = logging.getLogger(__name__)



# boot templates per engine; ceiling flags live in signals.CEILING_FLAG
def _boot(engine: str, model: str, port: int, extra: list[str]) -> list[str]:
    if engine == "vllm":
        return ["vllm", "serve", model, "--port", str(port), *extra]
    if engine == "sglang":
        return [sys.executable, "-m", "sglang.launch_server", "--model-path", model,
                "--port", str(port), "--enable-metrics", *extra]
    if engine == "llamacpp":
        return ["llama-server", "-m", model, "--port", str(port), "--metrics", *extra]
    raise ValueError(f"no boot template for engine {engine!r} — pass cmd= instead")


class Engine:
    """Spawn a serving engine in its own process group; health-gate; kill on exit."""

    def __init__(self, model: str, engine: str = "vllm", port: int = 8000,
                 extra_args: list[str] | None = None, boot_timeout: int = 1800,
                 cmd: list[str] | None = None, ready_route: str = "/chat/completions",
                 ready_payload: dict | None = None, ready_accept=None):
        self.model, self.engine, self.port = model, engine, port
        self.extra_args = extra_args or []
        self.boot_timeout = boot_timeout
        self.cmd = cmd  # full command override (e.g. sgl-omni, vendor images)
        self.ready_route, self.ready_payload = ready_route, ready_payload  # workload probe (r5)
        self.ready_accept = ready_accept  # response validator, e.g. lambda r: r.status_code == 200
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> str:
        if _port_in_use(self.port):  # readiness would be answered by that server, not ours
            raise RuntimeError(f"engine: port {self.port} is already in use — another server would answer "
                               "the readiness checks for this one; stop it or pass port=")
        cmd = self.cmd or _boot(self.engine, self.model, self.port, self.extra_args)
        log.info(f"engine: {' '.join(cmd)}")
        # own process group; stdout=2 -> engine chatter to stderr, protecting the CONTRACT §7 stdout line
        self.proc = subprocess.Popen(cmd, start_new_session=True, stdout=2)
        try:
            wait_for_health(self.endpoint, self.boot_timeout, proc=self.proc,
                            route=self.ready_route, payload=self.ready_payload,
                            accept=self.ready_accept)
        except BaseException:
            self.__exit__()  # a failed boot must not orphan the server process group
            raise
        return self.endpoint

    def __exit__(self, *exc) -> None:
        if not self.proc:
            return
        try:  # own session => pgid == leader pid; resolve while the leader is still in the table
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            pgid = self.proc.pid
        if self.proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except ProcessLookupError:
                pass  # r6: group exited between poll() and the signal — that IS teardown
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc.wait()  # reap the leader so a zombie can't hold the group open below
        for _ in range(50):  # r5: the GROUP is the teardown unit — a leader that exited on
            try:            # SIGTERM can leave children behind; kill and confirm they're gone
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                return  # group empty: teardown verified
            time.sleep(0.1)
        log.warning(f"engine: process group {pgid} still has members after SIGKILL sweep")


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_health(endpoint: str, timeout_s: int = 1800, proc: subprocess.Popen | None = None,
                    consecutive: int = 3, headers: dict | None = None,
                    poll_interval: float = 1.0, route: str = "/chat/completions",
                    payload: dict | None = None, accept=None) -> None:
    """Readiness = N consecutive health 200s + one server-generated response on
    the API route. The default acceptance (<500) means ALIVE, deliberately not
    "this workload works" — a 404/400 is a live server. To gate on the workload
    itself, pass route=/payload= AND accept= (a response predicate, e.g.
    `lambda r: r.status_code == 200`) — readiness then requires YOUR route to
    answer the way you expect (r6)."""
    base = endpoint.rsplit("/v1", 1)[0]
    deadline = time.time() + timeout_s
    ok_streak = 0
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"engine died during boot (exit {proc.returncode})")
        try:
            if _get(f"{base}/health", timeout=5, headers=headers).status_code == 200:
                ok_streak += 1
            else:
                ok_streak = 0
        except _UNANSWERED:
            ok_streak = 0
        if ok_streak >= consecutive:
            try:  # trial request: even a 400 proves the API path is alive
                r = _post(f"{endpoint.rstrip('/')}{route}",
                          payload if payload is not None else probe_body(None),
                          timeout=30, headers={**(headers or {}), **PROBE_HEADERS})
                if accept(r) if accept is not None else r.status_code < 500:
                    log.info(f"engine ready (health x{consecutive} + trial {r.status_code})")
                    return
            except _UNANSWERED:
                pass
            ok_streak = 0  # trial failed: not actually ready
        time.sleep(poll_interval)
    raise TimeoutError(f"engine not ready after {timeout_s}s")


# Boot-time checks are a few synchronous requests: the standard library is enough. They follow
# the same rules as HttpClient: the environment's proxy (never for localhost), no redirects.
_UNANSWERED = (OSError, http.client.HTTPException)  # refused, timed out, dropped, not HTTP


def _get(url: str, timeout: float, headers: dict | None = None) -> Reply:
    return _send(urllib.request.Request(url, headers=headers or {}), timeout)


def _post(url: str, body: dict, timeout: float, headers: dict | None = None) -> Reply:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", **(headers or {})})
    return _send(request, timeout)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # urllib then raises HTTPError for the 3xx, which _send returns as a reply


def _send(request: urllib.request.Request, timeout: float) -> Reply:
    """The server's reply; an error status is a reply too, not an exception."""
    parts = urlsplit(request.full_url)
    proxy = environment_proxy(parts.scheme, parts.hostname or "", parts.port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({parts.scheme: proxy} if proxy else {}),
                                         _NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as r:
            return Reply(r.status, CIMultiDict(r.headers.items()), r.read())
    except urllib.error.HTTPError as e:
        return Reply(e.code, CIMultiDict(e.headers.items()), e.read())
