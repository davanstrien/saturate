"""Engine: optional in-job server lifecycle (boot templates, readiness gate,
process-group kill). Only used when saturate launches the server; pointing at
an already-running endpoint bypasses this module entirely.

GGUF rule: download weights to local disk first — mmap-over-FUSE-mount stalls.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx2 as httpx

from saturate.transport import PROBE_HEADERS, probe_body


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)


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
        cmd = self.cmd or _boot(self.engine, self.model, self.port, self.extra_args)
        _log(f"engine: {' '.join(cmd)}")
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
        _log(f"engine: process group {pgid} still has members after SIGKILL sweep")


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
            if httpx.get(f"{base}/health", timeout=5, headers=headers).status_code == 200:
                ok_streak += 1
            else:
                ok_streak = 0
        except httpx.HTTPError:
            ok_streak = 0
        if ok_streak >= consecutive:
            try:  # trial request: even a 400 proves the API path is alive
                r = httpx.post(f"{endpoint.rstrip('/')}{route}",
                               json=payload if payload is not None else probe_body(None),
                               timeout=30, headers={**(headers or {}), **PROBE_HEADERS})
                if accept(r) if accept is not None else r.status_code < 500:
                    _log(f"engine ready (health x{consecutive} + trial {r.status_code})")
                    return
            except httpx.HTTPError:
                pass
            ok_streak = 0  # trial failed: not actually ready
        time.sleep(poll_interval)
    raise TimeoutError(f"engine not ready after {timeout_s}s")
