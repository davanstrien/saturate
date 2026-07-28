"""Engine: optional in-job server lifecycle (boot templates, readiness gate,
process-group kill). Only used when pumpjack launches the server; pointing at
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
                 cmd: list[str] | None = None):
        self.model, self.engine, self.port = model, engine, port
        self.extra_args = extra_args or []
        self.boot_timeout = boot_timeout
        self.cmd = cmd  # full command override (e.g. sgl-omni, vendor images)
        self.proc: subprocess.Popen | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> str:
        cmd = self.cmd or _boot(self.engine, self.model, self.port, self.extra_args)
        _log(f"engine: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, start_new_session=True)  # own process group
        wait_for_health(self.endpoint, self.boot_timeout, proc=self.proc)
        return self.endpoint

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)


def wait_for_health(endpoint: str, timeout_s: int = 1800, proc: subprocess.Popen | None = None,
                    consecutive: int = 3, headers: dict | None = None,
                    poll_interval: float = 1.0) -> None:
    """Readiness = N consecutive health 200s + one server-generated response on
    the API route. Health endpoints lie during warm-up (the 35B GGUF incident:
    one 200 while the completion path was still dark); any status <500 on a
    real POST is a server that parsed a request — the only readiness signal
    worth trusting."""
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
                r = httpx.post(f"{endpoint.rstrip('/')}/chat/completions",
                               json={"model": "readiness-probe",
                                     "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": 1},
                               timeout=30, headers=headers)
                if r.status_code < 500:
                    _log(f"engine ready (health x{consecutive} + trial {r.status_code})")
                    return
            except httpx.HTTPError:
                pass
            ok_streak = 0  # trial failed: not actually ready
        time.sleep(poll_interval)
    raise TimeoutError(f"engine not ready after {timeout_s}s")
