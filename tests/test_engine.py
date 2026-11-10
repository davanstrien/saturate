"""A readiness failure inside Engine.__enter__ tears down the spawned server: Python does not
call __exit__ when __enter__ raises, so __enter__ calls it itself (the server runs in its own
session). The GROUP is the teardown unit: teardown kills every member of the process group,
not only the leader."""

import os
import socket
import sys

import pytest

from saturate import Engine


def test_failed_boot_kills_process_group():
    eng = Engine(model="x", cmd=["sleep", "60"], boot_timeout=1, port=59999)
    with pytest.raises(TimeoutError):
        eng.__enter__()
    assert eng.proc is not None
    assert eng.proc.poll() is not None  # reaped, not orphaned
    with pytest.raises(ProcessLookupError):
        os.killpg(eng.proc.pid, 0)  # whole group gone (own session => pgid == leader pid)


def test_accept_validator_gates_readiness(monkeypatch):
    """route=/payload= alone accepts a 404 as ready. The accept= predicate makes readiness
    require the workload's expected answer; the default keeps the alive-only semantics."""
    import saturate.engine as engmod
    from saturate import wait_for_health

    class R:
        def __init__(self, code):
            self.status_code = code

    monkeypatch.setattr(engmod, "_get", lambda *a, **k: R(200))
    monkeypatch.setattr(engmod, "_post", lambda *a, **k: R(404))
    with pytest.raises(TimeoutError):  # strict: 404 on the workload route is NOT ready
        wait_for_health(
            "http://x/v1",
            timeout_s=1,
            poll_interval=0.05,
            route="/embeddings",
            payload={"input": "hi"},
            accept=lambda r: r.status_code == 200,
        )
    # default readiness: a live server answering 404 counts as alive
    wait_for_health("http://x/v1", timeout_s=5, poll_interval=0.05)


def test_group_members_killed_when_leader_exits():
    """The leader exits immediately, leaving a SIGTERM-ignoring child in the
    group — teardown must sweep the group, not just wait on the leader."""
    eng = Engine(model="x", cmd=["sh", "-c", "(trap '' TERM; sleep 60) & exit 0"], boot_timeout=5, port=59998)
    with pytest.raises(RuntimeError, match="died during boot"):
        eng.__enter__()
    assert eng.proc is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(eng.proc.pid, 0)  # the child is gone too


def test_a_port_already_in_use_is_refused_before_anything_is_spawned():
    """Another server on the port would answer the readiness checks for this one (a stale
    vLLM from an earlier run), and the run would talk to it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        eng = Engine(model="x", cmd=["sleep", "60"], boot_timeout=5, port=port)
        with pytest.raises(RuntimeError, match="already in use"):
            eng.__enter__()
        assert eng.proc is None  # nothing was spawned


def _fake_python(directory, has_module: bool) -> str:
    """A `python3` in `directory` that answers the has-module check with yes or no."""
    directory.mkdir()
    python = directory / "python3"
    python.write_text(f"#!/bin/sh\nexit {0 if has_module else 1}\n")
    python.chmod(0o755)
    return str(python)


def test_python_with_is_this_python_when_it_has_the_module():
    from saturate.engine import python_with

    assert python_with("json") == sys.executable


def test_python_with_finds_the_first_python_on_path_that_has_the_module(tmp_path, monkeypatch):
    """The script's own environment has no sglang (as under `uv run` on the SGLang image): the
    SGLang boot command uses the first python3 on PATH that has it, past one that has not."""
    from saturate.engine import _boot, python_with

    _fake_python(tmp_path / "script-env", has_module=False)
    image_python = _fake_python(tmp_path / "image-env", has_module=True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "script-env"), str(tmp_path / "image-env")]))
    assert python_with("no_such_module_here") == image_python

    monkeypatch.setattr("saturate.engine.python_with", lambda module: image_python)
    assert _boot("sglang", "org/model", 8000, ["--x"])[:3] == [image_python, "-m", "sglang.launch_server"]


def test_python_with_falls_back_to_this_python(tmp_path, monkeypatch):
    from saturate.engine import python_with

    _fake_python(tmp_path / "script-env", has_module=False)
    monkeypatch.setenv("PATH", str(tmp_path / "script-env"))
    assert python_with("no_such_module_here") == sys.executable
