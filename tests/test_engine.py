"""A readiness failure inside Engine.__enter__ tears down the spawned server: Python does not
call __exit__ when __enter__ raises, so __enter__ calls it itself (the server runs in its own
session). The GROUP is the teardown unit: teardown kills every member of the process group,
not only the leader."""

import os
import socket

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
