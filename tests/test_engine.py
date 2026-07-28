"""Codex r3 #6 / r5 #5: a readiness failure inside Engine.__enter__ previously
leaked the spawned server (own session, __exit__ never ran), and teardown only
waited on the group leader. The GROUP is the teardown unit."""

import os

import pytest

from pumpjack import Engine


def test_failed_boot_kills_process_group():
    eng = Engine(model="x", cmd=["sleep", "60"], boot_timeout=1, port=59999)
    with pytest.raises(TimeoutError):
        eng.__enter__()
    assert eng.proc.poll() is not None  # reaped, not orphaned
    with pytest.raises(ProcessLookupError):
        os.killpg(eng.proc.pid, 0)  # whole group gone (own session => pgid == leader pid)


def test_accept_validator_gates_readiness(monkeypatch):
    """Codex r6 blocker #5: route=/payload= alone still accepted a 404 — the
    accept= predicate makes readiness require the workload's expected answer,
    while the default keeps the documented alive-only semantics."""
    import pumpjack.engine as engmod
    from pumpjack import wait_for_health

    class R:
        def __init__(self, code):
            self.status_code = code

    monkeypatch.setattr(engmod.httpx, "get", lambda *a, **k: R(200))
    monkeypatch.setattr(engmod.httpx, "post", lambda *a, **k: R(404))
    with pytest.raises(TimeoutError):  # strict: 404 on the workload route is NOT ready
        wait_for_health("http://x/v1", timeout_s=1, poll_interval=0.05,
                        route="/embeddings", payload={"input": "hi"},
                        accept=lambda r: r.status_code == 200)
    # default semantics unchanged: a live server answering 404 counts as alive
    wait_for_health("http://x/v1", timeout_s=5, poll_interval=0.05)


def test_group_members_killed_when_leader_exits():
    """r5: leader exits immediately, leaving a SIGTERM-ignoring child in the
    group — teardown must sweep the group, not just wait on the leader."""
    eng = Engine(model="x", cmd=["sh", "-c", "(trap '' TERM; sleep 60) & exit 0"],
                 boot_timeout=5, port=59998)
    with pytest.raises(RuntimeError, match="died during boot"):
        eng.__enter__()
    with pytest.raises(ProcessLookupError):
        os.killpg(eng.proc.pid, 0)  # the child is gone too
