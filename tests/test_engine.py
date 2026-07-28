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


def test_group_members_killed_when_leader_exits():
    """r5: leader exits immediately, leaving a SIGTERM-ignoring child in the
    group — teardown must sweep the group, not just wait on the leader."""
    eng = Engine(model="x", cmd=["sh", "-c", "(trap '' TERM; sleep 60) & exit 0"],
                 boot_timeout=5, port=59998)
    with pytest.raises(RuntimeError, match="died during boot"):
        eng.__enter__()
    with pytest.raises(ProcessLookupError):
        os.killpg(eng.proc.pid, 0)  # the child is gone too
