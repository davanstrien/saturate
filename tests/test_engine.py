"""A readiness failure inside Engine.__enter__ tears down the spawned server: Python does not
call __exit__ when __enter__ raises, so __enter__ calls it itself (the server runs in its own
session). The GROUP is the teardown unit: teardown kills every member of the process group,
not only the leader."""

import os
import socket
import subprocess

import pytest

from saturate import Engine
from saturate.engine import _group_has_live_members


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


def _fake_proc(root, processes):
    (root / "self").mkdir()
    (root / "self" / "stat").write_text("1 (self) R 0 1 1")
    (root / "not-a-pid").mkdir()
    for pid, command, state, process_group in processes:
        (root / str(pid)).mkdir()
        (root / str(pid) / "stat").write_text(f"{pid} ({command}) {state} 1 {process_group} {process_group}")
    return str(root)


def test_a_group_of_only_zombies_has_no_live_members(tmp_path):
    root = _fake_proc(tmp_path, [(10, "vllm", "Z", 10), (11, "worker", "X", 10), (12, "other", "S", 99)])
    assert not _group_has_live_members(10, proc_root=root)


def test_a_running_member_counts_even_with_spaces_in_its_name(tmp_path):
    root = _fake_proc(tmp_path, [(10, "vllm", "Z", 10), (11, "a b) c", "S", 10)])
    assert _group_has_live_members(10, proc_root=root)


def test_without_proc_the_group_counts_as_live(tmp_path):
    assert _group_has_live_members(10, proc_root=str(tmp_path / "missing"))
    assert _group_has_live_members(10, proc_root=str(tmp_path))  # a directory, but not a /proc


def test_teardown_does_not_warn_when_only_zombies_remain(monkeypatch, caplog):
    """killpg keeps finding a zombie that PID 1 never reaps. Teardown ends at once, with no warning."""
    import saturate.engine as engmod

    eng = Engine(model="x", cmd=["true"], port=59997)
    eng.proc = subprocess.Popen(["true"], start_new_session=True)
    eng.proc.wait()
    monkeypatch.setattr(engmod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(engmod.os, "killpg", lambda pgid, sig: None)  # the zombie is still found
    monkeypatch.setattr(engmod, "_group_has_live_members", lambda pgid: False)
    eng.__exit__()
    assert "still has members" not in caplog.text


def test_teardown_warns_when_a_member_stays_alive(monkeypatch, caplog):
    import saturate.engine as engmod

    eng = Engine(model="x", cmd=["true"], port=59996)
    eng.proc = subprocess.Popen(["true"], start_new_session=True)
    eng.proc.wait()
    monkeypatch.setattr(engmod.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(engmod.os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(engmod, "_group_has_live_members", lambda pgid: True)
    monkeypatch.setattr(engmod.time, "sleep", lambda s: None)
    eng.__exit__()
    assert "still has members" in caplog.text
