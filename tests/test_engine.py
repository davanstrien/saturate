"""Codex r3 finding #6: a readiness failure inside Engine.__enter__ previously
leaked the spawned server (own session, __exit__ never ran). Boot failure must
kill and reap the process group."""

import pytest

from pumpjack import Engine


def test_failed_boot_kills_process_group():
    eng = Engine(model="x", cmd=["sleep", "60"], boot_timeout=1, port=59999)
    with pytest.raises(TimeoutError):
        eng.__enter__()
    assert eng.proc.poll() is not None  # reaped, not orphaned
