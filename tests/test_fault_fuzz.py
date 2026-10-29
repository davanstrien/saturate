"""Fault fuzzing: whatever mix of server failures a run meets, every row is accounted for, the
output holds exactly what the failures allow, and a resumed run against a healthy server ends
with every row stored once as a success."""

import collections
import glob
import tempfile

import pyarrow.parquet as pq
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from stub_server import StubServer

import saturate.core
import saturate.transport
from saturate import Fixed, pump

# what the server does with a row: (status of the first attempt, whether later attempts succeed)
FAULTS = {
    "ok": (200, True),
    "500 once": (500, True),
    "429 once": (429, True),
    "dropped once": ("drop", True),
    "not JSON once": ("html", True),
    "400 always": (400, False),  # a poison row: never retried
    "500 always": (500, False),  # retried until its budget runs out
}
RECOVERS = {name for name, (_, later_ok) in FAULTS.items() if later_ok}


class FaultyServer:
    """Answers each row by its planned fault; anything that is not a row (a breaker probe) is
    answered normally. The stub asks drop_for first, once per request, so the attempt is
    decided there; status_for and raw_for read that decision."""

    def __init__(self, stub: StubServer, plan: dict[str, str]):
        self.plan, self.attempts = plan, collections.Counter()
        self.current: dict[str, int | str] = {}  # each row's fault (a status, "drop" or "html")
        stub.drop_for = self.start_attempt
        stub.status_for = lambda request: self.status(text_of(request))
        stub.raw_for = self.body

    def body(self, request: dict) -> bytes | None:
        """A proxy's busy page for a "not JSON" attempt; otherwise the stub's normal reply."""
        return b"<html>busy</html>" if self.fault(text_of(request)) == "html" else None

    def start_attempt(self, request: dict) -> bool:
        text = text_of(request)
        if text not in self.plan:
            self.current.pop(text, None)
            return False
        self.attempts[text] += 1
        first, later_ok = FAULTS[self.plan[text]]
        self.current[text] = first if self.attempts[text] == 1 or not later_ok else 200
        return self.current[text] == "drop"

    def fault(self, text: str):
        return self.current.get(text, 200)

    def status(self, text: str) -> int:
        fault = self.fault(text)
        return fault if isinstance(fault, int) else 200


def text_of(request: dict) -> str:
    return str((request.get("messages") or [{}])[-1].get("content", ""))


def records(out: str) -> list[dict]:
    """Every record in the parts, error rows and repeats included (read_output hides both, and
    the test counts them)."""
    found = []
    for path in sorted(glob.glob(f"{out}/part-*.parquet")):
        found += pq.read_table(path).to_pylist()
    return found


def run(stub, out, rows, retry_errors=False):
    return pump(
        rows,
        lambda r: {"model": "stub", "messages": [{"role": "user", "content": r["text"]}]},
        lambda r, b: {"reply": b["choices"][0]["message"]["content"]},
        endpoint=stub.endpoint,
        output=out,
        window=Fixed(4),
        fail_fast=0,
        retry_budget_s=30.0,
        retry_errors=retry_errors,
    )  # generous: recovery, not timing


@pytest.fixture
def quick(monkeypatch):
    class QuickBreaker(saturate.transport.Breaker):
        def __init__(self):
            super().__init__(threshold=3, probe_interval=0.02, max_open_s=5.0)

    monkeypatch.setattr(saturate.core, "Breaker", QuickBreaker)
    monkeypatch.setattr(saturate.transport, "BACKOFF_BASE_S", 0.01)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.sampled_from(sorted(FAULTS)), min_size=1, max_size=12))
def test_any_mix_of_faults_ends_with_every_row_accounted_and_resumable(quick, faults):
    rows = [(f"r{i}", {"text": f"row {i}"}) for i in range(len(faults))]
    plan = {f"row {i}": fault for i, fault in enumerate(faults)}
    with tempfile.TemporaryDirectory(prefix="fuzz-") as out, StubServer() as stub:
        server = FaultyServer(stub, plan)
        stats = run(stub, out, rows)  # raises RowsLostError if a row went missing
        assert stats.rows_unaccounted == 0

        stored = {r["id"]: r for r in records(out)}
        assert sorted(stored) == sorted(id_ for id_, _ in rows)  # every row, once
        for i, fault in enumerate(faults):
            succeeded = stored[f"r{i}"]["error"] is None
            assert succeeded == (fault in RECOVERS), (fault, stored[f"r{i}"]["error"])

        server.plan = {}  # the server is healthy again
        run(stub, out, rows, retry_errors=True)
        best = collections.defaultdict(list)
        for r in records(out):
            best[r["id"]].append(r["error"] is None)
        assert all(sum(oks) == 1 for oks in best.values())  # each row succeeded exactly once
        assert sorted(best) == sorted(id_ for id_, _ in rows)
