"""On a failed or timed-out test, report what the stub server and the run's telemetry saw, so a
slow test on CI says where its time went (issue #53). A long stretch with nothing in flight at
the stub means the client was not sending; a long service time means the stub was slow.
Timeouts are covered with pytest-timeout's signal method (the Linux default): the thread
method exits the process before any report is made."""

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    try:
        lines = _diagnostics(getattr(item, "funcargs", {}))
    except Exception as e:  # a diagnostic must never take the test session down
        lines = [f"diagnostics failed: {e!r}"]
    if lines:
        report.sections.append(("saturate diagnostics", "\n".join(lines)))


def _diagnostics(funcargs: dict) -> list[str]:
    lines = []
    stub = funcargs.get("stub")
    if stub is not None and hasattr(stub, "longest_idle_s"):
        lines.append(f"stub: {stub.requests} row requests, {stub.probes} probes, peak in flight "
                     f"{stub.peak_inflight}, being handled now {stub.busy}; longest stretch with "
                     f"nothing in flight {stub.longest_idle_s:.2f} s; longest single request "
                     f"{stub.longest_service_s:.2f} s")
    tmp_path = funcargs.get("tmp_path")
    if tmp_path is not None:
        for path in sorted(tmp_path.rglob("telemetry-*.jsonl")):
            ticks = [line for line in path.read_text().splitlines() if line.strip()]
            # written every few ticks while a run goes: the file can lag the run's end
            lines.append(f"{path.name}: {len(ticks)} ticks written so far; the last ones:")
            lines += [f"  {tick}" for tick in ticks[-5:]]
    return lines
