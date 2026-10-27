"""On a failed or timed-out test, report what the stub server and the run's telemetry saw, so a
slow test on CI says where its time went: between requests (the client was not sending) or
inside them (the server was slow). See issue #53."""

import json

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    funcargs = getattr(item, "funcargs", {})
    lines = []
    stub = funcargs.get("stub")
    if stub is not None and hasattr(stub, "longest_gap_s"):
        lines.append(f"stub: {stub.requests} row requests, {stub.probes} probes, peak in flight "
                     f"{stub.peak_inflight}, now in flight {stub.inflight}, longest gap between "
                     f"row requests {stub.longest_gap_s:.2f} s")
    tmp_path = funcargs.get("tmp_path")
    if tmp_path is not None:
        for path in sorted(tmp_path.rglob("telemetry-*.jsonl")):
            ticks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            lines.append(f"{path.name}: {len(ticks)} ticks; the last ones:")
            lines += [f"  {tick}" for tick in ticks[-5:]]
    if lines:
        report.sections.append(("saturate diagnostics", "\n".join(lines)))
