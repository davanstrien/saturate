"""On a failed or timed-out test, report what the stub server and the run's telemetry saw, so a
slow test on CI says where its time went (issue #53). A long stretch with nothing in flight at
the stub means the client was not sending; a long service time means the stub was slow.
Timeouts are covered with pytest-timeout's signal method (the Linux default): the thread
method exits the process before any report is made."""

import os

import pytest
from hypothesis import settings

import saturate.controller

# property tests without their own max_examples: 60 examples, or 1000 with SATURATE_DEEP=1
settings.register_profile("default", max_examples=60, deadline=None)
settings.register_profile("deep", max_examples=1000, deadline=None)
settings.load_profile("deep" if os.environ.get("SATURATE_DEEP") else "default")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run the tests marked slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seeded_auto(monkeypatch):
    """An Auto built without a seed draws its trial jitter from the OS; in tests it gets seed 0,
    so a run is repeatable. A test that wants several schedules passes `seed` itself."""
    build = saturate.controller.Auto.__init__

    def seeded(self, *args, seed=None, **kwargs):
        build(self, *args, seed=0 if seed is None else seed, **kwargs)

    monkeypatch.setattr(saturate.controller.Auto, "__init__", seeded)


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
        lines.append(
            f"stub: {stub.requests} row requests, {stub.probes} probes, peak in flight "
            f"{stub.peak_inflight}, being handled now {stub.busy}; longest stretch with "
            f"nothing in flight {stub.longest_idle_s:.2f} s; longest single request "
            f"{stub.longest_service_s:.2f} s"
        )
    tmp_path = funcargs.get("tmp_path")
    if tmp_path is not None:
        for path in sorted(tmp_path.rglob("telemetry-*.jsonl")):
            ticks = [line for line in path.read_text().splitlines() if line.strip()]
            # written every few ticks while a run goes: the file can lag the run's end
            lines.append(f"{path.name}: {len(ticks)} ticks written so far; the last ones:")
            lines += [f"  {tick}" for tick in ticks[-5:]]
    return lines
