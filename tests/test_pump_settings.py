"""pump() refuses a setting that cannot work before the run starts: no request is sent, no
output is written, and the message names the setting, its range and the value given."""

import math
import re
from typing import Any

import pytest

from saturate import Fixed, check_pump_settings, pump

GOOD: dict[str, Any] = dict(flush_every=None, read_timeout=1800.0, retry_budget_s=None,
                            prepare_workers=0, fail_fast=32, timeouts=None, signal_source="auto")


def run(tmp_path, **settings):
    """A pump() call that would fail at once if it ever reached the network (port 9: discard)."""
    return pump([{"text": "a"}], lambda row: {"messages": []}, lambda row, body: {},
                endpoint="http://127.0.0.1:9/v1", output=str(tmp_path / "out"), window=Fixed(1), **settings)


@pytest.mark.parametrize("settings,words", [
    (dict(flush_every=0), "1 <= flush_every <= 1,000,000,000, a whole number or None, not 0"),
    (dict(flush_every=2.5), "flush_every"),
    (dict(flush_every=True), "flush_every"),
    (dict(read_timeout=0), "0 < read_timeout < inf, a number, not 0"),
    (dict(read_timeout=-1.0), "read_timeout"),
    (dict(read_timeout=math.inf), "read_timeout"),
    (dict(read_timeout=math.nan), "read_timeout"),
    (dict(read_timeout="30"), "read_timeout"),
    (dict(retry_budget_s=0), "retry_budget_s"),
    (dict(prepare_workers=-1), "0 <= prepare_workers <= 1,024, a whole number, not -1"),
    (dict(prepare_workers=4.0), "prepare_workers"),
    (dict(fail_fast=-1), "fail_fast"),
    (dict(timeouts={"read": 30}), "unknown phase"),
    (dict(timeouts={"connect": 0}), "0 < connect < inf"),
    (dict(timeouts={"pool": "5"}), "pool"),
    (dict(timeouts=[("connect", 5)]), "timeouts to be a dict or None"),
    (dict(signal_source="metrics"), "signal_source in ('auto', 'none'), not 'metrics'"),
])
def test_a_setting_that_cannot_work_is_refused_before_the_run(tmp_path, settings, words):
    with pytest.raises(ValueError, match=r"^pump\(\).*" + re.escape(words)):
        run(tmp_path, **settings)
    assert not (tmp_path / "out").exists()  # nothing was written


@pytest.mark.parametrize("settings", [
    dict(flush_every=1), dict(flush_every=10**9), dict(read_timeout=0.001), dict(read_timeout=7),
    dict(retry_budget_s=0.5), dict(prepare_workers=1024), dict(fail_fast=0),
    dict(timeouts={"connect": 1, "write": None, "pool": 2.5}), dict(timeouts={}), dict(signal_source="none"),
])
def test_a_setting_in_range_is_taken(settings):
    check_pump_settings(**{**GOOD, **settings})
