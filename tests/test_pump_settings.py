"""pump() refuses a setting that cannot work before the run starts (no request is sent, no
output is written), and runs every setting that works: 0 and inf where they mean "no limit",
and numbers of any kind (Decimal, Fraction, NumPy), converted to Python's own before use."""

import math
import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from stub_server import StubServer

import saturate
from saturate import Fixed, pump


def call(output, endpoint="http://127.0.0.1:9/v1", rows=1, **settings):
    rows_in = [{"text": f"row {i}"} for i in range(rows)]
    return pump(rows_in, lambda row: {"messages": [{"role": "user", "content": row["text"]}]},
                lambda row, body: {"ok": True}, endpoint=endpoint, output=str(output), window=Fixed(2),
                **settings)


@pytest.mark.parametrize("settings,words", [
    (dict(flush_every=-1), "flush_every >= 0, a whole number or None, not -1"),
    (dict(flush_every=2.5), "flush_every"),
    (dict(flush_every=True), "flush_every"),
    (dict(read_timeout=-1.0), "0 <= read_timeout < inf, a number or None, not -1.0"),
    (dict(read_timeout=math.inf), "read_timeout"),
    (dict(timeouts={"write": math.inf}), "0 <= write < inf"),
    (dict(read_timeout=math.nan), "read_timeout"),
    (dict(read_timeout="30"), "read_timeout"),
    (dict(read_timeout=None), "retry_budget_s when read_timeout is None"),
    (dict(retry_budget_s=-0.5), "retry_budget_s"),
    (dict(prepare_workers=-1), "prepare_workers >= 0, a whole number, not -1"),
    (dict(prepare_workers=4.0), "prepare_workers"),
    (dict(fail_fast=-1), "fail_fast"),
    (dict(timeouts={"read": 30}), "unknown phase"),
    (dict(timeouts={"connect": -1}), "0 <= connect < inf"),
    (dict(timeouts={"pool": "5"}), "pool"),
    (dict(timeouts=[("connect", 5)]), "timeouts to be a dict or None"),
    (dict(signal_source="metrics"), "signal_source in ('auto', 'none'), not 'metrics'"),
])
def test_a_setting_that_cannot_work_is_refused_before_the_run(tmp_path, monkeypatch, settings, words):
    def never(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr(saturate, "_pump", never)
    with pytest.raises(ValueError, match=r"^pump\(\).*" + re.escape(words)):
        call(tmp_path / "out", **settings)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("settings", [
    dict(flush_every=0), dict(flush_every=np.int64(3)), dict(read_timeout=0),
    dict(read_timeout=Decimal("30")), dict(read_timeout=None, retry_budget_s=60),
    dict(retry_budget_s=0), dict(retry_budget_s=math.inf), dict(retry_budget_s=Decimal("60")),
    dict(prepare_workers=np.int64(2)), dict(prepare_workers=2000), dict(fail_fast=0),
    dict(timeouts={"connect": Decimal("5"), "write": Fraction(5), "pool": np.float64(5.0)}),
    dict(timeouts={"connect": 0, "write": None}), dict(signal_source="none"),
])
def test_a_setting_that_works_runs(tmp_path, settings):
    """Through the code that uses each setting: the run completes every row."""
    with StubServer() as stub:
        stats = call(tmp_path / "out", endpoint=stub.endpoint, rows=6, **settings)
    assert (stats.rows_processed, stats.rows_failed) == (6, 0)
