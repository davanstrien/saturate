"""saturate.settings: the checks behind every entry point's refused settings."""

import math
from fractions import Fraction

import pytest

from saturate.settings import Setting, check

COUNT = Setting("step", whole=True, low=1, high=10**9)
SHARE = Setting("kv_hi", whole=False, low=0, high=1, low_open=True)
OPEN = Setting("improve", whole=False, low=1, low_open=True, high_open=True)
SEED = Setting("seed", whole=True, optional=True)


@pytest.mark.parametrize("setting,value,ok", [
    (COUNT, 1, True), (COUNT, 10**9, True), (COUNT, 10**400, False), (COUNT, 0, False),
    (COUNT, 2.0, False), (COUNT, True, False),
    (COUNT, None, False), (COUNT, "8", False),
    (SHARE, 1, True), (SHARE, 0.5, True), (SHARE, Fraction(1, 3), True), (SHARE, 0, False),
    (SHARE, math.nan, False), (SHARE, 1.01, False),
    (OPEN, 1.05, True), (OPEN, 1, False), (OPEN, math.inf, False),
    (SEED, None, True), (SEED, -7, True), (SEED, 1.5, False),
])
def test_a_setting_allows_exactly_its_range(setting, value, ok):
    assert setting.allows(value) is ok


@pytest.mark.parametrize("setting,words", [
    (COUNT, "1 <= step <= 1,000,000,000, a whole number"), (SHARE, "0 < kv_hi <= 1, a number"),
    (OPEN, "1 < improve < inf, a number"), (SEED, "seed, a whole number or None"),
])
def test_the_requirement_reads_as_the_range(setting, words):
    assert setting.requirement() == words


def test_check_names_the_owner_the_setting_and_the_value():
    with pytest.raises(ValueError, match=r"^Auto requires 0 < kv_hi <= 1, a number, not 0$"):
        check("Auto", (COUNT, SHARE), dict(step=8, kv_hi=0))
