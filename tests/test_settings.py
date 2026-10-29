"""saturate.settings: the checks behind every entry point's refused settings."""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from saturate.controller import AUTO_SETTINGS, MAX_COUNT, Auto
from saturate.settings import Setting, check

COUNT = Setting("step", whole=True, low=1, high=10**9)
SHARE = Setting("kv_hi", whole=False, low=0, high=1, low_open=True)
OPEN = Setting("improve", whole=False, low=1, low_open=True, high_open=True)
SEED = Setting("seed", whole=True, optional=True)


@pytest.mark.parametrize(
    "setting,value,ok",
    [
        (COUNT, 1, True),
        (COUNT, 10**9, True),
        (COUNT, 10**400, False),
        (COUNT, 0, False),
        (COUNT, 2.0, False),
        (COUNT, True, False),
        (COUNT, None, False),
        (COUNT, "8", False),
        (SHARE, 1, True),
        (SHARE, 0.5, True),
        (SHARE, Fraction(1, 3), True),
        (SHARE, 0, False),
        (SHARE, math.nan, False),
        (SHARE, 1.01, False),
        (SHARE, Decimal("0.5"), True),
        (SHARE, Decimal("NaN"), False),
        (SHARE, Decimal("sNaN"), False),
        (SHARE, np.float32(0.5), True),
        (SHARE, np.True_, False),
        (SHARE, 0.5j, False),
        (SHARE, np.array([0.5]), False),
        (COUNT, np.int64(8), True),
        (OPEN, 1.05, True),
        (OPEN, 1, False),
        (OPEN, math.inf, False),
        (SEED, None, True),
        (SEED, -7, True),
        (SEED, 1.5, False),
    ],
)
def test_a_setting_allows_exactly_its_range(setting, value, ok):
    assert setting.allows(value) is ok


@pytest.mark.parametrize(
    "setting,words",
    [
        (COUNT, "1 <= step <= 1,000,000,000, a whole number"),
        (SHARE, "0 < kv_hi <= 1, a number"),
        (OPEN, "1 < improve < inf, a number"),
        (SEED, "seed, a whole number or None"),
    ],
)
def test_the_requirement_reads_as_the_range(setting, words):
    assert setting.requirement() == words


def test_check_names_the_owner_the_setting_and_the_value():
    with pytest.raises(ValueError, match=r"^Auto requires 0 < kv_hi <= 1, a number, not 0$"):
        check("Auto", (COUNT, SHARE), dict(step=8, kv_hi=0))


@pytest.mark.parametrize("value", [Fraction(1, 2), Decimal("0.5"), np.float64(0.5), np.float32(0.5), 0.5])
def test_auto_takes_a_share_of_any_real_kind(value):
    Auto(kv_hi=value, hits_lo=value, bp_frac=value, improve=1 + value)


@pytest.mark.parametrize("name", [s.name for s in AUTO_SETTINGS if s.whole and s.name != "seed"])
def test_auto_takes_counts_up_to_the_cap_and_refuses_past_it(name):
    """min_limit <= max_limit must also hold: min_limit is tried under the largest max_limit."""

    def build(n):
        return Auto(min_limit=n, max_limit=MAX_COUNT) if name == "min_limit" else Auto(**{name: n})

    build(MAX_COUNT)
    build(np.int64(MAX_COUNT))
    with pytest.raises(ValueError, match=f"{name} <= 1,000,000,000"):
        build(MAX_COUNT + 1)


@pytest.mark.parametrize(
    "setting,words",
    [
        (Setting("x", whole=False, low_open=True), "x > -inf, a number"),
        (
            Setting("x", whole=False, low=0.123456789, high=0.12345679),
            "0.123456789 <= x <= 0.12345679, a number",
        ),
        (Setting("x", whole=True, low=10**400), f"x >= {10**400:,}, a whole number"),
    ],
)
def test_unusual_bounds_are_written_exactly(setting, words):
    assert setting.requirement() == words
