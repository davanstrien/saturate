"""Checks for the numbers a caller passes in: one table per entry point, one check for all.

A setting that cannot work is refused when the object is built, with a message that names the
setting, the range it needs and the value given, not discovered mid-run.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from decimal import Decimal
from typing import cast


@dataclass(frozen=True)
class Setting:
    """One numeric setting: whole or real, and its range. `low_open` / `high_open` exclude
    that end; `optional` allows None."""

    name: str
    whole: bool
    low: float = -math.inf
    high: float = math.inf
    low_open: bool = False
    high_open: bool = False
    optional: bool = False

    def requirement(self) -> str:
        """What the setting needs, in words, e.g. `0 < kv_hi <= 1, a number`."""
        low = "<" if self.low_open else "<="
        high = "<" if self.high_open else "<="
        if self.low == -math.inf and self.high == math.inf and not (self.low_open or self.high_open):
            text = self.name
        elif self.high == math.inf and not self.high_open:  # no upper bound: `step >= 1`
            text = f"{self.name} {'>' if self.low_open else '>='} {_number(self.low)}"
        else:
            text = f"{_number(self.low)} {low} {self.name} {high} {_number(self.high)}"
        text += ", a whole number" if self.whole else ", a number"
        return text + (" or None" if self.optional else "")

    def allows(self, value: object) -> bool:
        """None only when optional; a whole setting takes an int (Python's or NumPy's); a real
        one also takes a float, Fraction or Decimal. Never a bool, a string, nan or an array."""
        if value is None:
            return self.optional
        if isinstance(value, bool):  # True is an int to Python, never a window or a share
            return False
        kinds = (numbers.Integral,) if self.whole else (numbers.Real, Decimal)
        if not isinstance(value, kinds):
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        if isinstance(value, Decimal) and value.is_nan():  # NaN or sNaN: neither compares
            return False
        number = cast(float, value)  # for the type checker: a real number, comparable with the bounds
        above = number > self.low if self.low_open else number >= self.low
        below = number < self.high if self.high_open else number <= self.high
        return bool(above and below)  # a NumPy comparison gives numpy.bool_


def _number(bound: float) -> str:
    """A bound as written in the table: 1, 1,000,000,000, 0.5, inf (repr keeps every digit)."""
    if isinstance(bound, int):
        return f"{bound:,}"
    return repr(bound)


def check(owner: str, table: tuple[Setting, ...], values: dict[str, object]) -> None:
    """Raise ValueError for the first setting in `table` whose value in `values` it does not
    allow."""
    for setting in table:
        value = values[setting.name]
        if not setting.allows(value):
            raise ValueError(f"{owner} requires {setting.requirement()}, not {value!r}")
