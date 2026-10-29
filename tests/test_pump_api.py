"""pump() is the front door: its docstring must document every parameter and the
shard= contract (selects input, with an escape hatch)."""

import inspect

from saturate import pump


def test_pump_docstring_covers_every_parameter():
    doc = pump.__doc__
    assert doc
    for name in inspect.signature(pump).parameters:
        assert f"{name}:" in doc, name


def test_pump_docstring_says_shard_selects_input():
    doc = pump.__doc__
    assert doc is not None
    assert "processes only" in doc and "shard_of" in doc
    assert "split by `shard_of`" in doc
