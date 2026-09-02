"""pump() is the front door: its docstring must document every parameter and the
shard= contract (labels output, never selects input)."""

import inspect

from saturate import pump


def test_pump_docstring_covers_every_parameter():
    doc = pump.__doc__
    assert doc
    for name in inspect.signature(pump).parameters:
        assert f"{name}:" in doc, name


def test_pump_docstring_warns_that_shard_does_not_select_input():
    doc = pump.__doc__
    assert "shard_select" in doc
    assert "does" in doc and "not select input rows" in doc
