"""Package metadata: the runtime version comes from the installed distribution,
so it cannot drift from pyproject (the User-Agent header derives from it)."""

import re
from pathlib import Path

import saturate


def test_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.M)
    assert match is not None
    expected = match.group(1)
    assert saturate.__version__ == expected
    assert saturate.USER_AGENT == f"saturate/{expected}"
