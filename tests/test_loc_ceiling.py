"""Decision 1: the package stays small. Ceiling enforced in CI (docs/decisions.md
records the 800 figure and the renegotiation rule — trim at M3, don't delete
docstrings to sneak under)."""

from pathlib import Path

CEILING = 1050  # renegotiated again 2026-07-28: +probe-and-revert (the 10k-parity fix; see docs/decisions.md)
PKG = Path(__file__).parent.parent / "pumpjack"


def test_loc_ceiling():
    counts = {p.name: sum(1 for line in p.read_text().splitlines() if line.strip())
              for p in sorted(PKG.glob("*.py"))}
    total = sum(counts.values())
    detail = ", ".join(f"{k}={v}" for k, v in counts.items())
    assert total <= CEILING, f"package is {total} non-blank LOC (ceiling {CEILING}): {detail}"
