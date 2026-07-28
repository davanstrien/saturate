"""Decision 1: the package stays small. The ceiling is a feature-creep tripwire,
not a line budget: it exists to force a documented decision before new surface
area (a fourth building block, a DAG layer) lands — never to squeeze correctness
fixes or make comments fight for space. When a legitimate fix trips it, raise it
and record why in docs/decisions.md; don't golf docstrings to sneak under."""

from pathlib import Path

CEILING = 1200  # renegotiated 2026-07-28 (5th): codex r5 robustness batch (see docs/decisions.md)
PKG = Path(__file__).parent.parent / "pumpjack"


def test_loc_ceiling():
    counts = {p.name: sum(1 for line in p.read_text().splitlines() if line.strip())
              for p in sorted(PKG.glob("*.py"))}
    total = sum(counts.values())
    detail = ", ".join(f"{k}={v}" for k, v in counts.items())
    assert total <= CEILING, f"package is {total} non-blank LOC (ceiling {CEILING}): {detail}"
