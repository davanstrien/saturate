"""Decision 1 (amended 2026-07-28, owner steer): the LOC figure is a
FEATURE-CREEP NOTE, not a gate. This check never fails CI — it warns when the
package grows past the noted figure, so genuinely new surface area (a fourth
building block, a DAG layer, live-cluster features) gets a deliberate decision
and a docs/decisions.md entry. Correctness fixes, validation, and comments are
never its target: update NOTE alongside the decision entry, don't golf."""

import warnings
from pathlib import Path

NOTE = 1216  # last noted size (2026-07-28, codex r5/r6 robustness rounds); see docs/decisions.md
PKG = Path(__file__).parent.parent / "pumpjack"


def test_loc_note():
    counts = {p.name: sum(1 for line in p.read_text().splitlines() if line.strip())
              for p in sorted(PKG.glob("*.py"))}
    total = sum(counts.values())
    if total > NOTE:
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        warnings.warn(f"package grew past the noted {NOTE} non-blank LOC (now {total}): {detail} "
                      "— new surface area? record a decision in docs/decisions.md and bump NOTE",
                      stacklevel=1)
