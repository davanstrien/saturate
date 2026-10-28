"""Auto against simulated engines: one table, any versions of the controller.

    UV_NO_CONFIG=1 uv run python bench/controller_sim.py origin/main .

Each argument is a git ref (or `.` for the working tree); its saturate/controller.py is loaded
and run against the scenarios below with tests/sim_engine.py, three seeds each. Columns per
version: `best` is the share of the best fixed window's throughput after the first 100 ticks,
`1st-min` the share over the first 30 ticks (a minute), `past` the share of ticks with the
window more than an eighth past the window the engine can use.
"""

from __future__ import annotations

import importlib.util
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from sim_engine import LINEAR, PREEMPTING, SGLANG_LIKE, Engine, run, share_of_best, share_past_useful  # noqa: E402

SCENARIOS: list[tuple[str, dict]] = []
for gauges in (False, True):
    for name, curve, cap in (("sglang-like", SGLANG_LIKE, 256), ("linear-256", LINEAR, 256),
                             ("linear-100", [(100, 200.0)], 100), ("linear-48", [(48, 60.0)], 48),
                             ("preempting", PREEMPTING, 512)):
        for noise in (0.0, 0.07):
            SCENARIOS.append((f"{'gauges' if gauges else 'blind'} {name} noise={noise}",
                              dict(curve=curve, cap=cap, gauges=gauges, noise=noise)))
for limit in (64, 100, 127, 200, 500):
    SCENARIOS.append((f"hosted, 429 past {limit}", dict(curve=[(1024, 400.0)], cap=1024, rate_limit=limit,
                                                        gauges=False, noise=0.05)))
SCENARIOS.append(("blind, 32 s requests", dict(curve=[(256, 8.0)], cap=256, gauges=False, noise=0.05)))
SCENARIOS.append(("blind, 12 s requests, 60 s timeout", dict(curve=[(100, 8.0)], cap=100, gauges=False,
                                                             noise=0.05, read_timeout_s=60)))


def load_auto(ref: str):
    """The Auto class of saturate/controller.py at `ref` (`.` = the working tree)."""
    if ref == ".":
        path = ROOT / "saturate" / "controller.py"
    else:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:saturate/controller.py"],
                                check=True, capture_output=True, text=True).stdout
        path = Path(tempfile.mkdtemp()) / "controller.py"
        path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"controller_{abs(hash(ref))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Auto


def score(auto, kwargs: dict) -> tuple[float, float, float]:
    best, first, past = [], [], []
    for seed in range(3):
        engine = Engine(**kwargs, seed=seed)
        ticks = run(auto(), engine, 600)
        best.append(share_of_best(ticks[100:], engine))
        first.append(share_of_best(ticks[:30], engine))
        past.append(share_past_useful(ticks[100:], engine))
    return statistics.mean(best), statistics.mean(first), statistics.mean(past)


def main(refs: list[str]) -> None:
    autos = [(ref, load_auto(ref)) for ref in refs]
    print("| scenario | " + " | ".join(f"{ref}: best / 1st-min / past" for ref, _ in autos) + " |")
    print("|---|" + "---|" * len(autos))
    totals: dict[str, list[tuple[float, float, float]]] = {ref: [] for ref, _ in autos}
    for name, kwargs in SCENARIOS:
        cells = []
        for ref, auto in autos:
            b, f, p = score(auto, kwargs)
            totals[ref].append((b, f, p))
            cells.append(f"{b:.0%} / {f:.0%} / {p:.0%}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    means = []
    for ref, _ in autos:
        means.append(" / ".join(f"{statistics.mean(x[i] for x in totals[ref]):.0%}" for i in range(3)))
    print("| **mean** | " + " | ".join(means) + " |")
    worst = [f"{min(x[0] for x in totals[ref]):.0%}" for ref, _ in autos]
    print("| **worst best** | " + " | ".join(worst) + " |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["origin/main", "."])
