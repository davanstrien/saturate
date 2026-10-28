"""Auto against simulated engines: one table, any versions of the controller.

    UV_NO_CONFIG=1 uv run python bench/controller_sim.py origin/main .

Each argument is a git ref (or `.` for the working tree); its saturate/controller.py is loaded
and run against the scenarios below with tests/sim_engine.py, three seeds each. Columns per
version, all after the first 100 ticks unless noted: `best` is the delivered share of the best
fixed window's throughput, `1st-min` the same over the first 30 ticks (a minute), `lat` the
latency as a multiple of the latency at the smallest window that reaches the best throughput,
`fail` the share of rows rejected or timed out over the whole run.
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
from sim_engine import (  # noqa: E402
    LINEAR,
    PREEMPTING,
    SGLANG_LIKE,
    Engine,
    failed_share,
    latency_inflation,
    run,
    share_of_best,
)

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
for slots, rate in ((1, 0.5), (4, 0.8), (8, 3.0)):  # llama.cpp: a few slots, the rest queue
    SCENARIOS.append((f"blind llama.cpp-like, {slots} slots", dict(curve=[(slots, rate)], cap=slots,
                                                                   gauges=False, noise=0.05)))

# mid-run changes at tick 300: (name, before, after)
SHIFTS = [
    ("blind, capacity 256 -> 128", dict(curve=LINEAR, cap=256), dict(curve=[(128, 100.0)], cap=128)),
    ("blind, capacity 128 -> 256", dict(curve=[(128, 100.0)], cap=128), dict(curve=LINEAR, cap=256)),
    ("hosted, 429 limit 100 -> 200", dict(curve=[(1024, 400.0)], cap=1024, rate_limit=100),
     dict(curve=[(1024, 400.0)], cap=1024, rate_limit=200)),
    ("blind sglang-like, prompts twice as long", dict(curve=SGLANG_LIKE, cap=256),
     dict(curve=[(x, y / 2) for x, y in SGLANG_LIKE], cap=256, tokens_per_row=200.0)),
]


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


def score(auto, kwargs: dict, after: dict | None = None) -> tuple[float, float, float, float]:
    best, first, lat, fail = [], [], [], []
    for seed in range(3):
        engine = Engine(**kwargs, seed=seed)
        then = None if after is None else (300, Engine(**{**kwargs, **after}, seed=seed))
        ticks = run(auto(), engine, 600, then=then)
        tail = ticks[100:] if then is None else ticks[300:]  # after a change: how well it adapted
        best.append(share_of_best(tail))
        first.append(share_of_best(ticks[:30]))
        lat.append(latency_inflation(tail))
        fail.append(failed_share(ticks))
    return statistics.mean(best), statistics.mean(first), statistics.mean(lat), statistics.mean(fail)


def main(refs: list[str]) -> None:
    autos = [(ref, load_auto(ref)) for ref in refs]
    print("| scenario | " + " | ".join(f"{ref}: best / 1st-min / lat / fail" for ref, _ in autos) + " |")
    print("|---|" + "---|" * len(autos))
    totals: dict[str, list[tuple[float, float, float, float]]] = {ref: [] for ref, _ in autos}
    rows = [(name, kwargs, None) for name, kwargs in SCENARIOS]
    rows += [(name, {**before, "gauges": False, "noise": 0.05}, after) for name, before, after in SHIFTS]
    for name, kwargs, after in rows:
        cells = []
        for ref, auto in autos:
            b, f, la, fa = score(auto, kwargs, after)
            totals[ref].append((b, f, la, fa))
            cells.append(f"{b:.0%} / {f:.0%} / {la:.1f}x / {fa:.1%}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    means = []
    for ref, _ in autos:
        t = totals[ref]
        means.append(f"{statistics.mean(x[0] for x in t):.0%} / {statistics.mean(x[1] for x in t):.0%} / "
                     f"{statistics.mean(x[2] for x in t):.1f}x / {statistics.mean(x[3] for x in t):.1%}")
    print("| **mean** | " + " | ".join(means) + " |")
    worst = [f"{min(x[0] for x in totals[ref]):.0%}" for ref, _ in autos]
    print("| **worst best** | " + " | ".join(worst) + " |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["origin/main", "."])
