# Developing

```bash
uv sync
uv run ruff check .
uv run pytest tests/ -q        # unit + regression tests, fast
uv run --with blacken-docs blacken-docs -l 100 README.md CONTRACT.md docs/*.md  # snippet formatting (CI-enforced)
```

The acceptance oracle lives in a sibling repo (local dir `pumpjack-oracle`,
named for the pre-rename codename) and treats this package as one of several
possible implementations behind an adapter:

```bash
cd ../pumpjack-oracle
ORACLE_ADAPTER=clean uv run pytest -q    # 9 tests, ~50s, spawns fake engines
```

Rules of the road: the oracle is the review gate for any core change (9/9
before and after). New public surface requires a recorded decision in
[decisions.md](history/decisions.md) (the scope rule; there is no numeric LOC
ceiling). Real-workload receipts go in `spikes/RESULTS.md` with job ids.
