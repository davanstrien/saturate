# Developing

```bash
uv sync
uv run ruff check .
uv run pytest tests/ -q        # unit + regression tests, fast
```

The acceptance oracle lives in a sibling repo (`saturate-oracle`) and treats
this package as one of several possible implementations behind an adapter:

```bash
cd ../saturate-oracle
ORACLE_ADAPTER=clean uv run pytest -q    # 9 tests, ~50s, spawns fake engines
```

Rules of the road: the oracle is the review gate for any core change (9/9
before and after). The LOC ceiling (`tests/test_loc_ceiling.py`) is enforced —
renegotiate it in [decisions.md](decisions.md), never dodge it by stripping
docstrings. Real-workload receipts go in `spikes/RESULTS.md` with job ids.
