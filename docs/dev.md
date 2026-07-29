# Developing

Python ≥3.10. CI runs exactly these three (plus an `[hf]` import check):

```bash
uv sync
uv run ruff check .
uv run pytest tests/ -q                  # unit + regression tests; -k name for one test
uv run --with blacken-docs blacken-docs -l 100 README.md CONTRACT.md docs/*.md  # snippet formatting
```

**The acceptance oracle** is a differential test suite in a separate, currently-private
repo: nine end-to-end scenarios (kill/resume, breaker behavior, fan-out uniqueness, …) run
against fake engines, written so that any implementation of the pump API can sit behind its
adapter — the original POC and this package both pass it, which is how the clean rewrite
was verified against the validated prototype. It is the review gate for core changes (9/9
before and after). External contributors can't run it yet; the tests in `tests/` plus CI
are the visible bar, and a core PR will get an oracle run from the maintainer.

```bash
cd ../pumpjack-oracle            # local dir keeps the pre-rename codename
ORACLE_ADAPTER=clean uv run pytest -q    # "clean" = this package; "poc" = the frozen prototype
```

**Contribution posture**: this is a proof of concept shared for feedback — issues and
design comments are the most useful contribution right now; small PRs are welcome, big
ones deserve an issue first. New public surface needs a recorded decision in
[history/decisions.md](history/decisions.md) (the scope rule; there is no numeric LOC
ceiling). Real-workload claims go in `spikes/RESULTS.md` with job ids.
