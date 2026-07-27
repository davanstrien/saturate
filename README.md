# pumpjack

**PRIVATE — pre-release.** Adaptive batch-inference client: dataset in, any OpenAI-compatible
endpoint, resumable parquet out.

One client between your rows and an endpoint you may or may not own. It ramps concurrency
like congestion control (delivered throughput primary, engine `/metrics` gauges as an
accelerator, blind AIMD as the floor), writes crash-safe append-only parquet with exact
anti-join resume, and is safe to `kill -9` and re-run.

```python
from pumpjack import pump, Auto

stats = pump(
    rows,                      # iterable of dicts or (id, dict); streams, never materializes
    to_request=lambda row: {"model": "m", "messages": [{"role": "user", "content": row["text"]}]},
    parse=lambda row, resp: {"out": resp["choices"][0]["message"]["content"]},
    endpoint="http://127.0.0.1:8000/v1",
    output="out/",             # local path, hf://datasets/..., hf://buckets/...
    window=Auto(),             # or Fixed(n)
)
```

- **CONTRACT.md** — the frozen storage protocol (what any reader/writer can rely on).
- **DECISIONS.md** — build decisions and their receipts.
- Acceptance oracle: `../pumpjack-oracle` (`ORACLE_ADAPTER=clean uv run pytest`).

Apache-2.0. Not on PyPI yet; name registration deliberately deferred.
