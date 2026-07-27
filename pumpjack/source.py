"""Source: normalize any row iterable into a lazy (id, row) stream.

Streaming by default — the pump never materializes the input (the EBDC
disk-death lesson: a naive load_dataset killed a Job on disk). Works with any
iterable of dicts, including a streaming HF `IterableDataset`.

The v1 default id is a content hash (CONTRACT §2): order-independent,
re-shard-safe, no cleverness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator


def content_id(row: dict, keys: list[str] | None = None) -> str:
    """Stable 16-hex content hash of the row (or of `keys` fields only)."""
    src = {k: row[k] for k in keys} if keys else row
    blob = json.dumps(src, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def normalize(rows: Iterable, id_key: str | None = None,
              id_keys: list[str] | None = None) -> Iterator[tuple[str, dict]]:
    """Yield (id, row) lazily. Accepts (id, row) tuples as-is; dicts get an id
    from `id_key`, else a content hash (over `id_keys` fields if given)."""
    for item in rows:
        if isinstance(item, tuple) and len(item) == 2:
            yield str(item[0]), item[1]
        elif isinstance(item, dict):
            if id_key is not None:
                yield str(item[id_key]), item
            else:
                yield content_id(item, id_keys), item
        else:
            raise TypeError(f"row must be (id, dict) or dict, got {type(item).__name__}")


def shard_select(rows: Iterable[tuple[str, dict]], rank: int = 0, world: int = 1,
                 skip: int = 0) -> Iterator[tuple[str, dict]]:
    """Strided fan-out assignment over the global stream (CONTRACT §2):
    keep(idx) = (idx - skip) % world == rank."""
    for idx, pair in enumerate(rows):
        if idx >= skip and (idx - skip) % world == rank:
            yield pair
