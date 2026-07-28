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


def normalize(rows: Iterable, id_key: str | None = None, id_keys: list[str] | None = None,
              id_fn=None) -> Iterator[tuple[str, dict]]:
    """Yield (id, row) lazily. Accepts (id, row) tuples as-is; dicts get an id
    from `id_fn(row)`, else `id_key`, else a content hash (over `id_keys`)."""
    for item in rows:
        if isinstance(item, tuple) and len(item) == 2:
            yield str(item[0]), item[1]
        elif isinstance(item, dict):
            if id_fn is not None:
                yield str(id_fn(item)), item
            elif id_key is not None:
                yield str(item[id_key]), item
            else:
                yield content_id(item, id_keys), item
        else:
            raise TypeError(f"row must be (id, dict) or dict, got {type(item).__name__}")


stream = normalize  # the README-facing name


def skip_done(rows: Iterable[tuple[str, dict]], done, retry_errors: bool = False,
              stats=None) -> Iterator[tuple[str, dict]]:
    """Exact resume filter + within-run dedup (first admission wins).

    `done` may be: a set of ids, a sink/object with existing_ids(), or an
    output-dir string (resolved via the CONTRACT ParquetSink). Counts land on
    `stats` when given (rows_total / rows_done_prior / rows_deduped)."""
    if isinstance(done, str):
        from pumpjack.sink import ParquetSink

        done = ParquetSink(done).existing_ids(retry_errors=retry_errors)
    elif hasattr(done, "existing_ids"):
        done = done.existing_ids(retry_errors=retry_errors)
    seen: set[str] = set()
    for id_, row in rows:
        if stats is not None:
            stats.rows_total += 1
        if id_ in done:
            if stats is not None:
                stats.rows_done_prior += 1
            continue
        if id_ in seen:
            if stats is not None:
                stats.rows_deduped += 1
            continue
        seen.add(id_)
        yield id_, row


def shard_select(rows: Iterable[tuple[str, dict]], rank: int = 0, world: int = 1,
                 skip: int = 0) -> Iterator[tuple[str, dict]]:
    """Strided fan-out assignment over the global stream (CONTRACT §2):
    keep(idx) = (idx - skip) % world == rank."""
    for idx, pair in enumerate(rows):
        if idx >= skip and (idx - skip) % world == rank:
            yield pair
