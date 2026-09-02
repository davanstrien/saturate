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
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial


def _reject_non_json(o):
    raise TypeError(
        f"row contains a non-JSON value ({type(o).__name__}) — its str() is not stable "
        "across runs, so a content id would break resume. Pass (id, row) tuples, "
        "id_key=, or id_fn= for rows carrying objects (images, audio, tensors); "
        "from dataset_rows use ids=<key column>, a callable, or the index default."
    )


def content_id(row: dict, keys: list[str] | None = None) -> str:
    """Stable 16-hex content hash of the row (or of `keys` fields only).
    Strict by design: refuses non-JSON values rather than hashing an unstable
    repr (a PIL image's str() embeds its memory address)."""
    src = {k: row[k] for k in keys} if keys else row
    blob = json.dumps(src, sort_keys=True, default=_reject_non_json, ensure_ascii=False)
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
        from saturate.sink import ParquetSink

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


def rolling_map(items: Iterable, fn: Callable, executor: Executor, window: int) -> Iterator:
    """`map(fn, items)` on `executor`, in order, with at most `window` calls in flight:
    one result out, one call in. The consumer's pace bounds the look-ahead, so a slow
    consumer never accumulates results in RAM. The first exception from `fn` is raised
    at the consumer and the calls still queued are cancelled."""
    window = max(1, window)
    pending: deque = deque()
    it = iter(items)
    try:
        for item in it:  # prime the window
            pending.append(executor.submit(fn, item))
            if len(pending) >= window:
                break
        while pending:  # rolling: one out, one in — never more than `window` in flight
            fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(executor.submit(fn, nxt))
            yield fut.result()
    finally:
        for fut in pending:
            fut.cancel()


def _apply_to_row(fn: Callable[[dict], dict], item: tuple[str, dict]) -> tuple[str, dict]:
    id_, row = item
    return id_, fn(row)


def prepare_ahead(rows: Iterable[tuple[str, dict]], fn: Callable[[dict], dict], workers: int = 4,
                  executor: Executor | None = None) -> Iterator[tuple[str, dict]]:
    """(id, row) -> (id, fn(row)) computed ahead of the consumer on `executor`, order
    preserved, at most `workers` calls in flight. The default executor is a
    `ThreadPoolExecutor(workers)` owned by the iterator; pass your own (a
    `ProcessPoolExecutor` for pure-Python CPU work, or anything `concurrent.futures`-shaped)
    to control its lifetime and size — with a process pool, `fn` and the rows must be
    picklable (a module-level function, not a lambda). An exception in `fn` is raised at
    the consumer. Work done here happens before the pump sees the row, so a pump fed by
    prepare_ahead reads that cost as source wait (SOURCE-BOUND), not prep."""

    apply = partial(_apply_to_row, fn)  # module-level + partial: picklable for a process pool
    if executor is not None:
        yield from rolling_map(rows, apply, executor, workers)
        return
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="saturate-prepare-ahead")
    try:
        yield from rolling_map(rows, apply, pool, workers)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def shard_select(rows: Iterable[tuple[str, dict]], rank: int = 0, world: int = 1,
                 skip: int = 0) -> Iterator[tuple[str, dict]]:
    """Strided fan-out assignment over the global stream (CONTRACT §2):
    keep(idx) = (idx - skip) % world == rank."""
    for idx, pair in enumerate(rows):
        if idx >= skip and (idx - skip) % world == rank:
            yield pair
