"""HF dataset -> (id, row) stream: the input half of "dataset in, endpoint out".

Streaming by default. `load_dataset(streaming=True)` now runs at local-SSD
speed for exactly this access pattern — one sequential pass, no shuffle, no
epochs (persistent file cache, bundled resolution, prefetching:
hf.co/blog/streaming-datasets) — and a naive materializing load killed a real
Job on disk (EBDC input-side notes, 2026-07-16), so materializing is never the
default. `streaming=False` stays one flag away for small datasets and
repeated local runs.

Ids are a trust contract, not an enforced policy: whichever strategy (or
callable) produces them, the caller is asserting uniqueness within the output
and stability across resume — the pump trusts what it is handed, and
downstream resume/dedup are only as good as that promise (CONTRACT §2).
The one thing this module enforces is loudness: a strategy that produces no
id (None/empty) raises instead of silently colliding rows on "None".

Needs `datasets` (the [hf] extra); the import is lazy so the core never pays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from pumpjack.source import content_id


def dataset_rows(
    dataset,
    config: str | None = None,
    split: str = "train",
    *,
    columns: list[str] | None = None,
    limit: int | None = None,
    streaming: bool = True,
    revision: str | None = None,
    ids: str | Callable = "index",
    token: str | None = None,
) -> Iterator[tuple[str, dict]]:
    """Rows from a Hub dataset (repo id) or an already-loaded
    Dataset/IterableDataset object (pass your own for custom loading; tests do).
    `revision` and `token` apply only to repo-id loading — passing them with an
    already-loaded object raises (they could not take effect, and a silent
    no-op on `revision` would quietly void the index-id stability caveat below).

    ids — a default per workload, never an enforced approach:
    "index" (default) — `{split}-{i:09d}`, cheap and safe for any column type
    (images, audio); stable for a fixed dataset+revision+iteration order, so
    pin `revision` if the repo may move under a resume.
    "content" — order-independent content hash via `content_id` over the
    *yielded* row (after `columns` filtering — that is the lever for keeping
    the hash JSON-safe; strict, refuses images. Dedups identical rows,
    receipt: 38 real dolly dupes caught).
    A column name — a natural key, looked up on the FULL example, so the id
    column does not need to be in `columns`.
    A callable `example -> id` — your own scheme, also given the full
    example, trusted as-is (must return something non-empty).

    `columns` selects a subset into the yielded row (slims what flows
    downstream; also the JSON-safety lever for ids="content").
    """
    if isinstance(dataset, str):
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "dataset_rows() needs the `datasets` package — install pumpjack[hf]"
            ) from e
        dataset = load_dataset(
            dataset, name=config, split=split, streaming=streaming,
            revision=revision, token=token,
        )
    elif revision is not None or token is not None:
        raise ValueError(
            "revision=/token= only apply when `dataset` is a repo id string; "
            "with an already-loaded dataset they cannot take effect (and a "
            "silently ignored revision would void the index-id stability caveat)"
        )
    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            return
        row = {k: ex[k] for k in columns} if columns is not None else dict(ex)
        if callable(ids):
            rid = ids(ex)
        elif ids == "index":
            rid = f"{split}-{i:09d}"
        elif ids == "content":
            rid = content_id(row)
        else:
            rid = ex[ids]
        if rid is None or str(rid) == "":
            raise ValueError(
                f"id strategy produced no id ({rid!r}) at row {i} — refusing to "
                'collide rows on "None"/""; fix the callable/column or use ids="index"'
            )
        yield str(rid), row
