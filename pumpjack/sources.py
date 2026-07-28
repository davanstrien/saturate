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

    ids — a default per workload, never an enforced approach:
    "index" (default) — `{split}-{i:09d}`, cheap and safe for any column type
    (images, audio); stable for a fixed dataset+revision+iteration order, so
    pin `revision` if the repo may move under a resume.
    "content" — order-independent content hash via `content_id` (JSON-safe
    columns only — strict, refuses images; dedups identical rows, receipt:
    38 real dolly dupes caught).
    A column name — a natural key your dataset already carries.
    A callable `row -> id` — your own scheme, trusted as-is.

    `columns` selects a subset into the row (slims passthrough; also the way
    to keep rows JSON-safe for ids="content" when extra columns carry objects).
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
    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            return
        row = {k: ex[k] for k in columns} if columns else dict(ex)
        if callable(ids):
            rid = str(ids(row))
        elif ids == "index":
            rid = f"{split}-{i:09d}"
        elif ids == "content":
            rid = content_id(row)
        else:
            rid = str(row[ids])
        yield rid, row
