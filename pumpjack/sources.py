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


def bucket_rows(
    pattern: str,
    *,
    ids: str | Callable = "path",
    skip: set[str] | None = None,
    limit: int | None = None,
    read: bool = True,
    prefetch: int = 8,
) -> Iterator[tuple[str, dict]]:
    """Objects matching an fsspec glob -> (id, {"path", "bytes"}) — the raw-object
    input shape (an input dir of images, scans, PDFs...). Works on any fsspec
    URI: hf://buckets/org/name/scans/**/*.png, s3://..., or a local directory
    (which is what the tests use).

    Chosen over datasets/imagefolder deliberately: measured 1.6x faster on a
    real bucket and — decisively — streaming imagefolder drops the file path,
    leaving no stable id to hang resume on. Here the id IS the path relative
    to the glob's static prefix ("path", default; or any callable path -> id).

    `skip` filters paths BEFORE any bytes are read — id-first resume for
    buckets: pass `existing_ids(output)` and a re-run pays listing only, not
    transfer. `read=False` yields {"path"} rows without fetching (manifest /
    counting passes). `prefetch` reads ahead with a BOUNDED window of that
    many in-flight fetches — sequential remote reads would starve a fast
    consumer, and an unbounded prefetch would hold a bucket's worth of bytes
    in RAM (the vision-OOM lesson).

    No decode, no transforms (PDF-to-pages etc. stay a caller concern —
    decode explodes one object into N rows, which silently changes id
    semantics; see docs/decisions.md).
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(pattern)
    # ids are relative to the static prefix (chars before the first wildcard)
    static = pattern.split("*")[0].split("?")[0]
    base = static[: static.rfind("/") + 1]
    _, base_path = fsspec.core.url_to_fs(base + "x")  # resolve scheme prefix
    base_path = base_path[:-1]
    raw = pattern.split("://")[-1] if "://" in pattern else pattern
    infos = fs.glob(raw, detail=True)
    paths = sorted(p for p, i in infos.items() if i.get("type") != "directory")

    def rid_of(path: str) -> str:
        rel = path[len(base_path):] if path.startswith(base_path) else path
        return str(ids(path)) if callable(ids) else rel

    selected = []
    for p in paths:
        rid = rid_of(p)
        if not rid:
            raise ValueError(f"id strategy produced no id for {p!r}")
        if skip and rid in skip:
            continue
        selected.append((rid, p))
        if limit is not None and len(selected) >= limit:
            break

    if not read:
        for rid, p in selected:
            yield rid, {"path": p}
        return

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def fetch(item):
        rid, p = item
        with fs.open(p, "rb") as f:
            return rid, {"path": p, "bytes": f.read()}

    window = max(1, prefetch)
    with ThreadPoolExecutor(max_workers=window) as pool:
        pending: deque = deque()
        it = iter(selected)
        for item in it:  # prime the window
            pending.append(pool.submit(fetch, item))
            if len(pending) >= window:
                break
        while pending:  # rolling: one out, one in — never > window bytes-in-flight
            fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(fetch, nxt))
            yield fut.result()
