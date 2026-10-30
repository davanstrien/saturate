"""HF dataset -> (id, row) stream: the input half of "dataset in, endpoint out".

Streaming by default. `load_dataset(streaming=True)` runs at local-SSD
speed for exactly this access pattern — one sequential pass, no shuffle, no
epochs (persistent file cache, bundled resolution, prefetching:
hf.co/blog/streaming-datasets) — and materializing a large input can exhaust
a Job's local disk, so materializing is never the default. `streaming=False`
stays one flag away for small datasets and repeated local runs.

Ids are a trust contract, not an enforced policy: whichever strategy (or
callable) produces them, the caller is asserting uniqueness within the output
and stability across resume — the pump trusts what it is handed, and
downstream resume/dedup are only as good as that promise (CONTRACT §2).
The one thing this module enforces is loudness: a strategy that produces no
id (None/empty) raises instead of silently colliding rows on "None".

Both sources take `shard=(rank, world)` and select the rows of that shard as
early as they can: before an object is fetched, before an example is decoded.

Needs `datasets` (the [hf] extra); the import is lazy so the core never pays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from saturate.source import check_shard, content_id, rolling_map, shard_of


def _index_id_prefix(repo: str | None, revision: str | None, config: str | None, split: str) -> str:
    """What names the data an index id points into: `{repo}[@{revision}]:{config}:{split}`,
    leaving out the parts that are not known."""
    parts = []
    if repo is not None:
        parts.append(f"{repo}@{revision}" if revision else repo)
    if config:
        parts.append(config)
    parts.append(split)
    return ":".join(parts)


def _row_id(ids: str | Callable, example: dict, row: dict | None, i: int, prefix: str) -> str:
    """The id of example `i` under the `ids` strategy. `row` (the yielded row) is needed only
    for ids="content"."""
    if callable(ids):
        rid = ids(example)
    elif ids == "index":
        rid = f"{prefix}:{i:09d}"
    elif ids == "content":
        rid = content_id(row if row is not None else example)
    else:
        rid = example[ids]
    if rid is None or str(rid) == "":
        raise ValueError(
            f"id strategy produced no id ({rid!r}) at row {i} — refusing to "
            'collide rows on "None"/""; fix the callable/column or use ids="index"'
        )
    return str(rid)


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
    shard: tuple[int, int] | None = None,
) -> Iterator[tuple[str, dict]]:
    """Rows from a Hub dataset (repo id) or an already-loaded
    Dataset/IterableDataset object (pass your own for custom loading; tests do).
    `revision` and `token` apply only to repo-id loading — passing them with an
    already-loaded object raises (they could not take effect, and a silent
    no-op on `revision` would quietly void the index ids below).

    ids — a default per workload, never an enforced approach:
    "index" (default) — the row's position, prefixed with what names the data:
    `{repo}[@{revision}]:{config}:{split}:{i:09d}`, e.g.
    "org/books:en:train:000000012". Two configs, splits or datasets written to
    one output never share an id. The revision is part of the id only when
    you pin one: a position names the same row only within one snapshot, so a
    new pin re-runs every row, while an unpinned run follows the branch and
    trusts that rows were not reordered. For an already-loaded object the
    prefix is `{config}:{split}` from the arguments you pass (`{split}` alone
    without `config`); feed several loaded objects into one output with a
    column or callable id instead. Cheap and safe for any column type (images, audio).
    "content" — order-independent content hash via `content_id` over the
    *yielded* row (after `columns` filtering). Raw bytes hash by content: an
    Image or Audio column loaded with `decode=False` yields {"bytes", "path"}
    and works. A decoded value (PIL image, audio array) is refused; drop it
    with `columns` or load the column undecoded. Dedups identical rows.
    A column name — a natural key, looked up on the FULL example, so the id
    column does not need to be in `columns`.
    A callable `example -> id` — your own scheme, also given the full
    example, trusted as-is (must return something non-empty).

    `columns` selects a subset into the yielded row (slims what flows
    downstream; also how ids="content" leaves out a decoded column).
    `limit` stops after the first `limit` rows of the dataset (before sharding,
    so K shards with one limit cover those rows between them).

    `shard=(rank, world)` yields only the rows whose id belongs to that shard
    (`shard_of`, CONTRACT §2). When `dataset` is a repo id and the id is an
    index or a column, an example of another shard is skipped before it is
    decoded (images, audio). An already-loaded object is iterated as it is:
    every example is decoded, then the other shards' rows are dropped. Content
    and callable ids need the decoded example, so they decode every example
    too. The rows are the same either way; only the cost differs. Every shard
    still streams the whole dataset. Pass the same `shard` to pump().
    """
    if shard is not None:
        shard = check_shard(shard)
    repo = None
    if isinstance(dataset, str):
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError("dataset_rows() needs the `datasets` package — install saturate[hf]") from e
        repo = dataset
        dataset = load_dataset(
            repo,
            name=config,
            split=split,
            streaming=streaming,
            revision=revision,
            token=token,
        )
        prefix = _index_id_prefix(repo, revision, config or dataset.info.config_name or "default", split)
    elif revision is not None or token is not None:
        raise ValueError(
            "revision=/token= only apply when `dataset` is a repo id string; "
            "with an already-loaded dataset they cannot take effect (and a "
            "silently ignored revision would void the index ids)"
        )
    else:
        prefix = _index_id_prefix(None, None, config, split)
    if columns is not None and not callable(ids):
        # column pushdown: select BEFORE iterating so unused features are
        # neither downloaded nor decoded (embedding the text column of an
        # image-bearing dataset would otherwise decode every image, and need Pillow).
        # A column-name id joins the selection; callable ids see the full
        # example, so no pushdown there. An object passed in without
        # select_columns is iterated whole.
        keep = set(columns) | ({ids} if ids not in ("index", "content") else set())
        if keep and hasattr(dataset, "select_columns"):
            try:
                dataset = dataset.select_columns(sorted(keep))
            except Exception:
                pass  # unknown names surface naturally at first iteration
    # Decoding only this shard's examples needs the id before the decode (an index or a
    # column), a dataset loaded here (so nothing upstream decoded it already), and known
    # features to decode with (a JSON or CSV repo without dataset_info has none).
    if (
        shard is not None
        and repo is not None
        and not callable(ids)
        and ids != "content"
        and dataset.features is not None
    ):
        from datasets import Dataset

        if isinstance(dataset, Dataset):  # streaming=False
            if len(dataset) == 0:  # to_iterable_dataset() cannot shard an empty Dataset
                return iter(())
            dataset = dataset.to_iterable_dataset()
        return _iter_decoding_late(dataset, repo, token, ids, columns, limit, prefix, shard)
    return _iter_dataset(dataset, ids, columns, limit, prefix, shard)


def _iter_dataset(
    dataset: Iterable[dict],
    ids: str | Callable,
    columns: list[str] | None,
    limit: int | None,
    prefix: str,
    shard: tuple[int, int] | None,
) -> Iterator[tuple[str, dict]]:
    """(id, row) for each example as `dataset` yields it, keeping only shard `shard`'s rows."""
    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            return
        row = {k: ex[k] for k in columns} if columns is not None else dict(ex)
        rid = _row_id(ids, ex, row, i, prefix)
        if shard is not None and shard_of(rid, shard[1]) != shard[0]:
            continue
        yield rid, row


def _iter_decoding_late(
    dataset,
    repo: str,
    token: str | None,
    ids: str,
    columns: list[str] | None,
    limit: int | None,
    prefix: str,
    shard: tuple[int, int],
) -> Iterator[tuple[str, dict]]:
    """Like `_iter_dataset` for an IterableDataset loaded here from `repo`, with an index or
    column id: the id is known before the example is decoded, so the examples are read
    undecoded and only shard `shard`'s are decoded (an image or audio decode is the cost)."""
    features = dataset.features
    token_per_repo_id = {repo: token} if token is not None else None
    for i, ex in enumerate(dataset.decode(False)):
        if limit is not None and i >= limit:
            return
        rid = _row_id(ids, ex, None, i, prefix)
        if shard_of(rid, shard[1]) != shard[0]:
            continue
        ex = features.decode_example(ex, token_per_repo_id=token_per_repo_id)
        yield rid, ({k: ex[k] for k in columns} if columns is not None else dict(ex))


def bucket_rows(
    pattern: str,
    *,
    ids: str | Callable = "path",
    root: str | None = None,
    skip: set[str] | None = None,
    limit: int | None = None,
    read: bool = True,
    prefetch: int = 8,
    shard: tuple[int, int] | None = None,
) -> Iterator[tuple[str, dict]]:
    """Objects matching an fsspec glob -> (id, {"path", "bytes"}) — the raw-object
    input shape (an input dir of images, scans, PDFs...). Works on any fsspec
    URI: hf://buckets/org/name/scans/**/*.png, s3://..., or a local directory
    (which is what the tests use).

    Chosen over datasets/imagefolder deliberately: measured 1.6x faster on a
    real bucket, and streaming imagefolder drops the file path,
    leaving no stable id to hang resume on. Here the id IS the path ("path",
    default; or any callable path -> id): the path fsspec lists, without the
    scheme ("buckets/org/b/scans/vol1/p1.png" for
    "hf://buckets/org/b/scans/vol1/p1.png", "bucket/key" for s3, the absolute
    path for a local file). The id does not depend on the glob, so vol1/*.png
    and vol2/*.png never share an id, and widening a glob to **/*.png keeps
    every id it already had. Pass `root` (a URI or path that every match lies
    under) to drop that prefix from the ids: shorter ids, or ids that survive
    moving a local directory. A match outside `root` raises.

    `skip` filters paths BEFORE any bytes are read — id-first resume for
    buckets: pass `existing_ids(output)` and a re-run pays listing only, not
    transfer. `shard=(rank, world)` keeps only the paths whose id belongs to
    that shard (`shard_of`, CONTRACT §2), also before any read: each shard
    lists the whole glob and fetches only its own objects. `limit` keeps the
    first `limit` objects of the sorted listing, before shard and skip, as
    `limit` in dataset_rows does: K shards with one limit cover those objects
    between them, and a re-run with the same limit finishes the same objects
    rather than taking new ones. `read=False` yields {"path"} rows
    without fetching (manifest / counting passes). `prefetch` reads ahead with
    a BOUNDED window of that many in-flight fetches — sequential remote reads
    would starve a fast consumer, and an unbounded prefetch would hold a
    bucket's worth of bytes in RAM.

    No decode, no transforms (PDF-to-pages etc. stay a caller concern —
    decode explodes one object into N rows, which silently changes id
    semantics; see docs/history/decisions.md).
    """
    import fsspec

    if not callable(ids) and ids != "path":
        raise ValueError(
            f'bucket_rows ids must be "path" or a callable, got {ids!r} — '
            "silently falling back to path ids would violate the caller's resume scheme"
        )
    if shard is not None:
        shard = check_shard(shard)

    # url_to_fs resolves the scheme (and chained URIs like zip://...::archive.zip)
    # and returns the filesystem-native path — glob and id derivation both use IT,
    # never a re-parse of the original URI. Globbing, IO and URI chaining are all
    # fsspec/HfFileSystem's.
    fs, glob_path = fsspec.core.url_to_fs(pattern)
    base_path = ""
    if root is not None:
        root_fs, root_path = fsspec.core.url_to_fs(root)
        if type(root_fs) is not type(fs):
            raise ValueError(
                f"root {root!r} is not on the same filesystem as {pattern!r}: "
                "give it with the same scheme (e.g. hf://buckets/org/name/scans)"
            )
        base_path = root_path.rstrip("/") + "/"
    infos = fs.glob(glob_path, detail=True)
    paths = sorted(p for p, i in infos.items() if i.get("type") != "directory")
    if not paths:
        return iter(())

    def rid_of(path: str) -> str:
        if callable(ids):
            return str(ids(path))
        if not path.startswith(base_path):
            raise ValueError(f"{path!r} is not under root {base_path!r}")
        return path[len(base_path) :]

    if limit is not None:
        paths = paths[: max(0, limit)]  # the first `limit` objects, before shard and skip
    selected = []
    for p in paths:
        rid = rid_of(p)
        if not rid:
            raise ValueError(f"id strategy produced no id for {p!r}")
        if shard is not None and shard_of(rid, shard[1]) != shard[0]:
            continue
        if skip and rid in skip:
            continue
        selected.append((rid, p))

    return _fetch(fs, selected, prefetch) if read else _listed(selected)


def _listed(selected: list[tuple[str, str]]) -> Iterator[tuple[str, dict]]:
    for rid, p in selected:
        yield rid, {"path": p}


def _fetch(fs, selected: list[tuple[str, str]], prefetch: int) -> Iterator[tuple[str, dict]]:
    from concurrent.futures import ThreadPoolExecutor

    def fetch(item):
        rid, p = item
        with fs.open(p, "rb") as f:
            return rid, {"path": p, "bytes": f.read()}

    window = max(1, prefetch)
    with ThreadPoolExecutor(max_workers=window) as pool:
        yield from rolling_map(selected, fetch, pool, window)  # never > window bytes-in-flight
