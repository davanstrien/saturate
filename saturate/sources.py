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

Both sources take `shard=(rank, world)` and select the rows of that shard as
early as they can: before an object is fetched, before an example is decoded.

Needs `datasets` (the [hf] extra); the import is lazy so the core never pays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from saturate.source import ShardedRows, check_shard, content_id, rolling_map, shard_of


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
    prefix is `{config}:{split}` from its `info.config_name` (or `{split}`
    alone); feed several loaded objects into one output with a column or
    callable id instead. Cheap and safe for any column type (images, audio).
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
    `limit` stops after the first `limit` rows of the dataset (before sharding,
    so K shards with one limit cover those rows between them).

    `shard=(rank, world)` yields only the rows whose id belongs to that shard
    (`shard_of`, CONTRACT §2). With index or column ids an example of another
    shard is skipped before it is decoded (images, audio); with content or
    callable ids the id needs the decoded example, so every example is decoded
    and the others are then dropped. Every shard still streams the whole
    dataset. Pass the same `shard` to pump().
    """
    if shard is not None:
        shard = check_shard(shard)
    if isinstance(dataset, str):
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "dataset_rows() needs the `datasets` package — install saturate[hf]"
            ) from e
        repo = dataset
        dataset = load_dataset(
            repo, name=config, split=split, streaming=streaming,
            revision=revision, token=token,
        )
        loaded_config = getattr(getattr(dataset, "info", None), "config_name", None)
        prefix = _index_id_prefix(repo, revision, config or loaded_config or "default", split)
    elif revision is not None or token is not None:
        raise ValueError(
            "revision=/token= only apply when `dataset` is a repo id string; "
            "with an already-loaded dataset they cannot take effect (and a "
            "silently ignored revision would void the index ids)"
        )
    else:
        loaded_config = getattr(getattr(dataset, "info", None), "config_name", None)
        prefix = _index_id_prefix(None, None, loaded_config, split)
    if columns is not None and not callable(ids):
        # column pushdown: select BEFORE iterating so unused features are
        # neither downloaded nor decoded — found live embedding the text
        # column of an image-bearing dataset (392k page scans would have been
        # decoded just to be discarded; ImportError: Pillow was the symptom).
        # A column-name id joins the selection; callable ids see the full
        # example, so no pushdown there.
        keep = set(columns) | ({ids} if ids not in ("index", "content") else set())
        if keep and hasattr(dataset, "select_columns"):
            try:
                dataset = dataset.select_columns(sorted(keep))
            except Exception:
                pass  # unknown names surface naturally at first iteration
    rows = _iter_dataset(dataset, ids, columns, limit, prefix, shard)
    return ShardedRows(rows, shard) if shard is not None else rows


def _iter_dataset(dataset, ids, columns, limit, prefix, shard) -> Iterator[tuple[str, dict]]:
    if shard is not None and hasattr(dataset, "to_iterable_dataset"):
        dataset = dataset.to_iterable_dataset()  # a map-style Dataset: shard it the streaming way
    # An index or column id is known before the example is decoded, so a sharded run
    # iterates the raw examples and decodes only its own (an image decode is the cost).
    id_before_decode = not callable(ids) and ids != "content"
    features = getattr(dataset, "features", None)
    decode_late = (shard is not None and id_before_decode and features is not None
                   and hasattr(dataset, "decode"))
    examples = dataset.decode(False) if decode_late else dataset
    for i, ex in enumerate(examples):
        if limit is not None and i >= limit:
            return
        rid = None
        if decode_late:
            rid = _row_id(ids, ex, None, i, prefix)
            if shard_of(rid, shard[1]) != shard[0]:
                continue
            # the same token the dataset would use to read a private repo's files
            ex = features.decode_example(ex, token_per_repo_id=getattr(dataset, "_token_per_repo_id", None))
        row = {k: ex[k] for k in columns} if columns is not None else dict(ex)
        if rid is None:
            rid = _row_id(ids, ex, row, i, prefix)
            if shard is not None and shard_of(rid, shard[1]) != shard[0]:
                continue
        yield rid, row


def _store_root(fs, path: str) -> str:
    """The root of the store that holds `path`, as a prefix of it: the bucket or repo for
    Hugging Face paths (a repo's revision included), the bucket for object stores that
    expose split_path (s3, gcs, azure), and the filesystem root otherwise (local disk)."""
    in_store = None
    if hasattr(fs, "resolve_path"):  # HfFileSystem
        resolved = fs.resolve_path(path)
        in_store = getattr(resolved, "path_in_repo", None)
        if in_store is None:
            in_store = getattr(resolved, "path", None)
    elif hasattr(fs, "split_path"):  # s3fs, gcsfs, adlfs: (bucket, key, version)
        in_store = fs.split_path(path)[1]
    if in_store and path.endswith(in_store):
        return path[: len(path) - len(in_store)]
    return getattr(fs, "root_marker", "")


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
    real bucket and — decisively — streaming imagefolder drops the file path,
    leaving no stable id to hang resume on. Here the id IS the path ("path",
    default; or any callable path -> id), relative to `root`. The default
    root is the store: the bucket or repo for hf:// (so
    "hf://buckets/org/b/scans/vol1/p1.png" has the id "scans/vol1/p1.png"),
    the bucket for s3/gcs/azure, the filesystem root for a local path (the
    absolute path without its leading "/"). The id does not depend on the
    glob, so vol1/*.png and vol2/*.png never share an id, and widening a glob
    to **/*.png keeps every id it already had. Pass `root` (a URI or path
    that every match lies under) for shorter ids, or ids that survive moving
    a local directory; a match outside it raises.

    `skip` filters paths BEFORE any bytes are read — id-first resume for
    buckets: pass `existing_ids(output)` and a re-run pays listing only, not
    transfer. `shard=(rank, world)` keeps only the paths whose id belongs to
    that shard (`shard_of`, CONTRACT §2), also before any read: each shard
    lists the whole glob and fetches only its own objects. `limit` counts the
    objects left after shard and skip. `read=False` yields {"path"} rows
    without fetching (manifest / counting passes). `prefetch` reads ahead with
    a BOUNDED window of that many in-flight fetches — sequential remote reads
    would starve a fast consumer, and an unbounded prefetch would hold a
    bucket's worth of bytes in RAM (the vision-OOM lesson).

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
    infos = fs.glob(glob_path, detail=True)
    paths = sorted(p for p, i in infos.items() if i.get("type") != "directory")
    if not paths:
        return ShardedRows(iter(()), shard) if shard is not None else iter(())

    if root is not None:
        base_path = fsspec.core.url_to_fs(root)[1].rstrip("/") + "/"
    else:
        base_path = _store_root(fs, paths[0])

    def rid_of(path: str) -> str:
        if callable(ids):
            return str(ids(path))
        if not path.startswith(base_path):
            raise ValueError(f"{path!r} is not under root {base_path!r}")
        return path[len(base_path):]

    selected = []
    if limit is None or limit > 0:  # limit<=0 selects (and fetches) nothing
        for p in paths:
            rid = rid_of(p)
            if not rid:
                raise ValueError(f"id strategy produced no id for {p!r}")
            if shard is not None and shard_of(rid, shard[1]) != shard[0]:
                continue
            if skip and rid in skip:
                continue
            selected.append((rid, p))
            if limit is not None and len(selected) >= limit:
                break

    rows = _fetch(fs, selected, prefetch) if read else _listed(selected)
    return ShardedRows(rows, shard) if shard is not None else rows


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
