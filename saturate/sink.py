"""Sinks: where the Done stream lands.

The resume contract is one invariant (CONTRACT §3): an id returned by
existing_ids() implies its record is durable; an id not returned implies
re-processing it is safe. Any object satisfying `existing_ids/append/flush`
(duck-typed) is a sink; markers and telemetry hooks are optional. A sink that offers
`write_telemetry(shard, lines)` receives the run's cumulative tick lines repeatedly
(periodically from a worker thread, once more at exit): implement it as a whole-file
rewrite, never an append, and keep it independent of the append/flush state.

ParquetSink — the full storage CONTRACT (parts + manifest sidecar + error rows
+ healing). FileSink — one file per row, named by id: the filesystem is the
manifest, overwrites are idempotent; failed rows leave no record (they retry).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)


def _untyped(t: pa.DataType) -> bool:
    """A type with no typed leaf (null, list<null>, struct<>, ...): inferred from values that
    carry no information, so it can never be a pin — a later real value must be free to set it."""
    if pa.types.is_null(t):
        return True
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _untyped(t.value_type)
    if pa.types.is_struct(t):
        return all(_untyped(f.type) for f in t)
    return False


def _list_parts(fs, root: str) -> list[str]:
    """The data parts of an output dir, in write order (the CONTRACT part-name pattern)."""
    return sorted(fs.glob(f"{root}/part-*.parquet"))


def _error_row(id_, exc: BaseException) -> dict:
    """The record a row that cannot be stored leaves behind (CONTRACT §4)."""
    return {"id": str(id_), "error": f"parse output not storable: {exc}"}


def _has_untyped_leaf(t: pa.DataType) -> bool:
    """Whether some leaf of `t` is null-typed (a struct field only ever seen as None)."""
    if pa.types.is_null(t):
        return True
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _has_untyped_leaf(t.value_type)
    if pa.types.is_struct(t):
        return any(_has_untyped_leaf(f.type) for f in t)
    return False


def _storable(t: pa.DataType, top: bool = True) -> bool:
    """Parquet cannot write a struct with no fields below the top level (a nested `{}`);
    a top-level one is an untyped column and is simply dropped."""
    if pa.types.is_struct(t):
        return (top or len(t) > 0) and all(_storable(f.type, False) for f in t)
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _storable(t.value_type, False)
    return True


def _accepts(got: pa.DataType, pinned: pa.DataType) -> bool:
    """Whether values of type `got` may be stored in a column pinned to `pinned`: same type,
    untyped (nulls), or the same kind of thing with the value-level check left to a safe cast
    (2 fits int32; 0.5 into int64 is refused by the cast). Kinds never cross: a float into an
    int column, an int into a string column, a bool into an int column are conflicts, because
    a cast there rewrites the value rather than records it. Recursive through lists and structs
    (a struct may lack pinned fields, never carry extra ones)."""
    if got == pinned or _untyped(got):
        return True
    num = lambda t: pa.types.is_integer(t) or pa.types.is_floating(t)  # noqa: E731 (bool is not an integer here)
    if num(got) and num(pinned):
        return not (pa.types.is_floating(got) and pa.types.is_integer(pinned))
    for same in (pa.types.is_decimal, pa.types.is_timestamp, pa.types.is_date, pa.types.is_time,
                 pa.types.is_duration, pa.types.is_boolean,
                 lambda t: pa.types.is_string(t) or pa.types.is_large_string(t),
                 lambda t: pa.types.is_binary(t) or pa.types.is_large_binary(t)):
        if same(got) and same(pinned):
            return True
    if (pa.types.is_list(got) or pa.types.is_large_list(got)) and \
            (pa.types.is_list(pinned) or pa.types.is_large_list(pinned)):
        return _accepts(got.value_type, pinned.value_type)
    if pa.types.is_struct(got) and pa.types.is_struct(pinned):
        names = {f.name for f in pinned}
        return all(f.name in names and _accepts(f.type, pinned.field(f.name).type) for f in got)
    return False


class ParquetSink:
    def __init__(self, out_uri: str, flush_every: int = 10, schema: pa.Schema | None = None,
                 read_workers: int = 16):
        import fsspec

        self.fs, self.root = fsspec.url_to_fs(out_uri)
        self.flush_every = flush_every
        self.read_workers = read_workers  # resume read concurrency; 1 = sequential
        self._buf: list[dict] = []
        # declared mode (r5): an immutable schema every part is cast to — the only fully
        # schema-stable option for arbitrary parse output. Must carry contract-typed id + error.
        if schema is not None:
            if not {"id", "error"} <= set(schema.names):
                raise ValueError("declared schema must include the id and error columns")
            if (schema.field("id").type != pa.string() or schema.field("error").type != pa.string()
                    or not schema.field("error").nullable):
                raise ValueError("declared schema: id and error must be string, error nullable")
            bad = [f.name for f in schema if f.name != "id" and not f.nullable]
            if bad:  # r7: error rows write user fields as null — non-nullable would crash the flush
                raise ValueError(f"declared schema: fields must be nullable (error rows are sparse): {bad}")
        self._declared = schema
        self._schema: pa.Schema | None = None  # dynamic mode: pinned at first sight, widened by new columns
        self._pinned: dict[str, pa.DataType] = {}  # name -> pinned type (empty until pinned)
        if schema is not None:
            self._pin(schema)
        self._seeded = schema is not None  # dynamic mode reads its pin back from existing parts
        self.rows_demoted = 0  # rows flush turned into error rows (type conflicts, cumulative)
        self._seq = 0  # per-sink flush counter: sorted(part names) == write order on ms collisions
        self._telemetry_names: dict[int, str] = {}  # shard rank -> this run's telemetry file
        self._accepted: dict[tuple[str, pa.DataType], bool] = {}  # (name, got) -> _accepts verdict
        self._buf_schema: pa.Schema | None = None  # types of the rows buffered so far (dynamic mode)
        self._parts: list[str] | None = None  # part listing from existing_ids(), reused by the seed
        self._probe_cache: tuple[dict, pa.Schema] | None = None  # the record probe() last typed
        proto = getattr(self.fs, "protocol", ())  # fsspec: a str or a tuple of aliases — match exactly
        protos = set(proto) if isinstance(proto, (tuple, list)) else {proto}
        local = "file" in protos
        # periodic telemetry cadence in controller ticks (~2 s each): about a minute locally;
        # on a remote store every rewrite is a commit, so about five minutes there
        self.telemetry_every_ticks = 30 if local else 150
        self.rows_written = 0
        # hf:// outputs: ensure the dataset repo / bucket exists (private) at
        # construction — a fresh output path must not crash existing_ids()
        # (live failure: RepositoryNotFoundError on a never-created repo). A
        # typo'd path becomes an empty private repo instead of a hard crash;
        # exist_ok makes re-runs no-ops. Failure here is logged, not fatal —
        # the later glob raises with the real story if the repo truly can't exist.
        if "hf" in protos and self.root.startswith(("datasets/", "buckets/")):
            parts = self.root.split("/")
            if len(parts) >= 3:
                repo = f"{parts[1]}/{parts[2]}"
                try:
                    from huggingface_hub import HfApi

                    if parts[0] == "datasets":
                        HfApi().create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                    else:
                        HfApi().create_bucket(repo, private=True, exist_ok=True)
                except Exception as e:
                    _log(f"could not ensure output repo {repo}: {e}")
        try:
            self.fs.makedirs(self.root, exist_ok=True)
        except Exception:
            pass  # HfFileSystem: directories are implicit beyond the repo itself

    def _read_parquet(self, path: str, reader, attempts: int = 3):
        """`reader(file)` on an opened path, or None if the file is unreadable. A remote read can
        fail transiently (a 429 under concurrent reads, a reset), and an unreadable file re-pays a
        whole flush, so I/O errors are retried; bytes that are not parquet (crash-truncated) are
        not."""
        for i in range(attempts):
            try:
                with self.fs.open(path, "rb") as f:
                    return reader(f)
            except pa.ArrowException as e:
                _log(f"skipping unreadable {path}: {e}")
                return None
            except Exception as e:
                if i + 1 < attempts:
                    time.sleep(0.5 * 2**i)
                else:
                    _log(f"skipping unreadable {path}: {e}")
        return None

    def _read_id_error(self, path: str) -> tuple[set[str], set[str]] | None:
        """(done ids, failed ids) of one manifest or part; None if the file is unreadable."""
        t = self._read_parquet(path, lambda f: pq.read_table(f, columns=["id", "error"]))
        if t is None:
            return None
        done: set[str] = set()
        failed: set[str] = set()
        for id_, err in zip(t["id"].to_pylist(), t["error"].to_pylist(), strict=True):
            (failed if err else done).add(id_)
        return done, failed

    def _read_many(self, paths: list[str]) -> dict[str, tuple[set[str], set[str]]]:
        """{path: (done, failed)} for every readable file, in path order. Remote resume is
        bound by per-file round-trips, not parsing, so the files are read on a thread pool;
        fsspec is synchronous."""
        if self.read_workers <= 1 or len(paths) <= 1:
            got = [self._read_id_error(p) for p in paths]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(min(self.read_workers, len(paths))) as ex:
                got = list(ex.map(self._read_id_error, paths))
        return {p: g for p, g in zip(paths, got, strict=True) if g is not None}

    def existing_ids(self, retry_errors: bool = False) -> set[str]:
        """Manifest-first, exact: any part lacking its manifest sidecar is
        scanned individually (kill-safe). retry_errors=True excludes ids whose
        only record is an error."""
        manifests = sorted(self.fs.glob(f"{self.root}/_manifest/ids-*.parquet"))
        if len(manifests) > 1000:
            _log(f"resume: reading {len(manifests)} manifest files "
                 "(consider larger flush_every for remote outputs)")
        read = self._read_many(manifests)
        covered = {path.rsplit("/", 1)[-1][4:] for path in read}
        self._parts = _list_parts(self.fs, self.root)  # one listing serves the seed too
        read.update(self._read_many([p for p in self._parts
                                     if p.rsplit("/", 1)[-1] not in covered]))
        done: set[str] = set()
        failed: set[str] = set()
        for d, f in read.values():
            done |= d
            failed |= f
        if self._declared is None and not self._seeded:
            self._seed_schema()  # now, before any request is in flight, not at the first probe
        return done if retry_errors else done | failed

    def _pin(self, schema: pa.Schema) -> None:
        self._schema = schema
        self._pinned = {f.name: f.type for f in schema if not _untyped(f.type)}  # untyped never pins

    def _seed_schema(self) -> None:
        """Dynamic mode, once per sink, before the first probe/flush: read the pin back from
        the newest existing part (footer only), so a resumed run keeps the on-disk types.
        Otherwise a run whose first row is one the previous run error-rowed (a float into an
        int column) would pin the other type and write a part unreadable beside the old ones."""
        self._seeded = True
        try:
            parts = self._parts if self._parts is not None else _list_parts(self.fs, self.root)
        except Exception as e:
            _log(f"could not list existing parts, starting unpinned: {e}")
            return
        for path in reversed(parts):  # newest first; a crash-truncated newest part is skipped
            schema = self._read_parquet(path, lambda f: pq.read_schema(f).remove_metadata())
            if schema is not None:
                self._pin(schema)
                return
            _log(f"could not read the schema of {path}, trying an older part")
        if parts:
            _log("no existing part has a readable schema, starting unpinned")

    def _conform_type(self, name: str, got: pa.DataType) -> pa.DataType | None:
        """The pinned type values of type `got` in column `name` must be cast to, or None when
        `got` already is it. `got` is acceptable when it is the same kind of thing (`_accepts`):
        int64 -> double, int64 -> int32 (the cast checks the value), list<int64> -> list<double>,
        struct<a> -> struct<a, b>, all-null -> anything. Everything else raises: a float into an
        int column, an int into a string column, a struct field the pin lacks."""
        pinned = self._pinned[name]
        if got == pinned:
            return None
        if not _storable(got):
            raise TypeError(f"{name}: {got} has a nested empty object, which parquet cannot store")
        if _has_untyped_leaf(pinned):  # a struct field only ever seen as None: the pin fills in
            widened = pa.unify_schemas([pa.schema([pa.field(name, pinned)]),
                                        pa.schema([pa.field(name, got)])]).field(0).type  # raises on a clash
            if widened != pinned:
                self._pinned[name] = widened
                self._schema = self._schema.set(self._schema.get_field_index(name), pa.field(name, widened))
            return widened
        key = (name, got)
        ok = self._accepted.get(key)
        if ok is None:
            ok = self._accepted[key] = _accepts(got, pinned)
        if not ok:
            raise TypeError(f"{name}: {got} does not widen into the pinned type {pinned}")
        return pinned  # the caller's safe cast refuses values that do not fit (0.5 -> int64, overflow)

    def _conform(self, table: pa.Table) -> pa.Table:
        """Cast the columns the table shares with the pin to their pinned types; raise
        TypeError/ValueError when a column does not widen into its pin (or the cast refuses,
        e.g. an int64 beyond 2**53 into a double column)."""
        for i, f in enumerate(table.schema):
            if f.name in self._pinned:
                to = self._conform_type(f.name, f.type)
                if to is not None:
                    table = table.set_column(i, pa.field(f.name, to), table.column(i).cast(to))
            elif not _storable(f.type):
                raise TypeError(f"{f.name}: {f.type} has a nested empty object: parquet cannot store it")
        return table

    def probe(self, record: dict) -> None:
        """Raise (TypeError/ValueError/OverflowError) if this record cannot serialize into
        THIS sink's schema — drain turns that into an error row before buffering, with the
        row's tokens still counted. flush() is the guarantee (a row it cannot store becomes
        an error row there too); probe is the early check that keeps stats exact."""
        if self._declared is None:
            if not self._seeded:
                self._seed_schema()
            arr = pa.array([record])  # one serialisation: a struct typed per field
            for i, f in enumerate(arr.type):  # equal types (the common case) cost a dict lookup
                if f.name in self._pinned and not pa.types.is_null(f.type):  # a null fits any pin
                    to = self._conform_type(f.name, f.type)
                    if to is not None:
                        arr.field(i).cast(to)
                elif not _storable(f.type):
                    raise TypeError(f"{f.name}: {f.type} has a nested empty object: parquet cannot store it")
            # and against the rows already buffered: two types for one not-yet-pinned column
            # inside one batch cannot both be stored — the newcomer is the row that does not fit
            got = pa.schema(list(arr.type))
            self._widen_buffer(got)
            self._probe_cache = (record, got)  # append() reuses the types without re-serialising
            return
        extra = record.keys() - set(self._declared.names)
        if extra:
            raise ValueError(f"fields outside the declared schema: {sorted(extra)}")
        pa.Table.from_pylist([{n: record.get(n) for n in self._declared.names}], schema=self._declared)

    def _widen_buffer(self, got: pa.Schema) -> pa.Schema:
        """The buffer's types widened by one more row's; raises (ArrowTypeError) on a clash."""
        if self._buf_schema is None:
            return got
        return pa.unify_schemas([self._buf_schema, got], promote_options="permissive")

    def append(self, record: dict) -> None:
        if self._declared is None:  # track the buffer's types so probe() can see the whole batch
            cached = self._probe_cache
            got = cached[1] if cached and cached[0] is record else pa.schema(list(pa.array([record]).type))
            try:
                self._buf_schema = self._widen_buffer(got)
            except (TypeError, ValueError):
                pass  # a direct append of a conflicting row: flush demotes it (the guarantee)
        self._buf.append(record)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"part-{int(time.time() * 1000)}-{self._seq:06d}-{uuid.uuid4().hex[:8]}.parquet"
        self._seq += 1
        if self._declared is not None:  # declared mode: exact frame — missing fields null, extras raise
            extra = {k for r in self._buf for k in r} - set(self._declared.names)
            if extra:
                raise ValueError(f"rows carry fields outside the declared schema: {sorted(extra)}")
            table = pa.Table.from_pylist([{n: r.get(n) for n in self._declared.names}
                                          for r in self._buf], schema=self._declared)
        else:
            if not self._seeded:
                self._seed_schema()
            try:  # the batch as a whole: infer over the union of keys, conform pinned columns
                table = self._conform(self._batch(self._buf))
            except (TypeError, ValueError, OverflowError):  # some row does not fit: find it
                table = self._salvage()
            # an all-null user column with no pinned type yet is unknowable — drop it rather
            # than guess a type a later real value (float, list) would conflict with on disk.
            # Once a type IS pinned, the column is kept and materialized below. Residual gap
            # (§8): a column whose first-ever appearance is all-null stays absent from that
            # part until its first typed value arrives.
            table = table.drop_columns([f.name for f in table.schema
                                        if _untyped(f.type) and f.name != "error"
                                        and f.name not in self._pinned])
            if pa.types.is_null(table.schema.field("error").type):  # error is required: pin to string
                i = table.schema.get_field_index("error")
                table = table.set_column(i, pa.field("error", pa.string()),
                                         table["error"].cast(pa.string()))
            # pin seen columns at first sight; pinned columns already conform, so the unify
            # only ever appends new columns — a pinned type never moves, since widening only
            # the NEW part would break multi-part reads
            self._pin(table.schema if self._schema is None else
                      pa.unify_schemas([self._schema, table.schema]))
            # every part carries the FULL pinned schema — absent or all-null columns become
            # typed all-null arrays (#12): pyarrow.dataset takes the FIRST fragment's schema
            # by default, so any part missing a pinned column loses that column for the
            # whole cross-part read whenever it sorts first
            for f in self._schema:
                if f.name not in table.schema.names:
                    table = table.append_column(f, pa.nulls(len(table), type=f.type))
            table = table.select(self._schema.names).cast(self._schema)
        try:
            with self.fs.open(f"{self.root}/{name}", "wb") as f:
                pq.write_table(table, f, compression="zstd")
        except BaseException:  # never leave a partial part behind (it would cost a scan on every resume)
            try:
                self.fs.rm(f"{self.root}/{name}")
            except Exception:
                pass
            raise
        try:  # manifest second: a crash in between leaves an uncovered part -> scanned
            self.fs.makedirs(f"{self.root}/_manifest", exist_ok=True)
            with self.fs.open(f"{self.root}/_manifest/ids-{name}", "wb") as f:
                pq.write_table(table.select(["id", "error"]), f, compression="zstd")
        except Exception as e:
            _log(f"manifest write failed (non-fatal, part covered by scan fallback): {e}")
        self.rows_written += len(self._buf)
        self._buf.clear()
        self._buf_schema = None

    @staticmethod
    def _batch(rows: list[dict]) -> pa.Table:
        keys = {k for r in rows for k in r}  # union: row 0 alone must not set the schema
        return pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])

    def _salvage(self) -> pa.Table:
        """Per-row fallback when the batch does not build or conform as a whole: each row is
        built alone and checked against the running schema (the pin, widened by the rows
        accepted before it in this batch — first typed value wins, as batch inference would);
        a row that fails is demoted in place to an error row. A type conflict costs one row,
        never the buffer, and flush never raises for it."""
        running = self._schema
        tables: list[pa.Table] = []
        bad = 0
        for i, rec in enumerate(self._buf):
            try:
                t = self._conform(pa.Table.from_pylist([rec]))
                # THE check for not-yet-pinned columns: the unify raises (ArrowTypeError, a
                # TypeError) when this row's type clashes with an earlier row's in the batch
                running = (t.schema if running is None else
                           pa.unify_schemas([running, t.schema], promote_options="permissive"))
            except (TypeError, ValueError, OverflowError) as e:
                self._buf[i] = _error_row(rec.get("id"), e)
                t = pa.Table.from_pylist([self._buf[i]])
                bad += 1
            tables.append(t)
        self.rows_demoted += bad
        _log(f"{bad} of {len(self._buf)} rows do not fit the pinned schema: written as error rows")
        return pa.concat_tables(tables, promote_options="permissive")

    def begin_run(self, shard: tuple[int, int]) -> None:
        """A new run on this shard: its telemetry lands in a new file (a crashed run's partial
        trajectory is kept; the marker alone cannot end a run that never returned)."""
        self._telemetry_names.pop(shard[0], None)

    def write_marker(self, shard: tuple[int, int]) -> None:
        self.fs.makedirs(f"{self.root}/completions", exist_ok=True)
        with self.fs.open(f"{self.root}/completions/shard-{shard[0]}.done", "wb") as f:
            f.write(b"done")
        self._telemetry_names.pop(shard[0], None)  # the marker ends a run: the next one gets a new file

    def write_stats(self, shard: tuple[int, int], stats_json: str) -> None:
        """Console-facing run summary beside the marker (CONTRACT §5): exact
        final counts + shard geometry, so a storage-only reader never has to
        approximate from telemetry ticks or open parquet."""
        import json as _json

        payload = _json.loads(stats_json)
        payload["rank"], payload["world"] = shard[0], shard[1]
        self.fs.makedirs(f"{self.root}/completions", exist_ok=True)
        with self.fs.open(f"{self.root}/completions/stats-{shard[0]}.json", "wb") as f:
            f.write(_json.dumps(payload).encode())

    def write_telemetry(self, shard: tuple[int, int], lines: list[str]) -> None:
        """One telemetry file per run and shard: the name is chosen on the first call and every
        later call rewrites the whole file from the full tick list, until write_marker() ends
        the run. Whole-file rewrite rather than append: object stores have no append, and a
        rewrite is one operation on every fsspec backend."""
        name = self._telemetry_names.get(shard[0])
        if name is None:
            name = f"telemetry-shard{shard[0]}-{int(time.time())}-{uuid.uuid4().hex[:6]}.jsonl"
            self._telemetry_names[shard[0]] = name
        with self.fs.open(f"{self.root}/{name}", "wb") as f:
            f.write("\n".join(lines).encode())


class FileSink:
    """One file per successful row: {outdir}/{id}{ext} <- record[key].
    Exact resume (filenames are the manifest, writes are idempotent); failed
    rows leave no record and simply retry on the next run."""

    def __init__(self, outdir: str, ext: str = ".txt", key: str = "text"):
        if not ext.startswith(".") or "/" in ext or ".." in ext:  # ext lands in filenames + globs
            raise ValueError(f"FileSink: unsafe ext {ext!r}")
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ext, self.key = ext, key
        self.rows_written = 0

    def existing_ids(self, retry_errors: bool = False) -> set[str]:
        return {p.name.removesuffix(self.ext) for p in self.dir.glob(f"*{self.ext}")}

    def append(self, record: dict) -> None:
        if record.get("error") is not None:
            return
        id_ = str(record["id"])
        if not id_ or id_ != Path(id_).name or "\\" in id_ or id_.startswith("."):  # unsafe filename
            raise ValueError(f"FileSink: id {id_!r} is not a safe filename")
        # dot-prefix + .tmp never match existing_ids; uuid keeps concurrent duplicate-id writers
        # from colliding on the temp name (last replace wins, both files complete)
        tmp = self.dir / f".{id_}.{uuid.uuid4().hex[:6]}{self.ext}.tmp"
        tmp.write_text(str(record[self.key]))
        os.replace(tmp, self.dir / f"{id_}{self.ext}")  # atomic: no truncated files on crash
        self.rows_written += 1

    def flush(self) -> None:
        pass  # write-through


def as_sink(output, flush_every: int = 10, schema: pa.Schema | None = None):
    """A string or Path is a ParquetSink (the CONTRACT); anything with
    existing_ids/append/flush passes through."""
    if isinstance(output, (str, Path)):
        return ParquetSink(str(output), flush_every, schema=schema)
    if schema is not None:  # r6: never silently ignore a declared schema
        raise ValueError("schema= applies to ParquetSink outputs; pass it to your sink directly")
    return output


async def drain(results: AsyncIterator, sink, shard: tuple[int, int] = (0, 1),
                stats=None):
    """Terminal stage: persist a Done stream; returns Stats with counts filled in.
    Completion markers are pump()'s job (written last, after stats/telemetry) —
    drain alone writes no marker."""
    from saturate import Stats  # local import: avoid cycle

    stats = stats if stats is not None else Stats()
    # r7: validation is the SINK'S contract — only a sink that supplies probe() gets it.
    # A generic duck-typed sink (FileSink str()s its values) must not inherit Arrow's rules.
    probe = getattr(sink, "probe", None)
    demoted0 = getattr(sink, "rows_demoted", 0)  # rows flush demotes were counted as processed here
    try:
        async for done in results:
            if done.error is None:
                # id/error are reserved columns and ALWAYS win over parse output —
                # a parse returning e.g. the OpenAI response id must not break resume
                rec = {**done.out, "id": done.id, "error": None}
                if probe is not None:
                    try:  # r5/r6 #6: validate against the sink's ACTUAL schema before buffering —
                        probe(rec)  # a bad value must become an error row, not a flush crash
                    except (TypeError, ValueError, OverflowError) as e:
                        stats.rows_failed += 1
                        stats.prompt_tokens += done.usage.get("prompt_tokens", 0)  # tokens were spent
                        stats.completion_tokens += done.usage.get("completion_tokens", 0)
                        sink.append(_error_row(done.id, e))
                        continue
                stats.rows_processed += 1
                stats.prompt_tokens += done.usage.get("prompt_tokens", 0)
                stats.completion_tokens += done.usage.get("completion_tokens", 0)
                sink.append(rec)
            else:
                stats.rows_failed += 1
                sink.append({"id": done.id, "error": done.error})
    finally:
        sink.flush()  # a fatal abort must still land the rows already paid for
        # reconcile: rows flush demoted landed as error rows, not successes. Capped at what this
        # drain counted — rows buffered before it started were never in these stats
        demoted = min(getattr(sink, "rows_demoted", 0) - demoted0, stats.rows_processed)
        stats.rows_processed -= demoted
        stats.rows_failed += demoted
    return stats


def read_output(out_uri: str) -> Iterator[tuple[str, dict]]:
    """Read a saturate output dir back as an (id, row) source, applying the
    CONTRACT §4.2 reader rule: the error-IS-NULL record wins; error-only ids
    are skipped. Materializes the id->record map (fine to ~1M rows; document)."""
    import fsspec

    fs, root = fsspec.url_to_fs(out_uri)
    best: dict[str, dict] = {}
    for path in _list_parts(fs, root):
        try:
            with fs.open(path, "rb") as f:
                rows = pq.read_table(f).to_pylist()
        except Exception as e:
            _log(f"read_output: skipping unreadable {path}: {e}")
            continue
        for rec in rows:
            if rec.get("error") is None or rec["id"] not in best:
                best[rec["id"]] = rec
    for id_, rec in best.items():
        if rec.get("error") is None:
            yield id_, {k: v for k, v in rec.items() if k not in ("id", "error")}
