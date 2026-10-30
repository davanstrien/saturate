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

import logging
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    cast_columns,
    check_storable,
    fit,
    is_nested,
    is_untyped,
    outside_declared,
    without_untyped,
    without_untyped_schema,
)

log = logging.getLogger(__name__)


# Everything saturate writes beside the data parts lives under one directory whose name starts
# with "__": datasets' loader and pyarrow's dataset reader both skip such paths, so
# load_dataset(out) and pq.read_table(out) see the parts alone (CONTRACT §1).
SIDECAR_DIR = "__saturate"
MANIFEST_DIR = f"{SIDECAR_DIR}/manifest"
OLD_MANIFEST_DIR = "_manifest"  # outputs written before 0.3: read on resume, never written
COMPLETIONS_DIR = f"{SIDECAR_DIR}/completions"
TELEMETRY_DIR = f"{SIDECAR_DIR}/telemetry"


LOCAL_FLUSH_INTERVAL_S = 10.0  # a local output's oldest buffered row is written within this
REMOTE_FLUSH_INTERVAL_S = 60.0  # a remote output's: each write there is a slow commit or upload


def _list_parts(fs, root: str) -> list[str]:
    """The data parts of an output dir, in write order (the CONTRACT part-name pattern)."""
    return sorted(fs.glob(f"{root}/part-*.parquet"))


# what a value that cannot be stored raises: Python's conversion errors, and Arrow's own (an
# ArrowNotImplementedError, e.g. for a uuid.UUID, is a RuntimeError, not a TypeError)
UNSTORABLE = (TypeError, ValueError, OverflowError, pa.ArrowException)


def error_row(id_, exc: BaseException) -> dict:
    """The record a row that cannot be stored leaves behind (CONTRACT §4)."""
    return {"id": str(id_), "error": f"parse output not storable: {exc}"}


class ParquetSink:
    def __init__(
        self,
        out_uri: str,
        flush_every: int | None = None,
        schema: pa.Schema | None = None,
        read_workers: int = 16,
        flush_interval_s: float | None = None,
        create: bool = True,
    ):
        """`create=False` opens the output for reading only: no repo, bucket or directory is
        created, and a missing output reads as empty (resume checks, existing_ids())."""
        import fsspec

        self.fs, self.root = fsspec.url_to_fs(out_uri)
        proto = getattr(self.fs, "protocol", ())  # fsspec: a str or a tuple of aliases — match exactly
        protos = set(proto) if isinstance(proto, (tuple, list)) else {proto}
        local = "file" in protos
        self._local = local
        # a flush writes two objects (a part and its manifest): ~1 s per write on hf://, and
        # 10-row local flushes capped a fast run (embeddings) at ~1,100 rows/s. Outputs flush
        # by size or age: 1000 rows, or the oldest buffered row 10 s old locally (60 s remotely,
        # where every write is slow), counted from the oldest buffered row whether or not rows
        # keep arriving (the writer flushes a quiet buffer on the same clock): a crash loses at
        # most that many seconds of rows, which a resume re-requests.
        # Rows are counted, not bytes: if parse returns large payloads (images), pass a smaller
        # flush_every.
        self.flush_every = flush_every if flush_every is not None else 1000
        self.flush_interval_s = (
            flush_interval_s
            if flush_interval_s is not None
            else (LOCAL_FLUSH_INTERVAL_S if local else REMOTE_FLUSH_INTERVAL_S)
        )
        self.read_workers = read_workers  # resume read concurrency; 1 = sequential
        self._buf: list[dict] = []
        self._buf_since: float | None = None  # monotonic time of the oldest buffered row
        # declared mode: an immutable schema every part is cast to — the only fully
        # schema-stable option for arbitrary parse output. Must carry contract-typed id + error.
        if schema is not None:
            if not {"id", "error"} <= set(schema.names):
                raise ValueError("declared schema must include the id and error columns")
            if (
                schema.field("id").type != pa.string()
                or schema.field("error").type != pa.string()
                or not schema.field("error").nullable
            ):
                raise ValueError("declared schema: id and error must be string, error nullable")
            bad = [f.name for f in schema if f.name != "id" and not f.nullable]
            if bad:  # error rows write user fields as null: a non-nullable field would crash the flush
                raise ValueError(f"declared schema: fields must be nullable (error rows are sparse): {bad}")
        self._declared = schema
        # dynamic mode: pinned at first sight, extended by new columns and new nested fields
        self._schema: pa.Schema | None = None
        self._pinned: dict[str, pa.DataType] = {}  # name -> pinned type (empty until pinned)
        self._fits: dict[tuple[str, pa.DataType], pa.DataType | None] = {}  # (column, value type) -> fit()
        if schema is not None:
            self._pin(schema)
        self._seeded = schema is not None  # dynamic mode reads its pin back from existing parts
        self.rows_demoted = 0  # rows flush turned into error rows (type conflicts, cumulative)
        self._seq = 0  # per-sink flush counter: sorted(part names) == write order on ms collisions
        self._telemetry_names: dict[int, str] = {}  # shard rank -> this run's telemetry file
        self._buf_schema: pa.Schema | None = None  # types of the rows buffered so far (dynamic mode)
        self._parts: list[str] | None = None  # part listing from existing_ids(), reused by the seed
        self._probe_cache: tuple[dict, pa.Schema] | None = None  # the record probe() last typed
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
        if create and "hf" in protos and self.root.startswith(("datasets/", "buckets/")):
            parts = self.root.split("/")
            if len(parts) >= 3:
                repo = f"{parts[1]}/{parts[2]}"
                try:
                    from huggingface_hub import HfApi

                    if parts[0] == "datasets":
                        HfApi().create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
                    else:
                        HfApi().create_bucket(repo, private=True, exist_ok=True)
                    # fsspec shares one filesystem instance per process: a read-only check that
                    # found no repo left "missing" in its cache, and every write would fail on it
                    self.fs.invalidate_cache()
                except Exception as e:
                    log.warning(f"could not ensure output repo {repo}: {e}")
        if create:
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
                log.warning(f"skipping unreadable {path}: {e}")
                return None
            except Exception as e:
                if i + 1 < attempts:
                    time.sleep(0.5 * 2**i)
                else:
                    log.warning(f"skipping unreadable {path}: {e}")
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
        try:
            manifests = sorted(
                self.fs.glob(f"{self.root}/{MANIFEST_DIR}/ids-*.parquet")
                + self.fs.glob(f"{self.root}/{OLD_MANIFEST_DIR}/ids-*.parquet")
            )
        except FileNotFoundError:  # an output that does not exist yet (a repo never created)
            self.error_only_ids = set()
            return set()
        if len(manifests) > 1000:
            log.info(
                f"resume: reading {len(manifests)} manifest files "
                "(fewer, larger parts: a larger flush_every, or flush_interval_s on ParquetSink)"
            )
        read = self._read_many(manifests)
        covered = {path.rsplit("/", 1)[-1][4:] for path in read}
        self._parts = _list_parts(self.fs, self.root)  # one listing serves the seed too
        read.update(self._read_many([p for p in self._parts if p.rsplit("/", 1)[-1] not in covered]))
        done: set[str] = set()
        failed: set[str] = set()
        for d, f in read.values():
            done |= d
            failed |= f
        if self._declared is None and not self._seeded:
            self._seed_schema()  # now, before any request is in flight, not at the first probe
        self.error_only_ids = failed - done  # skipped unless retry_errors: resume says how many
        return done if retry_errors else done | failed

    def _pin(self, schema: pa.Schema) -> pa.Schema:
        """Pins the typed part of `schema` (`without_untyped`): an untyped column or nested field
        never pins, so its first typed value is free to set it."""
        if self._declared is None:
            schema = without_untyped_schema(schema)
        self._schema = schema
        self._pinned = {f.name: f.type for f in schema if not is_untyped(f.type)}
        self._fits = {}  # the cached verdicts were for the old pin
        return schema

    def _seed_schema(self) -> None:
        """Dynamic mode, once per sink, before the first probe/flush: read the pin back from
        the newest existing part (footer only), so a resumed run keeps the on-disk types.
        Otherwise a run whose first row is one the previous run error-rowed (a float into an
        int column) would pin the other type and write a part unreadable beside the old ones."""
        self._seeded = True
        try:
            parts = self._parts if self._parts is not None else _list_parts(self.fs, self.root)
        except Exception as e:
            log.warning(f"could not list existing parts, starting unpinned: {e}")
            return
        for path in reversed(parts):  # newest first; a crash-truncated newest part is skipped
            schema = self._read_parquet(path, lambda f: pq.read_schema(f).remove_metadata())
            if schema is not None:
                self._pin(schema)
                return
            log.warning(f"could not read the schema of {path}, trying an older part")
        if parts:
            log.warning("no existing part has a readable schema, starting unpinned")

    def _conform_type(self, name: str, value_type: pa.DataType) -> pa.DataType | None:
        """The type that values of type `value_type` in column `name` are cast to before they are
        stored; None when `value_type` already is the pinned type. The rules are `fit`'s: int64 ->
        double, int64 -> int32 (the cast checks the value), list<int64> -> list<double>,
        struct<a> -> struct<a, b>, all-null -> anything, and struct<a, c> -> struct<a, b, c>
        when the pin is struct<a, b> (a struct field the pin lacks is appended). Everything
        else raises, at any depth: a float into an int column, an int into a string column.
        This only reads the pin: an extension joins the pin when a flush writes a part that
        carries it."""
        pinned_type = self._pinned[name]
        if value_type == pinned_type:
            return None
        check_storable(name, value_type)
        key = (name, value_type)
        if key not in self._fits:
            self._fits[key] = fit(value_type, pinned_type)
        stored_type = self._fits[key]
        if stored_type is None:
            hint = (
                (
                    "; a nested field's type is fixed by its first typed value: declare it with "
                    "schema=, or store this object as a JSON string"
                )
                if is_nested(pinned_type)
                else ""
            )
            raise TypeError(f"{name}: {value_type} does not fit the pinned type {pinned_type}{hint}")
        return stored_type  # the caller's safe cast refuses values that do not fit (0.5 -> int64, overflow)

    def _conform(self, table: pa.Table) -> pa.Table:
        """Cast the columns the table shares with the pin to the types they are stored as;
        raise TypeError/ValueError when a column does not fit its pin (or the cast refuses,
        e.g. an int64 beyond 2**53 into a double column). The pin is not changed."""
        for i, field in enumerate(table.schema):
            if field.name in self._pinned:
                stored_type = self._conform_type(field.name, field.type)
                if stored_type is not None:
                    stored = table.column(i).cast(stored_type)
                    table = table.set_column(i, pa.field(field.name, stored_type), stored)
            else:
                check_storable(field.name, field.type)
        return table

    def probe(self, record: dict) -> None:
        """Raise (TypeError/ValueError/OverflowError) if this record cannot serialize into
        THIS sink's schema — drain turns that into an error row before buffering, with the
        row's tokens still counted. flush() is the guarantee (a row it cannot store becomes
        an error row there too); probe is the early check that keeps stats exact."""
        if self._declared is None:
            if not self._seeded:
                self._seed_schema()
            stored = self._stored_types(pa.array([record]))  # one serialisation
            # and against the rows already buffered: two types for one not-yet-pinned column
            # (or nested field) inside one batch cannot both be stored — the newcomer is the row
            # that does not fit
            self._widen_buffer(stored)
            self._probe_cache = (record, stored)  # append() reuses the types without re-serialising
            return
        outside = outside_declared(record, self._declared)
        if outside:
            raise ValueError(f"fields outside the declared schema: {outside}")
        pa.Table.from_pylist([{n: record.get(n) for n in self._declared.names}], schema=self._declared)

    def _stored_types(self, row: pa.StructArray) -> pa.Schema:
        """The types one row (`row`, a struct array of length 1) is stored as, as flush would
        conform it: a pinned column takes its conformed type (`_conform_type`), so a nested
        field with no type yet is left out exactly as flush leaves it out. Raises
        (TypeError/ValueError) when the row does not fit the pin. Only reads the pin, so a
        rejected row leaves no trace on it."""
        fields = []
        for i, field in enumerate(row.type):  # equal types (the common case) cost a dict lookup
            if field.name in self._pinned and not pa.types.is_null(field.type):  # a null fits any pin
                stored_type = self._conform_type(field.name, field.type)
                if stored_type is not None:
                    row.field(i).cast(stored_type)  # the values must fit too (0.5 -> int64, overflow)
                    fields.append(pa.field(field.name, stored_type))
                    continue
            else:
                check_storable(field.name, field.type)
            typed = without_untyped(field.type)
            if typed is not None:  # what has no type yet is never written, so it cannot clash
                fields.append(pa.field(field.name, typed))
        return pa.schema(fields)

    def _widen_buffer(self, stored: pa.Schema) -> pa.Schema:
        """The buffer's types widened by one more row's; raises (ArrowTypeError) on a clash."""
        if self._buf_schema is None:
            return stored
        return pa.unify_schemas([self._buf_schema, stored], promote_options="permissive")

    def append(self, record: dict) -> None:
        if self._declared is None:  # track the buffer's types so probe() can see the whole batch
            cached = self._probe_cache
            try:
                stored = (
                    cached[1] if cached and cached[0] is record else self._stored_types(pa.array([record]))
                )
                self._buf_schema = self._widen_buffer(stored)
            except UNSTORABLE:
                pass  # a direct append of a row that does not fit: flush demotes it (the guarantee)
        if not self._buf:
            self._buf_since = time.monotonic()
        self._buf.append(record)
        if self._flush_due():
            self.flush()

    def _flush_due(self) -> bool:
        """Enough rows buffered, or the oldest buffered row older than the flush interval."""
        if len(self._buf) >= self.flush_every:
            return True
        if self.flush_interval_s is None or self._buf_since is None:
            return False
        return time.monotonic() - self._buf_since >= self.flush_interval_s

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"part-{int(time.time() * 1000)}-{self._seq:06d}-{uuid.uuid4().hex[:8]}.parquet"
        self._seq += 1
        if self._declared is not None:  # declared mode: exact frame — missing fields null, extras raise
            extra = set()
            for r in self._buf:
                extra.update(outside_declared(r, self._declared))
            if extra:
                raise ValueError(f"rows carry fields outside the declared schema: {sorted(extra)}")
            table = pa.Table.from_pylist(
                [{n: r.get(n) for n in self._declared.names} for r in self._buf], schema=self._declared
            )
        else:
            if not self._seeded:
                self._seed_schema()
            try:  # the batch as a whole: infer over the union of keys, conform pinned columns
                table = self._conform(self._batch(self._buf))
            except UNSTORABLE:  # some row does not fit: find it
                table = self._salvage()
            table = self._drop_untyped(table)
            if pa.types.is_null(table.schema.field("error").type):  # error is required: pin to string
                i = table.schema.get_field_index("error")
                table = table.set_column(i, pa.field("error", pa.string()), table["error"].cast(pa.string()))
            # pin seen columns at first sight; pinned columns already conform, so the unify
            # only ever appends new columns and new struct fields — a pinned type never moves,
            # since retyping only the NEW part would break multi-part reads
            schema = self._pin(
                table.schema if self._schema is None else pa.unify_schemas([self._schema, table.schema])
            )
            # every part carries the FULL pinned schema — absent or all-null columns become
            # typed all-null arrays (#12): pyarrow.dataset takes the FIRST fragment's schema
            # by default, so any part missing a pinned column loses that column for the
            # whole cross-part read whenever it sorts first
            for f in schema:
                if f.name not in table.schema.names:
                    table = table.append_column(f, pa.nulls(len(table), type=f.type))
            table = cast_columns(table.select(schema.names), schema).cast(schema)
        self._write(f"{self.root}/{name}", table)
        try:  # manifest second: a crash in between leaves an uncovered part -> scanned
            self.fs.makedirs(f"{self.root}/{MANIFEST_DIR}", exist_ok=True)
            self._write(f"{self.root}/{MANIFEST_DIR}/ids-{name}", table.select(["id", "error"]))
        except Exception as e:
            log.warning(f"manifest write failed (non-fatal, part covered by scan fallback): {e}")
        self.rows_written += len(self._buf)
        self._buf.clear()
        self._buf_schema = None
        self._buf_since = None

    def _write(self, path: str, table: pa.Table, attempts: int = 4) -> None:
        """Write one parquet file. A remote write can fail transiently (a 503, a reset); failing
        the run for it re-pays every request in flight, so I/O errors are retried under the
        SAME name — a retry of a write that did land overwrites it, never duplicates it."""
        for i in range(attempts):
            try:
                if self._local:  # a kill mid-write must not leave a truncated part under its real
                    tmp = f"{path.rsplit('/', 1)[0]}/.{path.rsplit('/', 1)[1]}.{uuid.uuid4().hex[:8]}.tmp"
                    try:  # name (dot-prefixed temp names match no reader's glob)
                        with self.fs.open(tmp, "wb") as f:
                            pq.write_table(table, f, compression="zstd")
                        self.fs.mv(tmp, path)
                    finally:
                        if self.fs.exists(tmp):
                            self.fs.rm(tmp)
                else:  # remote object stores publish a file only when it is closed
                    with self.fs.open(path, "wb") as f:
                        pq.write_table(table, f, compression="zstd")
                return
            except BaseException as e:  # never leave a partial file behind (a scan on every resume)
                try:
                    self.fs.rm(path)
                except Exception:
                    pass
                if not isinstance(e, Exception) or i + 1 == attempts:
                    raise
                log.warning(f"write failed, retrying ({i + 1}/{attempts - 1}): {path}: {e}")
                time.sleep(2**i)

    def _drop_untyped(self, table: pa.Table) -> pa.Table:
        """`table` with its untyped columns and untyped nested fields removed: a user column
        with no pinned type whose values are all null (or empty lists) is dropped, and struct
        fields only ever seen as None are cast out of their column (`without_untyped`).
        Guessing a type there would conflict with a later real value (a float, a list) on
        disk, and writing the null type makes the parts unreadable together once a later part
        types it. Residual gap (§8): such a column or
        field is absent from the parts written before its first typed value arrives."""
        for field in list(table.schema):
            if field.name == "error":
                continue
            typed = without_untyped(field.type)
            if typed is None:
                if field.name not in self._pinned:  # a pinned column is materialized by the caller
                    table = table.drop_columns([field.name])
            elif typed != field.type:
                i = table.schema.get_field_index(field.name)
                table = table.set_column(i, pa.field(field.name, typed), table.column(i).cast(typed))
        return table

    @staticmethod
    def _batch(rows: list[dict]) -> pa.Table:
        keys = {k for r in rows for k in r}  # union: row 0 alone must not set the schema
        return pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])

    def _salvage(self) -> pa.Table:
        """Per-row fallback when the batch does not build or conform as a whole. Each row is
        built alone and conformed to the pin; a row that does not conform, or whose types clash
        with an earlier row's in this batch (first typed value wins, as batch inference would),
        is demoted in place to an error row. The rows kept are then cast to the types they
        share; a row whose VALUES do not fit those (2**60 into a double column that a later
        row's 0.5 made) is demoted too, and the shared types are worked out again without it,
        so a demoted row never shapes the part, nor the pin taken from it. A type conflict
        costs one row, never the buffer, and flush never raises for it."""
        running = self._schema
        conformed: list[pa.Table | None] = []
        bad = 0
        for i, rec in enumerate(self._buf):
            try:
                # without what has no type yet: it is never written, so it cannot clash
                t = self._drop_untyped(self._conform(pa.Table.from_pylist([rec])))
                # THE check for not-yet-pinned columns: the unify raises (ArrowTypeError, a
                # TypeError) when this row's type clashes with an earlier row's in the batch
                running = (
                    t.schema
                    if running is None
                    else pa.unify_schemas([running, t.schema], promote_options="permissive")
                )
            except UNSTORABLE as e:
                self._buf[i] = error_row(rec.get("id"), e)
                t = None
                bad += 1
            conformed.append(t)
        tables = self._cast_to_shared_types(conformed)
        bad += sum(1 for c, t in zip(conformed, tables, strict=True) if c is not None and t is None)
        tables = [pa.Table.from_pylist([self._buf[i]]) if t is None else t for i, t in enumerate(tables)]
        self.rows_demoted += bad
        log.warning(f"{bad} of {len(self._buf)} rows do not fit the pinned schema: written as error rows")
        return pa.concat_tables(tables, promote_options="permissive")

    def _cast_to_shared_types(self, conformed: list[pa.Table | None]) -> list[pa.Table | None]:
        """Each kept row cast to the types the kept rows share. The first row whose values do
        not fit is demoted in place (None in the result, an error row in the buffer), and the
        shared types are worked out again from the rows still kept before any other row is
        checked: a type only the demoted row brought (a double that 2**60 does not fit) must
        not cost another row. Repeats until every kept row fits."""
        kept = list(conformed)
        while True:
            schemas = [t.schema for t in kept if t is not None]
            if not schemas:
                return kept
            shared = pa.unify_schemas(schemas, promote_options="permissive")
            cast: list[pa.Table | None] = []
            demoted = False
            for i, t in enumerate(kept):
                if t is None:
                    cast.append(None)
                    continue
                try:
                    cast.append(cast_columns(t, shared))
                except UNSTORABLE as e:
                    self._buf[i] = error_row(self._buf[i].get("id"), e)
                    kept[i] = None
                    demoted = True
                    break  # the shared types may change without this row: work them out again
            if not demoted:
                return cast

    def begin_run(self, shard: tuple[int, int]) -> None:
        """A new run on this shard: its telemetry lands in a new file (a crashed run's partial
        trajectory is kept; the marker alone cannot end a run that never returned)."""
        self._telemetry_names.pop(shard[0], None)

    def write_marker(self, shard: tuple[int, int]) -> None:
        self.fs.makedirs(f"{self.root}/{COMPLETIONS_DIR}", exist_ok=True)
        with self.fs.open(f"{self.root}/{COMPLETIONS_DIR}/shard-{shard[0]}.done", "wb") as f:
            f.write(b"done")
        self._telemetry_names.pop(shard[0], None)  # the marker ends a run: the next one gets a new file

    def write_stats(self, shard: tuple[int, int], stats_json: str) -> None:
        """Console-facing run summary beside the marker (CONTRACT §5): exact
        final counts + shard geometry, so a storage-only reader never has to
        approximate from telemetry ticks or open parquet."""
        import json as _json

        payload = _json.loads(stats_json)
        payload["rank"], payload["world"] = shard[0], shard[1]
        self.fs.makedirs(f"{self.root}/{COMPLETIONS_DIR}", exist_ok=True)
        with self.fs.open(f"{self.root}/{COMPLETIONS_DIR}/stats-{shard[0]}.json", "wb") as f:
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
            self.fs.makedirs(f"{self.root}/{TELEMETRY_DIR}", exist_ok=True)
        with self.fs.open(f"{self.root}/{TELEMETRY_DIR}/{name}", "wb") as f:
            f.write("\n".join(lines).encode())


FIRST_ROWS = 32  # FileSink: this many rejected records before any accepted one stop the run
LOGGED_REJECTIONS = 10


class FileSink:
    """One file per successful row: {outdir}/{id}{ext} <- record[key].
    Exact resume (filenames are the manifest, writes are idempotent); failed
    rows leave no record and simply retry on the next run."""

    def __init__(self, outdir: str, ext: str = ".txt", key: str = "text"):
        if not ext.startswith(".") or "/" in ext or ".." in ext:  # ext lands in filenames + globs
            raise ValueError(f"FileSink: unsafe ext {ext!r}")
        if "://" in str(outdir):  # Path() would quietly make a local "hf:/..." directory
            raise ValueError(
                f"FileSink writes to a local directory, not {outdir!r}: pass the URI "
                "as pump(output=...) for a ParquetSink, or write locally and upload"
            )
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ext, self.key = ext, key
        self.rows_written = 0
        self._accepted = False  # a record has passed probe: rejections are now the data's
        self._rejected = 0

    def begin_run(self, shard: tuple[int, int]) -> None:
        """A new pump() run: the first-rows check starts again (a sink can serve several runs)."""
        self._accepted = False
        self._rejected = 0

    def existing_ids(self, retry_errors: bool = False) -> set[str]:
        return {p.name.removesuffix(self.ext) for p in self.dir.glob(f"*{self.ext}")}

    def probe(self, record: dict) -> None:
        """Raise ValueError if this record cannot become a file: its id is not a safe filename,
        or its `key` field is missing or None. drain turns that into an error row: no file is
        written, so the row runs again on the next run. The first rejections are logged. When
        the first FIRST_ROWS records are all rejected, the sink is misconfigured (a wrong
        `key=`, ids that are paths): probe raises RuntimeError, which stops the run."""
        reason = self._rejection(record)
        if reason is None:
            self._accepted = True
            return
        self._rejected += 1
        if self._rejected <= LOGGED_REJECTIONS:
            log.warning(f"FileSink: row {record['id']!r} not written: {reason}; it runs again next run")
        if not self._accepted and self._rejected >= FIRST_ROWS:
            raise RuntimeError(
                f"FileSink: the first {FIRST_ROWS} rows could not be written "
                f"(the last: {reason}); check FileSink(key=...) and the row ids"
            )
        raise ValueError(f"FileSink: {reason}")

    def _rejection(self, record: dict) -> str | None:
        id_ = str(record["id"])
        if not id_ or id_ != Path(id_).name or "\\" in id_ or id_.startswith("."):
            return f"id {id_!r} is not a safe filename"
        if record.get(self.key) is None:
            return f"parse returned no {self.key!r} value (FileSink(key={self.key!r}))"
        return None

    def append(self, record: dict) -> None:
        if record.get("error") is not None:
            return
        self.probe(record)  # a sink called directly, without drain, checks here
        id_ = str(record["id"])
        # dot-prefix + .tmp never match existing_ids; uuid keeps concurrent duplicate-id writers
        # from colliding on the temp name (last replace wins, both files complete)
        tmp = self.dir / f".{id_}.{uuid.uuid4().hex[:6]}{self.ext}.tmp"
        tmp.write_text(str(record[self.key]))
        os.replace(tmp, self.dir / f"{id_}{self.ext}")  # atomic: no truncated files on crash
        self.rows_written += 1

    def flush(self) -> None:
        pass  # write-through


def as_sink(output, flush_every: int | None = None, schema: pa.Schema | None = None):
    """A string or Path is a ParquetSink (the CONTRACT); anything with
    existing_ids/append/flush passes through."""
    if isinstance(output, (str, Path)):
        return ParquetSink(str(output), flush_every, schema=schema)
    if schema is not None:  # a declared schema is never silently ignored
        raise ValueError("schema= applies to ParquetSink outputs; pass it to your sink directly")
    return output


def __getattr__(name: str):
    """drain() lives in saturate.writer, next to the writer thread; it is importable from here too."""
    if name == "drain":
        from saturate.writer import drain

        return drain
    raise AttributeError(f"module 'saturate.sink' has no attribute {name!r}")


def read_output(out_uri: str) -> Iterator[tuple[str, dict]]:
    """Read a saturate output dir back as an (id, row) source, applying the
    CONTRACT §4 reader rule: the error-IS-NULL record wins; error-only ids
    are skipped. Holds every successful id -> record in memory before it yields
    (comfortable to about 1M rows)."""
    import fsspec

    fs, root = fsspec.url_to_fs(out_uri)
    best: dict[str, dict] = {}
    for path in _list_parts(fs, root):
        try:
            with fs.open(path, "rb") as f:
                rows = pq.read_table(f).to_pylist()
        except Exception as e:
            log.warning(f"read_output: skipping unreadable {path}: {e}")
            continue
        for rec in rows:
            if rec.get("error") is None:
                best[rec["id"]] = rec
    for id_, rec in best.items():
        yield id_, {k: v for k, v in rec.items() if k not in ("id", "error")}
