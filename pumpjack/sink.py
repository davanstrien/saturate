"""Sinks: where the Done stream lands.

The resume contract is one invariant (CONTRACT §3): an id returned by
existing_ids() implies its record is durable; an id not returned implies
re-processing it is safe. Any object satisfying `existing_ids/append/flush`
(duck-typed) is a sink; markers and telemetry hooks are optional.

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


class ParquetSink:
    def __init__(self, out_uri: str, flush_every: int = 10, schema: pa.Schema | None = None):
        import fsspec

        self.fs, self.root = fsspec.url_to_fs(out_uri)
        self.flush_every = flush_every
        self._buf: list[dict] = []
        # declared mode (r5): an immutable schema every part is cast to — the only fully
        # schema-stable option for arbitrary parse output. Must carry id + error.
        if schema is not None and not {"id", "error"} <= set(schema.names):
            raise ValueError("declared schema must include the id and error columns")
        self._declared = schema
        self._schema: pa.Schema | None = schema  # dynamic mode: pinned/unified across flushes
        self.rows_written = 0
        try:
            self.fs.makedirs(self.root, exist_ok=True)
        except Exception:
            pass  # HfFileSystem: directories are implicit; repo must already exist

    def _read_id_error(self, path: str, done: set, failed: set) -> bool:
        try:
            with self.fs.open(path, "rb") as f:
                t = pq.read_table(f, columns=["id", "error"])
            for id_, err in zip(t["id"].to_pylist(), t["error"].to_pylist(), strict=True):
                (failed if err else done).add(id_)
            return True
        except Exception as e:  # truncated file from a crash: skip, re-pay
            _log(f"skipping unreadable {path}: {e}")
            return False

    def existing_ids(self, retry_errors: bool = False) -> set[str]:
        """Manifest-first, exact: any part lacking its manifest sidecar is
        scanned individually (kill-safe). retry_errors=True excludes ids whose
        only record is an error."""
        done: set[str] = set()
        failed: set[str] = set()
        covered: set[str] = set()
        for path in sorted(self.fs.glob(f"{self.root}/_manifest/ids-*.parquet")):
            if self._read_id_error(path, done, failed):
                covered.add(path.rsplit("/", 1)[-1][4:])
        for path in sorted(self.fs.glob(f"{self.root}/part-*.parquet")):
            if path.rsplit("/", 1)[-1] not in covered:
                self._read_id_error(path, done, failed)
        return done if retry_errors else done | failed

    def append(self, record: dict) -> None:
        self._buf.append(record)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"part-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.parquet"
        if self._declared is not None:  # declared mode: exact frame — missing fields null, extras raise
            extra = {k for r in self._buf for k in r} - set(self._declared.names)
            if extra:
                raise ValueError(f"rows carry fields outside the declared schema: {sorted(extra)}")
            table = pa.Table.from_pylist([{n: r.get(n) for n in self._declared.names}
                                          for r in self._buf], schema=self._declared)
        else:
            keys = {k for r in self._buf for k in r}  # union: row 0 alone must not set the schema
            table = pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in self._buf])
            # dynamic mode writes SPARSE parts (§1/§8): an all-null user column carries no
            # information and its type is unknowable yet — drop it rather than guess a type
            # that a later real value (float, list) would then conflict with on disk
            table = table.drop_columns([f.name for f in table.schema
                                        if pa.types.is_null(f.type) and f.name != "error"])
            if pa.types.is_null(table.schema.field("error").type):  # error is required: pin to string
                i = table.schema.get_field_index("error")
                table = table.set_column(i, pa.field("error", pa.string()),
                                         table["error"].cast(pa.string()))
            # pin seen columns at first sight, cast later parts (mid-run type change raises — §8)
            self._schema = (table.schema if self._schema is None else
                            pa.unify_schemas([self._schema, table.schema], promote_options="permissive"))
            table = table.cast(pa.schema([self._schema.field(n) for n in table.schema.names]))
        with self.fs.open(f"{self.root}/{name}", "wb") as f:
            pq.write_table(table, f, compression="zstd")
        try:  # manifest second: a crash in between leaves an uncovered part -> scanned
            self.fs.makedirs(f"{self.root}/_manifest", exist_ok=True)
            with self.fs.open(f"{self.root}/_manifest/ids-{name}", "wb") as f:
                pq.write_table(table.select(["id", "error"]), f, compression="zstd")
        except Exception as e:
            _log(f"manifest write failed (non-fatal, part covered by scan fallback): {e}")
        self.rows_written += len(self._buf)
        self._buf.clear()

    def write_marker(self, shard: tuple[int, int]) -> None:
        self.fs.makedirs(f"{self.root}/completions", exist_ok=True)
        with self.fs.open(f"{self.root}/completions/shard-{shard[0]}.done", "wb") as f:
            f.write(b"done")

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
        name = f"telemetry-shard{shard[0]}-{int(time.time())}-{uuid.uuid4().hex[:6]}.jsonl"
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
        if not id_ or id_ != Path(id_).name or id_.startswith("."):  # filenames; dotfiles hide from resume
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
    return ParquetSink(str(output), flush_every, schema=schema) if isinstance(output, (str, Path)) else output


async def drain(results: AsyncIterator, sink, shard: tuple[int, int] = (0, 1),
                stats=None):
    """Terminal stage: persist a Done stream; returns Stats with counts filled in.
    Completion markers are pump()'s job (written last, after stats/telemetry) —
    drain alone writes no marker."""
    from pumpjack import Stats  # local import: avoid cycle

    stats = stats if stats is not None else Stats()
    try:
        async for done in results:
            if done.error is None:
                # id/error are reserved columns and ALWAYS win over parse output —
                # a parse returning e.g. the OpenAI response id must not break resume
                rec = {**done.out, "id": done.id, "error": None}
                try:
                    pa.array([rec])  # r5 #6: probe values BEFORE buffering — a non-serializable
                except (TypeError, ValueError) as e:  # value must be an error row, not a flush crash
                    stats.rows_failed += 1
                    sink.append({"id": done.id, "error": f"parse output not arrow-serializable: {e}"})
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
    return stats


def read_output(out_uri: str) -> Iterator[tuple[str, dict]]:
    """Read a pumpjack output dir back as an (id, row) source, applying the
    CONTRACT §4.2 reader rule: the error-IS-NULL record wins; error-only ids
    are skipped. Materializes the id->record map (fine to ~1M rows; document)."""
    import fsspec

    fs, root = fsspec.url_to_fs(out_uri)
    best: dict[str, dict] = {}
    for path in sorted(fs.glob(f"{root}/part-*.parquet")):
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
