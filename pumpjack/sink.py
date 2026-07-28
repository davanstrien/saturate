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
    def __init__(self, out_uri: str, flush_every: int = 10):
        import fsspec

        self.fs, self.root = fsspec.url_to_fs(out_uri)
        self.flush_every = flush_every
        self._buf: list[dict] = []
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
        table = pa.Table.from_pylist(self._buf)
        # pin `error` to string: an all-null column otherwise infers as null type,
        # which breaks cross-part schema unions for external readers (viewer,
        # pq.read_table over the dir) the moment another part has real errors
        i = table.schema.get_field_index("error")
        table = table.set_column(i, pa.field("error", pa.string()), table["error"].cast(pa.string()))
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

    def write_telemetry(self, shard: tuple[int, int], lines: list[str]) -> None:
        name = f"telemetry-shard{shard[0]}-{int(time.time())}.jsonl"
        with self.fs.open(f"{self.root}/{name}", "wb") as f:
            f.write("\n".join(lines).encode())


class FileSink:
    """One file per successful row: {outdir}/{id}{ext} <- record[key].
    Exact resume (filenames are the manifest, writes are idempotent); failed
    rows leave no record and simply retry on the next run."""

    def __init__(self, outdir: str, ext: str = ".txt", key: str = "text"):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ext, self.key = ext, key
        self.rows_written = 0

    def existing_ids(self, retry_errors: bool = False) -> set[str]:
        return {p.name.removesuffix(self.ext) for p in self.dir.glob(f"*{self.ext}")}

    def append(self, record: dict) -> None:
        if record.get("error") is None:
            (self.dir / f"{record['id']}{self.ext}").write_text(str(record[self.key]))
            self.rows_written += 1

    def flush(self) -> None:
        pass  # write-through


def as_sink(output, flush_every: int = 10):
    """A string is a ParquetSink (the CONTRACT); anything with existing_ids/
    append/flush passes through."""
    return ParquetSink(output, flush_every) if isinstance(output, str) else output


async def drain(results: AsyncIterator, sink, shard: tuple[int, int] = (0, 1),
                stats=None):
    """Terminal stage: persist a Done stream. Returns the Stats it was given
    (or a fresh one), with processed/failed/token counts filled in."""
    from pumpjack import Stats  # local import: avoid cycle

    stats = stats if stats is not None else Stats()
    async for done in results:
        if done.error is None:
            stats.rows_processed += 1
            stats.prompt_tokens += done.usage.get("prompt_tokens", 0)
            stats.completion_tokens += done.usage.get("completion_tokens", 0)
            # id/error are reserved columns and ALWAYS win over parse output —
            # a parse returning e.g. the OpenAI response id must not break resume
            sink.append({**done.out, "id": done.id, "error": None})
        else:
            stats.rows_failed += 1
            sink.append({"id": done.id, "error": done.error})
    sink.flush()
    if hasattr(sink, "write_marker"):
        sink.write_marker(shard)
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
