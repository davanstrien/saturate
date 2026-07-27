"""Sink: append-only parquet parts + 1:1 manifest sidecars + exact resume.

Resume is manifest-first but always exact: any part whose manifest sidecar is
missing (crash between part-write and manifest-write) is scanned individually,
so a kill can never produce duplicate work. See CONTRACT.md §3.
"""

from __future__ import annotations

import sys
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq


def _log(msg: str) -> None:
    print(f"[pump] {msg}", file=sys.stderr, flush=True)


class Sink:
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
        """Ids already durably written: manifests first, then any part lacking
        its manifest sidecar (scanned individually — exact, kill-safe).
        retry_errors=True excludes ids whose only record is an error."""
        done: set[str] = set()
        failed: set[str] = set()
        covered: set[str] = set()  # part basenames covered by a manifest
        for path in sorted(self.fs.glob(f"{self.root}/_manifest/ids-*.parquet")):
            if self._read_id_error(path, done, failed):
                covered.add(path.rsplit("/", 1)[-1][4:])  # "ids-<part>" -> "<part>"
        for path in sorted(self.fs.glob(f"{self.root}/part-*.parquet")):
            if path.rsplit("/", 1)[-1] not in covered:
                self._read_id_error(path, done, failed)
        return done if retry_errors else done | failed

    def append(self, row: dict) -> None:
        self._buf.append(row)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"part-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.parquet"
        table = pa.Table.from_pylist(self._buf)
        with self.fs.open(f"{self.root}/{name}", "wb") as f:
            pq.write_table(table, f, compression="zstd")
        # manifest second: a crash in between leaves an uncovered part, which
        # existing_ids scans individually — still exact
        try:
            self.fs.makedirs(f"{self.root}/_manifest", exist_ok=True)
            slim = table.select(["id", "error"])
            with self.fs.open(f"{self.root}/_manifest/ids-{name}", "wb") as f:
                pq.write_table(slim, f, compression="zstd")
        except Exception as e:
            _log(f"manifest write failed (non-fatal, part is covered by scan fallback): {e}")
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
