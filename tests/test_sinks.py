"""Smoke tests for surfaces outside the oracle: FileSink resume, read_output's
healing rule, skip_done with an external done-set."""

import pyarrow as pa
import pyarrow.parquet as pq

from pumpjack import FileSink, Stats, read_output, skip_done


def test_filesink_resume_roundtrip(tmp_path):
    sink = FileSink(tmp_path / "out", ext=".md", key="markdown")
    sink.append({"id": "a", "markdown": "# A", "error": None})
    sink.append({"id": "b", "error": "http 400: nope"})  # failed: no file
    sink.flush()
    assert (tmp_path / "out" / "a.md").read_text() == "# A"
    assert sink.existing_ids() == {"a"}  # the filesystem is the manifest
    sink.append({"id": "a", "markdown": "# A2", "error": None})  # idempotent overwrite
    assert (tmp_path / "out" / "a.md").read_text() == "# A2"


def test_read_output_healing_rule(tmp_path):
    rows = [
        {"id": "x", "out": None, "error": "transport: boom"},   # healed later
        {"id": "x", "out": "good", "error": None},
        {"id": "y", "out": None, "error": "http 500 after retries"},  # error-only: skipped
        {"id": "z", "out": "fine", "error": None},
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "part-1-abc.parquet")
    got = dict(read_output(str(tmp_path)))
    assert got == {"x": {"out": "good"}, "z": {"out": "fine"}}


def test_skip_done_with_external_set():
    stats = Stats()
    rows = [("a", {}), ("b", {}), ("b", {}), ("c", {})]
    kept = [i for i, _ in skip_done(iter(rows), done={"a"}, stats=stats)]
    assert kept == ["b", "c"]
    assert (stats.rows_total, stats.rows_done_prior, stats.rows_deduped) == (4, 1, 1)
