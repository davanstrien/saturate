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


def test_mixed_success_error_parts_union(tmp_path):
    """A dir holding success-only AND error-only parts must stay readable by
    external consumers (the shapes-run finding: null-typed error column)."""
    import pyarrow.dataset as ds

    from pumpjack import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "ok1", "response": "fine", "error": None})   # part 1: all-null error
    sink.append({"id": "bad1", "error": "http 400: nope"})          # part 2: string error
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))  # the CONTRACT reader pattern
    t = ds.dataset(parts, format="parquet").to_table()
    recs = {r["id"]: r for r in t.to_pylist()}
    assert recs["ok1"]["error"] is None and recs["bad1"]["error"].startswith("http 400")


def test_skip_done_with_external_set():
    stats = Stats()
    rows = [("a", {}), ("b", {}), ("b", {}), ("c", {})]
    kept = [i for i, _ in skip_done(iter(rows), done={"a"}, stats=stats)]
    assert kept == ["b", "c"]
    assert (stats.rows_total, stats.rows_done_prior, stats.rows_deduped) == (4, 1, 1)


def test_reserved_columns_win_over_parse(tmp_path):
    """Codex finding #2: parse returning its own 'id' must not break resume."""
    import asyncio

    from pumpjack import ParquetSink
    from pumpjack.core import Done
    from pumpjack.sink import drain

    async def results():
        yield Done("row-1", {}, {"id": "chatcmpl-xyz", "text": "hi"}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    asyncio.run(drain(results(), sink))
    assert sink.existing_ids() == {"row-1"}


def test_content_id_rejects_objects():
    """Codex finding #3: unstable reprs must be refused, not silently hashed."""
    import pytest

    from pumpjack.source import content_id

    class FakeImage:
        pass

    with pytest.raises(TypeError, match="non-JSON"):
        content_id({"image": FakeImage()})


def test_retry_after_http_date():
    """Codex finding #6: RFC 9110 HTTP-date form."""
    from pumpjack.transport import _parse_retry_after

    assert _parse_retry_after("3.5") == 3.5
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # past date clamps
    assert _parse_retry_after("not-a-date") is None
