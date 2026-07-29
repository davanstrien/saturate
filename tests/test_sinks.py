"""Smoke tests for surfaces outside the oracle: FileSink resume, read_output's
healing rule, skip_done with an external done-set."""

import pyarrow as pa
import pyarrow.parquet as pq

from saturate import FileSink, Stats, read_output, skip_done


def test_filesink_resume_roundtrip(tmp_path):
    sink = FileSink(tmp_path / "out", ext=".md", key="markdown")
    sink.append({"id": "a", "markdown": "# A", "error": None})
    sink.append({"id": "b", "error": "http 400: nope"})  # failed: no file
    sink.flush()
    assert (tmp_path / "out" / "a.md").read_text() == "# A"
    assert sink.existing_ids() == {"a"}  # the filesystem is the manifest
    sink.append({"id": "a", "markdown": "# A2", "error": None})  # idempotent overwrite
    assert (tmp_path / "out" / "a.md").read_text() == "# A2"


def test_filesink_rejects_unsafe_ids_and_writes_atomically(tmp_path):
    """Codex r3 #3/#15: ids become filenames — traversal/dotfile ids must raise
    (never write outside the dir or invisibly to resume); writes are atomic."""
    import pytest

    sink = FileSink(tmp_path / "out")
    for bad in ("../evil", "a/b", "", ".", "..", ".hidden"):
        with pytest.raises(ValueError, match="not a safe filename"):
            sink.append({"id": bad, "text": "x", "error": None})
    assert not (tmp_path / "evil.txt").exists()
    sink.append({"id": "ok", "text": "x", "error": None})
    assert sink.existing_ids() == {"ok"}
    assert not list((tmp_path / "out").glob("*.tmp"))  # atomic replace leaves no temp files


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

    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "ok1", "response": "fine", "error": None})   # part 1: all-null error
    sink.append({"id": "bad1", "error": "http 400: nope"})          # part 2: string error
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))  # the CONTRACT reader pattern
    t = ds.dataset(parts, format="parquet").to_table()
    recs = {r["id"]: r for r in t.to_pylist()}
    assert recs["ok1"]["error"] is None and recs["bad1"]["error"].startswith("http 400")


def test_error_row_first_keeps_success_columns(tmp_path):
    """Codex r3 #2: from_pylist takes the schema from row 0 — an error row at
    the head of a batch must not silently drop later success columns."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=2)
    sink.append({"id": "bad", "error": "http 400: nope"})
    sink.append({"id": "ok", "text": "hi", "error": None})
    part = sorted(tmp_path.glob("part-*.parquet"))[0]
    recs = {r["id"]: r for r in pq.read_table(part).to_pylist()}
    assert recs["ok"]["text"] == "hi" and recs["bad"]["text"] is None


def test_null_first_then_typed_column_evolves(tmp_path):
    """Codex r4/r5 blocker #1: all-null columns are dropped from their part
    (sparse, §1) so a later real type — string OR float — can never conflict
    with a premature guess already on disk. read_output unifies across parts."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "text": None, "score": None, "error": None})  # part 1: sparse {id,error}
    sink.append({"id": "b", "text": "hi", "score": 1.5, "error": None})   # part 2: real types appear
    sink.append({"id": "c", "error": "http 400: nope"})                   # part 3: error-only
    for part in tmp_path.glob("part-*.parquet"):
        pq.read_table(part)  # every part individually readable, no null/typed conflicts
    got = dict(read_output(str(tmp_path)))
    assert got == {"a": {}, "b": {"text": "hi", "score": 1.5}}


def test_declared_schema_is_stable_in_any_order(tmp_path):
    """Codex r5 blocker #1: a declared schema is the fully-stable option — every
    part frames to it (missing fields null), whatever order rows arrive in."""
    import pyarrow.dataset as ds
    import pytest

    from saturate import ParquetSink

    schema = pa.schema([("id", pa.string()), ("error", pa.string()),
                        ("text", pa.string()), ("score", pa.float64())])
    sink = ParquetSink(str(tmp_path), flush_every=1, schema=schema)
    sink.append({"id": "e", "error": "http 400: nope"})                  # error-only first
    sink.append({"id": "a", "score": None, "error": None})               # null before type
    sink.append({"id": "b", "text": "hi", "score": 1.5, "error": None})  # full row last
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    t = ds.dataset(parts, format="parquet").to_table()
    assert t.schema.field("score").type == pa.float64() and len(t) == 3
    with pytest.raises(ValueError, match="outside the declared schema"):
        sink.append({"id": "x", "surprise": 1, "error": None})
        sink.flush()
    with pytest.raises(ValueError, match="id and error"):
        ParquetSink(str(tmp_path), schema=pa.schema([("text", pa.string())]))


def test_declared_type_mismatch_becomes_error_row(tmp_path):
    """Codex r6 blocker #1: declared score: float64 + {'score': 'not-a-number'}
    previously passed the inferred-schema probe, counted as processed, then
    died at flush with no record. The probe now validates against the DECLARED
    schema; overflow ints and extra fields are error rows too."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    schema = pa.schema([("id", pa.string()), ("error", pa.string()), ("score", pa.float64())])

    async def results():
        yield Done("bad-type", {}, {"score": "not-a-number"}, None, {})
        yield Done("bad-overflow", {}, {"score": 10**1000}, None, {})
        yield Done("bad-extra", {}, {"score": 1.0, "surprise": 1}, None, {})
        yield Done("good", {}, {"score": 1.5}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1, schema=schema)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 3)
    assert sink.existing_ids(retry_errors=True) == {"good"}  # bad rows durable + healable


def test_declared_schema_requires_contract_types(tmp_path):
    """Codex r6 blocker #1: declared id/error must satisfy the contract."""
    import pytest

    from saturate import ParquetSink

    with pytest.raises(ValueError, match="must be string"):
        ParquetSink(str(tmp_path), schema=pa.schema([("id", pa.int64()), ("error", pa.string())]))


def test_declared_schema_rejects_nonnullable_user_fields(tmp_path):
    """Codex r7 #1: error rows write user fields as null — a declared
    non-nullable field would crash the Parquet write with no durable record."""
    import pytest

    from saturate import ParquetSink

    with pytest.raises(ValueError, match="nullable"):
        ParquetSink(str(tmp_path), schema=pa.schema([
            ("id", pa.string()), ("error", pa.string()),
            pa.field("score", pa.float64(), nullable=False)]))


def test_declared_schema_error_row_is_durable(tmp_path):
    """Codex r7 #1: an ordinary error Done under a declared schema must land as
    a durable sparse error row."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    schema = pa.schema([("id", pa.string()), ("error", pa.string()), ("score", pa.float64())])
    sink = ParquetSink(str(tmp_path), flush_every=1, schema=schema)
    asyncio.run(drain(iter_async([Done("x", {}, None, "http 500 after retries", {})]), sink))
    assert sink.existing_ids() == {"x"}


def iter_async(items):
    async def gen():
        for x in items:
            yield x
    return gen()


def test_generic_sink_not_subject_to_arrow_rules(tmp_path):
    """Codex r7 #2: FileSink str()s its values — a stringable object its own
    append() can persist must NOT be rejected by an Arrow probe it never asked
    for (validation belongs to sinks that supply probe())."""
    import asyncio

    from saturate.core import Done
    from saturate.sink import drain

    class Rendered:
        def __str__(self):
            return "# rendered"

    sink = FileSink(tmp_path / "out", ext=".md", key="markdown")
    stats = asyncio.run(drain(iter_async([Done("a", {}, {"markdown": Rendered()}, None, {})]), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 0)
    assert (tmp_path / "out" / "a.md").read_text() == "# rendered"


def test_dynamic_type_change_raises_at_flush(tmp_path):
    """Codex r6 blocker #4: int64 -> double was permissively widened only in
    the NEW part, breaking multi-part reads while CONTRACT §8 promised a raise.
    Strict unify makes the documented behavior true."""
    import pytest

    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "score": 1, "error": None})
    with pytest.raises((TypeError, ValueError)):
        sink.append({"id": "b", "score": 1.5, "error": None})


def test_schema_rejected_for_custom_sink_objects():
    """Codex r6 follow-up (fixed now): pump(schema=) must never be silently
    ignored when the output is a custom sink object."""
    import pytest

    from saturate.sink import as_sink

    class Custom:
        def existing_ids(self, retry_errors=False):
            return set()

        def append(self, r):
            pass

        def flush(self):
            pass

    with pytest.raises(ValueError, match="pass it to your sink"):
        as_sink(Custom(), schema=pa.schema([("id", pa.string()), ("error", pa.string())]))


def test_nonserializable_parse_value_becomes_error_row(tmp_path):
    """Codex r5 blocker #6: {'value': object()} passed the dict check but died
    at flush with no durable record — must become a healable error row."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("bad", {}, {"value": object()}, None, {})
        yield Done("good", {}, {"value": 1}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 1)
    assert sink.existing_ids(retry_errors=True) == {"good"}  # bad is healable


def test_filesink_rejects_unsafe_ext(tmp_path):
    """Codex r5: ext lands in filenames and the resume glob."""
    import pytest

    for bad in ("txt", ".t/xt", "..txt"):
        with pytest.raises(ValueError, match="unsafe ext"):
            FileSink(tmp_path / "out", ext=bad)


def test_all_null_column_inherits_pinned_type(tmp_path):
    """Codex r3 #5: an all-null user column in a later part must keep the type
    pinned at first flush, so cross-part dataset reads stay schema-stable."""
    import pyarrow.dataset as ds

    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "text": "hi", "error": None})     # part 1 pins text: string
    sink.append({"id": "b", "text": None, "error": None})     # part 2: all-null text
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    t = ds.dataset(parts, format="parquet").to_table()
    assert t.schema.field("text").type == pa.string()
    assert dict(zip(t["id"].to_pylist(), t["text"].to_pylist(), strict=True)) == {"a": "hi", "b": None}


def test_all_null_part_carries_full_pinned_schema(tmp_path, monkeypatch):
    """#12: with a part-name ms collision, sorted() order was uuid-decided — and the
    all-null part did not carry the pinned column at all, so first-fragment readers
    (pyarrow.dataset's default) lost it whenever that part sorted first. Every part
    must materialize the full pinned schema; the seq counter makes order deterministic."""
    import pyarrow.dataset as ds

    from saturate import ParquetSink
    from saturate import sink as sink_mod

    monkeypatch.setattr(sink_mod.time, "time", lambda: 1_234_567_890.0)  # force the collision
    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "text": "hi", "error": None})     # part 1 pins text: string
    sink.append({"id": "b", "text": None, "error": None})     # part 2: all-null text
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    assert len(parts) == 2
    for p in parts:  # the #12 guarantee: each part individually carries text: string
        assert pq.read_schema(p).field("text").type == pa.string()
    for order in (parts, parts[::-1]):  # first-fragment reads survive EITHER order
        t = ds.dataset(order, format="parquet").to_table()
        assert dict(zip(t["id"].to_pylist(), t["text"].to_pylist(), strict=True)) == {"a": "hi", "b": None}
    # seq counter: same-ms parts still sort in write order
    t = ds.dataset(parts, format="parquet").to_table()
    assert t["id"].to_pylist() == ["a", "b"]


def test_skip_done_with_external_set():
    stats = Stats()
    rows = [("a", {}), ("b", {}), ("b", {}), ("c", {})]
    kept = [i for i, _ in skip_done(iter(rows), done={"a"}, stats=stats)]
    assert kept == ["b", "c"]
    assert (stats.rows_total, stats.rows_done_prior, stats.rows_deduped) == (4, 1, 1)


def test_reserved_columns_win_over_parse(tmp_path):
    """Codex finding #2: parse returning its own 'id' must not break resume."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("row-1", {}, {"id": "chatcmpl-xyz", "text": "hi"}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    asyncio.run(drain(results(), sink))
    assert sink.existing_ids() == {"row-1"}
    # stats sidecar (CONTRACT §5): exact counts + geometry for storage-only readers
    import json as _json
    sink.write_stats((2, 4), _json.dumps({"rows_processed": 1, "rows_failed": 0}))
    got = _json.loads((tmp_path / "completions" / "stats-2.json").read_text())
    assert (got["rank"], got["world"], got["rows_processed"]) == (2, 4, 1)


def test_content_id_rejects_objects():
    """Codex finding #3: unstable reprs must be refused, not silently hashed."""
    import pytest

    from saturate.source import content_id

    class FakeImage:
        pass

    with pytest.raises(TypeError, match="non-JSON"):
        content_id({"image": FakeImage()})


def test_retry_after_http_date():
    """Codex finding #6: RFC 9110 HTTP-date form."""
    from saturate.transport import _parse_retry_after

    assert _parse_retry_after("3.5") == 3.5
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # past date clamps
    assert _parse_retry_after("not-a-date") is None


def test_hf_buckets_uri_resolves_offline():
    """#7: bucket output paths must work at the packaged [hf] floor — an
    hf://buckets/... URI resolves to HfFileSystem with the bucket path intact,
    no network and no token involved."""
    import fsspec
    from huggingface_hub import HfFileSystem

    fs, root = fsspec.url_to_fs("hf://buckets/owner/name/runs/out")
    assert isinstance(fs, HfFileSystem)
    assert root.endswith("buckets/owner/name/runs/out")


def test_hf_output_repo_autocreated(tmp_path, monkeypatch):
    """0.1.1: a fresh hf:// output path must not crash existing_ids() — the sink
    ensures the dataset repo / bucket exists (private, exist_ok) at construction
    (live failure: RepositoryNotFoundError from the Slack OCR snippet's first run)."""
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw):
            calls.append(("repo", repo, kw.get("repo_type"), kw.get("private")))

        def create_bucket(self, repo, **kw):
            calls.append(("bucket", repo, kw.get("private")))

    import huggingface_hub

    from saturate import ParquetSink

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    ParquetSink("hf://datasets/owner/name/runs/out")
    ParquetSink("hf://buckets/owner/name/runs/out")
    assert ("repo", "owner/name", "dataset", True) in calls
    assert ("bucket", "owner/name", True) in calls
    ParquetSink(str(tmp_path / "datasets/owner/name"))  # local lookalike: untouched
    assert len(calls) == 2
