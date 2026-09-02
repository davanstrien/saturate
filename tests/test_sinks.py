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


def test_dynamic_type_change_becomes_error_row_at_flush(tmp_path):
    """A direct sink append bypasses probe(): a same-run type change (int64 pinned,
    double arriving) is caught at flush and written as an error row — the pinned type
    never moves (widening only the new part would break multi-part reads, CONTRACT §8)
    and flush never raises for a type conflict."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "score": 1, "error": None})
    sink.append({"id": "b", "score": 1.5, "error": None})
    assert sink.existing_ids() == {"a", "b"} and sink.existing_ids(retry_errors=True) == {"a"}
    recs = {r["id"]: r for _, r in ((r["id"], r) for p in tmp_path.glob("part-*.parquet")
                                      for r in pq.read_table(p).to_pylist())}
    assert recs["b"]["error"].startswith("parse output not storable") and recs["b"]["score"] is None
    for part in tmp_path.glob("part-*.parquet"):
        assert pq.read_schema(part).field("score").type == pa.int64()


def test_dynamic_type_conflict_becomes_error_row_under_drain(tmp_path):
    """Under drain, a row whose type conflicts with the pinned dynamic schema must
    become a durable error row: previously probe() only checked serializability, so
    the conflict surfaced at flush, discarded the buffered rows and crashed the run
    (and every resume of it, since the same rows arrive in the same order)."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        for i in range(5):
            yield Done(f"int-{i}", {}, {"score": 1}, None, {})
        for i in range(5):
            yield Done(f"float-{i}", {}, {"score": 0.5}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=5)
    stats = asyncio.run(drain(results(), sink))
    ints, floats = {f"int-{i}" for i in range(5)}, {f"float-{i}" for i in range(5)}
    assert (stats.rows_processed, stats.rows_failed) == (5, 5)
    assert sink.existing_ids() == ints | floats  # every row durable
    assert sink.existing_ids(retry_errors=True) == ints  # the float rows are healable errors
    for part in tmp_path.glob("part-*.parquet"):
        assert pq.read_schema(part).field("score").type == pa.int64()  # the pinned type held


def test_dynamic_lossless_widening_is_accepted(tmp_path):
    """An int arriving in a pinned double column is lossless (1 -> 1.0): both rows
    durable, no error rows, and every part carries the pinned double type."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("f", {}, {"score": 0.5}, None, {})
        yield Done("i", {}, {"score": 1}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (2, 0)
    assert sink.existing_ids(retry_errors=True) == {"f", "i"}
    parts = list(tmp_path.glob("part-*.parquet"))
    assert len(parts) == 2
    for part in parts:
        assert pq.read_schema(part).field("score").type == pa.float64()
    assert dict(read_output(str(tmp_path))) == {"f": {"score": 0.5}, "i": {"score": 1.0}}


def test_dynamic_lossy_cast_is_error_row(tmp_path):
    """The reverse is lossy (0.5 into a pinned int64): the float row is an error row."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("i", {}, {"score": 1}, None, {})
        yield Done("f", {}, {"score": 0.5}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 1)
    assert sink.existing_ids() == {"i", "f"} and sink.existing_ids(retry_errors=True) == {"i"}


def test_dynamic_integral_float_into_int_is_conflict(tmp_path):
    """Widening is directional: 2.0 into a pinned int64 column is a float into an int
    column (CONTRACT §8), not a widening, even though the value happens to be integral —
    an error row under drain and at flush for a direct append."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("i", {}, {"score": 1}, None, {})
        yield Done("f", {}, {"score": 2.0}, None, {})

    sink = ParquetSink(str(tmp_path / "drain"), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 1)
    assert sink.existing_ids() == {"i", "f"} and sink.existing_ids(retry_errors=True) == {"i"}
    direct = ParquetSink(str(tmp_path / "direct"), flush_every=1)
    direct.append({"id": "a", "score": 1, "error": None})
    direct.append({"id": "b", "score": 2.0, "error": None})  # direct append: error row at flush
    assert direct.existing_ids(retry_errors=True) == {"a"}


def test_dynamic_value_rewrites_are_conflicts(tmp_path):
    """Only numeric widening is accepted into a pinned column. Arrow would happily cast
    1 -> "1" or True -> 1, but those rewrite the value — record, never guess — so an int
    into a pinned string column and a bool into a pinned int column are error rows."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("s", {}, {"label": "a", "n": 1}, None, {})
        yield Done("int-into-string", {}, {"label": 1}, None, {})
        yield Done("bool-into-int", {}, {"n": True}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 2)
    assert sink.existing_ids() == {"s", "int-into-string", "bool-into-int"}
    assert sink.existing_ids(retry_errors=True) == {"s"}


def test_dynamic_value_rewrite_is_error_row_at_flush(tmp_path):
    """The same rule on the direct-append path: int -> string is an error row at flush."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "label": "a", "error": None})
    sink.append({"id": "b", "label": 1, "error": None})
    assert sink.existing_ids() == {"a", "b"} and sink.existing_ids(retry_errors=True) == {"a"}


def test_mixed_types_in_first_buffer_cost_one_row(tmp_path, capsys):
    """Before any type is pinned, probe cannot see the buffer: {score: 1} then {score: "a"}
    in ONE buffer used to fail batch inference at flush and lose every buffered row. flush
    now falls back to a per-row pass: the first typed value pins, the conflicting row is
    written as an error row, and the flush is logged once with the count."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("i", {}, {"score": 1}, None, {})
        yield Done("s", {}, {"score": "a"}, None, {})
        yield Done("j", {}, {"score": 2}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=10)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (2, 1)
    assert sink.rows_demoted == 0  # probe sees the buffered rows' types: caught before buffering
    assert sink.existing_ids() == {"i", "s", "j"} and sink.existing_ids(retry_errors=True) == {"i", "j"}
    assert dict(read_output(str(tmp_path))) == {"i": {"score": 1}, "j": {"score": 2}}
    # the same rows appended directly (no probe) reach flush, which demotes instead of raising
    direct = ParquetSink(str(tmp_path / "direct"), flush_every=10)
    for rec in ({"id": "i", "score": 1, "error": None}, {"id": "s", "score": "a", "error": None},
                {"id": "j", "score": 2, "error": None}):
        direct.append(rec)
    direct.flush()
    assert direct.rows_demoted == 1 and direct.existing_ids(retry_errors=True) == {"i", "j"}
    assert "1 of 3 rows do not fit the pinned schema" in capsys.readouterr().err


def test_mixed_types_in_new_column_after_pin_cost_one_row(tmp_path):
    """After a pin, a NEW column with mixed types inside one buffer is the same case: the
    batch cannot be built, the per-row pass keeps the first typed value and demotes the
    conflicting row. The pinned column is untouched and every part stays readable."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("a", {}, {"score": 1}, None, {})                # buffer 1 pins score: int64
        yield Done("a2", {}, {"score": 1}, None, {})
        yield Done("b", {}, {"score": 2, "tag": 7}, None, {})      # buffer 2: tag int64 ...
        yield Done("c", {}, {"score": 3, "tag": "x"}, None, {})    # ... then tag string: demoted

    sink = ParquetSink(str(tmp_path), flush_every=2)
    asyncio.run(drain(results(), sink))
    assert sink.existing_ids() == {"a", "a2", "b", "c"}
    assert sink.existing_ids(retry_errors=True) == {"a", "a2", "b"}
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    assert len(parts) == 2
    second = pq.read_schema(parts[1])  # tag first appears in part 2: pinned there as int64
    assert second.field("tag").type == pa.int64() and second.field("score").type == pa.int64()
    assert dict(read_output(str(tmp_path)))["b"] == {"score": 2, "tag": 7}


def test_nested_widening_follows_the_scalar_rule(tmp_path):
    """The conform rule is recursive: [1] widens into a pinned list<double> exactly as 1
    widens into double; a struct missing a pinned field is filled with null; a struct with
    a field the pin lacks, or a retyped field, is a conflict."""
    import asyncio

    import pyarrow.dataset as ds

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("p", {}, {"v": [1.5], "m": {"a": 1, "b": "x"}}, None, {})  # pins list<double>, struct<a,b>
        yield Done("w", {}, {"v": [1], "m": {"a": 2}}, None, {})              # widens: list<int64>, struct<a>
        yield Done("x", {}, {"v": [1.5], "m": {"a": 3, "c": 1}}, None, {})    # struct field the pin lacks
        yield Done("y", {}, {"v": [1.5], "m": {"a": "z"}}, None, {})          # struct field retyped
        yield Done("z", {}, {"v": [0.5, "s"]}, None, {})                      # unserializable list

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (2, 3)
    assert sink.existing_ids(retry_errors=True) == {"p", "w"}
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    t = ds.dataset(parts, format="parquet").to_table()
    assert t.schema.field("v").type == pa.list_(pa.float64())
    got = dict(read_output(str(tmp_path)))
    assert got["w"] == {"v": [1.0], "m": {"a": 2, "b": None}}


def test_pinned_schema_is_read_back_on_resume(tmp_path):
    """The pin is a property of the output dir, not of one sink object: a fresh sink on a
    dir with parts reads the newest part's schema before its first flush. Otherwise a
    retry run whose first row is the float the previous run error-rowed would pin double
    and write a part unreadable beside the int64 ones."""
    import asyncio

    import pyarrow.dataset as ds

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    first = ParquetSink(str(tmp_path), flush_every=1)  # run 1 pins score: int64
    asyncio.run(drain(iter_async([Done("a", {}, {"score": 1}, None, {})]), first))
    again = ParquetSink(str(tmp_path), flush_every=1)  # run 2: the float first
    stats = asyncio.run(drain(iter_async([Done("f", {}, {"score": 0.5}, None, {})]), again))
    assert stats.rows_failed == 1 and again.existing_ids(retry_errors=True) == {"a"}
    third = ParquetSink(str(tmp_path), flush_every=1)  # run 3: an int, written as int64
    asyncio.run(drain(iter_async([Done("b", {}, {"score": 2}, None, {})]), third))
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    assert len(parts) == 3
    for p in parts:
        assert pq.read_schema(p).field("score").type == pa.int64()
    assert ds.dataset(parts, format="parquet").to_table().num_rows == 3  # readable as one dataset


def test_unreadable_newest_part_falls_back_to_an_older_pin(tmp_path, capsys):
    """A crash-truncated newest part must not defeat the seed: the pin comes from the newest
    part whose footer reads. Only when no part is readable does the sink start unpinned."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    ParquetSink(str(tmp_path), flush_every=1).append({"id": "a", "score": 1, "error": None})  # int64
    (tmp_path / "part-9999999999999-000000-deadbeef.parquet").write_bytes(b"not parquet")
    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(iter_async([Done("f", {}, {"score": 0.5}, None, {})]), sink))
    assert stats.rows_failed == 1  # the older part's int64 pin held: a float is a conflict
    assert "trying an older part" in capsys.readouterr().err
    only_bad = tmp_path / "only-bad"
    only_bad.mkdir()
    (only_bad / "part-1-000000-deadbeef.parquet").write_bytes(b"not parquet")
    ParquetSink(str(only_bad), flush_every=1).append({"id": "a", "score": 1, "error": None})
    assert "starting unpinned" in capsys.readouterr().err


def test_empty_lists_never_pin(tmp_path):
    """A column seen only as empty lists (list<null>) carries no type: it must not pin, or
    every later real value in that column would be an error row for the rest of the run."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("a", {}, {"tags": []}, None, {})
        yield Done("b", {}, {"tags": []}, None, {})
        yield Done("c", {}, {"tags": ["x"]}, None, {})
        yield Done("d", {}, {"tags": ["y"]}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=2)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (4, 0)
    assert dict(read_output(str(tmp_path)))["c"] == {"tags": ["x"]}
    fresh = ParquetSink(str(tmp_path), flush_every=1)  # the pin read back is list<string>
    asyncio.run(drain(iter_async([Done("e", {}, {"tags": ["z"]}, None, {})]), fresh))
    assert fresh.existing_ids(retry_errors=True) >= {"c", "d", "e"}


def test_narrow_pinned_types_accept_values_that_fit(tmp_path):
    """A declared run leaves int32/float32 parts behind; a dynamic resume infers int64/double.
    Same kind of thing and the value fits: accepted and cast. A float into the int column
    stays a conflict."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    declared = pa.schema([("id", pa.string()), ("error", pa.string()),
                          ("n", pa.int32()), ("sc", pa.float32())])
    ParquetSink(str(tmp_path), flush_every=1, schema=declared).append(
        {"id": "a", "n": 1, "sc": 0.5, "error": None})
    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(iter_async([Done("b", {}, {"n": 2, "sc": 0.25}, None, {}),
                                          Done("c", {}, {"n": 2.5}, None, {})]), sink))
    assert (stats.rows_processed, stats.rows_failed) == (1, 1)
    parts = sorted(str(p) for p in tmp_path.glob("part-*.parquet"))
    assert all(pq.read_schema(p).field("n").type == pa.int32() for p in parts)


def test_demoted_row_keeps_a_string_id(tmp_path):
    """A conflict in the id column itself must still land as an error row, not a raise."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=2)
    sink.append({"id": "b", "score": 2, "error": None})
    sink.append({"id": 7, "score": 3, "error": None})
    sink.flush()
    assert sink.existing_ids() == {"b", "7"} and sink.existing_ids(retry_errors=True) == {"b"}


def test_begin_run_gives_each_run_its_own_telemetry_file(tmp_path):
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path))
    sink.begin_run((0, 1))
    sink.write_telemetry((0, 1), ["{}"])  # run 1 crashes before its marker
    sink.begin_run((0, 1))
    sink.write_telemetry((0, 1), ["{}"])
    assert len(list(tmp_path.glob("telemetry-shard0-*.jsonl"))) == 2


def test_manifest_read_retries_transient_errors(tmp_path, monkeypatch):
    """A transient remote failure (a 429 under concurrent reads) must not mark a manifest
    unreadable and re-pay its rows: the read is retried before giving up."""
    from saturate import ParquetSink
    from saturate import sink as sink_mod

    sink = ParquetSink(str(tmp_path), flush_every=1)
    sink.append({"id": "a", "text": "x", "error": None})
    monkeypatch.setattr(sink_mod.time, "sleep", lambda s: None)
    real_open, calls = sink.fs.open, {"n": 0}

    def flaky_open(path, *a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError("429 Too Many Requests")
        return real_open(path, *a, **k)

    monkeypatch.setattr(sink.fs, "open", flaky_open)
    assert sink.existing_ids() == {"a"}


def test_dynamic_probe_accepts_nulls_new_fields_and_empty_lists(tmp_path):
    """The pinned-type check must not reject rows flush would accept: nulls, empty
    lists and fields the schema has not seen yet all merge into the pinned schema."""
    import asyncio

    from saturate import ParquetSink
    from saturate.core import Done
    from saturate.sink import drain

    async def results():
        yield Done("a", {}, {"score": 1, "tags": ["x"]}, None, {})
        yield Done("b", {}, {"score": None, "tags": [], "extra": "new"}, None, {})

    sink = ParquetSink(str(tmp_path), flush_every=1)
    stats = asyncio.run(drain(results(), sink))
    assert (stats.rows_processed, stats.rows_failed) == (2, 0)
    assert dict(read_output(str(tmp_path)))["b"] == {"score": None, "tags": [], "extra": "new"}


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


def test_hf_autocreate_needs_exact_protocol_match(monkeypatch):
    """Only the HF filesystem gets a repo auto-created: a protocol name that merely
    contains "hf" is not it, while HfFileSystem's alias tuple is."""
    calls = []

    class FakeApi:
        def create_repo(self, repo, **kw):
            calls.append(repo)

        def create_bucket(self, repo, **kw):
            calls.append(repo)

    class FakeFs:
        def __init__(self, protocol):
            self.protocol = protocol

        def makedirs(self, path, exist_ok=False):
            pass

    import fsspec
    import huggingface_hub

    from saturate import ParquetSink

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(fsspec, "url_to_fs", lambda uri: (FakeFs("shfs"), "datasets/owner/name/runs"))
    ParquetSink("shfs://datasets/owner/name/runs")
    assert calls == []
    monkeypatch.setattr(fsspec, "url_to_fs", lambda uri: (FakeFs(("hf",)), "datasets/owner/name/runs"))
    ParquetSink("hf://datasets/owner/name/runs")
    assert calls == ["owner/name"]

def test_existing_ids_concurrent_reads_match_sequential(tmp_path):
    """Resume reads manifest sidecars on a thread pool: the result must be exactly
    the sequential one — an uncovered part (sidecar deleted) is scanned, a corrupt
    sidecar is skipped and its part scanned as fallback, every id is recovered."""
    from saturate import ParquetSink

    sink = ParquetSink(str(tmp_path), flush_every=1)
    ids = {f"row-{i:03d}" for i in range(200)}
    for i, id_ in enumerate(sorted(ids)):
        sink.append({"id": id_, "error": "http 500: nope" if i % 7 == 0 else None})
    manifests = sorted((tmp_path / "_manifest").glob("ids-part-*.parquet"))
    assert len(manifests) == 200
    manifests[3].unlink()  # uncovered part: scanned individually
    manifests[150].write_bytes(b"not parquet")  # unreadable sidecar: its part is scanned
    errored = {f"row-{i:03d}" for i in range(200) if i % 7 == 0}
    for workers in (16, 1):
        s = ParquetSink(str(tmp_path), read_workers=workers)
        assert s.existing_ids() == ids
        assert s.existing_ids(retry_errors=True) == ids - errored


def test_existing_ids_warns_on_many_manifest_files(tmp_path, capsys):
    """Past ~1000 sidecars, resume cost on a remote store is the per-file round-trip;
    say so once and point at the knob (flush_every)."""
    from saturate import ParquetSink

    mdir = tmp_path / "_manifest"
    mdir.mkdir()
    table = pa.Table.from_pylist([{"id": "x", "error": None}])
    for i in range(1001):
        pq.write_table(table, mdir / f"ids-part-{i:05d}.parquet")
    assert ParquetSink(str(tmp_path)).existing_ids() == {"x"}
    err = capsys.readouterr().err
    assert "resume: reading 1001 manifest files" in err and "flush_every" in err
