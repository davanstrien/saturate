"""dataset_rows: id stability, strategies, limit/columns, streaming parity.

All in-memory (Dataset.from_dict / to_iterable_dataset) — no network.
"""

from typing import Any

import pytest
from datasets import Dataset

from saturate import content_id, dataset_rows


def ds():
    return Dataset.from_dict(
        {
            "text": ["alpha", "beta", "gamma", "alpha"],
            "key": ["a1", "b2", "c3", "a4"],
            "extra": [1, 2, 3, 4],
        }
    )


def test_index_ids_stable_across_iterations():
    one = list(dataset_rows(ds(), split="train"))
    two = list(dataset_rows(ds(), split="train"))
    assert one == two
    assert [i for i, _ in one] == [f"train:{n:09d}" for n in range(4)]
    assert one[0][1] == {"text": "alpha", "key": "a1", "extra": 1}


def test_streaming_object_matches_materialized():
    materialized = list(dataset_rows(ds()))
    streamed = list(dataset_rows(ds().to_iterable_dataset()))
    assert streamed == materialized


def test_limit_and_columns():
    rows = list(dataset_rows(ds(), columns=["text"], limit=2))
    assert rows == [("train:000000000", {"text": "alpha"}), ("train:000000001", {"text": "beta"})]


def test_content_ids_dedup_identical_rows():
    rows = list(dataset_rows(ds(), columns=["text"], ids="content"))
    assert rows[0][0] == content_id({"text": "alpha"})
    # rows 0 and 3 are identical content -> same id -> pump admits once
    assert rows[0][0] == rows[3][0]
    assert len({i for i, _ in rows}) == 3


def test_content_id_hashes_raw_bytes_by_content():
    """The same bytes give the same id in any buffer type and key order; different bytes give a
    different id; a JSON-only row keeps the id main gives, so resume holds."""
    audio = b"RIFF\x00\x01fake-wav"
    first = content_id({"audio": audio, "lang": "en"})
    assert content_id({"lang": "en", "audio": bytes(audio)}) == first
    assert content_id({"audio": bytearray(audio), "lang": "en"}) == first
    assert content_id({"audio": memoryview(audio), "lang": "en"}) == first
    assert content_id({"audio": audio + b"!", "lang": "en"}) != first
    assert content_id({"text": "row 7"}) == "0e88386f40ff2287"


def test_content_id_hashes_nested_and_strided_bytes():
    nested = {"file": {"bytes": b"abc", "path": "a.wav"}, "parts": [b"x", b"y"]}
    assert content_id(nested) == content_id(
        {"parts": [b"x", b"y"], "file": {"path": "a.wav", "bytes": b"abc"}}
    )
    assert content_id({"b": memoryview(b"aXbXcX")[::2]}) == content_id({"b": b"abc"})


def test_a_row_with_bytes_never_shares_an_id_with_a_json_row():
    """A string that spells out a bytes hash is not the bytes: the hashed text names where the
    bytes are, so the two rows never share an id."""
    import hashlib

    spelled_out = {"x": hashlib.sha1(b"abc").hexdigest()}
    assert content_id({"x": b"abc"}) != content_id(spelled_out)
    # both rows hold bytes; only where the bytes are tells them apart
    abc, def_ = hashlib.sha1(b"abc").hexdigest(), hashlib.sha1(b"def").hexdigest()
    assert content_id({"x": b"abc", "y": def_}) != content_id({"x": abc, "y": b"def"})


def test_content_id_refuses_views_whose_items_are_not_single_bytes():
    from array import array

    with pytest.raises(TypeError, match="non-JSON"):
        content_id({"a": memoryview(array("i", [1, 2]))})


def test_column_ids():
    assert [i for i, _ in dataset_rows(ds(), ids="key")] == ["a1", "b2", "c3", "a4"]


def test_callable_ids_trusted_as_is():
    rows = list(dataset_rows(ds(), ids=lambda ex: f"pg-{ex['key']}"))
    assert [i for i, _ in rows] == ["pg-a1", "pg-b2", "pg-c3", "pg-a4"]


def test_unknown_id_column_raises():
    with pytest.raises(KeyError):
        list(dataset_rows(ds(), ids="nope"))


def test_revision_or_token_with_loaded_object_raises():
    # a silent no-op would void the index-id stability caveat
    with pytest.raises(ValueError, match="repo id string"):
        list(dataset_rows(ds(), revision="some-branch"))
    with pytest.raises(ValueError, match="repo id string"):
        list(dataset_rows(ds(), token="hf_x"))


def test_empty_columns_means_empty_rows_not_all():
    # columns=[] is not treated as columns=None
    rows = list(dataset_rows(ds(), columns=[], limit=1))
    assert rows == [("train:000000000", {})]


def test_id_strategy_producing_none_raises_loudly():
    # a broken callable does not collide every row on "None"
    with pytest.raises(ValueError, match="produced no id"):
        list(dataset_rows(ds(), ids=lambda ex: None))


def test_id_column_need_not_be_in_columns():
    # id lookup happens on the full example, before the column filter
    rows = list(dataset_rows(ds(), columns=["text"], ids="key", limit=2))
    assert rows == [("a1", {"text": "alpha"}), ("b2", {"text": "beta"})]


def test_a_none_or_empty_id_raises_instead_of_collapsing_rows():
    """A None or empty id raises. As the id "None", the first such row would run and the rest
    would be silently deduped."""
    import pytest

    from saturate.source import normalize

    with pytest.raises(ValueError, match="non-empty id"):
        list(normalize([{"k": None, "x": 1}], id_key="k"))
    with pytest.raises(ValueError, match="non-empty id"):
        list(normalize([{"x": 1}], id_fn=lambda row: ""))
    assert [i for i, _ in normalize([{"k": 0}, {"k": "a"}], id_key="k")] == ["0", "a"]  # 0 is an id


def test_id_key_with_id_carrying_tuples_raises_instead_of_being_ignored():
    import pytest

    from saturate.source import normalize

    with pytest.raises(TypeError, match="already carry an id"):
        list(normalize([("a", {"page_id": "p1"})], id_key="page_id"))


def test_rolling_map_keeps_none_items_and_what_follows():
    from concurrent.futures import ThreadPoolExecutor

    from saturate.source import rolling_map

    with ThreadPoolExecutor(2) as pool:
        # a None after the first window is an item, not "exhausted": nothing after it is lost
        assert list(rolling_map([1, 2, 3, None, 5], lambda x: x, pool, window=2)) == [1, 2, 3, None, 5]


def test_index_ids_name_the_dataset_config_and_revision(monkeypatch):
    """Index ids name the dataset, config and revision, so a second config (or another dataset)
    written to the same output is not skipped as done (#34)."""
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: ds())
    en = [i for i, _ in dataset_rows("org/books", "en", limit=2)]
    fr = [i for i, _ in dataset_rows("org/books", "fr", limit=2)]
    other = [i for i, _ in dataset_rows("org/papers", "en", limit=2)]
    pinned = [i for i, _ in dataset_rows("org/books", "en", revision="abc123", limit=2)]
    assert en == ["org/books:en:train:000000000", "org/books:en:train:000000001"]
    assert pinned[0] == "org/books@abc123:en:train:000000000"
    assert len(set(en) | set(fr) | set(other) | set(pinned)) == 8


def decode_counter(monkeypatch):
    """Record the "key" of every example datasets decodes."""
    import datasets

    calls = []
    real = datasets.Features.decode_example

    def counting(self, example, *a, **kw):
        calls.append(example.get("key"))
        return real(self, example, *a, **kw)

    monkeypatch.setattr(datasets.Features, "decode_example", counting)
    return calls


def test_shard_keeps_only_its_rows_and_decodes_nothing_else(monkeypatch):
    """Decoding before shard selection would waste the work on (K-1)/K of the rows (#34). With
    shard= and a repo id, a row of another shard is never decoded, streaming or not. An
    already-loaded object is iterated as it is: every row is decoded, and the shards are still
    disjoint and complete."""
    import datasets

    from saturate import shard_of

    calls = decode_counter(monkeypatch)
    big = Dataset.from_dict({"key": [f"k{i}" for i in range(40)]})
    whole = [i for i, _ in dataset_rows(big)]
    calls.clear()
    slices = [list(dataset_rows(big, shard=(r, 4))) for r in range(4)]
    assert sorted(i for s in slices for i, _ in s) == sorted(whole)  # disjoint and complete
    for rank, s in enumerate(slices):
        assert all(shard_of(i, 4) == rank for i, _ in s)
    assert len(calls) == 4 * 40  # an object you loaded: every shard decodes every row

    for loaded in (big.to_iterable_dataset(), big):  # streaming=True, streaming=False
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, loaded=loaded, **kw: loaded)
        expected = [(i, row) for i, row in dataset_rows("org/keys") if shard_of(i, 4) == 2]
        calls.clear()
        streamed = list(dataset_rows("org/keys", shard=(2, 4)))
        assert streamed == expected
        assert len(calls) == len(streamed)  # only this shard's rows were decoded


def test_fractional_or_bool_shards_raise():
    """Fractional or bool shards raise: truncating (0.9, 2) to (0, 2) or (1.5, 1.8) to (1, 1)
    would quietly run a wrong shard."""
    from saturate import shard_select

    bad_shards: list[tuple[Any, Any]] = [(0.9, 2), (1.5, 1.8), (True, 2), (0, "4")]  # wrong on purpose
    for bad in bad_shards:
        with pytest.raises((ValueError, TypeError), match="shard"):
            shard_select(iter(()), *bad)


def test_transformed_iterable_is_never_decoded_twice(monkeypatch):
    """A filter or map on an IterableDataset decodes the example itself; decoding it again
    breaks images ('PngImageFile' object is not subscriptable). Such pipelines are iterated
    as they are, and every example is decoded exactly once."""
    from saturate import shard_of

    calls = decode_counter(monkeypatch)
    big = Dataset.from_dict({"key": [f"k{i}" for i in range(40)]})
    for transformed in (
        big.to_iterable_dataset().filter(lambda ex: True),
        big.to_iterable_dataset().map(lambda ex: ex),
    ):
        calls.clear()
        plain = [i for i, _ in dataset_rows(big)]
        calls.clear()
        got = list(dataset_rows(transformed, shard=(0, 4)))
        assert [i for i, _ in got] == [i for i in plain if shard_of(i, 4) == 0]
        assert len(calls) == len(set(calls))  # no example decoded twice


def test_sharded_image_dataset_through_a_filter():
    pytest.importorskip("PIL")
    import io

    from datasets import Features, Image, Value
    from PIL import Image as PILImage

    png = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(png, format="PNG")
    images = Dataset.from_dict(
        {"img": [{"bytes": png.getvalue(), "path": None}] * 8, "key": [f"k{i}" for i in range(8)]},
        features=Features({"img": Image(), "key": Value("string")}),
    )
    rows = list(dataset_rows(images.to_iterable_dataset().filter(lambda ex: True), shard=(0, 1)))
    assert len(rows) == 8 and all(hasattr(row["img"], "size") for _, row in rows)


def test_empty_and_transformed_datasets_shard():
    """An empty Dataset shards to no rows, though to_iterable_dataset cannot make one shard from
    zero rows. A Dataset with a transform shards without NotImplementedError."""
    empty = Dataset.from_dict({"text": []})
    assert list(dataset_rows(empty, shard=(0, 1))) == []

    def shout(batch):
        return {"text": [t.upper() for t in batch["text"]]}

    shouted = list(dataset_rows(ds().with_transform(shout), columns=["text"], shard=(0, 1)))
    assert [row["text"] for _, row in shouted] == ["ALPHA", "BETA", "GAMMA", "ALPHA"]


def _loaded(monkeypatch, dataset):
    """dataset_rows("org/x") loads `dataset`, as if load_dataset had fetched it."""
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: dataset)


def test_a_sharded_load_decodes_only_the_selected_columns(monkeypatch):
    """With shard=, the late decode reads only the columns= columns: an unreadable image in a
    column left out does not raise UnidentifiedImageError, as in the unsharded run."""
    import datasets

    features = datasets.Features({"text": datasets.Value("string"), "img": datasets.Image()})
    bad = Dataset.from_dict(
        {"text": ["a", "b"], "img": [{"bytes": b"not an image", "path": None}] * 2}, features=features
    )
    for loaded in (bad, bad.to_iterable_dataset()):  # streaming=False, streaming=True
        _loaded(monkeypatch, loaded)
        rows = list(dataset_rows("org/x", columns=["text"], shard=(0, 1), streaming=loaded is not bad))
        assert [row for _, row in rows] == [{"text": "a"}, {"text": "b"}]


def test_a_sharded_load_without_known_features_is_selected_after_reading(monkeypatch):
    """A JSON or CSV repo without dataset_info loads with features=None, and decode(False)
    raises for it. Such rows have nothing to decode late, so they are read and selected."""
    from datasets import IterableDataset

    from saturate import shard_of

    untyped = IterableDataset.from_generator(lambda: ({"key": f"k{i}"} for i in range(20)))
    assert untyped.features is None
    _loaded(monkeypatch, untyped)
    whole = list(dataset_rows("org/x"))
    assert list(dataset_rows("org/x", shard=(1, 3))) == [(i, r) for i, r in whole if shard_of(i, 3) == 1]


def test_a_sharded_empty_load_yields_nothing(monkeypatch):
    """streaming=False with shard= on an empty dataset yields nothing; it does not raise
    "Unable to shard a dataset of size 0"."""
    _loaded(monkeypatch, Dataset.from_dict({"key": []}))
    assert list(dataset_rows("org/x", streaming=False, shard=(0, 4))) == []


def test_a_shard_that_is_not_a_pair_raises_a_clear_error():
    from saturate.source import check_shard

    bad_shards: list[Any] = [3, (1, 2, 3), None]  # wrong on purpose
    for bad in bad_shards:
        with pytest.raises(TypeError, match="pair of ints"):
            check_shard(bad)
