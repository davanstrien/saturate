"""dataset_rows: id stability, strategies, limit/columns, streaming parity.

All in-memory (Dataset.from_dict / to_iterable_dataset) — no network.
"""

import pytest
from datasets import Dataset

from pumpjack import content_id, dataset_rows


def ds():
    return Dataset.from_dict({
        "text": ["alpha", "beta", "gamma", "alpha"],
        "key": ["a1", "b2", "c3", "a4"],
        "extra": [1, 2, 3, 4],
    })


def test_index_ids_stable_across_iterations():
    one = list(dataset_rows(ds(), split="train"))
    two = list(dataset_rows(ds(), split="train"))
    assert one == two
    assert [i for i, _ in one] == [f"train-{n:09d}" for n in range(4)]
    assert one[0][1] == {"text": "alpha", "key": "a1", "extra": 1}


def test_streaming_object_matches_materialized():
    materialized = list(dataset_rows(ds()))
    streamed = list(dataset_rows(ds().to_iterable_dataset()))
    assert streamed == materialized


def test_limit_and_columns():
    rows = list(dataset_rows(ds(), columns=["text"], limit=2))
    assert rows == [("train-000000000", {"text": "alpha"}),
                    ("train-000000001", {"text": "beta"})]


def test_content_ids_dedup_identical_rows():
    rows = list(dataset_rows(ds(), columns=["text"], ids="content"))
    assert rows[0][0] == content_id({"text": "alpha"})
    # rows 0 and 3 are identical content -> same id -> pump admits once
    assert rows[0][0] == rows[3][0]
    assert len({i for i, _ in rows}) == 3


def test_column_ids():
    assert [i for i, _ in dataset_rows(ds(), ids="key")] == ["a1", "b2", "c3", "a4"]


def test_callable_ids_trusted_as_is():
    rows = list(dataset_rows(ds(), ids=lambda ex: f"pg-{ex['key']}"))
    assert [i for i, _ in rows] == ["pg-a1", "pg-b2", "pg-c3", "pg-a4"]


def test_unknown_id_column_raises():
    with pytest.raises(KeyError):
        list(dataset_rows(ds(), ids="nope"))


def test_revision_or_token_with_loaded_object_raises():
    # silent no-op would void the index-id stability caveat (codex HIGH)
    with pytest.raises(ValueError, match="repo id string"):
        list(dataset_rows(ds(), revision="some-branch"))
    with pytest.raises(ValueError, match="repo id string"):
        list(dataset_rows(ds(), token="hf_x"))


def test_empty_columns_means_empty_rows_not_all():
    # columns=[] must not be treated as columns=None (codex MEDIUM)
    rows = list(dataset_rows(ds(), columns=[], limit=1))
    assert rows == [("train-000000000", {})]


def test_id_strategy_producing_none_raises_loudly():
    # a broken callable must not collide every row on "None" (codex MEDIUM)
    with pytest.raises(ValueError, match="produced no id"):
        list(dataset_rows(ds(), ids=lambda ex: None))


def test_id_column_need_not_be_in_columns():
    # id lookup happens on the full example, pre-filter (codex MEDIUM)
    rows = list(dataset_rows(ds(), columns=["text"], ids="key", limit=2))
    assert rows == [("a1", {"text": "alpha"}), ("b2", {"text": "beta"})]
