"""bucket_rows: glob discovery, relative-path ids, skip-before-read, prefetch.

All on local tmp dirs (fsspec local filesystem) — no network.
"""

import pytest

from saturate import bucket_rows


def make_tree(tmp_path):
    (tmp_path / "scans" / "vol1").mkdir(parents=True)
    (tmp_path / "scans" / "vol2").mkdir(parents=True)
    (tmp_path / "scans" / "vol1" / "p1.png").write_bytes(b"one")
    (tmp_path / "scans" / "vol1" / "p2.png").write_bytes(b"two")
    (tmp_path / "scans" / "vol2" / "p3.png").write_bytes(b"three")
    (tmp_path / "scans" / "notes.txt").write_bytes(b"not an image")
    return tmp_path / "scans"


def test_glob_relative_ids_and_bytes(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/**/*.png"))
    assert [rid for rid, _ in rows] == ["vol1/p1.png", "vol1/p2.png", "vol2/p3.png"]
    assert rows[0][1]["bytes"] == b"one"
    assert rows[0][1]["path"].endswith("vol1/p1.png")


def test_glob_excludes_non_matching_and_dirs(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/*"))
    # notes.txt matches; the vol1/vol2 DIRECTORIES must not become rows
    assert [rid for rid, _ in rows] == ["notes.txt"]


def test_skip_filters_before_read(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/**/*.png", skip={"vol1/p1.png", "vol2/p3.png"}))
    assert [rid for rid, _ in rows] == ["vol1/p2.png"]


def test_read_false_lists_only(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/**/*.png", read=False))
    assert all("bytes" not in row for _, row in rows)
    assert len(rows) == 3


def test_limit_applies_after_skip(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/**/*.png", skip={"vol1/p1.png"}, limit=1))
    assert [rid for rid, _ in rows] == ["vol1/p2.png"]


def test_prefetch_preserves_order(tmp_path):
    root = tmp_path / "many"
    root.mkdir()
    for i in range(37):
        (root / f"f{i:03d}.bin").write_bytes(bytes([i]))
    rows = list(bucket_rows(f"{root}/*.bin", prefetch=5))
    assert [rid for rid, _ in rows] == [f"f{i:03d}.bin" for i in range(37)]
    assert all(row["bytes"] == bytes([i]) for i, (_, row) in enumerate(rows))


def test_callable_ids(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/**/*.png", ids=lambda p: p.rsplit("/", 1)[-1]))
    assert [rid for rid, _ in rows] == ["p1.png", "p2.png", "p3.png"]


def test_empty_glob_yields_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert list(bucket_rows(f"{tmp_path}/empty/*.png")) == []


def test_limit_zero_and_negative_select_nothing(tmp_path):
    root = make_tree(tmp_path)
    assert list(bucket_rows(f"{root}/**/*.png", limit=0)) == []
    assert list(bucket_rows(f"{root}/**/*.png", limit=-3)) == []


def test_unsupported_ids_string_raises(tmp_path):
    root = make_tree(tmp_path)
    with pytest.raises(ValueError, match="path.*or a callable"):
        list(bucket_rows(f"{root}/**/*.png", ids="content"))


def test_bracket_glob_gets_relative_ids(tmp_path):
    root = make_tree(tmp_path)
    rows = list(bucket_rows(f"{root}/[vw]*/*.png", read=False))
    # `[` must count as glob magic: ids relative to scans/, not absolute paths
    assert [rid for rid, _ in rows] == ["vol1/p1.png", "vol1/p2.png", "vol2/p3.png"]


def test_literal_brace_dirname_is_not_magic(tmp_path):
    d = tmp_path / "scan{raw}"
    d.mkdir()
    (d / "page.png").write_bytes(b"pg")
    rows = list(bucket_rows(f"{d}/*.png"))
    # fsspec doesn't brace-expand: { is a literal, base must include the dir
    assert [rid for rid, _ in rows] == ["page.png"]


def test_chained_uri_zip(tmp_path):
    import zipfile

    zp = tmp_path / "arch.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("a.txt", "alpha")
        z.writestr("sub/b.txt", "beta")
    rows = list(bucket_rows(f"zip://**/*.txt::{zp}"))
    assert sorted(rid for rid, _ in rows) == ["a.txt", "sub/b.txt"]
    assert dict(rows)["a.txt"]["bytes"] == b"alpha"
