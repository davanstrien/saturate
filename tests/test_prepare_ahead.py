"""prepare_ahead: a bounded, order-preserving look-ahead over an (id, row) stream that runs
`fn` on an executor of the caller's choosing (threads by default, processes on request)."""

import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from saturate import prepare_ahead


def double(row):
    return {**row, "n": row["n"] * 2}


def rows(n):
    return [(str(i), {"n": i}) for i in range(n)]


def test_order_is_preserved_with_a_thread_pool():
    def jittered(row):
        time.sleep(0.01 * (5 - row["n"] % 5))  # later rows finish first
        return double(row)

    out = list(prepare_ahead(rows(20), jittered, workers=4))
    assert out == [(str(i), {"n": i * 2}) for i in range(20)]


def test_works_with_a_process_pool():
    with ProcessPoolExecutor(2) as pool:
        out = list(prepare_ahead(rows(10), double, workers=2, executor=pool))
    assert out == [(str(i), {"n": i * 2}) for i in range(10)]


def test_look_ahead_is_bounded_by_workers():
    started = 0
    lock = threading.Lock()

    def counting(row):
        nonlocal started
        with lock:
            started += 1
        return row

    consumed = 0
    for _ in prepare_ahead(rows(50), counting, workers=3):
        consumed += 1
        time.sleep(0.005)  # a slow consumer: the look-ahead must not run away
        assert started <= consumed + 3, (started, consumed)
    assert consumed == 50


def test_exception_in_fn_surfaces_at_the_consumer():
    def boom(row):
        if row["n"] == 5:
            raise ValueError("bad row 5")
        return row

    it = prepare_ahead(rows(20), boom, workers=2)
    assert [r["n"] for _, r in (next(it) for _ in range(5))] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="bad row 5"):
        next(it)


def test_a_caller_owned_executor_is_not_shut_down():
    with ThreadPoolExecutor(2) as pool:
        assert len(list(prepare_ahead(rows(6), double, executor=pool))) == 6
        assert pool.submit(double, {"n": 1}).result() == {"n": 2}  # still usable
