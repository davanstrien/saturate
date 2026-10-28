"""The slow tier for Auto (`pytest --runslow -m slow`): several clients sharing one engine, as sharded Jobs
pumping one endpoint do, and wide seed sweeps of the single-client scenarios. Run before a
release and for any controller change; bench/controller_sim.py prints the single-client table."""

import random
import statistics

import pytest
from sim_engine import (
    LINEAR,
    Engine,
    cuts_per_minute,
    failed_share,
    run,
    run_shared,
    share_of_best,
    shared_share_of_best,
)

from saturate.controller import Auto

pytestmark = pytest.mark.slow

REAL_SGLANG = [(32, 64.0), (64, 64.0), (256, 160.0)]  # 2026-10-28 Jobs: flat 32-64, rising to the cap
SEEDS = range(6)


def starts_for(clients: int, together: bool, seed: int) -> list[int]:
    """Every client at tick 0, or each at a random tick in the first minute."""
    rng = random.Random(seed)
    return [0] * clients if together else [rng.randrange(30) for _ in range(clients)]


def shared(engine_kw: dict, clients: int, together: bool):
    """(summed share of the best fixed window, cuts per client-minute, mean failed share, mean total
    window) after the first 100 ticks, averaged over SEEDS."""
    best, cuts, failed, total = [], [], [], []
    for seed in SEEDS:
        engine = Engine(**engine_kw, seed=seed)
        autos = [Auto(seed=seed * 100 + k) for k in range(clients)]
        runs = run_shared(autos, engine, 600, starts_for(clients, together, seed))
        best.append(shared_share_of_best(runs, 100))
        cuts.append(cuts_per_minute(runs, 100))
        failed.append(statistics.mean(failed_share(r) for r in runs))
        total.append(statistics.mean(sum(r[i].limit for r in runs) for i in range(100, 600)))
    return statistics.mean(best), statistics.mean(cuts), statistics.mean(failed), statistics.mean(total)


@pytest.mark.parametrize("together", [True, False], ids=["together", "staggered"])
@pytest.mark.parametrize("clients", [2, 4, 8])
def test_clients_sharing_an_engine_with_gauges_do_not_saw_tooth(clients, together):
    """Clients that start together open their trials together; on a shared queue gauge they then
    all cut at once and all grow back (without jitter: 4-6 cuts a minute, 90% of the best).
    Jittered trials fall out of step: few cuts, the queue near the engine's cap."""
    engine_kw = dict(curve=REAL_SGLANG, cap=256, gauges=True, noise=0.05)
    best, cuts, _, total = shared(engine_kw, clients, together)
    assert best >= 0.97, best
    assert cuts <= 0.5, cuts
    assert total <= 256 * 1.25, total  # the queue band on top of the cap, not a herd's overshoot


@pytest.mark.parametrize("together", [True, False], ids=["together", "staggered"])
@pytest.mark.parametrize("clients", [2, 4, 8])
@pytest.mark.parametrize("rate_limit", [200, 500])
def test_clients_sharing_a_hosted_api_limit_reject_few_rows(rate_limit, clients, together):
    """A hosted API's 429 limit shared by several clients: together they stay near it and few
    rows are rejected (one client alone: about 1-2%)."""
    engine_kw = dict(curve=[(1024, 400.0)], cap=1024, rate_limit=rate_limit, gauges=False, noise=0.05)
    best, _, failed, total = shared(engine_kw, clients, together)
    assert best >= 0.92, best
    assert failed <= 0.03, failed
    assert total <= rate_limit * 1.05, total


@pytest.mark.xfail(strict=True, reason="known: blind clients sharing an engine each judge their own "
                   "throughput, and a larger share of a fixed total looks like a gain")
def test_blind_clients_sharing_an_engine_keep_the_queue_bounded():
    """Without gauges or 429s, nothing a client sees tells another client's requests from the
    engine's own limits: each one's throughput rises with its own window, so all grow to
    max_limit (4 clients: a total window of about 1,900 on a 256-request engine, latency x7).
    No surveyed controller separates the two blind; this pins the gap until one does."""
    engine_kw = dict(curve=LINEAR, cap=256, gauges=False, noise=0.05)
    _, _, _, total = shared(engine_kw, 4, together=True)
    assert total <= 256 * 2, total


def test_one_client_on_the_shared_driver_matches_the_single_driver():
    """run_shared with one client is run: the same windows, reasons and deliveries."""
    for seed in SEEDS:
        engine = Engine(curve=REAL_SGLANG, cap=256, gauges=False, noise=0.05, seed=seed)
        alone = run(Auto(seed=seed), engine, 300)
        (together,) = run_shared([Auto(seed=seed)], Engine(curve=REAL_SGLANG, cap=256, gauges=False,
                                                                 noise=0.05, seed=seed), 300)
        assert [(t.limit, t.reason) for t in alone] == [(t.limit, t.reason) for t in together]


@pytest.mark.parametrize("rate_limit", [64, 100, 127, 200, 500])
def test_a_hosted_api_limit_over_many_seeds(rate_limit):
    """The fast tier's hosted-API test over 40 noise draws and trial schedules."""
    best, failed = [], []
    for seed in range(40):
        engine = Engine(curve=[(1024, 400.0)], cap=1024, rate_limit=rate_limit, gauges=False, noise=0.05,
                        seed=seed)
        ticks = run(Auto(seed=seed), engine, 600)
        best.append(share_of_best(ticks[100:]))
        failed.append(failed_share(ticks))
    assert statistics.mean(best) >= 0.94 and min(best) >= 0.88, sorted(best)[:5]
    assert statistics.mean(failed) <= 0.03 and max(failed) <= 0.045, sorted(failed)[-5:]


@pytest.mark.parametrize("gauges", [False, True])
@pytest.mark.parametrize("curve,cap", [(REAL_SGLANG, 256), (LINEAR, 256), ([(100, 200.0)], 100)])
def test_single_client_engines_over_many_seeds(curve, cap, gauges):
    """One client, 30 noise draws and trial schedules: the mean and the worst run stay near the
    best fixed window."""
    shares = []
    for seed in range(30):
        engine = Engine(curve=curve, cap=cap, gauges=gauges, noise=0.07, seed=seed)
        shares.append(share_of_best(run(Auto(seed=seed), engine, 600)[100:]))
    assert statistics.mean(shares) >= 0.96 and min(shares) >= 0.9, sorted(shares)[:5]
