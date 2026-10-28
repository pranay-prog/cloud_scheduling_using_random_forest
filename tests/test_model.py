import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uagk.queueing import erlang_c, erlang_c_all, wq_mmk, wq_ggk, tail_ggk
from uagk.workload import Jobs, MIX_DEFAULT, make_jobs
from uagk.oracle import ServiceOracle, training_jobs, evaluate
from uagk.policies import StaticPolicy, make_policy
from uagk.simulator import Cluster, SimConfig, summarize


def test_erlang_c_known_values():
    # k=2, a=1: P_W = 1/3 ; k=1: P_W = rho
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3, rel=1e-9)
    assert erlang_c(1, 0.7) == pytest.approx(0.7, rel=1e-9)
    # textbook: k=10, a=8 -> 0.4092
    assert erlang_c(10, 8.0) == pytest.approx(0.4092, abs=5e-4)


def test_erlang_c_vector_matches_scalar_and_unstable_is_one():
    v = erlang_c_all(7.5, 20)
    for k in range(8, 21):
        assert v[k - 1] == pytest.approx(erlang_c(k, 7.5))
    assert np.all(v[:7] == 1.0)


def test_allen_cunneen_reduces_to_mmk():
    assert wq_ggk(12, 10.0, 1.0, 1.0, 1.0) == pytest.approx(wq_mmk(12, 10.0, 1.0))


def test_tail_mean_matches_allen_cunneen():
    # integral of the tail = mean wait
    k, lam, mu, ca2, cs2 = 12, 10.0, 1.0, 2.0, 3.0
    taus = np.linspace(0, 200, 200_001)
    integral = np.trapezoid(tail_ggk(taus, k, lam, mu, ca2, cs2), taus)
    assert integral == pytest.approx(wq_ggk(k, lam, mu, ca2, cs2), rel=1e-3)


def _mmk_jobs(seed, lam, mu, horizon):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1 / lam, int(lam * horizon * 1.1)))
    t = t[t < horizon]
    s = rng.exponential(1 / mu, len(t))
    z = np.zeros(len(t))
    return Jobs(t, z.astype(int), z + 1, z, z + 1, s, z + 1 / mu, z)


def test_simulator_matches_mmk_theory():
    lam, mu, k = 9.0, 1.0, 10
    cfg = SimConfig(horizon=60_000, warmup=2000, k0=k, k_min=k, k_max=k, b_max=10**9)
    res = Cluster(_mmk_jobs(3, lam, mu, cfg.horizon), None, StaticPolicy(k), cfg).run()
    sim_wq = np.mean(res.wait)
    sim_pw = np.mean(res.wait > 1e-12)
    assert sim_wq == pytest.approx(wq_mmk(k, lam, mu), rel=0.08)
    assert sim_pw == pytest.approx(erlang_c(k, lam / mu), rel=0.05)
    # P(W > 1 s) against the exact M/M/k tail
    assert np.mean(res.wait > 1.0) == pytest.approx(
        float(tail_ggk(1.0, k, lam, mu, 1.0, 1.0)), rel=0.1)


def test_vm_hours_static_equals_k_times_span():
    cfg = SimConfig(horizon=1000, warmup=100, k0=7, k_min=7, k_max=7)
    res = Cluster(_mmk_jobs(1, 3.0, 1.0, cfg.horizon), None, StaticPolicy(7), cfg).run()
    assert res.vm_seconds == pytest.approx(7 * 900, rel=1e-9)


def test_oracle_is_calibrated_and_correction_reduces_bias():
    o = ServiceOracle(level=0, seed=0).fit(training_jobs(0))
    test = make_jobs(np.random.default_rng(5), np.zeros(15000), MIX_DEFAULT)
    r = evaluate(o, test)
    assert 0.85 <= r["cover90"] <= 0.95
    assert abs(r["cs2_corrected"] - r["cs2_true"]) < abs(r["cs2_naive"] - r["cs2_true"])


def test_scaling_respects_bounds_and_boot_delay():
    cfg = SimConfig(horizon=400, warmup=50, k0=4, k_min=2, k_max=12, t_scale=10)
    jobs = _mmk_jobs(2, 30.0, 1.0, cfg.horizon)   # heavy overload forces scale-up

    class Up:
        def decide(self, obs, pred):
            return 50

    res = Cluster(jobs, None, Up(), cfg).run()
    kt = res.k_trace
    assert (kt[:, 1] + kt[:, 2]).max() <= 12
    # first control step at t=4 orders VMs; none can be active before t=14
    assert kt[kt[:, 0] < 14, 1].max() == 4
