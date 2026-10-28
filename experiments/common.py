"""Shared runner: one (scenario, policy, seed) simulation -> metrics dict."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uagk.oracle import ServiceOracle, training_jobs, evaluate
from uagk.policies import make_policy
from uagk.simulator import Cluster, SimConfig, summarize
from uagk.workload import make_scenario, scenario_mean_load

_ORACLES = {}


def get_oracle(seed, level=0, noise=1.0):
    key = (seed, level, noise)
    if key not in _ORACLES:
        _ORACLES[key] = ServiceOracle(level=level, seed=seed).fit(training_jobs(seed, noise_scale=noise))
    return _ORACLES[key]


def run_one(scenario, policy, seed, level=0, noise=1.0, base_rate=20.0, policy_kw=None, **cfg_kw):
    t0 = time.time()
    cfg = SimConfig(**cfg_kw)
    rng = np.random.default_rng(seed)
    jobs = make_scenario(scenario, rng, cfg.horizon, base_rate=base_rate, noise_scale=noise)
    oracle = get_oracle(seed, level, noise)
    pred = oracle.predict(jobs)
    static_k = math.ceil(scenario_mean_load(scenario, base_rate, noise) / 0.7)
    cfg.k0 = static_k
    pol = make_policy(policy, seed=seed, static_k=static_k, **(policy_kw or {}))
    res = Cluster(jobs, pred, pol, cfg).run()
    out = summarize(res, cfg.t_sla)
    out.update(scenario=scenario, policy=policy, seed=seed, level=level, noise=noise,
               t_scale=cfg.t_scale, **(policy_kw or {}), runtime_s=round(time.time() - t0, 2))
    if hasattr(pol, "diag") and pol.diag:
        d = np.array(pol.diag)
        out["mean_cs2_est"] = float(d[:, 3].mean())
        out["mean_ca2_est"] = float(d[:, 2].mean())
    out["cs2_trace_true"] = float(jobs.s_true.var() / jobs.s_true.mean() ** 2)
    return out
