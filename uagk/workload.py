"""Synthetic cloud workload: job features, true service times, arrival processes.

Service-time model (per job with features x):
    E[S | x] = m(x)                       (depends on type, cpu, ram, input size)
    log S    = log m(x) - sigma(x)^2/2 + sigma(x) Z,   Z ~ N(0,1)
so runtimes are heteroscedastic and heavy-tailed: ML jobs are long and noisy,
web requests short and predictable. sigma(x) also grows with RAM footprint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TYPE_NAMES = ("web", "batch", "ml")
BASE = np.array([0.20, 0.70, 1.60])      # seconds, mean runtime scale per type
SIGMA = np.array([0.35, 0.55, 0.70])     # log-scale noise per type
MIX_DEFAULT = np.array([0.6, 0.3, 0.1])
MIX_HEAVY = np.array([0.3, 0.3, 0.4])
CPU_LEVELS = np.array([1, 2, 4, 8])
CPU_PROBS = np.array([0.4, 0.3, 0.2, 0.1])


@dataclass
class Jobs:
    t_arr: np.ndarray        # arrival times (sorted)
    jtype: np.ndarray        # 0 web, 1 batch, 2 ml
    cpu: np.ndarray
    ram: np.ndarray
    size: np.ndarray
    s_true: np.ndarray       # realised service time
    m_true: np.ndarray       # true conditional mean E[S|x]
    sig_true: np.ndarray     # true log-scale sigma(x)
    t_store: np.ndarray = None   # storage-write time of the result (T_storage)

    def __len__(self):
        return len(self.t_arr)


def sample_features(rng, n, mix):
    jtype = rng.choice(3, size=n, p=mix)
    cpu = rng.choice(CPU_LEVELS, size=n, p=CPU_PROBS)
    ram = rng.uniform(0.5, 16.0, size=n)
    size = rng.lognormal(0.0, 0.5, size=n)
    return jtype, cpu, ram, size


def true_mean(jtype, cpu, ram, size):
    return BASE[jtype] * size ** 0.8 * (1.0 + 0.3 * ram / 16.0) * (1.3 - 0.15 * np.log2(cpu))


def true_sigma(jtype, ram, noise_scale=1.0):
    return SIGMA[jtype] * (0.8 + 0.4 * ram / 16.0) * noise_scale


def make_jobs(rng, t_arr, mix_per_job, noise_scale=1.0) -> Jobs:
    """mix_per_job: array (n, 3) of type probabilities, or a single mix vector."""
    n = len(t_arr)
    mix_per_job = np.asarray(mix_per_job)
    if mix_per_job.ndim == 1:
        jtype, cpu, ram, size = sample_features(rng, n, mix_per_job)
    else:
        u = rng.random(n)
        cum = np.cumsum(mix_per_job, axis=1)
        jtype = (u[:, None] > cum).sum(axis=1).clip(0, 2)
        _, cpu, ram, size = sample_features(rng, n, MIX_DEFAULT)
    m = true_mean(jtype, cpu, ram, size)
    sig = true_sigma(jtype, ram, noise_scale)
    s = m * np.exp(-0.5 * sig ** 2 + sig * rng.standard_normal(n))
    t_store = 0.05 + 0.04 * size          # result write time, known at admission from input size
    return Jobs(t_arr, jtype, cpu, ram, size, s, m, sig, t_store)


def mean_service(mix, noise_scale=1.0, n=400_000, seed=123):
    rng = np.random.default_rng(seed)
    j = make_jobs(rng, np.zeros(n), mix, noise_scale)
    return float(j.s_true.mean()), float(j.s_true.var() / j.s_true.mean() ** 2)


# ---------------------------------------------------------------- arrivals
def _h2_interarrivals(rng, n, mean, ca2):
    """Balanced-means hyperexponential with given mean and SCV (ca2 >= 1)."""
    if ca2 <= 1.0 + 1e-9:
        return rng.exponential(mean, n)
    p = 0.5 * (1.0 + np.sqrt((ca2 - 1.0) / (ca2 + 1.0)))
    m1, m2 = mean / (2 * p), mean / (2 * (1 - p))
    pick = rng.random(n) < p
    return np.where(pick, rng.exponential(m1, n), rng.exponential(m2, n))


def arrivals_poisson(rng, rate, horizon):
    n = rng.poisson(rate * horizon * 1.05) + 100
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    while t[-1] < horizon:
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / rate, n))])
    return t[t < horizon]


def arrivals_nhpp(rng, rate_fn, rate_max, horizon):
    t = arrivals_poisson(rng, rate_max, horizon)
    keep = rng.random(len(t)) < rate_fn(t) / rate_max
    return t[keep]


def arrivals_mmpp_h2(rng, rate_base, rate_burst, hold_base, hold_burst, ca2, horizon):
    """Two-state modulated process; inside each state, H2 interarrivals with SCV ca2."""
    out, t, state = [], 0.0, 0
    while t < horizon:
        hold = rng.exponential(hold_base if state == 0 else hold_burst)
        end = min(t + hold, horizon)
        rate = rate_base if state == 0 else rate_burst
        n = int(rate * (end - t) * 1.5) + 20
        ts = t + np.cumsum(_h2_interarrivals(rng, n, 1.0 / rate, ca2))
        out.append(ts[ts < end])
        t, state = end, 1 - state
    return np.concatenate(out)


# ---------------------------------------------------------------- scenarios
SCENARIOS = ("steady", "diurnal", "bursty", "mixshift")


def make_scenario(name, rng, horizon, base_rate=20.0, noise_scale=1.0) -> Jobs:
    if name == "steady":
        t = arrivals_poisson(rng, base_rate, horizon)
        return make_jobs(rng, t, MIX_DEFAULT, noise_scale)
    if name == "diurnal":
        f = lambda x: base_rate * (1.0 + 0.5 * np.sin(2 * np.pi * x / 900.0))
        t = arrivals_nhpp(rng, f, base_rate * 1.5, horizon)
        return make_jobs(rng, t, MIX_DEFAULT, noise_scale)
    if name == "bursty":
        t = arrivals_mmpp_h2(rng, 0.8 * base_rate, 2.2 * base_rate, 240.0, 45.0, 3.0, horizon)
        return make_jobs(rng, t, MIX_DEFAULT, noise_scale)
    if name == "mixshift":
        rate = 0.75 * base_rate
        t = arrivals_poisson(rng, rate, horizon)
        heavy = (np.floor(t / 400.0) % 2) == 1
        mix = np.where(heavy[:, None], MIX_HEAVY[None, :], MIX_DEFAULT[None, :])
        return make_jobs(rng, t, mix, noise_scale)
    raise ValueError(name)


def scenario_mean_load(name, base_rate=20.0, noise_scale=1.0):
    """Long-run offered load (erlangs) – used only to size the static baseline."""
    es_d, _ = mean_service(MIX_DEFAULT, noise_scale)
    es_h, _ = mean_service(MIX_HEAVY, noise_scale)
    if name in ("steady", "diurnal"):
        return base_rate * es_d
    if name == "bursty":
        frac_burst = 45.0 / (240.0 + 45.0)
        lam = (1 - frac_burst) * 0.8 * base_rate + frac_burst * 2.2 * base_rate
        return lam * es_d
    if name == "mixshift":
        return 0.75 * base_rate * 0.5 * (es_d + es_h)
    raise ValueError(name)
