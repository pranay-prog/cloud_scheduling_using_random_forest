"""Random-Forest service-time oracle with per-request uncertainty (step 3, N1).

Two forests, both trained on log runtime:
  * mean forest      mu(x)   ~ E[log S | x]
  * variance forest  s2(x)   ~ E[(log S - mu_oob(x))^2 | x], fitted on
                              out-of-bag residuals, so s2 covers both runtime
                              noise and the forest's own generalisation error.
Under a log-normal residual model the predictive distribution of S is
LogNormal(mu, s2), giving
  mean  S_hat   = exp(mu + s2/2)
  var   sig2    = (exp(s2) - 1) exp(2 mu + s2)
  P95           = exp(mu + 1.645 sqrt(s2)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .workload import Jobs, make_jobs, MIX_DEFAULT, MIX_HEAVY, arrivals_poisson

# Feature levels for the predictor-quality sweep (E2): higher level = fewer features.
FEATURE_LEVELS = {
    0: ("type", "cpu", "ram", "size"),
    1: ("type", "cpu", "ram"),
    2: ("type", "cpu"),
    3: ("type",),
    4: (),
}


def featurize(jobs: Jobs, level: int = 0) -> np.ndarray:
    cols = []
    names = FEATURE_LEVELS[level]
    if "type" in names:
        cols += [(jobs.jtype == i).astype(float) for i in range(3)]
    if "cpu" in names:
        cols.append(np.log2(jobs.cpu).astype(float))
    if "ram" in names:
        cols.append(jobs.ram)
    if "size" in names:
        cols.append(np.log(jobs.size))
    if not cols:
        cols.append(np.zeros(len(jobs)))
    return np.column_stack(cols)


@dataclass
class Prediction:
    mu: np.ndarray       # log-scale mean
    s2: np.ndarray       # log-scale variance
    mean: np.ndarray     # E[S|x]
    var: np.ndarray      # Var(S|x)
    p95: np.ndarray


class ServiceOracle:
    def __init__(self, level: int = 0, n_estimators: int = 150, seed: int = 0):
        self.level = level
        self.seed = seed
        self.rf_mu = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=20, oob_score=True,
            max_features=0.7, n_jobs=1, random_state=seed)
        self.rf_s2 = RandomForestRegressor(
            n_estimators=80, min_samples_leaf=60, max_features=0.7, n_jobs=1,
            random_state=seed + 1)

    def fit(self, jobs: Jobs):
        X = featurize(jobs, self.level)
        y = np.log(jobs.s_true)
        self.rf_mu.fit(X, y)
        resid2 = (y - self.rf_mu.oob_prediction_) ** 2
        self.rf_s2.fit(X, resid2)
        return self

    def predict(self, jobs: Jobs) -> Prediction:
        X = featurize(jobs, self.level)
        mu = self.rf_mu.predict(X)
        s2 = np.maximum(self.rf_s2.predict(X), 1e-4)
        mean = np.exp(mu + 0.5 * s2)
        var = np.expm1(s2) * np.exp(2 * mu + s2)
        p95 = np.exp(mu + 1.6449 * np.sqrt(s2))
        return Prediction(mu, s2, mean, var, p95)


def training_jobs(seed: int, n: int = 6000, noise_scale: float = 1.0) -> Jobs:
    """Historical jobs used to train the oracle (mixed, independent of test traces)."""
    rng = np.random.default_rng(10_000 + seed)
    t = np.sort(rng.uniform(0, 1, n))
    mix = np.where((rng.random(n) < 0.7)[:, None], MIX_DEFAULT[None, :], MIX_HEAVY[None, :])
    return make_jobs(rng, t, mix, noise_scale)


def evaluate(oracle: ServiceOracle, jobs: Jobs) -> dict:
    """Held-out accuracy and calibration of the oracle."""
    p = oracle.predict(jobs)
    s = jobs.s_true
    ly = np.log(s)
    r2_log = 1 - np.mean((ly - p.mu) ** 2) / np.var(ly)
    r2 = 1 - np.mean((s - p.mean) ** 2) / np.var(s)
    z = 1.6449 * np.sqrt(p.s2)
    cover90 = np.mean((ly >= p.mu - z) & (ly <= p.mu + z))
    cs2_true = s.var() / s.mean() ** 2
    cs2_naive = p.mean.var() / p.mean.mean() ** 2
    cs2_corr = (p.mean.var() + p.var.mean()) / p.mean.mean() ** 2
    return dict(r2=r2, r2_log=r2_log, mae=float(np.mean(np.abs(s - p.mean))),
                cover90=cover90, cs2_true=cs2_true, cs2_naive=cs2_naive,
                cs2_corrected=cs2_corr)
