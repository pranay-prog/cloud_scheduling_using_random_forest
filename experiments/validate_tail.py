"""E4: accuracy of the Allen-Cunneen mean and the exponential tail (N2) vs exact simulation.

Stationary G/G/k queues with gamma inter-arrivals (SCV ca2) and log-normal
service times (SCV cs2), simulated with the FCFS heap recursion
    W_n = max(0, F_min - t_n),  F_min <- start_n + S_n
for 400k customers per cell (first 20k discarded).
"""
from __future__ import annotations

import heapq
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uagk.queueing import wq_ggk, tail_ggk  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def simulate_ggk(k, lam, ca2, cs2, n=400_000, burn=20_000, seed=0):
    rng = np.random.default_rng(seed)
    shape = 1.0 / ca2
    ia = rng.gamma(shape, (1.0 / lam) / shape, n)
    s2 = np.log1p(cs2)
    s = np.exp(-0.5 * s2 + np.sqrt(s2) * rng.standard_normal(n))   # mean 1
    t = np.cumsum(ia)
    free = [0.0] * k
    w = np.empty(n)
    for i in range(n):
        f = free[0]
        wi = f - t[i] if f > t[i] else 0.0
        w[i] = wi
        heapq.heapreplace(free, t[i] + wi + s[i])
    return w[burn:]


def main():
    rows = []
    for k, rho, ca2, cs2 in itertools.product([5, 20], [0.7, 0.85, 0.95],
                                              [0.5, 1.0, 3.0], [0.5, 1.0, 2.5]):
        lam = rho * k
        w = simulate_ggk(k, lam, ca2, cs2, seed=k * 1000 + int(rho * 100))
        wq_sim = w.mean()
        wq_ac = wq_ggk(k, lam, 1.0, ca2, cs2)
        for mult in [1.0, 2.0, 4.0]:          # tau as multiples of the mean service time
            p_sim = np.mean(w > mult)
            p_ap = float(tail_ggk(mult, k, lam, 1.0, ca2, cs2))
            rows.append(dict(k=k, rho=rho, ca2=ca2, cs2=cs2, tau=mult, wq_sim=wq_sim,
                             wq_ac=wq_ac, p_sim=p_sim, p_approx=p_ap))
        print(k, rho, ca2, cs2, round(wq_sim, 3), round(wq_ac, 3), flush=True)
    df = pd.DataFrame(rows)
    df["wq_rel_err"] = (df.wq_ac - df.wq_sim) / df.wq_sim
    df["tail_abs_err"] = df.p_approx - df.p_sim
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "tail_validation.csv", index=False)


if __name__ == "__main__":
    main()
