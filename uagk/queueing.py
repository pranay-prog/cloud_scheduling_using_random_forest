"""Queueing formulas used by the planner.

Erlang-C (M/M/k), the Allen-Cunneen G/G/k correction, and the exponential
tail approximation of the waiting time used for percentile sizing (N2).
"""
from __future__ import annotations

import numpy as np


def erlang_c_all(a: float, k_max: int) -> np.ndarray:
    """P_W (probability of waiting) for k = 1..k_max servers, offered load a = lambda/mu.

    Uses the numerically stable Erlang-B recursion
        B(0) = 1,  B(n) = a B(n-1) / (n + a B(n-1)),
        C(k) = B(k) / (1 - rho (1 - B(k))),   rho = a / k.
    Entries with k <= a (unstable) are returned as 1.0.
    """
    out = np.ones(k_max)
    b = 1.0
    for n in range(1, k_max + 1):
        b = a * b / (n + a * b)
        rho = a / n
        if rho < 1.0:
            out[n - 1] = b / (1.0 - rho * (1.0 - b))
    return out


def erlang_c(k: int, a: float) -> float:
    if k <= 0 or a >= k:
        return 1.0
    return float(erlang_c_all(a, k)[k - 1])


def wq_mmk(k: int, lam: float, mu: float) -> float:
    """Mean waiting time in queue for M/M/k."""
    if lam >= k * mu:
        return np.inf
    return erlang_c(k, lam / mu) / (k * mu - lam)


def allen_cunneen_v(ca2: float, cs2: float) -> float:
    return 0.5 * (ca2 + cs2)


def wq_ggk(k: int, lam: float, mu: float, ca2: float, cs2: float) -> float:
    """Allen-Cunneen approximation of the mean wait for G/G/k."""
    return allen_cunneen_v(ca2, cs2) * wq_mmk(k, lam, mu)


def tail_ggk(tau, k: int, lam: float, mu: float, ca2: float, cs2: float):
    """Approximate P(W_q > tau) for G/G/k.

    Exponential tail  P_W * exp(-(k mu - lam) tau / V).  With V = 1 this is the
    exact M/M/k result (the Erlang-C service-level formula); for V != 1 its mean
    equals the Allen-Cunneen mean wait, so the two approximations are consistent.
    """
    if lam >= k * mu:
        return np.ones_like(np.asarray(tau, dtype=float))
    v = max(allen_cunneen_v(ca2, cs2), 1e-6)
    pw = erlang_c(k, lam / mu)
    tau = np.asarray(tau, dtype=float)
    return np.where(tau < 0, 1.0, pw * np.exp(-(k * mu - lam) * np.maximum(tau, 0) / v))
