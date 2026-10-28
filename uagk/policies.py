"""Scaling policies: the proposed planner, its ablations and baselines.

All model-based policies share the same actuation (step 10): scale up straight
to k*, scale down only after `down_patience` consecutive intervals with a lower
target, to the largest target seen in that period. This keeps the comparison
about the capacity model, not the actuator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .queueing import erlang_c_all
from .simulator import Obs


class Hysteresis:
    def __init__(self, down_patience: int = 3):
        self.p = down_patience
        self.pending = []

    def apply(self, target, current):
        if target >= current:
            self.pending = []
            return target
        self.pending.append(target)
        if len(self.pending) >= self.p:
            t = max(self.pending)
            self.pending = []
            return t
        return current


class StaticPolicy:
    name = "static"

    def __init__(self, k):
        self.k = k

    def decide(self, obs: Obs, pred):
        return self.k


class HPAPolicy:
    """Kubernetes-HPA-style utilisation targeting: k = ceil(k * util / target)."""
    name = "hpa"

    def __init__(self, target_util=0.7, down_patience=3):
        self.target = target_util
        self.h = Hysteresis(down_patience)

    def decide(self, obs: Obs, pred):
        current = obs.active + obs.booting - obs.retiring
        util = obs.util
        if obs.queue_len > 0:
            util = max(util, 1.0)
        desired = math.ceil(max(obs.active, 1) * util / self.target)
        desired = max(desired, obs.active if obs.queue_len > 0 else 0)
        return self.h.apply(desired, current)


@dataclass
class ModelSpec:
    service: str = "rf_corrected"   # "empirical" | "rf_naive" | "rf_corrected"
    criterion: str = "tail"         # "tail" (N2) | "mean_alpha" (original prototype) | "erlang_sl"
    variability: bool = True        # Allen-Cunneen V; False => M/M/k (V = 1)
    backlog: str = "projected"      # "none" | "current" | "projected" (N3)
    eps: float = 0.05               # allowed P(T_resp > T_SLA)
    eps_queue: float = 0.02         # queueing budget on top of the irreducible part
    objective: str = "min_wait"     # "min_wait": argmin lam*E[W_B+W_q] + c*k  s.t. SLA
                                    # "min_cost": smallest k that meets the SLA
    wait_cost: float = 0.10         # c: VM cost expressed in requests-waiting per VM
    samples_per_job: int = 3
    max_pool: int = 1500


class ModelPolicy:
    """Steps 2-9 of the pipeline: characterise, model the delay, solve for k*."""

    def __init__(self, name: str, spec: ModelSpec, seed: int = 0, down_patience: int = 3):
        self.name = name
        self.s = spec
        self.rng = np.random.default_rng(seed)
        self.h = Hysteresis(down_patience)
        self.last_mu = None
        self.last_cs2 = 1.0
        self.diag = []   # (t, lam, ca2, cs2, k*) for analysis

    # ---------------- step 2: arrivals
    @staticmethod
    def _arrival_stats(obs: Obs):
        a = obs.arr_times
        if len(a) < 10:
            return max(len(a), 1) / obs.cfg.window, 1.0
        ia = np.diff(a)
        lam = len(a) / obs.cfg.window
        ca2 = float(ia.var() / ia.mean() ** 2) if ia.mean() > 0 else 1.0
        return lam, float(np.clip(ca2, 0.2, 20.0))

    # ---------------- steps 3-4: service moments
    def _service_stats(self, obs: Obs, pred):
        """Returns mu, Cs^2 and a sample pool of post-queue time S + T_storage."""
        s = self.s
        st_mean = float(obs.t_store.mean()) if len(obs.t_store) else 0.0
        if s.service == "empirical":
            d = obs.done_s
            if len(d) < 10:
                return self.last_mu, self.last_cs2, (d + st_mean) if len(d) else None
            es = d.mean()
            cs2 = d.var() / es ** 2
            return 1.0 / es, cs2, d + st_mean
        sl = obs.pred_idx
        mean = pred.mean[sl]
        if len(mean) < 5:
            return self.last_mu, self.last_cs2, None
        es = mean.mean()
        if s.service == "rf_naive":
            var = mean.var()                       # point predictions only
            pool = mean + obs.t_store
        else:                                      # N1: law of total variance
            var = mean.var() + pred.var[sl].mean()
            mu_l, s2_l = pred.mu[sl], pred.s2[sl]
            n = len(mu_l)
            reps = max(1, min(s.samples_per_job, s.max_pool // max(n, 1)))
            idx = np.repeat(np.arange(n), reps)
            if len(idx) > s.max_pool:
                idx = self.rng.choice(idx, s.max_pool, replace=False)
            pool = np.exp(mu_l[idx] + np.sqrt(s2_l[idx]) * self.rng.standard_normal(len(idx)))
            pool = pool + obs.t_store[idx]
        return 1.0 / es, var / es ** 2, pool

    # ---------------- steps 5-9
    def decide(self, obs: Obs, pred):
        s, cfg = self.s, obs.cfg
        current = obs.active + obs.booting - obs.retiring
        lam, ca2 = self._arrival_stats(obs)
        mu, cs2, pool = self._service_stats(obs, pred)
        if mu is None:
            return current
        self.last_mu, self.last_cs2 = mu, cs2
        v = 0.5 * (ca2 + cs2) if s.variability else 1.0
        a = lam / mu

        # step 5: backlog (N3 = projected to the moment new capacity is ready)
        k_eff = obs.active + obs.booting - obs.retiring
        if s.backlog == "none":
            backlog = 0.0
        elif s.backlog == "current":
            backlog = float(obs.queue_len)
        else:
            # booting VMs do not serve yet: backlog grows at lam - active*mu until they are ready
            backlog = obs.queue_len + max(0.0, lam - obs.active * mu) * cfg.t_scale

        k_lo = max(cfg.k_min, int(math.floor(a)) + 1)
        ks = np.arange(k_lo, cfg.k_max + 1)
        if len(ks) == 0:
            return cfg.k_max
        pw = erlang_c_all(a, cfg.k_max)[ks - 1]          # step 6
        drain = backlog / (ks * mu)
        st_mean = float(obs.t_store.mean()) if len(obs.t_store) else 0.0
        es = 1.0 / mu + st_mean                           # mean post-queue time S + T_storage
        wq = v * pw / (ks * mu - lam)                     # step 7: Allen-Cunneen mean wait

        if s.criterion == "mean_alpha":                   # original prototype: mean W_total check
            alpha = 0.8 if ca2 > 1.5 else 1.0
            ok = wq + drain + es <= alpha * cfg.t_sla
        elif s.criterion == "erlang_sl":                  # classic Erlang-C service level
            tau = cfg.t_sla - es - drain
            p = np.where(tau < 0, 1.0, pw * np.exp(-(ks * mu - lam) * np.maximum(tau, 0) / v))
            ok = p <= s.eps
        else:                                   # N2: P(W_B + W_q + S + T_storage > T_SLA) <= eps
            if pool is None:
                pool = np.array([es])
            theta = (ks * mu - lam) / v
            tau = cfg.t_sla - pool[None, :] - drain[:, None]
            p = np.where(tau < 0, 1.0, pw[:, None] * np.exp(-theta[:, None] * np.maximum(tau, 0)))
            # SLA budget = irreducible part P(S + T_st > T_SLA) (no k can fix it) + queueing budget
            p_floor = float(np.mean(pool > cfg.t_sla))
            ok = p.mean(axis=1) <= max(s.eps, p_floor + s.eps_queue)
        hit = np.flatnonzero(ok)
        if len(hit) == 0:
            k_star = cfg.k_max
        elif s.objective == "min_wait":
            # step 9: minimise total waiting time per second plus VM cost.
            #   steady-state part (Little's law):        lam * Wq_GGk(k)
            #   transient backlog part (fluid drain):     Q~^2 / (2 (k mu - lam)) spread over the
            #   shortest billed lifetime of a new VM      L = T_scale + patience * T_ctrl
            horizon = cfg.t_scale + self.h.p * cfg.t_ctrl
            j = lam * wq + backlog ** 2 / (2.0 * (ks * mu - lam) * horizon) + s.wait_cost * ks
            j = np.where(ok, j, np.inf)
            k_star = int(ks[int(np.argmin(j))])
        else:
            k_star = int(ks[hit[0]])
        self.diag.append((obs.t, lam, ca2, cs2, k_star))
        return self.h.apply(k_star, current)                # step 10


POLICY_SPECS = {
    # external baselines
    "erlang_c":   ModelSpec(service="empirical", criterion="erlang_sl", variability=False,
                            backlog="none", objective="min_cost"),
    # the two variants of the original prototype (report v1)
    "proto_emp":  ModelSpec(service="empirical", criterion="mean_alpha", backlog="none", objective="min_cost"),
    "proto_rf":   ModelSpec(service="rf_naive", criterion="mean_alpha", backlog="none", objective="min_cost"),
    # proposed constraint handling but cheapest feasible k (previous version of the model)
    "proposed_mc": ModelSpec(service="rf_corrected", criterion="tail", backlog="projected", objective="min_cost"),
    # proposed and ablations
    "proposed":   ModelSpec(service="rf_corrected", criterion="tail", backlog="projected"),
    "no_N1":      ModelSpec(service="rf_naive", criterion="tail", backlog="projected"),
    "no_N2":      ModelSpec(service="rf_corrected", criterion="mean_alpha", backlog="projected"),
    "no_N3":      ModelSpec(service="rf_corrected", criterion="tail", backlog="current"),
}

LABELS = {
    "static": "Static", "hpa": "HPA (util 70%)", "erlang_c": "Erlang-C (M/M/k)",
    "proto_emp": "Prototype v1 (empirical)", "proto_rf": "Prototype v1 (RF)",
    "proposed": "Proposed", "proposed_mc": "Proposed (min-cost)", "no_N1": "Proposed − N1", "no_N2": "Proposed − N2",
    "no_N3": "Proposed − N3",
}


def make_policy(name, seed=0, static_k=20, **overrides):
    """overrides: eps / eps_queue for model policies, target_util for HPA."""
    if name == "static":
        return StaticPolicy(static_k)
    if name == "hpa":
        return HPAPolicy(target_util=overrides.get("target_util", 0.7))
    spec = replace(POLICY_SPECS[name], **{k: v for k, v in overrides.items()
                                          if k in ("eps", "eps_queue", "wait_cost")})
    return ModelPolicy(name, spec, seed=seed)
