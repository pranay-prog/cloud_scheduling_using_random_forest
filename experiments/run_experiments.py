"""Run all simulation experiments and write CSVs to results/.

  python experiments/run_experiments.py [main|predictor|boot|pareto|all] [--seeds N] [--procs P]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_one  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
ALL_POLICIES = ["static", "hpa", "erlang_c", "proto_emp", "proto_rf",
                "proposed_mc", "proposed", "no_N1", "no_N2", "no_N3"]


def _call(kw):
    return run_one(**kw)


def grid(stage, seeds):
    S = range(seeds)
    if stage == "main":            # E1: all policies, all scenarios
        return [dict(scenario=sc, policy=p, seed=s)
                for sc in ["steady", "diurnal", "bursty", "mixshift"]
                for p in ALL_POLICIES for s in S]
    if stage == "predictor":       # E2: predictor quality sweep (feature levels)
        return [dict(scenario=sc, policy=p, seed=s, level=lv)
                for sc in ["steady", "bursty"] for lv in [0, 1, 3, 4]
                for p in ["proposed", "no_N1", "proto_rf"] for s in S]
    if stage == "boot":            # E3: boot-delay sweep
        return [dict(scenario="bursty", policy=p, seed=s, t_scale=ts)
                for ts in [0.0, 10.0, 30.0, 60.0]
                for p in ["proposed", "no_N3", "hpa"] for s in S]
    if stage == "pareto":          # E5: waiting-time vs cost frontier
        jobs = [dict(scenario=sc, policy=p, seed=s, policy_kw=dict(wait_cost=c))
                for sc in ["steady", "bursty"] for c in [0.01, 0.03, 0.1, 0.3, 1.0]
                for p in ["proposed", "no_N1"] for s in S]
        jobs += [dict(scenario=sc, policy="proposed_mc", seed=s, policy_kw=dict(eps=e))
                 for sc in ["steady", "bursty"] for e in [0.02, 0.05, 0.10, 0.20] for s in S]
        jobs += [dict(scenario=sc, policy="hpa", seed=s, policy_kw=dict(target_util=u))
                 for sc in ["steady", "bursty"] for u in [0.5, 0.6, 0.7, 0.8, 0.9] for s in S]
        return jobs
    raise ValueError(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", default="all", nargs="?")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stages = ["main", "predictor", "boot", "pareto"] if a.stage == "all" else [a.stage]
    for st in stages:
        jobs = grid(st, a.seeds)
        t0 = time.time()
        # group by seed so each worker reuses its trained oracle
        jobs.sort(key=lambda j: (j["seed"], j.get("level", 0)))
        with Pool(a.procs) as pool:
            rows = pool.map(_call, jobs, chunksize=4)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"{st}.csv", index=False)
        print(f"{st}: {len(df)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
