# UA-G/G/k: Uncertainty-Aware G/G/k Autoscaling

Reference implementation of the pipeline in `uncertainty_aware_pipeline.pdf`
(Random-Forest-augmented queueing for cloud resource scheduling).

**Problem.** Choose the number of workers k(t) that minimises the total waiting
time of all requests plus VM cost, subject to P(W_total > T_SLA) ≤ ε with
W_total = W_B + W_q + S + T_storage, under bursty arrivals, heterogeneous job
runtimes and a VM boot delay T_scale.

**Novelty.**

| | Idea | Where in code |
|---|---|---|
| N1 | Propagate the RF's per-request prediction error into service variability: C²ₛ = (Var(Ŝ) + E[σ̂²(x)]) / E[Ŝ]² (law of total variance). Naive plug-in of point predictions drops E[σ̂²] and under-estimates C²ₛ. | `oracle.py`, `ModelPolicy._service_stats` |
| N2 | Size for the percentile SLA directly: P(W + S > T_SLA) ≈ E_S[P_W · exp(−(kμ−λ)(T_SLA − S − drain)/V)], averaged over the RF's predictive runtime distribution. The SLA budget is split into the irreducible part P(S > T_SLA) and a queueing budget. | `ModelPolicy.decide` (criterion `tail`) |
| N3 | Boot-delay lookahead: size for the backlog projected to when new VMs are ready, Q̃ = Q + max(0, λ − k_active μ)·T_scale. | `ModelPolicy.decide` (backlog `projected`) |

## Layout

```
uagk/
  workload.py    synthetic jobs (web / batch / ML), heteroscedastic log-normal runtimes,
                 4 arrival scenarios: steady, diurnal, bursty (MMPP + H2), mixshift
  oracle.py      RF on log-runtime + RF on out-of-bag squared residuals
                 -> per-job mean, variance, P95; held-out evaluation
  queueing.py    Erlang-C, Allen-Cunneen, exponential G/G/k tail
  policies.py    proposed planner, ablations (−N1, −N2, −N3), baselines
                 (static, HPA-style utilisation, Erlang-C service level,
                 original prototype v1 with empirical / RF moments)
  simulator.py   heap-based discrete-event cluster: FCFS line with capacity,
                 boot delay, billed booting VMs, graceful retirement
experiments/
  run_experiments.py   E1 main, E2 predictor quality, E3 boot delay, E5 target sweep
  validate_tail.py     E4 accuracy of Allen-Cunneen and the tail vs exact G/G/k simulation
  make_figures.py      figures (PNG + PDF) and tables (CSV + LaTeX)
tests/test_model.py    formula checks; simulator reproduces M/M/k theory
results/               CSVs, figures/, tables/
```

## Run

```bash
pip install numpy scipy pandas scikit-learn matplotlib
python demo.py                # 10-second demo: v1 vs HPA vs proposed on one bursty trace
python run_all.py --quick     # whole pipeline with 3 seeds (~5-10 min) to check everything works
python run_all.py             # full paper run, 20 seeds (~30-60 min depending on CPU cores)
```

`run_all.py` runs, in order: the tests, all simulations (`experiments/run_experiments.py`),
the tail-approximation check (`experiments/validate_tail.py`) and the figures and tables
(`experiments/make_figures.py`). Outputs:

| Where | What |
|---|---|
| `results/figures/fig1_main_benchmark` | SLA violations, total waiting, VM-hours for all policies and scenarios |
| `results/figures/fig2_wtotal_breakdown` | Mean W_total split into W_B, W_q, S, T_storage |
| `results/figures/fig3_waiting_cost_frontier` | Total waiting and violations vs VM-hours (cost-weight sweep) |
| `results/figures/fig4_predictor_quality` | Effect of RF accuracy; naive vs corrected C²ₛ |
| `results/figures/fig5_boot_delay` | Effect of VM boot delay |
| `results/figures/fig6_tail_validation` | Accuracy of the queueing approximations |
| `results/figures/fig7_trace_bursty` | k(t) over time during bursts |
| `results/figures/flow_v1`, `flow_proposed` | Algorithm flowcharts |
| `results/tables/table_main.csv` / `.tex` | Main results with 95% CIs and p-values |
| `results/tables/related_work_table.tex` | Literature comparison |

Each figure is saved as PNG (slides) and PDF (LaTeX).

## Default parameters

| | |
|---|---|
| Horizon / warm-up | 2400 s / 300 s (metrics on arrivals in [300, 2400]) |
| Control interval T_ctrl | 4 s |
| Boot delay T_scale | 10 s |
| SLA | response time ≤ 4 s for at least 95% of requests (ε = 0.05) |
| Queueing budget | 2% on top of the predicted irreducible part |
| Characterisation window | 30 s |
| Scale-down | after 3 consecutive lower targets (shared by all adaptive policies) |
| k bounds | 2 – 120 |
| Seeds | 20 per cell; 95% t-intervals; paired t-tests vs proposed |

## Limitations

* Synthetic workload; validate on a public trace with job durations
  (Google cluster, Alibaba, Azure Functions) before final submission.
* The log-normal residual model in the oracle is an assumption; with no
  informative features (level 4) the corrected C²ₛ still under-estimates the
  true value because the runtime mixture is not log-normal.
* The exponential tail is an approximation; see E4 for where it holds.
