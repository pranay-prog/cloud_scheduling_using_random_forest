# Results summary (minimum-total-waiting model)

These are the numbers in `results/` and the presentation:
- 20 seeds per cell, with 35 minutes of measured traffic per run.
- Intervals are 95% CIs; tests are paired t-tests against the proposed model (`results/tables/table_main.csv`).
- SLA: P(W_total > 4 s) ≤ 5%, where W_total = W_B + W_q + S + T_storage.
- About 1.8% of requests take longer than 4 s on service time alone, so 1.8% is the lowest violation rate any policy can reach.
- Proposed model: cost weight c = 0.1.

## E1: main benchmark (Fig. 1, Fig. 2, `table_main.tex`)

| Scenario | Policy | SLA violations | Total waiting | Mean W_total | VM-hours |
|---|---|---|---|---|---|
| Steady | Prototype v1 (RF) | 8.29% | 11.16 h | 1.76 s | 8.88 |
| | HPA (70% util) | 1.83% | 0.084 h | 0.81 s | 14.05 |
| | **Proposed** | **1.82%** | **0.063 h** | **0.81 s** | **12.84** |
| Diurnal | Prototype v1 (RF) | 34.73% | 31.91 h | 3.53 s | 8.98 |
| | HPA | 1.83% | 0.110 h | 0.81 s | 14.10 |
| | **Proposed** | **1.83%** | **0.081 h** | **0.81 s** | **12.92** |
| Bursty | Prototype v1 (RF) | 45.34% | 66.24 h | 6.15 s | 9.87 |
| | HPA | 11.17% | 8.72 h | 1.50 s | 15.97 |
| | **Proposed** | **6.40%** | **4.23 h** | **1.14 s** | **16.75** |
| Mix shift | Prototype v1 (RF) | 33.46% | 23.02 h | 3.77 s | 9.93 |
| | HPA | 4.48% | 0.284 h | 1.18 s | 15.65 |
| | **Proposed** | **4.62%** | **0.366 h** | **1.19 s** | **14.34** |

* **Against prototype v1:** total waiting is 94–99% lower and SLA violations are 4–19× lower, at a cost of 44–70% more VM-hours.
* **Against HPA:**
  * Steady and diurnal traffic: 25–26% less waiting and 8–9% fewer VM-hours.
  * Bursty traffic: 52% less waiting for 5% more VM-hours.
  * Mix shift: 8% fewer VM-hours but 29% more waiting.
* **Bursty traffic misses 5% for every policy.** Bursts last about 45 s, while new VMs need 10 s to boot.

## E5: waiting versus cost (Fig. 3)

The cost weight c was swept over 0.01–1 for the proposed model, and HPA's utilisation target over 50–90%.
* On steady and bursty traffic, the proposed curve lies below HPA's: less waiting and fewer violations at the same VM-hours.
* The previous minimum-cost model sits at the cheap end of the range, with 30–50× more waiting.

## Ablation (c = 0.1)

| Removed | Effect |
|---|---|
| N1 (naive C²ₛ) | Waiting +62% on steady, +4% bursty, +31% mix shift. It also uses 3% fewer VMs because it under-estimates variability. At equal VM-hours, N1 gives about 15% less waiting on steady traffic and no difference on bursty. |
| N2 (mean check) | No change except on mix shift. At c = 0.1 the objective already buys more headroom than the SLA needs, so the SLA constraint is slack. N2 matters at high c, where the constraint binds. |
| N3 (current backlog only) | No measurable change. |

## E2: predictor quality (Fig. 4)

* Corrected C²ₛ = 2.31, naive = 1.10, true = 2.30.
* The forest's 90% intervals cover 89.7% of held-out runtimes.
* With no features at all, the corrected estimate drops to 1.77 because the log-normal residual assumption breaks down.

## E3: boot delay (Fig. 5)

| Boot delay | Proposed: waiting / VM-h | HPA: waiting / VM-h |
|---|---|---|
| 0 s | 0.51 h / 14.7 | 0.54 h / 15.7 |
| 30 s | 19.6 h / 19.7 | 50.2 h / 15.3 |
| 60 s | 51.6 h / 23.1 | 105.2 h / 14.8 |

At long boot delays the proposed model trades VMs for waiting.

## E4: approximation accuracy (Fig. 6)

* The exponential tail P(W > τ) has a mean error of about 0 and a range of ±0.10 against exact G/G/k simulation, over 54 configurations.
* The Allen–Cunneen mean wait has a median error of 16%.

## Known limitations

1. **Synthetic workload only.** Validate on a public trace (Google, Alibaba or Azure) before submission.
2. **No proactive scaling.** Bursts shorter than about 4× the boot delay are not handled by any reactive policy; a burst forecaster would help.
3. **N2 and N3 only bind at high cost weights.** Present N1 as the core contribution.
4. **The cost weight c is a design choice.** Report the full frontier (Fig. 3) rather than one value.
