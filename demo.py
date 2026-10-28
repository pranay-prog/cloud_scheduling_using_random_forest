"""Quick demo: run one bursty trace with three policies and print the key numbers (~10 s).

    python demo.py
    python demo.py steady
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from common import run_one  # noqa: E402

scenario = sys.argv[1] if len(sys.argv) > 1 else "bursty"
print(f"Scenario: {scenario}  (SLA: 95% of requests within 4 s)\n")
print(f"{'policy':<24}{'SLA viol.':>10}{'total wait':>13}{'mean W_total':>14}{'VM-hours':>10}")
for pol, name in [("proto_rf", "Prototype v1 (RF)"), ("hpa", "HPA (util 70%)"), ("proposed", "Proposed")]:
    r = run_one(scenario, pol, seed=1)
    print(f"{name:<24}{r['sla_violation'] * 100:>9.2f}%{r['total_wait_hours']:>11.2f} h"
          f"{r['mean_resp']:>12.2f} s{r['vm_hours']:>10.2f}")
