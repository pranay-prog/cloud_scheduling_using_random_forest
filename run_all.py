"""One command to reproduce everything: tests -> experiments -> tail validation -> figures/tables.

    python run_all.py              # full paper run (20 seeds, ~30-60 min)
    python run_all.py --quick      # 3 seeds, ~5 min, to check everything works
    python run_all.py --procs 8    # use 8 CPU cores

Outputs:
    results/*.csv          raw per-run metrics
    results/figures/       fig1..fig7 (PNG for slides, PDF for LaTeX)
    results/tables/        table_main.csv / .tex, table_oracle_quality.csv, related_work_table.tex
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        print(f"\n{title} FAILED (exit {r.returncode}). Fix the error above and rerun.")
        sys.exit(r.returncode)
    print(f"--- done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 seeds instead of 20")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    seeds = "3" if a.quick else "20"
    step("1/4 Tests", ["run_tests.py"])
    step(f"2/4 Simulations ({seeds} seeds, {a.procs} processes)",
         ["experiments/run_experiments.py", "all", "--seeds", seeds, "--procs", str(a.procs)])
    step("3/4 Tail-approximation validation", ["experiments/validate_tail.py"])
    step("4/4 Figures and tables", ["experiments/make_figures.py"])
    print("\nAll done. Open results/figures/ for the graphs and results/tables/ for the tables.")


if __name__ == "__main__":
    main()
