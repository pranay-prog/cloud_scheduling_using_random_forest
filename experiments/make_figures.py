"""Paper figures and tables from results/*.csv  ->  results/figures/, results/tables/."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from uagk.policies import LABELS  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"

# Colour follows the entity (fixed categorical order, validated reference palette).
COLOR = {
    "proposed": "#2a78d6", "hpa": "#eb6834", "erlang_c": "#1baf7a", "proto_emp": "#eda100",
    "proto_rf": "#e87ba4", "proposed_mc": "#6250d6", "no_N1": "#e34948", "no_N3": "#008300",
    "no_N2": "#52514e", "static": "#8c8b86",
}
MARK = {"proposed": "o", "hpa": "s", "erlang_c": "^", "proto_emp": "v", "proto_rf": "<",
        "proposed_mc": "D", "no_N1": "P", "no_N2": "h", "no_N3": "X", "static": "h"}
MAIN = ["static", "hpa", "erlang_c", "proto_emp", "proto_rf", "proposed_mc", "proposed"]
TABLE_ORDER = MAIN[:-1] + ["no_N1", "no_N2", "no_N3", "proposed"]
SCEN = {"steady": "Steady", "diurnal": "Diurnal", "bursty": "Bursty", "mixshift": "Mix shift"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))


def agg(df, keys, cols):
    g = df.groupby(keys)
    out = g[cols].mean()
    for c in cols:
        out[c + "_ci"] = g[c].apply(ci95)
    return out.reset_index()


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


METRICS = ["viol_pct", "total_wait_hours", "vm_hours", "mean_resp", "p95", "p99",
           "mean_wait", "mean_wb", "mean_wq", "mean_service", "mean_storage", "drop_rate"]


def prep(df):
    df = df.copy()
    df["viol_pct"] = df.sla_violation * 100
    return df


# ------------------------------------------------------------------ E1
def fig_main(df):
    a = agg(df, ["scenario", "policy"], METRICS)
    scen = list(SCEN)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True)
    w = 0.11
    x = np.arange(len(scen))
    cols = [("viol_pct", "SLA violations (%)"), ("total_wait_hours", "Total waiting time (h, log)"),
            ("vm_hours", "VM-hours")]
    for i, p in enumerate(MAIN):
        sub = a[a.policy == p].set_index("scenario").loc[scen]
        off = (i - (len(MAIN) - 1) / 2) * (w + 0.006)
        for ax, (col, _) in zip(axes, cols):
            ax.bar(x + off, sub[col], w, color=COLOR[p], edgecolor="white", linewidth=0.8,
                   yerr=sub[col + "_ci"], error_kw=dict(lw=0.8, capsize=1.5, ecolor=MUTED),
                   label=LABELS[p] if col == "viol_pct" else None)
    for ax, (_, lab) in zip(axes, cols):
        ax.set_ylabel(lab)
    axes[0].axhline(5, color=INK, lw=0.9, ls="--", label="SLA target ε = 5%")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_yticks([0, 2, 5, 10, 20, 40])
    axes[0].yaxis.set_major_formatter(PLAIN)
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(PLAIN)
    axes[2].set_xticks(x, [SCEN[s] for s in scen])
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.36), fontsize=8)
    save(fig, "fig1_main_benchmark")
    return a


def fig_breakdown(a):
    """Mean W_total split into W_B, W_q, S, T_storage."""
    parts = [("mean_wb", "Buffer wait W_B", "#1e3350"), ("mean_wq", "Queue wait W_q", "#2a78d6"),
             ("mean_service", "Service S", "#b9b8b2"), ("mean_storage", "Storage T_st", "#dedcd6")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, sc in zip(axes, ["steady", "bursty"]):
        sub = a[a.scenario == sc].set_index("policy").loc[MAIN]
        left = np.zeros(len(MAIN))
        for col, lab, c in parts:
            ax.barh(np.arange(len(MAIN)), sub[col], left=left, color=c, edgecolor="white",
                    linewidth=1.0, height=0.7, label=lab)
            left += sub[col].values
        for i, v in enumerate(left):
            ax.text(v + 0.05, i, f"{v:.2f} s", va="center", fontsize=8, color=INK)
        ax.set_yticks(np.arange(len(MAIN)), [LABELS[p] for p in MAIN])
        ax.set_xlabel("Mean end-to-end latency W_total (s)")
        ax.set_title(SCEN[sc], loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, max(left) * 1.18)
        ax.grid(axis="y", visible=False)
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.22), fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_wtotal_breakdown")


def table_main(df, a):
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in SCEN:
        base = df[(df.scenario == sc) & (df.policy == "proposed")].set_index("seed")
        for p in TABLE_ORDER:
            r = a[(a.scenario == sc) & (a.policy == p)].iloc[0]
            row = dict(scenario=SCEN[sc], policy=LABELS[p],
                       sla_viol_pct=f"{r.viol_pct:.2f} ± {r.viol_pct_ci:.2f}",
                       total_wait_h=f"{r.total_wait_hours:.2f} ± {r.total_wait_hours_ci:.2f}",
                       mean_wtotal_s=f"{r.mean_resp:.3f}", W_B=f"{r.mean_wb:.3f}", W_q=f"{r.mean_wq:.3f}",
                       S=f"{r.mean_service:.3f}", T_st=f"{r.mean_storage:.3f}",
                       p95_s=f"{r.p95:.2f}", p99_s=f"{r.p99:.2f}",
                       vm_hours=f"{r.vm_hours:.2f} ± {r.vm_hours_ci:.2f}")
            if p != "proposed":
                o = df[(df.scenario == sc) & (df.policy == p)].set_index("seed")
                s = base.index.intersection(o.index)
                for m, nm in [("sla_violation", "p_viol"), ("total_wait_hours", "p_wait"), ("vm_hours", "p_vmh")]:
                    row[nm] = f"{stats.ttest_rel(o.loc[s, m], base.loc[s, m]).pvalue:.1g}"
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table_main.csv", index=False)
    lines = [r"\begin{tabular}{@{}l" + "ccc" * 4 + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{SCEN[s]}}}" for s in SCEN) + r" \\",
             " & " + " & ".join([r"Viol.\,\% & Wait (h) & VM-h"] * 4) + r" \\", r"\midrule"]
    for p in TABLE_ORDER:
        cells = []
        for sc in SCEN:
            r = a[(a.scenario == sc) & (a.policy == p)].iloc[0]
            vals = [f"{r.viol_pct:.1f}", f"{r.total_wait_hours:.2f}", f"{r.vm_hours:.1f}"]
            cells += [rf"\textbf{{{v}}}" for v in vals] if p == "proposed" else vals
        name = LABELS[p].replace("−", "$-$").replace("%", r"\%")
        lines.append((rf"\textbf{{{name}}}" if p == "proposed" else name) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "table_main.tex").write_text("\n".join(lines))
    return t


# ------------------------------------------------------------------ E5 frontier
def fig_frontier(df):
    df = df.copy()
    df["knob"] = df["wait_cost"].fillna(df["eps"]).fillna(df["target_util"])
    a = agg(df, ["scenario", "policy", "knob"], ["viol_pct", "total_wait_hours", "vm_hours"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    labs = {"proposed": "Proposed (c = 0.01–1)", "no_N1": "Proposed − N1 (c = 0.01–1)",
            "proposed_mc": "Proposed min-cost (ε = 2–20%)", "hpa": "HPA (util 50–90%)"}
    for i, sc in enumerate(["steady", "bursty"]):
        for j, (col, lab) in enumerate([("total_wait_hours", "Total waiting time (h, log)"),
                                        ("viol_pct", "SLA violations (%, log)")]):
            ax = axes[i, j]
            for p in ["hpa", "proposed_mc", "no_N1", "proposed"]:
                s = a[(a.scenario == sc) & (a.policy == p)].sort_values("vm_hours")
                ax.plot(s.vm_hours, s[col], color=COLOR[p], marker=MARK[p], ms=6, lw=2, label=labs[p])
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PLAIN)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("VM-hours")
            ax.set_ylabel(lab)
            ax.set_title(f"{SCEN[sc]}", loc="left", fontsize=10, color=INK)
            if col == "viol_pct":
                ax.axhline(5, color=INK, lw=0.8, ls="--")
                ax.set_yticks([2, 3, 5, 10, 20, 40])
                ax.yaxis.set_major_formatter(PLAIN)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_waiting_cost_frontier")
    return a


# ------------------------------------------------------------------ E2
def fig_predictor(df, quality):
    a = agg(df, ["scenario", "level", "policy"], ["viol_pct", "total_wait_hours", "vm_hours", "mean_cs2_est", "cs2_trace_true"])
    names = {0: "all", 1: "no size", 3: "type only", 4: "none"}
    qq = quality.set_index("level").r2_log
    xt = [f"{names[l]}\nR²={qq[l]:.2f}" for l in sorted(a.level.unique())]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for j, sc in enumerate(["steady", "bursty"]):
        ax = axes[j]
        for p in ["proto_rf", "no_N1", "proposed"]:
            s = a[(a.scenario == sc) & (a.policy == p)].sort_values("level")
            ax.errorbar(np.arange(len(s)), s.viol_pct, yerr=s.viol_pct_ci, color=COLOR[p], marker=MARK[p],
                        ms=6, lw=2, capsize=2, label=LABELS[p])
        ax.axhline(5, color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(xt)), xt, fontsize=7.5)
        ax.set_xlabel("Oracle features (accuracy drops →)")
        ax.set_ylabel("SLA violations (%)")
        ax.set_title(SCEN[sc], loc="left", fontsize=10, color=INK)
        if sc == "steady":
            ax.legend(fontsize=8)
    ax = axes[2]
    for p in ["no_N1", "proposed"]:
        s = a[(a.scenario == "steady") & (a.policy == p)].sort_values("level")
        ax.plot(np.arange(len(s)), s.mean_cs2_est, color=COLOR[p], marker=MARK[p], ms=6, lw=2,
                label=("naive plug-in" if p == "no_N1" else "corrected (N1)"))
    tru = a[(a.scenario == "steady")].cs2_trace_true.mean()
    ax.axhline(tru, color=INK, lw=1, ls="--")
    ax.text(2.0, tru + 0.08, f"true C²ₛ ≈ {tru:.2f}", fontsize=8, color=INK)
    ax.set_ylim(0, 3.0)
    ax.set_xticks(range(len(xt)), xt, fontsize=7.5)
    ax.set_xlabel("Oracle features (accuracy drops →)")
    ax.set_ylabel("Estimated C²ₛ used by planner")
    ax.set_title("Service variability estimate", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_predictor_quality")
    return a


# ------------------------------------------------------------------ E3
def fig_boot(df):
    a = agg(df, ["t_scale", "policy"], ["viol_pct", "total_wait_hours", "vm_hours"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for p in ["hpa", "no_N3", "proposed"]:
        s = a[a.policy == p].sort_values("t_scale")
        for ax, col in zip(axes, ["viol_pct", "total_wait_hours", "vm_hours"]):
            ax.errorbar(s.t_scale, s[col], yerr=s[col + "_ci"], color=COLOR[p], marker=MARK[p],
                        ms=6, lw=2, capsize=2, label=LABELS[p])
    for ax, lab in zip(axes, ["SLA violations (%)", "Total waiting time (h)", "VM-hours"]):
        ax.set_ylabel(lab)
        ax.set_xlabel("VM boot delay T_scale (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Bursty workload", x=0.02, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig5_boot_delay")
    return a


# ------------------------------------------------------------------ E4
def fig_tail(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cmap = {0.5: "#2a78d6", 1.0: "#1baf7a", 2.5: "#eb6834"}
    ax = axes[0]
    for cs2, g in df.drop_duplicates(["k", "rho", "ca2", "cs2"]).groupby("cs2"):
        ax.scatter(g.wq_sim, g.wq_ac, s=26, color=cmap[cs2], edgecolor="white", lw=0.6, label=f"C²ₛ = {cs2}")
    m = [df.wq_sim.min() * 0.7, df.wq_sim.max() * 1.4]
    ax.plot(m, m, color=INK, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Simulated mean wait"); ax.set_ylabel("Allen–Cunneen mean wait")
    ax.set_title("Mean wait (E[S] = 1)", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8)
    ax = axes[1]
    for cs2, g in df.groupby("cs2"):
        ax.scatter(g.p_sim, g.p_approx, s=22, color=cmap[cs2], edgecolor="white", lw=0.6)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.9, ls="--")
    ax.set_xlabel("Simulated P(W > τ)"); ax.set_ylabel("Approximation (N2)")
    ax.set_title("Waiting-time tail", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig6_tail_validation")


# ------------------------------------------------------------------ k(t) trace
def fig_trace():
    from common import get_oracle
    from uagk.policies import make_policy
    from uagk.simulator import Cluster, SimConfig
    from uagk.workload import make_scenario, scenario_mean_load
    seed = 3
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    cfg0 = SimConfig()
    jobs = make_scenario("bursty", np.random.default_rng(seed), cfg0.horizon)
    pred = get_oracle(seed).predict(jobs)
    k0 = math.ceil(scenario_mean_load("bursty") / 0.7)
    bins = np.arange(0, cfg0.horizon + 10, 10)
    axes[0].fill_between(bins[:-1], np.histogram(jobs.t_arr, bins)[0] / 10, step="post", color="#c7c6c0", lw=0)
    axes[0].set_ylabel("Arrivals / s")
    for p in ["hpa", "proto_emp", "proposed"]:
        res = Cluster(jobs, pred, make_policy(p, seed=seed, static_k=k0), SimConfig(k0=k0)).run()
        kt = res.k_trace
        axes[1].step(kt[:, 0], kt[:, 1] + kt[:, 2], where="post", color=COLOR[p], lw=1.6, label=LABELS[p])
    axes[1].set_ylabel("Workers k(t)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(300, 1500)
    axes[1].legend(ncol=3, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "fig7_trace_bursty")


def oracle_quality():
    from uagk.oracle import ServiceOracle, training_jobs, evaluate
    from uagk.workload import make_jobs, MIX_DEFAULT
    rows = []
    for lv in [0, 1, 3, 4]:
        for seed in range(5):
            o = ServiceOracle(level=lv, seed=seed).fit(training_jobs(seed))
            test = make_jobs(np.random.default_rng(900 + seed), np.zeros(20000), MIX_DEFAULT)
            r = evaluate(o, test)
            r.update(level=lv, seed=seed)
            rows.append(r)
    q = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    s = q.groupby("level").mean(numeric_only=True).drop(columns="seed").reset_index()
    s.to_csv(TAB / "table_oracle_quality.csv", index=False)
    return s


def main():
    out = {}
    if (RES / "main.csv").exists():
        df = prep(pd.read_csv(RES / "main.csv"))
        a = fig_main(df)
        fig_breakdown(a)
        out["main"] = table_main(df, a)
        fig_trace()
    q = oracle_quality()
    out["oracle"] = q
    if (RES / "predictor.csv").exists():
        out["predictor"] = fig_predictor(prep(pd.read_csv(RES / "predictor.csv")), q[["level", "r2_log"]])
    if (RES / "boot.csv").exists():
        out["boot"] = fig_boot(prep(pd.read_csv(RES / "boot.csv")))
    if (RES / "pareto.csv").exists():
        out["frontier"] = fig_frontier(prep(pd.read_csv(RES / "pareto.csv")))
    if (RES / "tail_validation.csv").exists():
        fig_tail(pd.read_csv(RES / "tail_validation.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for k, v in out.items():
        print(f"\n== {k}\n", v.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
