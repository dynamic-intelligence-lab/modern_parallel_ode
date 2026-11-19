"""Panel B of figure fig:ssm_exp: the contracting SSM on EigenWorms with and without the gamma head.

Reads results/contracting_ssm/runs.csv (export_runs.py) and draws, for quasi-DEER and Jacobi, the fraction of solver
calls that reached the tolerance in the last validation pass (left) and the test accuracy (right), mean and sample
standard deviation (ddof 1) over seeds: "stable" is the gamma_min sweep at gamma_min = 0.05 (5 seeds, iteration caps
200 and 1000), "unstable" the runs with --gamma_param off (2 seeds, caps 1000 and 5000). The paper's panel is this
drawing with its two axes placed one above the other in a drawing program, next to panel A, a diagram that is not
drawn by code in this repository.

    python experiments/appendix/contracting_ssm/figure_ablation.py [runs.csv]

Writes figs/contracting_ssm/ssm_ablation_bars.pdf.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "results", "contracting_ssm", "runs.csv")
OUT = os.path.join(ROOT, "figs", "contracting_ssm", "ssm_ablation_bars.pdf")
BASELINE_GAMMA_MIN = 0.05


def conditions(runs):
    runs = runs[runs["solver"].isin(["quasi", "jacobi"])].copy()
    stable = (runs["group"] == "gamma_min_sweep") & (runs["gamma_min"] == BASELINE_GAMMA_MIN)
    runs.loc[stable, "condition"] = "stable"
    runs.loc[runs["gamma_param"] == "off", "condition"] = "unstable"
    return runs.dropna(subset=["condition"])


def main():
    finished = conditions(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else RUNS))

    COND_ORDER = ["stable", "unstable"]
    SOLVER_ORDER = ["quasi", "jacobi"]
    COND_COLOR = {"stable": "#00B0F0", "unstable": "#C00000"}

    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.title_fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2), constrained_layout=True)
    for ax, metric, ylabel, ylim in [
        (axes[0], "converged_frac", "Fraction converged", (0, 1.05)),
        (axes[1], "test_acc", "Test accuracy", (0, 1.0)),
    ]:
        g = finished.groupby(["condition", "solver"])[metric].agg(["mean", "std", "count"]).reset_index()
        g["std"] = g["std"].fillna(0.0)
        width = 0.18
        xs = np.arange(len(SOLVER_ORDER)) * 0.55
        for j, cond in enumerate(COND_ORDER):
            sub = g[g["condition"] == cond].set_index("solver")
            for i, sv in enumerate(SOLVER_ORDER):
                ax.bar(xs[i] + (j - 0.5) * width, sub.loc[sv, "mean"], width=width, yerr=sub.loc[sv, "std"],
                       label=cond if i == 0 else None, color=COND_COLOR[cond], capsize=2, edgecolor="black",
                       linewidth=0.4, error_kw=dict(linewidth=0.6, capthick=0.6))
        ax.set_xticks(xs)
        ax.set_xticklabels(SOLVER_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8)
        print(metric, {(r.condition, r.solver): (round(r["mean"], 4), round(r["std"], 4), int(r["count"]))
                       for _, r in g.iterrows()})

    axes[0].set_title("Solver convergence")
    axes[1].set_title("Downstream training")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    # 0.357 is the line of the paper's figure, labelled about 0.36; the test split's majority class is 14 of 39
    # samples, 0.359. 0.20 is one of five classes.
    axes[1].axhline(0.357, ls=":", color="0.5", alpha=0.8, linewidth=0.8)
    axes[1].text(xs[-1] + 0.80, 0.365, "majority class\n(≈0.36)", color="0.4", ha="right", va="bottom", fontsize=7)
    axes[1].axhline(0.20, ls=":", color="0.5", alpha=0.7, linewidth=0.8)
    axes[1].text(xs[-1] + 0.80, 0.21, "uniform chance", color="0.4", ha="right", va="bottom", fontsize=7)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
