"""Table tab:eigenworms_gamma_min: EigenWorms with the contracting SSM, the sweep over gamma_min.

Reads results/contracting_ssm/runs.csv (export_runs.py; 75 runs: 5 values of gamma_min, 3 solvers, 5 seeds, one
NVIDIA B200 each) and prints the rows of the table as they are in the paper's LaTeX source:

    python experiments/appendix/contracting_ssm/table_gamma_min.py [runs.csv]

Columns, each a mean and a sample standard deviation (ddof 1) over the 5 seeds:
- test accuracy: the mean of the three solvers' mean accuracies; the deviation is that of the sequential runs;
- wall time per epoch: wall_time_per_epoch_s of the run, the time of its last epoch (training and validation);
- speedup: sequential time over parallel time, paired by seed.
A note follows the table for each row whose three solver mean accuracies differ by 0.5 points or more.
"""
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "results", "contracting_ssm", "runs.csv")


def table(runs):
    df = runs[runs["group"] == "gamma_min_sweep"]
    agg = df.groupby(["gamma_min", "solver"]).agg(
        n=("test_acc", "count"), acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
        wall_mean=("wall_time_per_epoch_s", "mean"), wall_std=("wall_time_per_epoch_s", "std"))
    pv = df.pivot_table(index=["gamma_min", "seed"], columns="solver", values="wall_time_per_epoch_s", aggfunc="first")
    for s in ("quasi", "jacobi"):
        pv[f"speedup_{s}"] = pv["seq"] / pv[s]
    sp = pv.groupby("gamma_min").agg(["mean", "std"])
    rows, notes = [], []
    for g in sorted(df["gamma_min"].unique()):
        a = agg.loc[g]
        assert (a["n"] == 5).all(), f"gamma_min {g}: {a['n'].to_dict()} runs"
        accs = a.loc[["seq", "quasi", "jacobi"], "acc_mean"]
        if accs.max() - accs.min() >= 0.005:
            notes.append(f"gamma_min {g:.2f}: mean accuracies " + ", ".join(f"{s} {100 * v:.2f}" for s, v in accs.items()))
        cells = [f"{g:.2f}", f"{100 * accs.mean():.1f}", f"{100 * a.loc['seq', 'acc_std']:.1f}",
                 f"{a.loc['seq', 'wall_mean']:.1f}", f"{a.loc['seq', 'wall_std']:.1f}",
                 f"{a.loc['quasi', 'wall_mean']:.2f}", f"{a.loc['quasi', 'wall_std']:.2f}",
                 f"{a.loc['jacobi', 'wall_mean']:.2f}", f"{a.loc['jacobi', 'wall_std']:.2f}",
                 f"{sp.loc[g, ('speedup_quasi', 'mean')]:.0f}$\\times$", f"{sp.loc[g, ('speedup_quasi', 'std')]:.1f}",
                 f"{sp.loc[g, ('speedup_jacobi', 'mean')]:.1f}$\\times$", f"{sp.loc[g, ('speedup_jacobi', 'std')]:.1f}"]
        rows.append(cells)
    return rows, notes


def main():
    runs = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else RUNS)
    rows, notes = table(runs)
    for cells in rows:
        print("    " + " & ".join(cells) + " \\\\")
    for n in notes:
        print("note:", n)


if __name__ == "__main__":
    main()
