"""Print the neural-controller table of the paper (tab:controllers) from the two result files of
run_controller_table.py.

kp = 3, seeds 0 to 4: bench_table.json holds seed 0, bench_table_seeds1-4_kp3.json seeds 1 to 4. A seed counts as
converged when its final RMSE is finite: the `status` field reads "converged" also for runs that became NaN during
training. Each cell is the mean and the sample standard deviation (ddof = 1) over the converged seeds; the same seeds
fail for both solvers. Final RMSE is that of the DEER-exp-trained controller; a speedup is the `train_speedup` of the
solver's row (mean time of the sequential training step over that of the parallel one).

Usage (from the repository root): python experiments/appendix/robot_control/neural/table_controllers.py [RESULTS_DIR]
"""
import json
import math
import os
import sys

import numpy as np

ARCHS = (("mlp_large", "MLP large"), ("mlp_small", "MLP small"), ("ssm_d16_contract", "SSM"))


def load_rows(res):
    rows = json.load(open(os.path.join(res, "bench_table.json")))
    for r in rows:
        r.setdefault("seed", 0)                  # the seed-0 file predates the seed field
    rows += json.load(open(os.path.join(res, "bench_table_seeds1-4_kp3.json")))
    return [r for r in rows if r["kp"] == 3.0]


def converged(r):
    return "final_rmse" in r and math.isfinite(r["final_rmse"])


def table(res):
    """The table's rows [controller, final RMSE, speedup DEER-exp, speedup Euler-DEER] as printed strings, and one
    line per controller with its converged seeds and the Newton iterations of its runs at the end of training."""
    rows = load_rows(res)
    pd_rmse = {r["pd_only_rmse"] for r in rows if "pd_only_rmse" in r}
    assert len(pd_rmse) == 1
    cells = [["PD only", f"{pd_rmse.pop():.3f}", "--", "--"]]
    notes = []
    for arch, name in ARCHS:
        sel = {s: [r for r in rows if r["arch"] == arch and r["par_solver"] == s] for s in ("exp", "euler")}
        seeds = {s: sorted(r["seed"] for r in sel[s] if converged(r)) for s in sel}
        assert seeds["exp"] == seeds["euler"], (arch, seeds)
        rmse = np.array([r["final_rmse"] for r in sel["exp"] if converged(r)])
        sp = {s: np.array([r["train_speedup"] for r in sel[s] if converged(r)]) for s in sel}
        n_params = {r["n_params"] for r in sel["exp"] if "n_params" in r}.pop()
        cells.append([f"{name} ({n_params / 1e3:.1f}k params)",
                      f"{rmse.mean():.3f} +- {rmse.std(ddof=1):.3f}",
                      f"{sp['exp'].mean():.1f}x +- {sp['exp'].std(ddof=1):.1f}",
                      f"{sp['euler'].mean():.1f}x +- {sp['euler'].std(ddof=1):.1f}"])
        iters = {s: sorted(r["par_solver_iters"] for r in sel[s] if converged(r)) for s in sel}
        notes.append(f"{name}: converged seeds {seeds['exp']} of {sorted(r['seed'] for r in sel['exp'])}; "
                     f"Newton iterations at the end of training: DEER-exp {iters['exp']}, Euler-DEER {iters['euler']}")
    return cells, notes


def main():
    cells, notes = table(sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "robot_control"))
    print(f"{'Controller':26s} & Final RMSE & Speedup (DEER-exp) & Speedup (Euler-DEER)")
    for c in cells:
        print(f"{c[0]:26s} & " + " & ".join(c[1:]))
    print("\n".join(notes))


if __name__ == "__main__":
    main()
