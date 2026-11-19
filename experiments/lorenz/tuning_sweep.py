"""Rerun planning for the Lorenz tuning sweep, and the loader and the outcome rule of its summaries.

Standard library only (csv, math), so it runs on a login node. H is the shipped sweep, results/lorenz/tuning_sweep/,
which the figure scripts read: stageA holds scale-ELK over K_GRID with DEER and LASER, stageB ELK over S_GRID, both
with at most 200 iterations, and stage{A,B}_cap2000 the runs of those that stopped at 200 iterations with a bounded
iterate (outcome cap), rerun with at most 2,000.

  python experiments/lorenz/tuning_sweep.py [SWEEP_DIR]

reads SWEEP_DIR/stage{A,B}/summary.csv, written by experiments/lorenz/slurm/tuning_sweep_{A,B}.slurm, and writes the
runs to rerun into SWEEP_DIR/stage{A,B}_cap2000/plan.txt, which tuning_sweep_{A,B}_cap2000.slurm read. SWEEP_DIR is
results/lorenz/check/tuning_sweep by default, where the templates write; results/lorenz/tuning_sweep rewrites the
shipped plan.txt files from the shipped summaries. If a stage has no runs yet, it stops, names the template to
submit first and writes nothing.

Outcome of one run:
  conv  status "ok" and residual < 1e-9
  nan   a non-finite iterate or residual (status "NaN")
  div   stopped at max_iter with max|X| > 10 x max|x| of the truth
  cap   stopped at max_iter with max|X| <= 10 x max|x| of the truth
  stop  finite, not converged, below max_iter (not expected for scale-elk or elk)
"""
import argparse
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.join(ROOT, "results", "lorenz", "tuning_sweep")
CHECK = os.path.join(ROOT, "results", "lorenz", "check", "tuning_sweep")
NS = [1000, 3000, 10000, 30000, 100000]
K_GRID = [0.03, 0.04, 0.045, 0.05, 0.051, 0.052, 0.053, 0.055, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
S_GRID = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
TOL = 1e-9
FAMILIES = {"A": ("scale-elk", K_GRID), "B": ("elk", S_GRID)}
LASER = "laser"


def mname(prefix, v):
    return f"{prefix}:{v:g}" if prefix == "scale-elk" else f"{prefix}:{int(v)}"


def load(path):
    """{(N, method): row}; a later row for the same key (a resubmitted configuration) replaces the earlier one."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(int(r["N"]), r["method"])] = r
    return out


def outcome(r, max_iter):
    res = float(r["residual"])
    if r["status"] == "ok" and res < TOL:
        return "conv"
    if r["status"] == "NaN" or not math.isfinite(res):
        return "nan"
    if int(r["iters"]) >= max_iter:
        return "div" if float(r["max_abs"]) > 10.0 * float(r["truth_max"]) else "cap"
    return "stop"


def plan(sweep):
    loaded = {stage: load(os.path.join(sweep, f"stage{stage}", "summary.csv")) for stage in FAMILIES}
    for stage, rows in loaded.items():
        if not rows:
            raise SystemExit(f"no runs in {os.path.join(sweep, f'stage{stage}', 'summary.csv')}: submit "
                             f"experiments/lorenz/slurm/tuning_sweep_{stage}.slurm first")
    for stage, (prefix, grid) in FAMILIES.items():
        rows = loaded[stage]
        lines = []
        for n in NS:
            ms = [mname(prefix, v) for v in grid if (n, mname(prefix, v)) in rows
                  and outcome(rows[(n, mname(prefix, v))], 200) == "cap"]
            missing = [mname(prefix, v) for v in grid if (n, mname(prefix, v)) not in rows]
            if missing:
                print(f"stage {stage} N {n}: MISSING rows {missing}")
            if ms:
                lines.append(f"{n} {','.join(ms)}")
        d = os.path.join(sweep, f"stage{stage}_cap2000")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "plan.txt"), "w") as f:
            f.write("".join(l + "\n" for l in lines))
        print(f"stage {stage}: {sum(len(l.split()[1].split(',')) for l in lines)} capped runs to rerun")
        for l in lines:
            print("  " + l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default=CHECK, metavar="SWEEP_DIR",
                    help="the folder that holds stageA/ and stageB/ (default results/lorenz/check/tuning_sweep)")
    plan(ap.parse_args().sweep)


if __name__ == "__main__":
    main()
