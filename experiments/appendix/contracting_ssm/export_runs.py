"""Export the run records of the paper's EigenWorms runs into a runs table (results/contracting_ssm/runs.csv).

Needs the run records, which are not in this repository: the Slurm logs of the runs (the "args:" line that
slurm/train.slurm prints, and the wandb run URL in the .err file) and the local wandb run folders
(files/wandb-summary.json and files/wandb-metadata.json). The paper's tables and figure were made from the same
summaries read through the wandb API; for the 75 runs of the gamma_min table the local summaries give the same test
accuracies and epoch times.

    python experiments/appendix/contracting_ssm/export_runs.py LOG_DIR WANDB_DIR [--out FILE]

--out defaults to results/contracting_ssm/check/runs.csv, which leaves the shipped runs.csv alone; the paper's
records give that file byte for byte.

The column args holds the run's command-line arguments without the wandb group name. The solver, the A and gamma
options, gamma_min (with the gamma head) and the two iteration caps are read from that line, so a run must pass them,
as the submit scripts do; no default of run_train.py is assumed. The Slurm job ids are not written.

Rows: the 75 runs of the gamma_min sweep (table tab:eigenworms_gamma_min and the "stable" bars of figure fig:ssm_exp
at gamma_min = 0.05) and the four runs without the gamma head (--gamma_param off, the "unstable" bars). For the
jacobi runs without the gamma head the runs repeated on an NVIDIA B200 (group ablation_stability_b200_*) replace the
first ones, as in the paper's figure. The records also hold runs with an unconstrained A (--A_param raw, an option
of the research code that made them, which this repository does not have); they are not in the paper and are left
out.
"""
import argparse
import csv
import glob
import json
import os
import re

# wandb group prefix -> label; a longer prefix is tried before a shorter one. v2_gamma_min_sweep is the group name of
# the paper's sweep runs, gamma_min_sweep that of runs of slurm/submit_gamma_min_sweep.sh; ablation_stability_b200 is
# that of the paper's two jacobi runs without the gamma head that were repeated on an NVIDIA B200 (set outside the
# scripts of this repository).
GROUPS = {"v2_gamma_min_sweep": "gamma_min_sweep", "gamma_min_sweep": "gamma_min_sweep",
          "ablation_stability_b200": "ablation_b200", "ablation_stability": "ablation"}
SUMMARY = {"test_acc": "test/acc", "wall_time_per_epoch_s": "wall_time_per_epoch_s",
           "converged_frac": "solver/converged_frac", "iters_median": "solver/iters_median",
           "epochs_completed": "wall_time/epochs_completed"}
COLUMNS = ["group", "solver", "gamma_param", "gamma_min", "seed", "max_iter_newton", "max_iter_jacobi",
           "gpu", *SUMMARY, "args"]


def runs(log_dir, wandb_dir):
    for out in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
        text = open(out, errors="ignore").read()
        m = re.search(r"^args:\s+(.*)$", text, re.M)
        if not m:
            continue
        toks = m.group(1).split()
        args = {toks[i][2:]: toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--")}
        group = next((lab for pre, lab in GROUPS.items() if args.get("wandb_group", "").startswith(pre + "_")), None)
        if group is None:
            continue
        run_id = re.search(r"runs/([A-Za-z0-9]+)", open(out[:-4] + ".err", errors="ignore").read()).group(1)
        (run_dir,) = glob.glob(os.path.join(wandb_dir, f"*run-*-{run_id}"))
        summary = json.load(open(os.path.join(run_dir, "files", "wandb-summary.json")))
        meta = json.load(open(os.path.join(run_dir, "files", "wandb-metadata.json")))
        row = dict(group=group, solver=args["solver"], gamma_param=args["gamma_param"], A_param=args["A_param"],
                   gamma_min=args["gamma_min"] if args["gamma_param"] == "sigmoid" else "",
                   seed=args["seed"], max_iter_newton=args["max_iter_newton"],
                   max_iter_jacobi=args["max_iter_jacobi"], gpu=meta.get("gpu", ""),
                   args=" ".join(t for i, t in enumerate(toks)          # all arguments but the wandb group name
                                 if t != "--wandb_group" and (i == 0 or toks[i - 1] != "--wandb_group")))
        row.update({k: summary[v] for k, v in SUMMARY.items()})
        yield row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("log_dir")
    p.add_argument("wandb_dir")
    p.add_argument("--out", default="results/contracting_ssm/check/runs.csv")
    a = p.parse_args()
    rows = list(runs(a.log_dir, a.wandb_dir))
    rows = [r for r in rows if r["A_param"] == "sigmoid"]
    b200 = {(r["solver"], r["seed"]) for r in rows if r["group"] == "ablation_b200"}
    rows = [r for r in rows if not (r["group"] == "ablation" and (r["solver"], r["seed"]) in b200)]
    rows.sort(key=lambda r: (r["group"] != "gamma_min_sweep", r["solver"], r["gamma_min"], int(r["seed"])))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
