"""Tables of the long runs of one paper cell
(the question "how do you know when to stop"): the suffix loss with LASER for 1e6 optimizer steps (cosine
learning rate 1e-3 to 1e-5 over the run, evaluated every 10,000 steps, the log thinned to one entry per 100 steps by
--log-every) and the velocity loss for 43,290 epochs (9,999,990 steps, evaluated every 500 epochs), against the
paper's 10,000-step suffix record and 1,000-epoch velocity record of the same cell. Records under
results/elcd/records/long/<system>/<dtype>/s<seed>/ (chain8z fp32 seed 1). The figure notebook draws the
Chamfer curves of part B from `evals`. numpy and pandas; no JAX.

Definitions:
- **Evaluation:** the driver's evaluation of the model at a step (experiments/elcd/train/metrics.py): the
  paper Chamfer (the paper's rollout protocol: data-space RK4 at dt 0.01 for twice the demonstration's duration,
  cut where the speed drops below 1e-5), the native Chamfer (latent RK4 at the demonstration's step for its
  length, decoded), the multi-start RMSE (rollouts from 128 recorded states to the demonstration's end, matched in
  time), each a mean over the training demonstrations or over the held-out ones. A paper-protocol rollout whose
  Chamfer is above 10 or not finite counts as diverged (`*_chamfer_paper_diverged`); the mean carries it, the
  median does not.
- **Clean evaluation:** one whose paper-protocol rollouts on that side (training or held-out) all converged; the
  floor and the minimum of a paper Chamfer are taken over clean evaluations only.
- **Floor:** the median of a metric over the last LAST clean evaluations of a run, the value the run settles at.
  `floor_table` also gives the minimum and the first evaluation within a margin of the floor.
- **Hours:** the sum of the logged walls before an evaluation (the suffix runs: the solver's steps, the fallback
  inside them, no evaluation; the velocity runs: the epochs), GPU hours on the same B200.
- The velocity driver logs its final evaluation's step as the epoch count; `evals` maps it to the last epoch's end
  step, as loss_comparison.load_velocity does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.elcd.figures import records as R                                   # noqa: E402
from experiments.elcd.record_names import record_file                               # noqa: E402

PAPER_ROOT = R.ROOT.parent
# vel_ref is the cosine arm, the schedule of vel_long; the figures' VEL_ARM (panel_data.py) is the constant-rate one
RUNS = {"suffix_long": ("long", "laser-cos1M"), "suffix_ref": ("main", "laser"),
        "vel_long": ("long", "vel-cos10M"), "vel_ref": ("velocity", "vel-cos1000")}
COLOR = {"suffix_long": "#1f5fbf", "suffix_ref": "#7fa8e0", "vel_long": "#d1600f", "vel_ref": "#f0a86a"}
METRICS = {"train_chamfer_paper": "train_chamfer_paper_mean", "heldout_chamfer_paper": "heldout_chamfer_paper_mean",
           "train_chamfer_native": "train_chamfer_native_mean", "heldout_chamfer_native": "heldout_chamfer_native_mean",
           "train_multistart_rmse": "train_multistart_rmse_mean", "heldout_multistart_rmse": "heldout_multistart_rmse_mean"}
DIVERGED = {"train": "train_chamfer_paper_diverged", "heldout": "heldout_chamfer_paper_diverged"}
LAST = 10                      # evaluations in the floor's median
MARGINS = (1.0, 0.5, 0.2, 0.1)  # floor_table: first evaluation within floor * (1 + margin)
LATE_FROM = 20000              # solver_table's late statistics start at this step


# ----------------------------------------------------------------------------- records

def load_records(root: Path = PAPER_ROOT, system: str = "chain8z", dtype: str = "fp32", seed: int = 1,
                 runs: Dict[str, tuple] = RUNS) -> Dict[str, dict]:
    """The records of `runs` (name -> (subdirectory of root, file stem)) of one cell that exist, by name."""
    out = {}
    for name, (sub, stem) in runs.items():
        rec = R._load(record_file(Path(root) / sub / system / dtype / f"s{seed}" / f"{stem}.json"))
        if rec:
            out[name] = rec
    return out


def is_velocity(rec: dict) -> bool:
    return (rec.get("config") or {}).get("loss") == "vel"


def evals(rec: dict) -> pd.DataFrame:
    """One row per evaluation of a record: step, hours (the logged walls before it), seconds (the evaluation's own
    wall) and every summary value, sorted by step. A velocity record's final evaluation, logged at the epoch count,
    is placed at the last epoch's end step; a suffix record's log entries end at their `step` (the block's last
    step), so an evaluation at step s counts the entries with step < s."""
    log = rec.get("log") or []
    ends = np.array([int(e["step"]) for e in log])
    cum = np.cumsum([float(e["wall"]) for e in log]) / 3600 if log else np.zeros(0)
    vel = is_velocity(rec)
    rows = []
    for i, ev in enumerate(rec.get("evals") or []):
        step = int(ev["step"])
        if vel and i == len(rec["evals"]) - 1 and step == len(log):
            step = int(ends[-1])
        k = int(np.searchsorted(ends, step, side="right" if vel else "left"))
        rows.append(dict(step=step, hours=float(cum[k - 1]) if k else 0.0, seconds=float(ev.get("seconds", np.nan)),
                         **ev["summary"]))
    df = pd.DataFrame(rows)
    return df.sort_values("step").reset_index(drop=True) if len(df) else df


def clean_rows(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """The evaluations whose value of `key` is finite and, for a paper Chamfer, whose side has no diverged rollout."""
    if key not in df:
        return df.iloc[0:0]
    ok = np.isfinite(df[key].to_numpy(float))
    side = "train" if key.startswith("train") else "heldout"
    if "chamfer_paper" in key and DIVERGED[side] in df:
        ok &= df[DIVERGED[side]].to_numpy() == 0
    return df[ok]


def log_frame(rec: dict) -> pd.DataFrame:
    """The log entries as a frame with x (the entry's last step + 1), steps (steps in the entry, 1 for a per-step
    log) and wall_per_step."""
    log = rec.get("log") or []
    df = pd.DataFrame(log)
    if len(df) == 0:
        return df
    if "steps" not in df:
        df["steps"] = 1
    df["x"] = df["step"].astype(int) + 1
    df["wall_per_step"] = df["wall"] / df["steps"]
    return df


# ----------------------------------------------------------------------------- tables

def floor_table(ev: Dict[str, pd.DataFrame], metrics: Dict[str, str] = METRICS, last: int = LAST,
                margins: Sequence[float] = MARGINS) -> pd.DataFrame:
    """Per run and metric: evaluations (all), clean (used), first_step and first (the first clean evaluation),
    floor (median over the last `last` clean evaluations) and floor_from_step (the first of them), minimum and
    at_step, final (the last clean evaluation), and for every margin m the first clean evaluation's step and hours
    with value <= floor * (1 + m) (-1 and NaN when none)."""
    rows = []
    for name, df in ev.items():
        for metric, key in metrics.items():
            d = clean_rows(df, key)
            if d.empty:
                continue
            tail = d.tail(last)
            floor = float(tail[key].median())
            row = dict(run=name, metric=metric, evaluations=len(df), clean=len(d), first_step=int(d["step"].iloc[0]),
                       first=float(d[key].iloc[0]), floor=floor, floor_from_step=int(tail["step"].iloc[0]),
                       minimum=float(d[key].min()), at_step=int(d.loc[d[key].idxmin(), "step"]), final=float(d[key].iloc[-1]))
            for m in margins:
                hit = d[d[key] <= floor * (1 + m)]
                tag = f"within_{int(round(m * 100))}pct"
                row[f"{tag}_step"] = int(hit["step"].iloc[0]) if len(hit) else -1
                row[f"{tag}_hours"] = float(hit["hours"].iloc[0]) if len(hit) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def solver_table(recs: Dict[str, dict], late_from: int = LATE_FROM) -> pd.DataFrame:
    """Per suffix run: steps, fallback_steps (optimizer steps that fell back to the loop), min_converged_frac (the
    lowest logged fraction of converged solves), iters_max (over the run), iters_mean_late and wall_per_step_late
    (step-weighted means over the entries from `late_from`), train_hours (the logged walls), eval_hours (the
    evaluations' seconds) and total_hours (the record's total_seconds)."""
    rows = []
    for name, rec in recs.items():
        if is_velocity(rec):
            continue
        lf = log_frame(rec)
        if len(lf) == 0:
            continue
        late = lf[lf["step"] >= late_from] if (lf["step"] >= late_from).any() else lf
        w = late["steps"].to_numpy(float)
        rows.append(dict(run=name, steps=int(lf["steps"].sum()),
                         fallback_steps=int(lf["seq_fallback"].sum()) if "seq_fallback" in lf else 0,
                         min_converged_frac=float(lf["converged_frac"].min()) if "converged_frac" in lf else np.nan,
                         iters_max=int(lf["iters_max"].max()) if "iters_max" in lf else -1,
                         iters_mean_late=float(np.average(late["iters_mean"], weights=w)) if "iters_mean" in lf else np.nan,
                         wall_per_step_late=float(late["wall"].sum() / w.sum()),
                         train_hours=float(lf["wall"].sum() / 3600),
                         eval_hours=float(sum(float(e.get("seconds", 0.0)) for e in rec.get("evals") or []) / 3600),
                         total_hours=float(rec.get("total_seconds", np.nan)) / 3600))
    return pd.DataFrame(rows)


def velocity_table(recs: Dict[str, dict], ev: Dict[str, pd.DataFrame], last_epochs: int = 1000) -> pd.DataFrame:
    """Per velocity run: epochs, steps, epochs_loss_above_1 (epochs whose mean loss exceeded 1), loss_last_median
    (median epoch loss over the last `last_epochs`), best_train_chamfer_paper and best_epoch (the lowest clean
    training paper Chamfer), first_diverged_epoch (the first evaluation with a diverged training rollout, -1 when
    none), evals_diverged_train and evals_diverged_heldout (evaluations with at least one), final_* (the last
    evaluation's means), train_hours and eval_hours."""
    rows = []
    for name, rec in recs.items():
        if not is_velocity(rec):
            continue
        lf = log_frame(rec)
        df = ev[name]
        per_epoch = int(lf["step"].iloc[0]) if len(lf) else 1
        clean = clean_rows(df, "train_chamfer_paper_mean")
        div = df[df.get(DIVERGED["train"], pd.Series(0, index=df.index)) > 0]
        last = df.iloc[-1]
        rows.append(dict(run=name, epochs=len(lf), steps=int(lf["step"].iloc[-1]) if len(lf) else 0,
                         epochs_loss_above_1=int((lf["loss"] > 1).sum()),
                         loss_last_median=float(lf["loss"].tail(last_epochs).median()),
                         best_train_chamfer_paper=float(clean["train_chamfer_paper_mean"].min()) if len(clean) else np.nan,
                         best_epoch=int(clean.loc[clean["train_chamfer_paper_mean"].idxmin(), "step"] / per_epoch) if len(clean) else -1,
                         first_diverged_epoch=int(div["step"].iloc[0] / per_epoch) if len(div) else -1,
                         evals_diverged_train=int(len(div)),
                         evals_diverged_heldout=int((df.get(DIVERGED["heldout"], pd.Series(0, index=df.index)) > 0).sum()),
                         final_train_chamfer_paper=float(last["train_chamfer_paper_mean"]),
                         final_train_chamfer_native=float(last["train_chamfer_native_mean"]),
                         final_heldout_chamfer_paper=float(last["heldout_chamfer_paper_mean"]),
                         final_train_multistart_rmse=float(last["train_multistart_rmse_mean"]),
                         train_hours=float(lf["wall"].sum() / 3600),
                         eval_hours=float(sum(float(e.get("seconds", 0.0)) for e in rec.get("evals") or []) / 3600)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=PAPER_ROOT, help="the records root (main/, velocity/, long/)")
    p.add_argument("--system", default="chain8z")
    p.add_argument("--dtype", default="fp32")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, required=True, help="directory for the tables (csv)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    recs = load_records(a.root, a.system, a.dtype, a.seed)
    print("records:", {k: f"{len(v.get('log') or [])} log entries, {len(v.get('evals') or [])} evaluations" for k, v in recs.items()})
    ev = {k: evals(v) for k, v in recs.items()}
    pd.set_option("display.width", 220, "display.max_columns", 40)
    tables = dict(floor=floor_table(ev), solver=solver_table(recs), velocity=velocity_table(recs, ev))
    for name, t in tables.items():
        print(f"\n== {name}\n{t.to_string(index=False) if len(t) else '(empty)'}")
        t.to_csv(a.out / f"long_{name}.csv", index=False)
    for name, df in ev.items():
        df.to_csv(a.out / f"long_evals_{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
