"""Tables of the ELCD paper runs: loaders and
tables over the records that experiments/elcd/train/run_cell.py writes. numpy and pandas only, no JAX.

Layout: <root>/<system>/<dtype>/s<seed>/<item>.json, with <item> deer, laser, scale-elk<g>, elk<s2>, loop_time<n>
or loop_full, plus probe_<kind>_<k>.json, snapshots <item>_at<S>.json and cell.json. Every loader returns None (or an
empty frame) when its input is missing. The paper's records name LASER by an earlier name, which item_name
(experiments/elcd/record_names.py) maps to "laser".

Definitions:
- **Step hours of a run:** every logged step's wall summed, the compiling step included, evaluations left out.
  The clock covers the warm buffer's copies, the jitted step and, for the paper protocol, the
  sequential fallback inside it.
- **The loop's step hours of a cell:** measured, its full run's, where one exists (every chain8z and pendulum
  cell); else projected from its timing run, the compiling step's wall plus the median of the other walls times the
  remaining steps of the parallel runs (loop_projection; the timed original tasks). See loop_hours.
- **Speedup:** the loop's step hours over a solver's, same cell. Higher is better; 1 is the loop.
- **Fallback hours:** the steps on which the loop re-solved the step, times the forward loop's measured wall over all
  starts (record solve.fallback_seconds.all): what the failures cost the solver, a part of its step hours.
- **Step 0 of a cell:** the cell's initial model, drawn from the seed alone, evaluated as the driver evaluates a
  trained one (run_init_eval.py; init_eval.json beside the records, read by load_init_evals). Every solver of a
  cell starts from it.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from experiments.elcd.record_names import item_name

ROOT = Path(os.environ.get("ELCD_PAPER_ROOT", "results/elcd/records/main"))
CAUSES = ("failed_nonfinite", "failed_max_iter", "failed_stalled")
METRICS = {"train_chamfer": "train_chamfer_native_mean", "heldout_chamfer": "heldout_chamfer_native_median",
           "multistart_rmse": "train_multistart_rmse_mean"}
LOOP_TIME = re.compile(r"^loop_time(?P<n>\d+)$")
SKIP = re.compile(r"^(cell|probe)|_at\d+$|\.stale-")
INIT_ITEM = "init_eval"


def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def solver_of(item: str) -> str:
    """The solver of an item name: 'seq' for the loop's runs, else the name ('scale-elk0.3', 'deer')."""
    return "seq" if item.startswith("loop_") else item


def run_row(path: Path) -> Optional[Dict]:
    """One run's numbers (see the module docstring); None for a record without a log."""
    rec = _load(path)
    if not rec or not rec.get("log"):
        return None
    log = rec["log"]
    walls = [e["wall"] for e in log]
    rest = [e["wall"] for e in log if not e.get("compile")]
    final = rec.get("final") or {}
    start = (rec.get("provenance") or {}).get("start") or {}
    gpus = start.get("gpus") or [{}]
    fb = ((rec.get("solve") or {}).get("fallback_seconds") or {}).get("all", {}).get("seconds")
    fb_steps = sum(int(e.get("seq_fallback", 0)) for e in log)
    item = item_name(path.stem)
    return dict(item=item, solver=solver_of(item), steps=len(log), steps_planned=rec["config"]["steps"],
                time_only=rec["config"].get("time_only", 0), step_hours=sum(walls) / 3600,
                compile_s=sum(e["wall"] for e in log if e.get("compile")),
                median_s=statistics.median(rest) if rest else None,
                fallback_steps=fb_steps, fallback_solves=sum(int(e.get("seq_fallback_solves", 0)) for e in log),
                fallback_hours=None if fb is None else fb_steps * fb / 3600,
                solves=sum(int(e.get("n_solves", 0)) for e in log),
                **{c: sum(int(e.get(c, 0)) for e in log) for c in CAUSES},
                iters_mean=float(np.mean([e["iters_mean"] for e in log])) if "iters_mean" in log[0] else None,
                **{k: final.get(v) for k, v in METRICS.items()},
                host=start.get("host"), gpu=gpus[0].get("uuid") if gpus else None, device_kind=rec.get("device_kind"))


def load_runs(root: Path = ROOT) -> pd.DataFrame:
    """One row per run record under root (probes, snapshots, summaries and stale records left out)."""
    rows = []
    for cell in sorted(Path(root).glob("*/*/s*")):
        for f in sorted(cell.glob("*.json")):
            if SKIP.search(f.stem):
                continue
            row = run_row(f)
            if row:
                rows.append(dict(system=cell.parent.parent.name, dtype=cell.parent.name, seed=int(cell.name[1:]), **row))
    return pd.DataFrame(rows)


def loop_projection(runs: pd.DataFrame, steps: Optional[int] = None) -> pd.DataFrame:
    """Per cell, the loop's projected step hours from its timing run: compile + median x (steps - 1), steps being
    the parallel runs' planned steps (or `steps`); and the full loop run's step hours where one exists."""
    out = []
    for (system, dtype, seed), g in runs.groupby(["system", "dtype", "seed"]):
        timing = g[g["item"].str.match(LOOP_TIME.pattern)]
        full = g[g["item"] == "loop_full"]
        par = g[g["solver"] != "seq"]
        n = steps or (int(par["steps_planned"].max()) if len(par) else None)
        row = dict(system=system, dtype=dtype, seed=seed, steps=n)
        if len(timing) and n:
            t = timing.iloc[0]
            row.update(loop_median_s=t["median_s"], loop_hours=(t["compile_s"] + t["median_s"] * (n - 1)) / 3600)
        if len(full):
            row["loop_full_hours"] = float(full.iloc[0]["step_hours"])
        out.append(row)
    return pd.DataFrame(out)


def loop_hours(runs: pd.DataFrame) -> pd.Series:
    """Per cell (system, dtype, seed), the loop's step hours: measured, its full run's, where one exists; else the
    projection from its timing run (loop_projection). A cell with neither has no entry."""
    proj = loop_projection(runs).set_index(["system", "dtype", "seed"]).reindex(columns=["loop_full_hours", "loop_hours"])
    return proj["loop_full_hours"].fillna(proj["loop_hours"]).dropna()


def speedups(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, parallel solver) of a complete run: its step hours, the loop's hours (loop_hours: measured
    where the loop ran in full) and the speedup, with the fallback steps and hours."""
    loop = loop_hours(runs)
    par = runs[(runs["solver"] != "seq") & (runs["time_only"] == 0)].copy()
    par["loop_hours"] = [loop.get(k, np.nan) for k in zip(par["system"], par["dtype"], par["seed"])]
    par["speedup"] = par["loop_hours"] / par["step_hours"]
    return par[["system", "dtype", "seed", "solver", "steps", "step_hours", "loop_hours", "speedup", "fallback_steps",
                "fallback_hours", *CAUSES, "train_chamfer", "heldout_chamfer"]]


def curves(root: Path = ROOT, metrics: Dict[str, str] = METRICS) -> pd.DataFrame:
    """One row per evaluation of every run: the step, the step hours before it (the walls of the steps the
    evaluated parameters had taken, the step-hours clock) and the metrics (column name -> summary key; METRICS by
    default); the final evaluation sits at the total."""
    rows = []
    for cell in sorted(Path(root).glob("*/*/s*")):
        for f in sorted(cell.glob("*.json")):
            if SKIP.search(f.stem):
                continue
            rec = _load(f)
            if not rec or not rec.get("log") or not rec.get("evals"):
                continue
            cum = np.concatenate([[0.0], np.cumsum([e["wall"] for e in rec["log"]])]) / 3600
            item = item_name(f.stem)
            for ev in rec["evals"]:
                k = min(int(ev["step"]), len(rec["log"]))
                rows.append(dict(system=cell.parent.parent.name, dtype=cell.parent.name, seed=int(cell.name[1:]),
                                 item=item, solver=solver_of(item), step=int(ev["step"]), hours=float(cum[k]),
                                 **{m: ev["summary"].get(v) for m, v in metrics.items()}))
    return pd.DataFrame(rows)


def load_init_evals(root: Path = ROOT, metrics: Dict[str, str] = METRICS) -> pd.DataFrame:
    """One row per cell with an init_eval.json: system, dtype, seed, the evaluation's seconds and the metrics
    (column name -> summary key, as curves takes them; NaN where the summary lacks one)."""
    rows = []
    for f in sorted(Path(root).glob(f"*/*/s*/{INIT_ITEM}.json")):
        rec = _load(f)
        if not rec or not rec.get("evals"):
            continue
        ev = rec["evals"][0]
        cell = f.parent
        vals = {m: ev["summary"].get(v) for m, v in metrics.items()}
        rows.append(dict(system=cell.parent.parent.name, dtype=cell.parent.name, seed=int(cell.name[1:]),
                         seconds=float(ev.get("seconds", np.nan)), **{m: np.nan if x is None else float(x) for m, x in vals.items()}))
    return pd.DataFrame(rows)
