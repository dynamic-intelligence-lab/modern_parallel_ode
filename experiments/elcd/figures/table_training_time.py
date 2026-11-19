"""The table of training hours of the ELCD paper runs (hours_table.csv), from the main records through
the loaders of records.py, and the helpers the paper's figures share (the order and labels of the methods, save).
pandas (and matplotlib in save), no JAX;

  python -m experiments.elcd.figures.table_training_time --out results/elcd/check

writes hours_table.csv and prints, per system and precision, the solver time, the fallback steps and the final
training Chamfer per seed and method, for the main records and for the four timed original tasks of the appendix
(APPENDIX_SYSTEMS, appendix/ beside the main records). Over main/ alone, line 99 of hours_table.csv (pendulum, fp64,
seed 2, scale-ELK) has the first run's 11.860 h; the shipped file has the 8.250 h of its rerun and is written with
--root a copy of main/ in which npend-8-N5000/fp64/s2/scale-elk0.03.json is the record of rerun_slow/
(experiments/elcd/DETAILS.md, section 4).

Quantities:
- **solver time**: the wall of the 10,000 training steps (solves, fallback loop, adjoint and optimizer update),
  hours; evaluations left out. The loop's is its full run's where one exists (every chain8z and pendulum cell), else
  projected from its 200 timed steps (compile + median x 9,999), the rule of records.loop_hours.
- **speedup**: the loop's solver time over a parallel solver's, same cell (same GPU model).
- **fallback steps**: training steps on which at least one parallel solve failed and the sequential loop re-solved
  the step. 0 is best.
- **training Chamfer**: the native Chamfer distance of the rollout from each training demonstration's first sample,
  mean over the 3 training demonstrations, at the last evaluation. Lower is better.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from experiments.elcd.figures import records as R

APPENDIX_SYSTEMS = ("lasa-Sshape", "lasa-JShape", "lasa-Angle", "pend-8")
APPENDIX_ROOT = R.ROOT.parent / "appendix"
X_ORDER = ("seq", "deer", "scale-elk", "laser")


# ----------------------------------------------------------------------------- tables

def family(solver: str) -> str:
    """'scale-elk0.3' -> 'scale-elk', 'elk1' -> 'elk', the loop's items ('loop_time200') -> 'seq', others as they
    are."""
    if solver.startswith("loop_"):
        return "seq"
    for f in ("scale-elk", "elk"):
        if solver.startswith(f) and solver not in ("deer", "laser", "seq"):
            return f
    return solver


def rate_of(solver: str) -> Optional[float]:
    """scale-ELK's rate g of a solver name ('scale-elk0.3' -> 0.3); None for other solvers."""
    if family(solver) != "scale-elk":
        return None
    try:
        return float(solver[len("scale-elk"):])
    except ValueError:
        return None


def _plain_log(axis) -> None:
    """Ticks of a log axis at 1, 2 and 5 times the powers of ten, written as plain numbers (0.2, 5, 10)."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def x_categories(solvers: Sequence[str]) -> list:
    """The methods in the order they sit on an x axis: loop, DEER, scale-ELK by rate, LASER."""
    return sorted(set(solvers), key=lambda s: (X_ORDER.index(family(s)), rate_of(s) or 0))


def x_label(solver: str, rate: bool = True) -> str:
    """The method's tick label; scale-ELK's carries its rate g on a second line unless `rate` is False."""
    g = rate_of(solver)
    return {"seq": "loop", "deer": "DEER", "laser": "LASER"}.get(solver, "scale-ELK" + (f"\ng = {g:g}" if g and rate else ""))


def hours_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, method): solver time in hours (the loop's from R.loop_hours: measured where it ran in full,
    else projected), the speedup (NaN for the loop), the fallback steps and the final training Chamfer (the loop's
    from its full run where one exists)."""
    runs = runs[runs["solver"] != "vel"]                 # the appendix's velocity-loss arm is no solver
    sp = R.speedups(runs)
    full = runs[runs["item"] == "loop_full"].set_index(["system", "dtype", "seed"])["train_chamfer"]
    rows = [dict(system=r.system, dtype=r.dtype, seed=r.seed, solver=r.solver, hours=r.step_hours, speedup=r.speedup,
                 fallback_steps=int(r.fallback_steps), train_chamfer=r.train_chamfer) for r in sp.itertuples()]
    for (system, dtype, seed), hours in R.loop_hours(runs).items():
        rows.append(dict(system=system, dtype=dtype, seed=seed, solver="seq", hours=hours, speedup=np.nan,
                         fallback_steps=0, train_chamfer=full.get((system, dtype, seed), np.nan)))
    return pd.DataFrame(rows)


MAIN_DROP = ("scale-elk0.01",)     # the main text keeps one scale-ELK arm per system: on chain8z g = 0.03, the better one


# ----------------------------------------------------------------------------- the runner

def save(fig, out: Path, name: str) -> None:
    if fig is None:
        print(f"{name}: nothing to draw")
        return
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"{name}: {out / name}.pdf and .png")


def print_tables(ht: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for (system, dtype), g in ht.groupby(["system", "dtype"]):
        print(f"\n== {system} {dtype}: solver time (h) per seed and method; then fallback steps; then final Chamfer")
        for col in ("hours", "fallback_steps", "train_chamfer"):
            p = g.pivot_table(index="seed", columns="solver", values=col)
            p = p[x_categories(p.columns)]
            print(f"-- {col}")
            print(p.round(3).to_string())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(R.ROOT), help="the root of the main records")
    p.add_argument("--out", required=True, help="directory of hours_table.csv")
    p.add_argument("--appendix", default=None,
                   help="the appendix records root (default: appendix/ beside ELCD_PAPER_ROOT, whatever --root is)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root)
    runs = R.load_runs(root)
    ht = hours_table(runs)
    ht.to_csv(out / "hours_table.csv", index=False)
    print_tables(ht)
    appendix = Path(a.appendix) if a.appendix else APPENDIX_ROOT
    if appendix.exists():
        runs_a = R.load_runs(appendix)
        ht_a = hours_table(runs_a) if len(runs_a) else pd.DataFrame()
        if len(ht_a):
            print_tables(ht_a[ht_a["system"].isin(APPENDIX_SYSTEMS)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
