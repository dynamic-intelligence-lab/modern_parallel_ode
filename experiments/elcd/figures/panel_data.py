"""The inputs of the paper's ELCD figure, which the figure notebook (build_figure_elcd_notebook.py) draws: chain8z
fp32, the paper cells under results/elcd/records (main, velocity, long).

- load_parts: the hours table of every cell (part C), the rollouts of two trained models (part B's trajectory
  drawings) and the evaluations of the long runs of one seed (part B's Chamfer curves). The models: training
  demonstration 0 and the rollouts of the model trained with the multi-start rollout loss (LASER, 10,000 steps) from
  the demonstration's first sample and four later states (STARTS), on the seed whose final training Chamfer is the
  lowest among SEEDS (0 to 9, best_seed); the same for the model trained with the velocity loss for 1,000 epochs
  (the paper's loss and recipe, ten times its epochs; VEL_ARM), its best seed likewise. "Best" uses the native
  Chamfer (latent RK4 at the demonstration's step, decoded), its mean over the training demonstrations
  (loss_comparison.METRICS). The rollouts run JAX (the CPU is enough).
- box_table: what part C draws, the training time of 10,000 steps per method over the seeds (the median, the
  quartiles, the minimum and the maximum), the loop's measured, one scale-ELK rate (table_training_time.MAIN_DROP
  leaves the other out). The hours are the logged walls of the training steps (the solver, its fallback, the backward
  pass and the optimizer; evaluations left out).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.elcd.figures import records as R                                   # noqa: E402
from experiments.elcd.record_names import record_file                               # noqa: E402
from experiments.elcd.figures import long_runs as LG                             # noqa: E402
from experiments.elcd.figures import loss_comparison as LF                             # noqa: E402
from experiments.elcd.figures import table_training_time as MF                             # noqa: E402

LOSS_LABEL = {"suffix": "multi-start rollout loss", "vel": "velocity loss"}          # the trajectory panels' titles
COLOR = {"suffix": LG.COLOR["suffix_long"], "vel": LG.COLOR["vel_long"]}            # the losses' colours, as the curves'
STARTS = {"chain8z": (0, 1513, 3087, 4600, 6174)}   # the rollouts' starts per system, else (0,)
VEL_ARM = "vel-const1000"                           # the velocity-loss arm drawn: 1,000 epochs, the paper's constant rate
METRIC = "train_chamfer"                            # loss_comparison.METRICS: the native Chamfer over the training demonstrations
SEEDS = tuple(range(10))                            # the seeds the trajectory panels' models are picked from


def best_seed(df: pd.DataFrame, item: str, metric: str = METRIC, seeds: Optional[Sequence[int]] = None) -> Tuple[int, float]:
    """The seed (among `seeds`, all when None) whose last evaluation of `item` has the lowest finite `metric` in
    `df` (one row per evaluation: seed, item, step, the metric; records.curves or
    loss_comparison.load_velocity), and that value."""
    d = df[df["item"] == item]
    if seeds is not None:
        d = d[d["seed"].isin(list(seeds))]
    d = d.sort_values("step").groupby("seed").tail(1)
    d = d[np.isfinite(d[metric].to_numpy(float))]
    if d.empty:
        raise ValueError(f"no final {metric} of {item}")
    r = d.loc[d[metric].idxmin()]
    return int(r["seed"]), float(r[metric])


def box_table(ht: pd.DataFrame, system: str = "chain8z", dtype: str = "fp32", drop: Sequence[str] = MF.MAIN_DROP) -> pd.DataFrame:
    """What part C draws, per method: seeds, the median, the quartiles (q1, q3), the minimum and the maximum of the
    training hours of 10,000 steps (the loop's measured)."""
    t = ht[(ht["system"] == system) & (ht["dtype"] == dtype) & ~ht["solver"].isin(drop)]
    rows = []
    for c in MF.x_categories(t["solver"].unique()):
        v = t.loc[t["solver"] == c, "hours"].to_numpy(float)
        v = v[np.isfinite(v)]
        rows.append(dict(method=MF.x_label(c, rate=False), solver=c, seeds=len(v), median=np.median(v),
                         q1=np.percentile(v, 25), q3=np.percentile(v, 75), min=v.min(), max=v.max()))
    return pd.DataFrame(rows)


def load_parts(root: Path = R.ROOT, system: str = "chain8z", dtype: str = "fp32", long_seed: int = 1,
               seeds: Sequence[int] = SEEDS) -> Tuple[pd.DataFrame, list, Dict[str, pd.DataFrame]]:
    """The figures' inputs: the hours table of every cell (part C), the two trajectory cases (the best seeds among
    `seeds` of the multi-start rollout loss with LASER and of the velocity loss, rolled out from STARTS; JAX on the
    CPU) and the long runs' evaluations of `long_seed` (part B); prints the seeds picked. velocity/ and long/ sit
    beside `root`."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no run records at {root} (ELCD_PAPER_ROOT); they are not in the repository: see "
                                f"section 4, \"Figures and tables\", of experiments/elcd/DETAILS.md")
    vroot, lroot = root.parent / "velocity", root.parent
    runs = R.load_runs(root)
    cell = runs[(runs["system"] == system) & (runs["dtype"] == dtype)]
    assert set(cell.loc[cell["item"] == "loop_full", "seed"]) == set(cell["seed"]), \
        f"part C shows the measured loop: a seed of {system} {dtype} has no loop_full record"
    ht = MF.hours_table(runs)
    cv = R.curves(root, LF.METRICS)
    cv = cv[(cv["system"] == system) & (cv["dtype"] == dtype)]
    vel = LF.load_velocity(vroot)
    vel = vel[(vel["system"] == system) & (vel["dtype"] == dtype)]
    s_suffix, c_suffix = best_seed(cv, "laser", seeds=seeds)
    s_vel, c_vel = best_seed(vel, VEL_ARM, seeds=seeds)
    print(f"== part B: the best seeds among {list(seeds)} by the final training Chamfer (native): {LOSS_LABEL['suffix']} "
          f"(LASER) seed {s_suffix} ({c_suffix:.3f}), {LOSS_LABEL['vel']} ({VEL_ARM}) seed {s_vel} ({c_vel:.3f}); "
          f"the long runs of seed {long_seed}")
    starts = STARTS.get(system, (0,))
    cases = [(LOSS_LABEL["suffix"], LF.rollouts(record_file(root / system / dtype / f"s{s_suffix}" / "laser.npz"), system,
                                                starts), COLOR["suffix"]),
             (LOSS_LABEL["vel"], LF.rollouts(vroot / system / dtype / f"s{s_vel}" / f"{VEL_ARM}.npz", system, starts),
              COLOR["vel"])]
    recs = LG.load_records(lroot, system, dtype, long_seed)
    ev = {k: LG.evals(v) for k, v in recs.items()}
    print("long runs:", {k: f"{len(v)} evaluations" for k, v in ev.items()})
    return ht, cases, ev
