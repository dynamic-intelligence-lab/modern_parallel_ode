"""Loaders and the trajectory panel of the paper's ELCD figures: the velocity-loss arms, and the rollouts of trained
models with the drawing of one trajectory panel. numpy and pandas (draw_trajectory draws on matplotlib axes); JAX
only inside rollouts().

Definitions:
- **Velocity arms** (results/elcd/records/velocity): the paper's loss trained with the paper's recipe,
  vel-const100 (100 epochs, constant learning rate, evaluated every 11 epochs); the same for 1,000 epochs,
  vel-const1000 (evaluated every 100); and 1,000 epochs with the suffix runs' cosine schedule, vel-cos1000. The two
  constant-rate arms are one trajectory (the same seed draws the same batches).
- **Training time:** the sum of the logged step walls (the suffix runs: the solver's steps, the fallback inside
  them, no evaluation; the velocity arms: the epochs' walls), in GPU hours on the same B200.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.elcd.figures import records as R                                   # noqa: E402

VELOCITY_ROOT = R.ROOT.parent / "velocity"
METRICS = {"train_chamfer": "train_chamfer_native_mean", "train_chamfer_paper": "train_chamfer_paper_mean",
           "heldout_chamfer": "heldout_chamfer_native_median", "heldout_chamfer_paper": "heldout_chamfer_paper_mean",
           "multistart_rmse": "train_multistart_rmse_mean"}
GRAY = "#bdbbb6"


# ----------------------------------------------------------------------------- loaders

def load_velocity(root: Path = VELOCITY_ROOT, metrics: Dict[str, str] = METRICS) -> pd.DataFrame:
    """One row per evaluation of every velocity-loss record: system, dtype, seed, item (the arm), epoch, step (the
    optimizer steps taken), hours (the walls of the epochs before it), final (the run's last evaluation) and the
    metrics. The driver logs one entry per epoch with its end step and wall; its final evaluation's `step` is the
    epoch count (100 for the paper's recipe), which this loader maps to the last epoch's end step."""
    rows = []
    for f in sorted(Path(root).glob("*/*/s*/vel*.json")):
        if R.SKIP.search(f.stem):
            continue
        rec = R._load(f)
        if not rec or not rec.get("log") or not rec.get("evals") or rec.get("config", {}).get("loss") != "vel":
            continue
        log = rec["log"]
        ends = np.array([int(e["step"]) for e in log])
        cum = np.cumsum([float(e["wall"]) for e in log]) / 3600
        per_epoch = int(ends[0])
        cell = f.parent
        for i, ev in enumerate(rec["evals"]):
            last = i == len(rec["evals"]) - 1
            step = int(ev["step"])
            if last and step == len(log):           # the driver's final evaluation logs the epoch count as its step
                step = int(ends[-1])
            k = int(np.searchsorted(ends, step, side="right"))          # epochs that ended at or before this step
            rows.append(dict(system=cell.parent.parent.name, dtype=cell.parent.name, seed=int(cell.name[1:]),
                             item=f.stem, epoch=step / per_epoch, step=step, hours=float(cum[k - 1]) if k else 0.0,
                             final=last, **{m: ev["summary"].get(v) for m, v in metrics.items()}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sample trajectories

def rollouts(npz: Path, system: str, starts: Sequence[int], demo: int = 0) -> Dict[str, np.ndarray]:
    """The trained model at `npz` rolled out from the recorded states `starts` of demonstration `demo` to the
    demonstration's end (metrics' native rollout: latent RK4 at the demonstration's step, decoded), in raw units.
    Returns demo (N, D), paths (list of (N - k, D) arrays, one per start), x_eq (D,), segments, pos_std."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from experiments.elcd.train import data as DATA
    from experiments.elcd.train import metrics as M
    from experiments.elcd.train.train_elcd import load_model
    demos = DATA.load_system(system)
    theta = load_model(str(npz))
    run = M._native_rollout_fn(demos.dt, demos.N - 1)
    x0s = jnp.asarray(demos.pos[demo, list(starts)], jnp.float64)
    X = np.asarray(run(theta, jnp.asarray(demos.x_eq, jnp.float64), x0s))        # (S, N, D), normalized
    std = np.asarray(demos.pos_std)
    paths = [X[i, : demos.N - k] * std for i, k in enumerate(starts)]
    return dict(demo=np.asarray(demos.pos[demo]) * std, paths=paths, x_eq=np.asarray(demos.x_eq) * std,
                segments=tuple(demos.segments), pos_std=std, starts=tuple(int(k) for k in starts))


def trajectory_frame(cases: Sequence[Tuple[str, Dict[str, np.ndarray], str]], frame: float = 0.25,
                     normalized: bool = True):
    """The frame the panels of a trajectory figure share: the demonstrations' box padded by `frame` times its span,
    as (lower corner, upper corner), in the normalized coordinates unless `normalized` is False."""
    pts = np.concatenate([r["demo"] / (r["pos_std"] if normalized else 1.0) for _, r, _ in cases])
    lo, hi = pts.min(0), pts.max(0)
    return (lo - frame * (hi - lo), hi + frame * (hi - lo))


def draw_trajectory(ax, label: str, r: Dict[str, np.ndarray], color: str, lim=None, normalized: bool = True,
                    title_loc: str = "left") -> None:
    """One trajectory panel on `ax` (figure_appendix.fig_seed_trajectories): the demonstration of the
    rollouts() result `r` in gray, the equilibrium a star, the rollout from the first sample in `color`, those from
    the later starts lighter, each start marked; `lim` (trajectory_frame) sets the frame, None autoscales; `label` is
    the title at the left."""
    scale = r["pos_std"] if normalized else np.ones_like(r["pos_std"])
    demo, x_eq = r["demo"] / scale, r["x_eq"] / scale
    ax.plot(demo[:, 0], demo[:, 1], color=GRAY, lw=2.0, zorder=1, solid_capstyle="round")
    ax.scatter(*x_eq, marker="*", s=70, color="black", zorder=5)
    for k, path in zip(r["starts"], r["paths"]):
        path = path / scale
        first = k == 0
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.1 if first else 0.9, alpha=1.0 if first else 0.65, zorder=3)
        ax.scatter(path[0, 0], path[0, 1], color=color, s=18 if first else 10, zorder=4, edgecolor="white", lw=0.4)
    ax.set_aspect("equal")
    if lim is not None:
        ax.set_xlim(lim[0][0], lim[1][0])
        ax.set_ylim(lim[0][1], lim[1][1])
    ax.set_title(label, loc=title_loc)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
