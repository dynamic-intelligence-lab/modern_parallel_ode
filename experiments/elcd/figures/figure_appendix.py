"""The ELCD appendix figures in the paper's style (DejaVu Sans, figs/colors.json, panel letters). fp32.

- **mechanism_chain8z_fp32_seed0** (fig:elcd_mechanism): seed 0, DEER against LASER over the 10,000 training steps. A: the time of
  each step (dots; lines: the rolling median over 51 steps); B: the Newton iterations of the step's slowest solve.
  The steps the sequential solver re-solved are drawn in its colour. The compiling step is left out. Read from the
  two run records: no JAX.
- **mechanism_pendulum_fp32_seed0** (fig:elcd_pendulum_mechanism): the same for the eight-link pendulum (seed 0). Dashed
  line in A: the median step time of the sequential solver's run of that cell.
- **seed_trajectories_chain8z_fp32** (fig:elcd_seed_trajectories): training demonstration 0 and the rollouts from five of
  its states (the panels of panel_data's part B, which shows the best seed of each loss), for the five seeds
  of each loss with the lowest final training Chamfer (native), the best at the left. A: the multi-start rollout
  loss (LASER, 10,000 steps); B: the velocity loss (1,000 epochs, constant rate). The rollouts need JAX (the CPU is
  enough: slurm/figure_appendix.slurm); those of all ten seeds are cached in <out>/seed_trajectories.pkl,
  and `--cached` redraws from the cache without JAX. The cache holds the rollouts only: the demonstration is read
  from the LASA dataset in both cases (the dataset may not be redistributed).
- **original_tasks_loss_fp32** (fig:elcd_lasa_loss): the original tasks of Jaffe et al. (the 30 LASA shapes and
  their 8-link pendulum; appendix/ beside the main root). One point per task, the mean over seeds 0 to 2 of the last
  evaluation: the rollout loss (LASER) against the velocity loss (their recipe, 100 epochs). A: training Chamfer
  (native); B: held-out Chamfer (native); C: multi-start RMSE on the training demonstrations. Read from the run
  records: no JAX.

  python -m experiments.elcd.figures.figure_appendix --out <dir> [--cached] [mechanism] [pendulum] [trajectories] [original]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.elcd.figures import records as R                                   # noqa: E402
from experiments.elcd.record_names import record_file                               # noqa: E402
from experiments.elcd.figures import panel_data as PF                                # noqa: E402
from experiments.elcd.figures import loss_comparison as LF                             # noqa: E402
from experiments.elcd.figures import table_training_time as MF                             # noqa: E402

PALETTE = json.loads((REPO / "figs/colors.json").read_text())       # the colors shared by the paper figures
INK = PALETTE["neutral"]["text"]
GRID = matplotlib.colors.to_hex([0.9 + 0.1 * v for v in matplotlib.colors.to_rgb(INK)])   # the ink, nine parts white
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
         "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.minor.width": 0.45, "ytick.minor.width": 0.45, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
         "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "lines.linewidth": 1.0, "grid.linewidth": 0.5,
         "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "axes.labelcolor": INK,
         "xtick.labelcolor": INK, "ytick.labelcolor": INK,
         "pdf.fonttype": 42}                     # the main figure's style (build_figure_elcd_notebook.STYLE)
LETTER_PT = 10                                   # the panel letters, bold capitals, as in the main figure
WIDTH = 6.6                                      # inches, the text width
SOLVERS = (("deer", "DEER"), ("laser", "LASER"))                     # the record's name, the name in colors.json
LOSSES = (("multi-start rollout loss", "laser"), ("velocity loss", PF.VEL_ARM))   # the name in colors.json, the record
CACHE = "seed_trajectories.pkl"
# figure: system, precision, seed, draw the sequential solver's step time, file name
MECHANISMS = {"mechanism": ("chain8z", "fp32", 0, False, "mechanism_chain8z_fp32_seed0"),
              "pendulum": ("npend-8-N5000", "fp32", 0, True, "mechanism_pendulum_fp32_seed0")}
# the original tasks: the summary's key and the panel's name; the loss and its record
ORIGINAL_METRICS = (("train_chamfer_native_mean", "Chamfer distance, training"),
                    ("heldout_chamfer_native_mean", "Chamfer distance, held-out"),
                    ("train_multistart_rmse_mean", "multi-start RMSE, training"))
ORIGINAL_ARMS = (("velocity", "vel.json"), ("rollout", "laser.json"))


def _letter(ax, letter: str, name: str, dy: float = 4.0) -> None:
    """The panel letter and the panel's name above the left end of `ax`, `dy` points above it."""
    ax.annotate(letter, (0, 1), xytext=(-2, dy), xycoords="axes fraction", textcoords="offset points",
                fontsize=LETTER_PT, fontweight="bold", ha="right", va="bottom")
    ax.annotate(name, (0, 1), xytext=(3, dy + 0.8), xycoords="axes fraction", textcoords="offset points",
                ha="left", va="bottom")


# ----------------------------------------------------------------------------- the solver over training

def step_frame(root: Path, system: str = "chain8z", dtype: str = "fp32", seed: int = 0) -> pd.DataFrame:
    """One row per training step of the DEER and LASER records of a cell, the compiling step left out: item, step,
    wall (s), iters_max (the Newton iterations of the step's slowest solve), fallback (the sequential solver
    re-solved the step)."""
    rows = []
    for item, _ in SOLVERS:
        rec = json.loads(record_file(Path(root) / system / dtype / f"s{seed}" / f"{item}.json").read_text())
        rows += [dict(item=item, step=int(e["step"]), wall=float(e["wall"]), iters_max=int(e["iters_max"]),
                      fallback=int(e.get("seq_fallback") or 0) > 0) for e in rec["log"] if not e.get("compile")]
    return pd.DataFrame(rows)


def seq_step_time(root: Path, system: str, dtype: str, seed: int) -> float:
    """The median time (s) of a training step in the sequential solver's run of a cell, the compiling step left out."""
    rec = json.loads((Path(root) / system / dtype / f"s{seed}" / "loop_full.json").read_text())
    return float(np.median([e["wall"] for e in rec["log"] if not e.get("compile")]))


def fig_mechanism(g: pd.DataFrame, window: int = 51, width: float = WIDTH, height: float = 2.3, seq_step: float = None):
    """A: the time of each training step, with its rolling median over `window` steps; B: the iterations of the
    step's slowest solve. `g`: step_frame(). `seq_step`: seq_step_time(), drawn in A as a dashed line."""
    seq = PALETTE["methods"]["seq"]
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(width, height))
        for item, name in SOLVERS:
            h = g[g["item"] == item].sort_values("step")
            c, fb = PALETTE["methods"][name], h["fallback"].to_numpy()
            for ax, y in ((a, "wall"), (b, "iters_max")):
                ax.scatter(h["step"][~fb], h[y][~fb], s=2.5, color=c, alpha=0.4, lw=0, zorder=2)
                ax.scatter(h["step"][fb], h[y][fb], s=3.5, color=seq, alpha=0.8, lw=0, zorder=3)
            a.plot(h["step"], h["wall"].rolling(window, center=True, min_periods=1).median(), color=c, zorder=4)
        if seq_step is not None:
            a.axhline(seq_step, color=seq, ls=(0, (4, 2)), zorder=1)
            a.annotate("sequential solver", (1, seq_step), xycoords=a.get_yaxis_transform(), xytext=(-2, 3),
                       textcoords="offset points", ha="right", va="bottom")
        for ax, letter, name, ylabel in ((a, "A", "time of each training step", "time (s)"),
                                         (b, "B", "iterations of the step's slowest solve", "Newton iterations")):
            ax.set(yscale="log", xlabel="training step", ylabel=ylabel, xlim=(-150, 10150))
            ax.set_xticks(range(0, 10001, 2000))
            ax.set_xticklabels([f"{t:,}" for t in range(0, 10001, 2000)])
            MF._plain_log(ax.yaxis)
            ax.grid(axis="y", color=GRID, zorder=0)
            ax.tick_params(which="minor", left=False)
            _letter(ax, letter, name)
        hs = [Line2D([], [], color=PALETTE["methods"][name], marker="o", ls="", ms=3.5, label=name) for _, name in SOLVERS]
        hs.append(Line2D([], [], color=seq, marker="o", ls="", ms=3.5, label="step re-solved by the sequential solver"))
        fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.6,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
    return fig


# ----------------------------------------------------------------------------- the rollouts of every seed

def final_chamfer(record: Path) -> float:
    """The training Chamfer (native) of the record's last evaluation."""
    return float(json.loads(Path(record).read_text())["evals"][-1]["summary"][LF.METRICS["train_chamfer"]])


def compute_rollouts(root: Path, system: str = "chain8z", dtype: str = "fp32", seeds: Sequence[int] = PF.SEEDS) -> Dict:
    """{loss: {seed: (final training Chamfer, loss_comparison.rollouts() of the trained model)}} for the two
    LOSSES; velocity/ sits beside `root`. JAX."""
    roots = {"laser": Path(root), PF.VEL_ARM: Path(root).parent / "velocity"}
    out = {}
    for loss, item in LOSSES:
        out[loss] = {}
        for seed in seeds:
            cell = roots[item] / system / dtype / f"s{seed}"
            out[loss][seed] = (final_chamfer(record_file(cell / f"{item}.json")),
                               LF.rollouts(record_file(cell / f"{item}.npz"), system, PF.STARTS[system]))
            print(f"{loss} ({item}) seed {seed}: final training Chamfer {out[loss][seed][0]:.3f}", flush=True)
    return out


def without_demo(roll: Dict) -> Dict:
    """`roll` without the demonstration of each entry, for the cache file."""
    return {loss: {seed: (c, {k: v for k, v in r.items() if k != "demo"}) for seed, (c, r) in by.items()}
            for loss, by in roll.items()}


def with_demo(roll: Dict, system: str = "chain8z", demo: int = 0) -> Dict:
    """The inverse of without_demo: the demonstration in raw units as loss_comparison.rollouts() returns it,
    rebuilt from the dataset. NumPy only."""
    from experiments.elcd.train import data as DATA
    pos = np.asarray(DATA.load_system(system).pos[demo])
    return {loss: {seed: (c, dict(r, demo=pos * r["pos_std"])) for seed, (c, r) in by.items()}
            for loss, by in roll.items()}


def best_seeds(roll: Dict, loss: str, n: int) -> Sequence[Tuple[int, Tuple[float, Dict]]]:
    """The `n` seeds of `loss` with the lowest final training Chamfer, the best first."""
    return sorted(roll[loss].items(), key=lambda kv: kv[1][0])[:n]


def fig_seed_trajectories(roll: Dict, n: int = 5, width: float = WIDTH, height: float = 3.3, frame: float = 0.12):
    """One row per loss, one panel per seed (best_seeds): the demonstration in gray, the rollouts in the loss's
    colour, every panel in the frame of the demonstration padded by `frame` (a rollout that leaves it is cut off).
    Above each panel: the seed and its final training Chamfer."""
    picked = {loss: best_seeds(roll, loss, n) for loss, _ in LOSSES}
    lim = LF.trajectory_frame([("", r, "") for loss in picked for _, (_, r) in picked[loss]], frame=frame)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(LOSSES), n, figsize=(width, height))
        for row, letter, (loss, _) in zip(axes, "AB", LOSSES):
            for ax, (seed, (chamfer, r)) in zip(row, picked[loss]):
                LF.draw_trajectory(ax, f"seed {seed}, {chamfer:.2f}", r, PALETTE["losses"][loss], lim,
                                   title_loc="center")
            _letter(row[0], letter, loss, dy=13.0)
        fig.tight_layout(w_pad=0.6, h_pad=1.6)
    return fig


# ----------------------------------------------------------------------------- the original tasks

def original_frame(root: Path, seeds: Sequence[int] = (0, 1, 2)) -> pd.DataFrame:
    """One row per task (the 30 LASA shapes and pend-8), precision, seed and loss under `root` (appendix/): the
    ORIGINAL_METRICS of the record's last evaluation."""
    rows = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for dtype in ("fp32", "fp64"):
            for seed in seeds:
                for arm, name in ORIGINAL_ARMS:
                    sm = json.loads(record_file(d / dtype / f"s{seed}" / name).read_text())["evals"][-1]["summary"]
                    rows.append(dict(system=d.name, dtype=dtype, seed=seed, arm=arm,
                                     **{k: float(sm[k]) for k, _ in ORIGINAL_METRICS}))
    df = pd.DataFrame(rows)
    assert len(df) == 31 * 2 * len(seeds) * len(ORIGINAL_ARMS), len(df)
    assert np.isfinite(df[[k for k, _ in ORIGINAL_METRICS]].to_numpy()).all()
    return df


def task_means(df: pd.DataFrame, dtype: str, metric: str) -> pd.DataFrame:
    """Index: task; columns: velocity, rollout; the mean of `metric` over the seeds."""
    return df[df["dtype"] == dtype].pivot_table(index="system", columns="arm", values=metric)


def fig_original_tasks(df: pd.DataFrame, dtype: str = "fp32", width: float = WIDTH, height: float = 2.45):
    """One panel per metric: the rollout loss against the velocity loss, one point per task (task_means), the two
    axes on the same logarithmic range. Dotted line: equal values. Diamond: the pendulum. `df`: original_frame()."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(ORIGINAL_METRICS), figsize=(width, height))
        for ax, letter, (metric, name) in zip(axes, "ABC", ORIGINAL_METRICS):
            m = task_means(df, dtype, metric)
            lasa, pend = m[m.index.str.startswith("lasa")], m.loc["pend-8"]
            lo, hi = m.to_numpy().min() / 1.5, m.to_numpy().max() * 1.5
            ax.plot([lo, hi], [lo, hi], color=INK, ls=":", lw=0.8, zorder=1)
            ax.scatter(lasa["velocity"], lasa["rollout"], s=16, color=PALETTE["losses"]["multi-start rollout loss"],
                       edgecolor="white", lw=0.5, zorder=3)
            ax.scatter([pend["velocity"]], [pend["rollout"]], s=22, marker="D", facecolor="white", edgecolor=INK,
                       lw=0.9, zorder=4)
            # the label goes below the diamond, or to its right when the diamond is in the lowest eighth of the axis
            right = np.log(pend["rollout"] / lo) < 0.125 * np.log(hi / lo)
            ax.annotate("pendulum", (pend["velocity"], pend["rollout"]), xytext=(6, 0) if right else (4, -8),
                        textcoords="offset points", ha="left" if right else "center", va="center" if right else "top")
            lower = int((lasa["rollout"] < lasa["velocity"]).sum())
            ax.text(0.04, 0.97, f"rollout loss lower\non {lower} of {len(lasa)} shapes", transform=ax.transAxes,
                    ha="left", va="top")
            ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="velocity loss")
            MF._plain_log(ax.xaxis)
            MF._plain_log(ax.yaxis)
            _letter(ax, letter, name)
        axes[0].set_ylabel("rollout loss")
        fig.tight_layout(w_pad=1.0)
    return fig


# ----------------------------------------------------------------------------- command line

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("figures", nargs="*", default=["mechanism", "pendulum", "trajectories", "original"],
                   help="mechanism, pendulum, trajectories, original (default: all four)")
    p.add_argument("--root", type=Path, default=R.ROOT,
                   help="the main records' root; velocity/ and appendix/ sit beside it")
    p.add_argument("--out", type=Path, required=True, help="directory for the figures (pdf and png) and the cache")
    p.add_argument("--cached", action="store_true",
                   help=f"draw the trajectories from <out>/{CACHE} (no JAX; the demonstration from the LASA dataset)")
    a = p.parse_args(argv)
    assert set(a.figures) <= {*MECHANISMS, "trajectories", "original"}, a.figures
    matplotlib.use("Agg")
    a.out.mkdir(parents=True, exist_ok=True)
    for name in [f for f in a.figures if f in MECHANISMS]:
        system, dtype, seed, with_seq, stem = MECHANISMS[name]
        g = step_frame(a.root, system, dtype, seed)
        for item, h in g.groupby("item"):
            fb, late = h[h["fallback"]], h[h["step"] >= 2100]
            print(f"{item}: {len(h)} steps, {len(fb)} re-solved by the sequential solver (last at step "
                  f"{fb['step'].max() if len(fb) else 'none'}), slowest solve {h['iters_max'].max()} iterations "
                  f"({late['iters_max'].max()} from step 2,100 on, median {late['iters_max'].median():g}), "
                  f"median step time {h['wall'].median():.4f} s")
        seq_step = seq_step_time(a.root, system, dtype, seed) if with_seq else None
        if with_seq:
            print(f"sequential solver: median step time {seq_step:.4f} s")
        MF.save(fig_mechanism(g, seq_step=seq_step), a.out, stem)
    if "trajectories" in a.figures:
        if a.cached:
            roll = with_demo(pickle.loads((a.out / CACHE).read_bytes()))
        else:
            roll = compute_rollouts(a.root)
            (a.out / CACHE).write_bytes(pickle.dumps(without_demo(roll)))
        for loss, _ in LOSSES:
            print(f"{loss}: best first", [(s, round(c, 3)) for s, (c, _) in best_seeds(roll, loss, len(roll[loss]))])
        MF.save(fig_seed_trajectories(roll), a.out, "seed_trajectories_chain8z_fp32")
    if "original" in a.figures:
        df = original_frame(a.root.parent / "appendix")
        for dtype in ("fp32", "fp64"):
            for metric, name in ORIGINAL_METRICS:
                m = task_means(df, dtype, metric)
                lasa, pend = m[m.index.str.startswith("lasa")], m.loc["pend-8"]
                print(f"{dtype} {name}: rollout loss lower on {int((lasa['rollout'] < lasa['velocity']).sum())} of "
                      f"{len(lasa)} shapes (velocity loss lower on "
                      f"{', '.join(i[5:] for i in lasa.index[lasa['rollout'] >= lasa['velocity']]) or 'none'}); "
                      f"pendulum {pend['rollout']:.4f} against {pend['velocity']:.4f}")
        MF.save(fig_original_tasks(df), a.out, "original_tasks_loss_fp32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
