"""Appendix figure: residual against iteration at N = 1e4 and 1e5, bistable Lorenz, best constants of the tuning sweep.

Data: results/lorenz/table_timing/residual_histories.json, written by run_residual_histories.py in the job
slurm/residual_histories.slurm (NVIDIA B200). Check: the iteration counts must be those of the tuning sweep
(results/lorenz/tuning_sweep, stages A and B).

  python experiments/lorenz/figure_residual.py [HISTORIES.json [OUT_PREFIX]]    # numpy and matplotlib; login node

Writes figs/lorenz/lorenz_residual_long_horizons.{pdf,png} by default, one panel per horizon of the file. The iteration
axis is cut at X_MAX; a solver that needs more iterations is marked at the cut with its count.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning_sweep as SA          # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5})
COLORS = json.load(open(os.path.join(SA.ROOT, "figs", "colors.json")))
INK, C = COLORS["neutral"]["text"], COLORS["methods"]
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                   os.path.join(SA.ROOT, "results", "lorenz", "table_timing", "residual_histories.json")))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SA.ROOT, "figs", "lorenz", "lorenz_residual_long_horizons")
assert (D["k"], D["sigmasq"]) == (0.053, 3e4), D
SERIES = (("deer", "DEER", C["DEER"], 1.2), ("scale_elk", r"scale-ELK, $k = 0.053$", C["scale-ELK"], 1.4),
          ("elk", r"ELK, $\sigma^2 = 3\times 10^4$", C["ELK"], 1.4), ("laser", "LASER", C["LASER"], 2.0))
Y_LO, Y_HI = 1e-13, 1e3
X_MAX = 250


def record_iters(n, key):
    """Iterations of the tuning sweep (200 iterations, then 2,000)."""
    stage, method = {"scale_elk": ("A", SA.mname("scale-elk", D["k"])), "elk": ("B", SA.mname("elk", D["sigmasq"])),
                     "laser": ("A", SA.LASER)}[key]
    r = SA.load(os.path.join(SA.H, f"stage{stage}", "summary.csv"))[(n, method)]
    if SA.outcome(r, 200) == "cap":
        r = SA.load(os.path.join(SA.H, f"stage{stage}_cap2000", "summary.csv"))[(n, method)]
    return int(r["iters"])


def main():
    ns = sorted(int(n) for n in D["histories"])
    fig, axes = plt.subplots(1, len(ns), figsize=(3.3 * len(ns), 2.5), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, letter, n in zip(axes, "ABC", ns):
        h = D["histories"][str(n)]
        for key, label, color, lw in SERIES:
            v = np.asarray(h[key], float)
            it = len(v) - 1
            if key == "deer":                     # non-finite after the first step: drawn leaving the axis
                assert it == 1 and not np.isfinite(v[1]), (n, v)
                ax.plot([0, 1], [v[0], Y_HI], color=color, lw=lw, label=label, zorder=3)
                ax.plot([1], [Y_HI], "x", color=color, ms=5, mew=1.4, clip_on=False, zorder=4)
                ax.annotate("non-finite", (1, Y_HI), xytext=(5, -1), textcoords="offset points", ha="left", va="top",
                            fontsize=7, color=INK)
                continue
            assert np.all(np.isfinite(v)) and v[-1] < D["tol"] and it == record_iters(n, key), (n, key, it)
            ax.plot(np.arange(it + 1), v, color=color, lw=lw, label=label, zorder=4 if key == "laser" else 3)
            if it > X_MAX:                        # ends beyond the cut: its iteration count at the right edge
                ax.annotate(f"{it:,} \u2192", (X_MAX, v[X_MAX]), xytext=(0, 4), textcoords="offset points", ha="right",
                            va="bottom", fontsize=7, color=INK)
                continue
            # iteration count at the end of the curve; the two baselines can end close together, so the earlier one
            # is labelled on its left and the later one on its right
            other = len(h["elk" if key == "scale_elk" else "scale_elk"]) - 1
            left = key != "laser" and it < other
            ax.annotate(f"{it:,}", (it, v[-1]), xytext=(-4, -5) if left else (5, 0 if key == "laser" else -5),
                        textcoords="offset points", ha="right" if left else "left", va="center", fontsize=7, color=INK)
        ax.axhline(D["tol"], color=INK, ls=":", lw=0.8, zorder=1)
        ax.set_xlim(-0.02 * X_MAX, X_MAX)
        ax.set_yscale("log")
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Newton iteration")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
        ax.text(0.02, 1.06, rf"$N = 10^{int(np.log10(n))}$", transform=ax.transAxes, fontsize=8.5, ha="left",
                va="bottom")
    axes[0].set_ylabel(r"residual $\max_n \|r_n\|_\infty$")
    axes[0].text(0.3 * X_MAX, D["tol"] * 2.0, "tolerance", ha="left", va="bottom", fontsize=7, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=4, frameon=False, fontsize=7.5,
               handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=1.2)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=300)
    print({n: {k: len(v) - 1 for k, v in h.items() if k != "laser_rejected"} for n, h in D["histories"].items()})
    print(OUT + ".pdf")


if __name__ == "__main__":
    main()
