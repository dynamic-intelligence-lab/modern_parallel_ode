"""Appendix figure: iterations to tolerance of scale-ELK and ELK against their constant, bistable Lorenz.

Bistable Lorenz (rho 10, x0 (1,1,1), explicit Euler dt 0.01, fp64), zero guess, tolerance 1e-9.
Data: results/lorenz/tuning_sweep, stages A and B (200 iterations, runs that had not converged
rerun with 2,000), read with the loader and the outcome rule of tuning_sweep.py.

  python experiments/lorenz/figure_tuning.py     # standard library, numpy and matplotlib; login node

Writes figs/lorenz/lorenz_baseline_tuning.{pdf,png}.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning_sweep as SA          # noqa: E402

NS = [1000, 10000, 100000]
# lambda_h of the zero guess: lam_zero of the rho = 10 rows of results/lorenz/spectra.csv (5.3217076), to the four
# decimals with which the shipped figure was drawn
LAMBDA_GUESS, DT = 5.3217, 0.01
Y_CAP, Y_NAN, Y_SPLIT, DODGE = 3500.0, 9000.0, 2350.0, 1.27   # rows above the axis: "> 2,000" and "non-finite"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5})
COLORS = json.load(open(os.path.join(SA.ROOT, "figs", "colors.json")))
INK, TEAL = COLORS["neutral"]["text"], COLORS["methods"]["LASER"]


def tint(c, t):
    """Color c mixed with a fraction t of white."""
    return tuple(t + (1.0 - t) * v for v in mcolors.to_rgb(c))


STYLE = {1000: dict(color=tint(INK, 0.66), marker="o"), 10000: dict(color=tint(INK, 0.36), marker="s"),
         100000: dict(color=INK, marker="^")}
ROWS = {s: (SA.load(os.path.join(SA.H, f"stage{s}", "summary.csv")),
            SA.load(os.path.join(SA.H, f"stage{s}_cap2000", "summary.csv"))) for s in "AB"}


def result(stage, n, method):
    """('conv', iterations), ('cap', None) for no convergence in 2,000 iterations, or ('nan', None)."""
    r = ROWS[stage][0][(n, method)]
    o = SA.outcome(r, 200)
    if o == "cap":
        r = ROWS[stage][1][(n, method)]
        o = SA.outcome(r, 2000)
    assert o in ("conv", "cap", "nan"), (stage, n, method, o)
    return o, int(r["iters"]) if o == "conv" else None


def panel(ax, stage, prefix, grid, xs):
    best = {}
    for n in NS:
        res = [result(stage, n, SA.mname(prefix, v)) for v in grid]
        conv = [(x, it) for x, (o, it) in zip(xs, res) if o == "conv"]
        st = STYLE[n]
        ax.plot(*zip(*conv), "-", color=st["color"], marker=st["marker"], ms=4.6, mec="white", mew=0.5, lw=1.3,
                label=rf"$N = 10^{int(np.log10(n))}$", zorder=3, clip_on=False)
        for o, y in (("cap", Y_CAP), ("nan", Y_NAN)):
            xf = [x for x, (oo, _) in zip(xs, res) if oo == o]
            ax.plot(xf, [y * DODGE ** (NS.index(n) - 1)] * len(xf), ls="none", color=st["color"], marker=st["marker"],
                    ms=3.6, mec="white", mew=0.4, zorder=3)
        best[n] = min(conv, key=lambda c: c[1])
    laser = [int(ROWS["A"][0][(n, SA.LASER)]["iters"]) + int(ROWS["A"][0][(n, SA.LASER)]["rejected"]) for n in NS]
    ax.axhline(np.sqrt(min(laser) * max(laser)), color=TEAL, lw=1.6, zorder=2)
    ax.axhspan(Y_SPLIT, 1e5, color=tint(INK, 0.95), lw=0, zorder=0)
    ax.axhline(Y_SPLIT, color=tint(INK, 0.7), lw=0.6, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(15, Y_NAN * DODGE ** 1.9)
    ax.set_yticks([20, 50, 100, 200, 500, 1000, 2000, Y_CAP, Y_NAN])
    ax.set_yticklabels(["20", "50", "100", "200", "500", "1,000", "2,000", "> 2,000", "non-finite"])
    ax.minorticks_off()
    for y in (50, 100, 200, 500, 1000, 2000):
        ax.axhline(y, color=tint(INK, 0.9), lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return best, laser


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.95), sharey=True)

    # A: scale-ELK. The tested k crowd around the threshold, so they are drawn equally spaced.
    xa = list(range(len(SA.K_GRID)))
    best_a, laser = panel(a, "A", "scale-elk", SA.K_GRID, xa)
    a.set_xticks(xa)
    a.set_xticklabels([f"{k:g}" for k in SA.K_GRID], rotation=90, fontsize=6.5)
    a.set_xlim(-0.6, len(xa) - 0.4)
    a.set_xlabel(r"scale-ELK constant $k$    (more damping $\rightarrow$)")
    a.set_ylabel("iterations to tolerance")
    k_star = 1.0 - np.exp(-LAMBDA_GUESS * DT)
    x_star = float(np.interp(k_star, SA.K_GRID, xa))
    a.plot([x_star, x_star], [15, Y_SPLIT], ":", color=INK, lw=0.8, zorder=1)
    a.text(x_star - 0.25, 620, r"$1 - e^{-\lambda_h \Delta t}$", ha="right", va="center", fontsize=7.5, color=INK)

    # B: ELK. The tested sigma^2 are evenly spaced on a log axis.
    best_b, _ = panel(b, "B", "elk", SA.S_GRID, SA.S_GRID)
    b.set_xscale("log")
    b.set_xlim(0.55, 1.9e5)
    b.set_xticks([10.0 ** p for p in range(6)])
    b.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xlabel(r"ELK $\sigma^2$    ($\leftarrow$ more damping)")

    for ax, best in ((a, best_a), (b, best_b)):            # the fewest iterations at each horizon
        for x, it in sorted(set(best.values())):
            xy, ha, va = ((0, 4.5), "center", "bottom") if ax is a else ((6, -4 if it < min(laser) else 0), "left", "center")
            ax.annotate(f"{it:,}", (x, it), xytext=xy, textcoords="offset points", ha=ha, va=va, fontsize=7, color=INK)
    acc = [int(ROWS["A"][0][(n, SA.LASER)]["iters"]) for n in NS]
    label = (f"LASER, every $N$: {min(laser)} or {max(laser)} steps\n"
             f"({min(acc)} or {max(acc)} accepted, {laser[0] - acc[0]} rejected)")
    a.text(len(xa) - 0.5, max(laser) * 1.12, label, ha="right", va="bottom", fontsize=7, color=TEAL, linespacing=1.15)
    b.text(0.62, max(laser) * 1.12, "LASER", ha="left", va="bottom", fontsize=7, color=TEAL)

    for ax, letter, name in ((a, "A", "scale-ELK"), (b, "B", "ELK")):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
        ax.text(0.02, 1.06, name, transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom")
    fig.legend(*a.get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=8,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.2)

    out = os.path.join(SA.ROOT, "figs", "lorenz", "lorenz_baseline_tuning")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    best_k = {n: (SA.K_GRID[x], it) for n, (x, it) in best_a.items()}     # panel A's x is the index into K_GRID
    print(f"best (constant, iterations): scale-ELK {best_k}, ELK {best_b}; LASER steps (accepted + rejected) {laser}")
    print(out + ".pdf")


if __name__ == "__main__":
    main()
