"""Build experiments/lorenz/figure_bistable.ipynb (run with python from the repository
root; execute on a compute node with tools/run_figure_notebooks.slurm).

The paper's bistable Lorenz figure (section "Bistable Dynamics with an Unstable Initial Guess"). The notebook holds
the complete drawing code: one cell per panel with its knobs at the top, then one cell that assembles the figure at
print width and writes figs/lorenz/lorenz_bistable.{pdf,png}. The numbers come from
experiments/lorenz/run_bistable_figure.py, cached in results/lorenz/bistable_figure/; RECOMPUTE = True reruns that script
first. The method colors come from figs/colors.json, shared with the ELCD figure notebook. Pure Python: nbformat
only. Edit this file, not the notebook.
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
DEFAULT_NB = HERE / "figure_bistable.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# The bistable Lorenz figure of the paper, panel by panel

**The task in plain words.** This is the figure of the section "Bistable Dynamics with an Unstable Initial Guess".
The problem is the Lorenz system at sigma = 10, rho = 10, beta = 8/3 from x0 = (1, 1, 1), discretized by explicit
Euler at dt = 0.01 with N = 1000 steps in fp64. At rho = 10 the origin is a saddle and C+ and C- are stable foci.
Every solver starts from the zero guess, the constant trajectory at the saddle, where the step Jacobians are
unstable.

- **A, the two basins** in the (x1, x3) plane. The sequential solution from x0 settles at C+, its mirror image from
  (-1, -1, 1) at C-. The text gives the horizon exponent lambda_h of the guess and of the solution.
- **B, the merit landscape at the guess.** The Gauss-Newton model of the merit function on the plane spanned by the
  right singular vectors of the residual Jacobian with the largest and the smallest singular value, without damping
  and with LASER's damping. kappa is the ratio of those two singular values.
- **C, the residual against Newton iteration** for DEER, scale-ELK at k = 0.05, ELK at the better of
  sigma^2 = 1e2 and 1e4, and LASER.

**How to use it.** `experiments/lorenz/run_bistable_figure.py` computes every number (a few minutes, JAX on the CPU);
the shipped numbers are cached in `results/lorenz/bistable_figure/`. The setup cell loads the cache; `RECOMPUTE = True`
first reruns the script into the cache (on a compute node). Each panel's cell holds its complete drawing code with
its knobs at the top. The last cell puts the panels together, exactly as wide as the paper's text (6.0 in), and writes
`figs/lorenz/lorenz_bistable.pdf` and `.png`. After
editing a panel's cell, run it and then the last cell.

**Colors.** The method colors of panel C come from `figs/colors.json`, the file the ELCD figure notebook reads too.
Panels A and B keep their own colors in their cells (the two basins, the saddle, the two landscapes).

**For Illustrator.** The text is TrueType (`pdf.fonttype 42`), so every label is editable text. No label has a white
halo: a halo is drawn as a white outline over the curves, so a dragged label would leave a white gap in the
trajectory. The C+ and C- labels sit outside the spirals; each can be moved on its own.
""")

md("## Setup: the style, the colors and the cached numbers")
code(r"""
RECOMPUTE = False                    # True: rerun experiments/lorenz/run_bistable_figure.py first (a few minutes, JAX on the CPU)
STYLE = {"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.major.size": 2.0, "ytick.major.size": 2.0,
         "axes.spines.top": False, "axes.spines.right": False,
         "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
         "pdf.fonttype": 42, "ps.fonttype": 42}       # TrueType in the PDF: Illustrator can edit the text

import io, json, os, re, subprocess, sys
from pathlib import Path
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "parode").is_dir())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from IPython.display import Image, display

plt.rcParams.update(STYLE)
CACHE = REPO / "results/lorenz/bistable_figure"
OUT = REPO / "figs/lorenz"
OUT.mkdir(parents=True, exist_ok=True)
PALETTE = json.loads((REPO / "figs/colors.json").read_text())       # the colors shared by the paper figures
M = PALETTE["methods"]
INK = PALETTE["neutral"]["text"]

if RECOMPUTE:
    subprocess.run([sys.executable, "experiments/lorenz/run_bistable_figure.py", "--outdir", str(CACHE)], cwd=REPO,
                   check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
q = json.loads((CACHE / "lorenz_bistable.json").read_text())        # the scalars: exponents, kappa, iteration counts
a = dict(np.load(CACHE / "lorenz_bistable_cache.npz"))               # the arrays: trajectories, landscapes, residuals


def sci(x, digits=1):
    # the mathtext body of a scientific label, e.g. 2.5e24 -> '2.5{\times}10^{24}' ('10^{4}' when the mantissa
    # rounds to one); the caller adds the dollars
    if not np.isfinite(x):
        return "non-finite"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    if f"{m:.{digits}f}" == f"{1.0:.{digits}f}":
        return f"10^{{{e}}}"
    return f"{m:.{digits}f}{{\\times}}10^{{{e}}}"
""")

md(r"""
The runs behind panel C. One row per solver. `iterations`: Newton iterations run, each one parallel solve of the
linearized system. `final_residual`: max_n ||r_n||_inf after the last iteration (lower is better; the tolerance is 1e-9).
`status`: converged (below the tolerance), nonfinite (an iterate overflowed), or max_iter (the cap of 200 reached).
Row `laser` reads: LASER reached the tolerance after 22 iterations.
""")
code(r"""
pd.DataFrame([dict(solver=k, iterations=v["iterations"], final_residual=v["final_residual"], status=v["status"])
              for k, v in q["runs"].items()])
""")

md(r"""
## Panel A: the two basins

The solution from x0 (triangle) in the color of C+, its mirror image in the color of C-, the fixed points as dots, the
saddle (the zero guess) as a cross. `C_LABEL` places the C+ and C- labels in data units, outside the spirals: the
spirals reach x1 = 6.1 below x3 = 5, so the lower outer corners are free.
""")
code(r"""
SOLUTION, MIRROR, SADDLE = "#0F6E73", "#3B5BA5", "#B3392B"    # C+ and its basin, C- and its basin, the saddle and the guess
XLIM_A, YLIM_A = (-10.5, 10.5), (-6.1, 22.0)
C_LABEL = {"+": (7.0, 3.5, "left"), "-": (-7.0, 3.5, "right")}   # (x1, x3, horizontal alignment) of the C+ and C- labels
X0_OFFSET = (3.0, -3.0)              # the x0 label's offset from its triangle, in points


def panel_a(ax):
    tr, trm = a["truth"], a["truth_mirror"]
    ax.plot(trm[:, 0], trm[:, 2], color=MIRROR, lw=0.8, zorder=3)
    ax.plot(tr[:, 0], tr[:, 2], color=SOLUTION, lw=0.8, zorder=3)
    Cp, Cm = q["C_plus"], q["C_minus"]
    ax.plot([Cp[0]], [Cp[2]], "o", color=SOLUTION, ms=3.2, mec="white", mew=0.5, zorder=6)
    ax.plot([Cm[0]], [Cm[2]], "o", color=MIRROR, ms=3.2, mec="white", mew=0.5, zorder=6)
    ax.plot([0], [0], "X", color=SADDLE, ms=4.4, mec="white", mew=0.5, zorder=6)
    ax.plot([q["x0"][0]], [q["x0"][2]], "^", color=SOLUTION, ms=3.0, mec="white", mew=0.4, zorder=6)
    ax.plot([q["x0_mirror"][0]], [q["x0_mirror"][2]], "^", color=MIRROR, ms=3.0, mec="white", mew=0.4, zorder=6)
    for sign, color in (("+", SOLUTION), ("-", MIRROR)):
        x, z, ha = C_LABEL[sign]
        ax.text(x, z, f"$C_{sign}$", color=color, ha=ha, va="center", zorder=7)
    ax.annotate("$x_0$", (q["x0"][0], q["x0"][2]), textcoords="offset points", xytext=X0_OFFSET,
                color=SOLUTION, va="top", zorder=7)
    ax.annotate("saddle,\nzero guess", (0, 0), textcoords="offset points", xytext=(0, -6.5),
                color=SADDLE, ha="center", va="top", zorder=7, linespacing=1.05)
    ax.text(0.02, 1.00, f"$\\lambda_h = {q['lambda_h_guess']:+.2f}$ (guess)", transform=ax.transAxes,
            color=SADDLE, va="top", ha="left", zorder=7)
    ax.text(0.02, 0.90, f"$\\lambda_h = {q['lambda_h_truth']:+.2f}$ (solution)",
            transform=ax.transAxes, color=SOLUTION, va="top", ha="left", zorder=7)
    ax.set_xlabel("$x_1$", labelpad=1.0)
    ax.set_ylabel("$x_3$", labelpad=1.0)
    ax.set_xlim(*XLIM_A)
    ax.set_ylim(*YLIM_A)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([0, 8, 16])
""")

md(r"""
## Panel B: the merit landscape at the guess, without and with damping

log10 of the Gauss-Newton model of the merit, L(d) = ||r + A d||^2 / (2 dt) with d = a v_max + b v_min, on
the same window in both panels. The axes are a = v_max^T d and b = v_min^T d in their own units, labelled as the
components of the step delta s (the cache calls them alpha and beta). The window is |a| <= 1.77 and |b| <= 1567
(twice ||r||_2 over the damped sigma_max and sigma_min), so the two axes differ in scale by the damped kappa. The
cross is the guess.
""")
code(r"""
LANDSCAPE = {"undamped": ("Reds", "#B3392B", "no damping"), "damped": ("Blues", "#3B5BA5", "with damping")}  # colormap, title color, title
N_LEVELS = 13                        # filled contour levels, evenly spaced in log10 L between its min and max
X_TICKS = (-1, 0, 1)                 # the step's component along v_max (the window is +-1.77)
Y_TICKS = ((-1000, r"$-10^3$"), (0, "0"), (1000, r"$10^3$"))   # along v_min (the window is +-1567): value, label
Y_LABEL_X = -0.27                    # the y label's right edge, in widths of a landscape left of its axis: beside the
                                     # "0" tick label, between the two longer ones (None: left of the longest tick label)


def panel_b(ax_undamped, ax_damped):
    alphas, betas = a["alphas"], a["betas"]
    for ax, tag in ((ax_undamped, "undamped"), (ax_damped, "damped")):
        cmap, color, title = LANDSCAPE[tag]
        L = np.log10(np.maximum(a[f"L_{tag}"], 1e-300))
        levels = np.linspace(float(np.min(L)), float(np.max(L)), N_LEVELS)
        ax.contourf(alphas, betas, L.T, levels=levels, cmap=cmap, extend="neither")
        ax.contour(alphas, betas, L.T, levels=levels, colors="white", linewidths=0.3, alpha=0.7)
        ax.plot([0], [0], "+", color=INK, ms=4.2, mew=0.9, zorder=5)
        ax.set_xticks(X_TICKS)
        ax.set_yticks([t for t, _ in Y_TICKS])
        ax.set_yticklabels([label for _, label in Y_TICKS])
        ax.set_xlabel(r"$v_{\max}^{\top}\delta s$", labelpad=1.0)
        ax.text(0.5, 1.17, title, transform=ax.transAxes, ha="center", va="bottom", color=color)
        ax.text(0.5, 1.01, f"$\\kappa = {sci(q[tag]['kappa'])}$", transform=ax.transAxes,
                ha="center", va="bottom", color=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.6)
    ax_undamped.set_ylabel(r"$v_{\min}^{\top}\delta s$", labelpad=0.5)
    if Y_LABEL_X is not None:
        ax_undamped.yaxis.set_label_coords(Y_LABEL_X, 0.5)
    ax_damped.set_yticklabels([])
""")

md(r"""
## Panel C: the residual against Newton iteration

max_n ||r_n||_inf after each iteration. A value above the window is marked by a triangle at its top, a non-finite iterate
by a cross, a converged run by a dot at its last iteration, a run that hit the cap by a square. The curves are
labelled in place (`LABEL_AT`); a legend box of four entries does not fit beside them.
""")
code(r"""
YLIM_C = (1e-16, 1e6)
ELK_KEY = f"elk_{q['elk_best_sigmasq']:g}"       # ELK at the better of the two sigma^2 tried
SERIES = [("deer", M["DEER"], 1.0), ("scale_elk_0.05", M["scale-ELK"], 1.0), (ELK_KEY, M["ELK"], 1.0),
          ("laser", M["LASER"], 1.4)]            # (run, color, line width)


def panel_c(ax):
    ybot, ytop = YLIM_C
    ends, nonfinite_at, above_val = {}, {}, {}
    for key, col, lw in SERIES:
        r = np.asarray(a[f"res_{key}"])
        it = np.arange(len(r))
        finite = np.isfinite(r)
        mk = "o" if int(finite.sum()) <= 5 else None                     # a two-point run needs markers
        ax.plot(it[finite], r[finite], "-", color=col, lw=lw, marker=mk, ms=2.4, zorder=4 if key == "laser" else 3)
        ends[key] = (int(it[-1]), float(r[finite][-1]))
        above = finite & (r > ytop)                                      # above the window: marked, not dropped
        for i, v in zip(it[above], r[above]):
            ax.plot([i], [ytop * 0.45], "^", color=col, ms=3.4, mec="white", mew=0.4, zorder=5)
            above_val[key] = float(v)
        if not finite.all():                                             # non-finite iterate
            i = int(np.argmax(~finite))
            ax.plot([i], [ytop * 0.45], "x", color=col, ms=4.4, mew=1.1, zorder=6)
            nonfinite_at[key] = i
        elif q["runs"][key]["status"] != "converged":                    # reached the iteration cap
            ax.plot([it[-1]], [r[-1]], "s", color=col, ms=3.2, mec="white", mew=0.4, zorder=5)
            ax.annotate(f"cap {q['max_iter']}", (it[-1], r[-1]), textcoords="offset points",
                        xytext=(-2, 6), color=col, ha="right", zorder=7)
        else:
            ax.plot([it[-1]], [r[-1]], "o", color=col, ms=2.6, mec="white", mew=0.4, zorder=5)
    ax.axhline(q["tol"], color=INK, ls=":", lw=0.6, zorder=1)
    ax.text(0.995, q["tol"], "tol", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(ybot, ytop)
    ax.set_xlim(-4, max(e[0] for e in ends.values()) + 14)
    ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5])
    ax.set_xlabel("Newton iteration", labelpad=1.0)
    ax.set_ylabel(r"$\max_n \|r_n\|_\infty$", labelpad=1.0)

    # the in-place labels: (x, y, text, color, horizontal alignment, vertical alignment)
    deer = ("DEER: $" + sci(above_val["deer"], 1) + "$,\nthen non-finite") if "deer" in nonfinite_at else "DEER"
    LABEL_AT = [(nonfinite_at.get("deer", 0) + 2.5, 1e5, deer, M["DEER"], "left", "top"),
                (0.42 * ends["scale_elk_0.05"][0], 3e-1, "scale-ELK, $k=0.05$", M["scale-ELK"], "left", "bottom"),
                (ends[ELK_KEY][0] + 3, 1e-12, "ELK, $\\sigma^2 = " + sci(q["elk_best_sigmasq"], 0) + "$", M["ELK"], "left", "center"),
                (ends["laser"][0] + 3, 3e-15, "LASER", M["LASER"], "left", "center")]
    for x, y, text, color, ha, va in LABEL_AT:
        ax.text(x, y, text, color=color, ha=ha, va=va, linespacing=1.05, zorder=7)
""")

md(r"""
## The figure

The panels side by side, `FIG_H` inches tall. The saved file is trimmed to its ink and is exactly `TEXT_WIDTH` wide:
the paper places it with `\includegraphics[width=\textwidth]`, so only at that width is it placed at 100 % and the
7 pt type printed at 7 pt (a narrower file is scaled up, a wider one down). The ink reaches beyond the grid of the
axes (the y label of panel A, the panel letters), so the width before trimming is not the saved width: the figure
is assembled and saved to memory as a PDF, the width of the PDF's page is read back, the width before trimming is corrected and the
figure is assembled again until the PDF is within 0.002 in of `TEXT_WIDTH`. The width is read from the PDF itself
because the screen renderer and the PDF renderer measure text slightly differently (0.02 in on this figure). The
printed line reports both widths. The damped landscape is moved next to the undamped one (`B_GAP` apart), and
panel C's left edge is placed so that its y label is `C_CLEAR` from the damped landscape; panel C takes the width
that is left.
""")
code(r"""
TEXT_WIDTH = 6.0                     # in: the paper's text width (L4DC, jmlr class) = the saved figure's width
FIG_H = 2.05                         # in, before trimming
WIDTHS = [1.15, 0.70, 0.70, 1.62]    # panel A, the two landscapes, panel C (ratios)
WSPACE = 0.46
B_GAP = 0.20                         # in: between the two landscapes of panel B
C_CLEAR = 0.20                       # in: between the damped landscape and panel C's y label
PAD = 0.01                           # in: the white margin around the ink in the saved files
LETTER_PT = 10                       # the panel letters A, B, C: bold capitals, as in the ELCD figure
NAME = "lorenz_bistable"


def assemble(fig_w):
    # the figure fig_w inches wide before trimming
    fig = plt.figure(figsize=(fig_w, FIG_H))
    gs = GridSpec(1, 4, width_ratios=WIDTHS, wspace=WSPACE, left=0.068, right=0.995, bottom=0.19, top=0.865)
    ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(4))
    panel_a(ax0)
    panel_b(ax1, ax2)
    panel_c(ax3)
    for ax, letter, dx in ((ax0, "A", -0.22), (ax1, "B", -0.34), (ax3, "C", -0.26)):
        ax.text(dx, 1.235, letter, transform=ax.transAxes, fontweight="bold", fontsize=LETTER_PT, va="top", ha="left")

    p1, p2 = ax1.get_position(), ax2.get_position()
    ax2.set_position([p1.x1 + B_GAP / fig_w, p2.y0, p2.width, p2.height])      # the two landscapes close together
    fig.canvas.draw()                # measure panel C's y label and put it C_CLEAR right of the damped landscape
    rend = fig.canvas.get_renderer()
    need = (ax2.get_window_extent(rend).x1 + C_CLEAR * fig.dpi) - ax3.yaxis.label.get_window_extent(rend).x0
    p = ax3.get_position()
    ax3.set_position([p.x0 + need / fig.bbox.width, p.y0, p.width - need / fig.bbox.width, p.height])
    return fig


fig_w = TEXT_WIDTH
for _ in range(5):
    fig = assemble(fig_w)
    pdf = io.BytesIO()               # the trial PDFs stay in memory; the file is written once the width has settled
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=PAD)
    x0, _, x1, _ = (float(v) for v in re.search(rb"MediaBox \[([^\]]*)\]", pdf.getvalue()).group(1).split())
    saved_w = (x1 - x0) / 72         # the PDF page's width (in)
    if abs(saved_w - TEXT_WIDTH) < 0.002:
        break
    plt.close(fig)
    fig_w += TEXT_WIDTH - saved_w
else:
    raise RuntimeError(f"the saved width did not settle: {saved_w:.3f} in")
(OUT / f"{NAME}.pdf").write_bytes(pdf.getvalue())
print(f"{fig_w:.3f} in wide before trimming; the PDF is {saved_w:.3f} in wide against a {TEXT_WIDTH} in text width; "
      f"type prints at {7.0 * TEXT_WIDTH / saved_w:.2f} pt")
fig.savefig(OUT / f"{NAME}.png", dpi=400, bbox_inches="tight", pad_inches=PAD)
plt.close(fig)
print(f"wrote {OUT / NAME}.pdf and .png")
display(Image(filename=str(OUT / f"{NAME}.png"), width=900))
""")


def main(argv=None):
    nb_path = Path(argv[0]) if argv else DEFAULT_NB
    nb = nbf.v4.new_notebook()
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:02d}"     # nbformat draws random ids; fixed ones make a rebuild reproduce the file
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, str(nb_path))
    print(f"wrote {nb_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main(sys.argv[1:])
