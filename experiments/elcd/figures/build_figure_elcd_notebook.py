"""Build experiments/elcd/figures/figure_elcd.ipynb (run with python from the repository root;
execute on a compute node with
  JAX_PLATFORMS=cpu PYTHONPATH=<repo> \\
      jupyter nbconvert --to notebook --execute --output-dir figs/notebooks <notebook>
and the HTML twin with --to html --output-dir figs/notebooks; or run
tools/run_figure_notebooks.slurm, which does all three).
The paper's ELCD figure, part by part, each part in its own cell with its own print size and the COMPLETE drawing code
in the cell (sizes, fonts and details are tuned there): the schematic (panel A, drawn by
experiments/elcd/schematic/make_schematic_C.py and make_nn_spaces.py with NAME=VALUE overrides), the trajectory
drawings and the Chamfer plot of panel B, and the training times (panel C, a linear axis broken between the
sequential loop and the parallel solvers). Every matplotlib part is a function that draws at a position on a page of
the paper's text width, so the last section places the same drawings on one page (the composed figure), with the
schematic SVGs put in at 100 %. Only the data loading is imported
(panel_data.load_parts: the records, the best seeds' rollouts, the long runs' evaluations). One STYLE for
every part: DejaVu Sans, 7 pt labels, 0.6 pt axes, 1 pt curves. Saved as PDF (and PNG) in figs/elcd/ for Illustrator,
where the parts are placed at 100 %. Pure Python: nbformat only. Importing this module builds the cells in memory
and writes nothing (main() writes).
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
DEFAULT_NB = HERE / "figure_elcd.ipynb"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# The ELCD paper figure, part by part at print size, then composed

**The task in plain words.** The paper's ELCD figure has three panels. Panel A is the schematic of one LASER solve.
Panel B compares the two training losses: the Chamfer distance to the demonstrations against the training time,
and rollouts of the two best models. Panel C is the training time per solver. A panel is built from parts, and a
part is one PDF written by this notebook. Every part is drawn at the size it has on the page, so that placed at
100 % all parts share one font size and one set of line widths. The last section places the parts on one page.

**Sizes.** The sizes follow an earlier layout of this figure, composed by hand in Illustrator. The paper's text
is 6.0 in wide (the `jmlr` class), so the page here is
`PAGE_W` = 6.0 in. In the hand layout the parts had been rescaled one by one, so their labels ran from 4.9 pt (the
trajectory titles) to 8.6 pt (the Chamfer axis labels). Here every part is drawn at its final size instead:

- **Labels:** `LABEL_PT` = 7 pt in every part (axis labels, ticks, legends, titles, the schematic's labels). The
  schematic's legend prints at 6/7 of it and its sub and superscripts at 5.5 pt, as designed. The encoder / decoder
  part is the exception: its text scales with its width only. At the hand layout's width, 1.87 in, its labels print
  at 5.9 pt; made wider, it covers the ghost arms behind its lower right corner.
- **Lines:** axes and ticks 0.6 pt; curves 1 pt (the Chamfer curves and the solution x*(t) on the desk, on the cards
  and in the encoder / decoder part); the learned field's stream lines 0.45 pt; slab, card and box outlines 0.3 pt.
- **Each part's size** is set at the top of its cell: the print width for a schematic part, the size of the axes
  (the data rectangle) for a matplotlib part. Positions are set only in the last section.

Every part is a PDF plus a PNG in `OUT` (`figs/elcd/` at the repository root), trimmed to its ink; place the PDF at
100 %, no rescaling, so the fonts keep the size set here. The colors of the methods and of the losses come from
`figs/colors.json`, the file the Lorenz figure reads too: edit a hex code there and re-run the cells.

- **A, in three parts.** The schematic is drawn by `experiments/elcd/schematic/make_schematic_C.py` from the
  chain8z iterates: `paper_A_desk` (the desk with the robot, the learned field, the demonstration, the solution
  x*(t) and the t_i labels) and `paper_A_cards` (the legend, the k axis and one card per LASER iteration shown, the
  iterate in orange over x*(t) in blue). `paper_A_spaces` is the encoder / decoder part (`make_nn_spaces.py`),
  which sits over the desk's upper left corner. The scripts take `NAME=VALUE` overrides; `A_OVERRIDES` holds the
  geometry of the paper's figure (three cards, a flatter desk, a smaller arm).
- **B, the trajectories** (`paper_B_trajectories`): the rollouts of the two best models (the velocity loss above,
  the multi-start rollout loss with LASER below; the best of the ten seeds by the final training Chamfer)
  from five starts on training demonstration 0 (gray), the star the equilibrium.
- **B, the Chamfer curves** (`paper_B_chamfer`): the training Chamfer (native: latent RK4 at the demonstration's
  step, decoded; the mean over the training demonstrations) of the long runs of seed 1 against the training time
  in hours, 1e6 steps of the rollout loss (25.0 h) and 1e7 of the velocity loss (10.7 h).
- **C** (`paper_C`): the training time of 10,000 steps per method over the ten chain8z fp32 seeds as box plots (the
  median black, the box the quartiles, the whiskers the minimum and the maximum) on a linear axis broken in two: the
  sequential loop (`seq`, measured: its full run of 10,000 steps) above, the parallel solvers below at their own range.
- **The composed figure** (`elcd_figure_traj_left`): all parts on one page, the trajectories left of the Chamfer
  plot (the hand layout had them right of it).
""")

md("## Setup: the page, the style, the records, the rollouts and the long runs (about a minute; JAX on the CPU)")
code(r"""
PAGE_W, PAGE_H = 6.0, 3.75           # in: the paper's text width (L4DC, jmlr class: 6.0 in) and the figure's height; both whole points
LABEL_PT = 7.0                       # every label of every part
FS_LABEL_PX = 28                     # FS_LABEL of make_schematic_C.py and of make_nn_spaces.py: a label's size in canvas px at FONT_SCALE 1
STROKE_PT = {"SW_XSTAR": 1.0, "SW_ITER": 0.875, "SW_STREAM": 0.45, "SW_SLAB": 0.3, "SW_CARD_SLAB": 0.3, "R_DEMO": 0.81}
                                     # the schematic as printed (pt), the same on every part: the widths of x*(t), of the iterate, of
                                     # the field's stream lines (desk and legend), of the desk slab's and of a card's outline, and the
                                     # radius of a demonstration dot (desk and legend)
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
         "mathtext.fontset": "dejavusans",                # the 10^k tick labels in the same font
         "font.size": LABEL_PT, "axes.titlesize": LABEL_PT, "axes.labelsize": LABEL_PT,
         "xtick.labelsize": LABEL_PT, "ytick.labelsize": LABEL_PT, "legend.fontsize": LABEL_PT,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.minor.width": 0.45, "ytick.minor.width": 0.45,
         "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
         "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3.0,
         "lines.linewidth": 1.0, "grid.linewidth": 0.5,
         "axes.spines.top": False, "axes.spines.right": False,
         "pdf.fonttype": 42}                              # TrueType in the PDF: Illustrator can edit the text (Type 3, the default, it cannot)

import json, logging, os, re, subprocess, sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)   # the CUDA plugin cannot start on a CPU node; JAX runs on the CPU here
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "parode").is_dir())
sys.path.insert(0, str(REPO))
os.chdir(REPO)                               # the data and record defaults are relative to the repository root
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from experiments.elcd.figures import panel_data as PF     # the data loading only

plt.rcParams.update(STYLE)
OUT = REPO / "figs/elcd"
OUT.mkdir(parents=True, exist_ok=True)
SCHEM = REPO / "experiments/elcd/schematic"
SAVED = []                                   # the parts written by this run, for the sizes table
PALETTE = json.loads((REPO / "figs/colors.json").read_text())       # the colors shared by the paper figures
COLOR = {"suffix": PALETTE["losses"]["multi-start rollout loss"], "vel": PALETTE["losses"]["velocity loss"]}
ht, cases, ev = PF.load_parts()              # the hours table; [(title, rollouts, colour)] of the two best models; the long runs
cases = [(title, r, PALETTE["losses"][title]) for title, r, _ in cases]   # the titles are the losses' names in colors.json


def page():
    # an empty figure of the page's size; a part is placed on it in inches from the page's top-left corner
    return plt.figure(figsize=(PAGE_W, PAGE_H))


def ax_in(fig, x, y, w, h):
    # axes w x h inches whose top-left corner is (x, y) inches from the page's top-left corner
    return fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])


def save(fig, name):
    # one part alone: trimmed to its ink plus 0.02 in, so the axes keep the size set in the part's cell
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else None)
    plt.close(fig)
    SAVED.append(name)
    display(Image(filename=str(OUT / f"{name}.png"), retina=True))


def _draw(script, name, args):
    # run a schematic script into OUT/name with NAME=VALUE args; return its size line
    r = subprocess.run([sys.executable, script, *args, f"OUT_STEM={OUT / name}"], cwd=SCHEM, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0, f"{script} failed"
    return next(l for l in r.stdout.splitlines() if l.startswith("print"))


def pt_per_px(name):
    # the print scale of the schematic part OUT/name.svg: the points one canvas px prints at, from its width (in) and its viewBox
    root = re.search(r"<svg\b[^>]*>", (OUT / f"{name}.svg").read_text()).group(0)
    width_in = float(re.search(r'\swidth="([\d.]+)in"', root).group(1))
    return width_in * 72 / float(re.search(r'viewBox="([^"]+)"', root).group(1).split()[2])


def _report(name, line, label_pt):
    # print the script's size line and the part's label size; fail when the script's own label size disagrees,
    # which means its FS_LABEL is no longer FS_LABEL_PX
    reported = float(re.search(r"text ([\d.]+) /", line).group(1))
    assert abs(reported - label_pt) < 0.1, f"{name}: the script reports {reported} pt labels, expected {label_pt:.2f} pt"
    print(line)
    print(f"{name}: {pt_per_px(name):.4f} pt per canvas px; labels {label_pt:.2f} pt")
    SAVED.extend([name, name + "_outlined"])


def schematic(name, width_in, **overrides):
    # make_schematic_C.py (PART selects the desk or the cards), the part width_in wide. The print scale is known
    # only once the part is cropped to its content, so it is drawn, FONT_SCALE and the sizes of STROKE_PT are set
    # from the scale, and it is drawn again until the labels are within 1 % of LABEL_PT
    args = dict(overrides, PRINT_W_IN=width_in, FONT_SCALE=1.0)
    for attempt in range(4):
        line = _draw("make_schematic_C.py", name, [f"{k}={v}" for k, v in args.items()])
        s = pt_per_px(name)
        if attempt and abs(FS_LABEL_PX * args["FONT_SCALE"] * s - LABEL_PT) < 0.01 * LABEL_PT:
            break
        args.update(FONT_SCALE=round(LABEL_PT / (FS_LABEL_PX * s), 3), **{k: round(pt / s, 2) for k, pt in STROKE_PT.items()})
    else:
        raise RuntimeError(f"{name}: the label size did not settle")
    _report(name, line, FS_LABEL_PX * args["FONT_SCALE"] * s)
    display(Image(filename=str(OUT / f"{name}_1x.png"), width=500))


def nn_spaces(name, width_in, card_scale=1.2):
    # make_nn_spaces.py: the encoder / decoder part (network variant, its two cards at card_scale) width_in wide.
    # Its text scales with the width only. Drawn twice: the canvas's width gives the print scale, then the curve,
    # the stream lines and the outlines are set to the printed sizes of STROKE_PT
    args = ["network", f"PRINT_W_IN={width_in}", f"NETWORK_SCALE={card_scale}"]
    _draw("make_nn_spaces.py", name, args)
    s = pt_per_px(name)
    line = _draw("make_nn_spaces.py", name, args + [f"CURVE_W={STROKE_PT['SW_XSTAR'] / s:.2f}", f"FLOW_W={STROKE_PT['SW_STREAM'] / s:.2f}",
                                                    f"FRAME_W={STROKE_PT['SW_SLAB'] / s:.2f}"])
    _report(name, line, FS_LABEL_PX * s)
    display(Image(filename=str(OUT / f"{name}_1x.png"), width=400))
""")

md(r"""
## Panel A: the schematic in three parts

`make_schematic_C.py` draws the desk with the robot and the cards of the LASER iterations; `PART` selects the desk
or the cards, each cropped to its content. `make_nn_spaces.py` draws the encoder / decoder part. Every part writes
`.svg`, `.pdf` (fonts embedded), `_outlined.pdf` (text as paths, for an Illustrator without DejaVu Sans), `.png`
(2x) and `_1x.png`. The width is set, the height follows the content.

Two lines are printed per part. The first, starting with `print`, is the script's own report: the print size and
three font sizes (labels / legend / k axis title for the desk and the cards; labels / second level / scripts for
the encoder / decoder part). The second is this notebook's: the print scale, which is the number of points one
canvas pixel of the script prints at, and the label size that follows from it.

The desk and the cards share `A_OVERRIDES`, so they keep the same drawing. Their print scales differ (the cards are
drawn smaller than the desk), so `schematic()` sets `FONT_SCALE` and the sizes of `STROKE_PT` per part: the labels,
x*(t), the iterate, the stream lines of the desk and of the legend, the slab outlines and the demonstration dots
then print alike on both. The field on the cards stays lighter (0.3 pt), as designed. The widths are the hand
layout's: the desk's slab prints 2.66 in wide and a card's slab 2.07 in.
""")
code(r"""
A_OVERRIDES = dict(CARD_GAP=18, TOP_GAP=26, LEGEND_ROW=32, K_SHOW=[0, 1, -1], DESK_H=150, ARM_SCALE=1.1,
                   FS_AXIS=28,       # FS_AXIS: the k axis title at the labels' size (the script's own is 26 px to 28)
                   NODE_LABEL_CARD_NUDGE={1: (-12, -3), 3: (4, 34)},   # canvas px on the k = 0 card: t_1 left, off the
                                     # rising branch of x*(t); t_3 right and down, under the two curves it sat on
                   ELLIPSIS_DY=10)   # canvas px: the dots of the skipped iterations clear the slab of the k = 1 card
A_DESK_WIDTH_IN = 2.75               # the desk part's print width (in); the height follows

schematic("paper_A_desk", A_DESK_WIDTH_IN, PART="desk", **A_OVERRIDES)
""")
code(r"""
A_CARDS_WIDTH_IN = 2.72              # the cards part's print width (in)

schematic("paper_A_cards", A_CARDS_WIDTH_IN, PART="cards", **A_OVERRIDES)
""")
code(r"""
A_SPACES_WIDTH_IN = 1.87             # the encoder / decoder part's print width (in), the hand layout's. Its labels print at
                                     # 3.16 pt per inch of width: 5.9 pt here, 7 pt at 2.22 in. From about 1.9 in on, at its
                                     # place on the page, it covers the ghost arms behind its lower right corner

nn_spaces("paper_A_spaces", A_SPACES_WIDTH_IN)
""")

md(r"""
## Panel B, the trajectories

Each drawing shows the demonstration in gray, the equilibrium as a star, the rollout from the demonstration's first
sample in the loss's colour and the rollouts from the four later starts lighter, each start a dot, in the normalized
coordinates the model sees (positions over the training demonstrations' standard deviation per coordinate). The
drawings share one frame, the demonstration's box padded by `FRAME` of its span, and are stacked in `TRAJ_ORDER`.
`TRAJ_H` is the height of one frame; its width follows from the demonstration's shape.
""")
code(r"""
TRAJ_H = 0.655                       # in: the height of one drawing's frame (the demonstration takes 1 / (1 + 2 FRAME) of it)
TRAJ_GAP = 0.25                      # in: between the two frames; the lower drawing's title sits in it, nearer to its own drawing
TRAJ_ORDER = ["velocity loss", "multi-start rollout loss"]                 # top to bottom
TRAJ_TITLE = {"multi-start rollout loss": "multi-start\nrollout loss"}     # in two lines: 1.06 in wide in one line at 7 pt
FRAME = 0.12                         # padding of the shared frame, as a fraction of the demonstration's span
GRAY = PALETTE["neutral"]["demonstration"]

pts = np.concatenate([r["demo"] / r["pos_std"] for _, r, _ in cases])
lo, hi = pts.min(0), pts.max(0)
lo, hi = lo - FRAME * (hi - lo), hi + FRAME * (hi - lo)
TRAJ_W = TRAJ_H * (hi[0] - lo[0]) / (hi[1] - lo[1])      # equal scales on both coordinates


def trajectories(fig, x, y):
    # the drawings stacked on the page; (x, y): the top-left corner of the upper frame, in inches
    by_title = {title: (r, color) for title, r, color in cases}
    for i, title in enumerate(TRAJ_ORDER):
        r, color = by_title[title]
        ax = ax_in(fig, x, y + i * (TRAJ_H + TRAJ_GAP), TRAJ_W, TRAJ_H)
        demo, x_eq = r["demo"] / r["pos_std"], r["x_eq"] / r["pos_std"]
        ax.plot(demo[:, 0], demo[:, 1], color=GRAY, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter(*x_eq, marker="*", s=28, color="black", lw=0, zorder=5)
        for k, path in zip(r["starts"], r["paths"]):
            path, first = path / r["pos_std"], k == 0
            ax.plot(path[:, 0], path[:, 1], color=color, lw=0.9 if first else 0.75, alpha=1.0 if first else 0.65, zorder=3)
            ax.scatter(path[0, 0], path[0, 1], color=color, s=7 if first else 4, zorder=4, edgecolor="white", lw=0.3)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        ax.set_title(TRAJ_TITLE.get(title, title), pad=1.5, linespacing=1.0)
        ax.set_axis_off()                # no frame, no ticks; the title stays


print(f"one frame: {TRAJ_W:.2f} x {TRAJ_H:.2f} in")
fig = page()
trajectories(fig, 1.0, 0.5)
save(fig, "paper_B_trajectories")
""")

md(r"""
## Panel B, the Chamfer curves

One line per long run with a dot per evaluation (the rollout loss every 10,000 steps after 500, 1,000, 2,000 and
5,000; the velocity loss every 500 epochs of 231 steps, so from step 115,500), log-log. The x axis is the training
time in hours at the evaluation: the logged wall-clock time of the optimizer steps before it, evaluations excluded
(the `hours` column of `long_runs.evals`). `CHAMFER_W` and `CHAMFER_H` are the size of the axes; the tick
labels and the axis labels come outside of it.
""")
code(r"""
CHAMFER_W, CHAMFER_H = 1.70, 1.28    # in: the axes (the data rectangle)
LEGEND = True                        # the legend inside, upper left, in a band above the curves
TIME_TICKS = (0.1, 1, 10)            # h: the labelled ticks of the time axis, written as plain numbers
HEADROOM = 3.0                       # with the legend: the top of the y axis is raised by this factor to make that band
CURVES = {"suffix_long": ("multi-start rollout loss", COLOR["suffix"]), "vel_long": ("velocity loss", COLOR["vel"])}
KEY = "train_chamfer_native_mean"    # the native Chamfer, its mean over the training demonstrations


def chamfer(fig, x, y):
    # the Chamfer plot on the page; (x, y): the top-left corner of its axes, in inches
    ax = ax_in(fig, x, y, CHAMFER_W, CHAMFER_H)
    for name, (label, color) in CURVES.items():
        df = ev[name]
        ax.plot(df["hours"].to_numpy(float), df[KEY].to_numpy(float), color=color, marker="o", ms=2.0, label=label)
    ax.set(xscale="log", yscale="log", xlabel="training time (h)", ylabel="Chamfer")
    ax.set_xticks(TIME_TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TIME_TICKS])
    ax.grid(True, alpha=0.25)
    if LEGEND:
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi * HEADROOM)
        ax.legend(loc="upper left", frameon=False, handlelength=2.0, borderaxespad=0.3, labelspacing=0.3)


fig = page()
chamfer(fig, 1.0, 0.5)
save(fig, "paper_B_chamfer")
""")

md(r"""
## Panel C: the training time

The table has one row per method. `method` is the name in the pipeline's table (`loop` is the sequential loop,
`seq` on the axis) and `solver` its key in the records. `seeds` is the number of runs (one per seed) behind the box.
`median`, `q1`, `q3`, `min` and `max` are the training hours of 10,000 steps (lower is better; the loop's are
measured too: its full run of 10,000 steps, which every seed has). Read the LASER row: 10 seeds, median 0.27 h, quartiles 0.26 and 0.29 h, range 0.24 to 0.30 h.

The axis is linear and broken: the upper axes hold the sequential loop over `Y_SEQ`, the lower the parallel solvers
over `Y_PAR`. `C_W` and `C_H` are the size of the two axes together, `C_BREAK` the gap between them and `RATIO` the
split of the rest of the height.
""")
code(r"""
C_W, C_H = 1.95, 1.28                # in: the two axes together (the data rectangles and the break)
C_BREAK = 0.10                       # in: the gap between the two axes at the break
C_YLABEL = 0.40                      # in: from the axes' left edge to the middle of the y label
Y_SEQ = (9.4, 11.0)                  # hours shown on the upper axes (the sequential loop: 9.56 to 10.74)
Y_PAR = (0.2, 0.52)                  # hours shown on the lower axes (the parallel solvers: 0.24 to 0.48)
RATIO = (1, 2.6)                     # heights of the upper and the lower axes
METHODS = ["seq", "deer", "scale-elk0.03", "laser"]
LABEL = {"seq": "seq", "deer": "DEER", "scale-elk0.03": "scale-ELK", "laser": "LASER"}
MCOLOR = {m: PALETTE["methods"][LABEL[m]] for m in METHODS}

print(PF.box_table(ht).to_string(index=False))
t = ht[(ht["system"] == "chain8z") & (ht["dtype"] == "fp32")]
hours = [t.loc[t["solver"] == m, "hours"].to_numpy(float) for m in METHODS]


def training_time(fig, x, y):
    # the box plots on the page; (x, y): the top-left corner of the upper axes, in inches
    h_top = (C_H - C_BREAK) * RATIO[0] / sum(RATIO)
    top = ax_in(fig, x, y, C_W, h_top)
    bot = ax_in(fig, x, y + h_top + C_BREAK, C_W, C_H - C_BREAK - h_top)
    for ax in (top, bot):
        bp = ax.boxplot(hours, positions=range(len(METHODS)), widths=0.5, whis=(0, 100), patch_artist=True, showfliers=False,
                        zorder=2, medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        boxprops=dict(lw=0.4))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(MCOLOR[m])
            patch.set_alpha(0.55)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
    top.set_ylim(*Y_SEQ)
    bot.set_ylim(*Y_PAR)
    top.spines["bottom"].set_visible(False)
    top.set_xticks([])
    bot.set_xticks(range(len(METHODS)))
    bot.set_xticklabels([LABEL[m] for m in METHODS])
    mark = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none", color="black", mew=0.6, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **mark)              # the break marks on the y axis
    bot.plot([0], [1], transform=bot.transAxes, **mark)
    fig.text((x - C_YLABEL) / PAGE_W, 1 - (y + C_H / 2) / PAGE_H, "training time (h)", rotation=90, ha="center", va="center")


fig = page()
training_time(fig, 1.0, 0.5)
save(fig, "paper_C")
""")

md(r"""
## The composed figure

All parts on one page `PAGE_W` x `PAGE_H`. `ARRANGEMENTS` holds, per arrangement, the top-left corner of every part
in inches from the page's top-left corner: for a matplotlib part the corner of its axes (so equal numbers align the
axes), for a schematic part the corner of its SVG, for a letter the corner of its text box. Sizes are not set here:
every part is drawn by its own cell's function at its own size, so the page and the parts are the same drawings.
The one arrangement, `elcd_figure_traj_left`, places the trajectories left of the Chamfer plot (the hand layout had
them right of it).

Three deliberate differences from the hand layout. The axes of B and C share their top and their bottom edge (the
bottom as long as `CHAMFER_H` = `C_H`). The desk's slab and the last card end on one line (in the hand layout the
card ended 0.04 in lower). The `LASER iterations k` arrow is where the script draws it, from the first card's
centre to the last card's (in the hand layout it had been moved up by 0.19 in).

How the page is put together: matplotlib draws B, C and the letters into an SVG with real text
(`svg.fonttype` = `none`); the three schematic SVGs are inserted as nested `<svg>` elements at their own print size
(the encoder / decoder part after the desk, so it lies on top); `rsvg-convert` writes the PDF (all vector, the text
as embedded TrueType) and a 300 dpi PNG. `svg_mathtext_like_pdf` corrects one matplotlib detail: with real SVG
text, a math label such as $10^{3}$ is drawn up to 1 pt above its place; the patch makes the SVG backend measure
math labels as the PDF backend does (checked on matplotlib 3.10.7: the tick labels then sit within 0.1 px at
300 dpi of matplotlib's own PDF). A part that leaves the page, or ink on the page's edge, stops the cell.
""")
code(r"""
LETTER_PT = 10                       # the panel letters A, B, C: bold capitals, as in the Lorenz figure
ROW_Y = 2.06                         # in: the top edge of the axes of B and of C
PANEL_A = dict(spaces=(0.15, 0.04), desk=(0.40, 0.38), cards=(3.20, 0.08))
LETTERS = {"A": (0.05, 0.03), "B": (0.05, 1.95), "C": (3.31, 1.95)}
ARRANGEMENTS = {
    "elcd_figure_traj_left": dict(PANEL_A, letters=LETTERS, trajectories=(0.34, ROW_Y + 0.04), chamfer=(1.45, ROW_Y),
                                  training_time=(3.85, ROW_Y)),
}


def svg_mathtext_like_pdf():
    # matplotlib's SVG backend with real text: measure a math label at its own size, as the PDF backend does
    from matplotlib.backends.backend_svg import RendererSVG
    measure = RendererSVG.get_text_width_height_descent

    def like_pdf(self, s, prop, ismath):
        if ismath is True:
            return self._text2path.mathtext_parser.parse(s, 72, prop)[:3]
        return measure(self, s, prop, ismath)

    RendererSVG.get_text_width_height_descent = like_pdf


def inject_svg(fig_svg, part_path, x, y, prefix):
    # the matplotlib SVG text fig_svg with the SVG file part_path added as a nested <svg> at its own print size,
    # its top-left corner (x, y) inches from the page's top-left corner. One user unit of a matplotlib SVG is 1 pt.
    part = Path(part_path).read_text(encoding="utf-8")
    assert "<style" not in part, f"{part_path}: a <style> block would restyle the whole page"
    m = re.search(r"<svg\b[^>]*>", part)
    root, body = m.group(0), part[m.end():part.rindex("</svg>")]
    view = re.search(r'viewBox="([^"]+)"', root).group(1)
    box = [float(v) for v in view.split()]
    w = 72 * float(re.search(r'\swidth="([\d.]+)in"', root).group(1))
    h = w * box[3] / box[2]
    assert x >= 0 and y >= 0 and 72 * x + w <= 72 * PAGE_W + 1e-6 and 72 * y + h <= 72 * PAGE_H + 1e-6, f"{part_path}: leaves the page"
    # matplotlib's stylesheet "*{stroke-linejoin: round; stroke-linecap: butt}" would restyle the part's strokes too:
    # keep it to matplotlib's own group (already done when a part was inserted before)
    fig_svg, n = re.subn(r"(<style[^>]*>)\*\{", r"\1#figure_1 *{", fig_svg)
    assert n == 1 or "#figure_1 *{" in fig_svg, "matplotlib's stylesheet rule was not found"
    # the part's first element is a white rectangle over its whole viewBox, which would hide the part below it
    r = re.match(r"\s*<rect\b([^>]*)/>", body)
    a = dict(re.findall(r'([\w:-]+)="([^"]*)"', r.group(1)))
    assert [float(a.get(k, 0)) for k in ("x", "y", "width", "height")] == box, f"{part_path}: no background rectangle"
    body = body[r.end():]
    # ids are unique per file only: prefix them and the references to them
    body = re.sub(r'(?<![\w:-])id="', f'id="{prefix}', body)
    body = re.sub(r'url\(\s*([\'"]?)#', rf"url(\1#{prefix}", body)
    body = re.sub(r'href="#', f'href="#{prefix}', body)
    nested = f'<svg id="{prefix}root" x="{72 * x:.3f}" y="{72 * y:.3f}" width="{w:.3f}" height="{h:.3f}" viewBox="{view}">{body}</svg>\n'
    i = fig_svg.rindex("</svg>")
    return fig_svg[:i] + nested + fig_svg[i:]


def compose(name, at):
    # the page of arrangement `at`, written to OUT/name.svg, .pdf and .png
    assert all(abs(72 * v - round(72 * v)) < 1e-9 for v in (PAGE_W, PAGE_H)), \
        "the page in whole points: rsvg-convert writes a PDF page in whole points and refits the drawing to it"
    fig = page()
    for letter, (x, y) in at["letters"].items():
        fig.text(x / PAGE_W, 1 - y / PAGE_H, letter, fontsize=LETTER_PT, fontweight="bold", ha="left", va="top")
    chamfer(fig, *at["chamfer"])
    trajectories(fig, *at["trajectories"])
    training_time(fig, *at["training_time"])
    with plt.rc_context({"svg.fonttype": "none"}):                  # real <text>, which rsvg-convert embeds as TrueType
        fig.savefig(OUT / f"{name}.svg", transparent=True)
    plt.close(fig)
    svg = (OUT / f"{name}.svg").read_text(encoding="utf-8")
    for part in ("desk", "spaces", "cards"):                        # a later part lies on top: the encoder / decoder part over the desk
        svg = inject_svg(svg, OUT / f"paper_A_{part}.svg", *at[part], prefix=f"{part}-")
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
    # 72 dpi: 1 px = 1 pt, so the PDF page is the page's size and nothing is rescaled
    subprocess.run(["rsvg-convert", "-f", "pdf", "--dpi-x", "72", "--dpi-y", "72", "-o", OUT / f"{name}.pdf",
                    OUT / f"{name}.svg"], check=True)
    subprocess.run(["rsvg-convert", "-f", "png", "--dpi-x", "72", "--dpi-y", "72", "-w", str(round(300 * PAGE_W)),
                    "-h", str(round(300 * PAGE_H)), "-b", "white", "-o", OUT / f"{name}.png", OUT / f"{name}.svg"], check=True)
    im = plt.imread(OUT / f"{name}.png")[..., :3]
    assert np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]).min() > 0.98, f"{name}: ink on the page's edge, a part is cut"
    SAVED.append(name)
    print(name)
    display(Image(filename=str(OUT / f"{name}.png"), width=900))


svg_mathtext_like_pdf()
for name, at in ARRANGEMENTS.items():
    compose(name, at)
""")

md(r"""
## The PDFs and their sizes

One row per PDF written above: `width_in` and `height_in` are its page size in inches. Read the first row:
`paper_A_desk.pdf` is 2.75 in wide and 1.50 in tall. The matplotlib parts and the schematic parts are trimmed to
their content; the composed figure is the whole page. Place each at 100 % in Illustrator.

One caveat on the schematic parts' PDFs: `rsvg-convert` writes a PDF page in whole points and fits the drawing into
it, so such a PDF is 0.1 to 0.6 % smaller than its SVG (the encoder / decoder part 0.6 %, which is 0.04 pt on its
labels). The SVGs and the composed figure are exact.
""")
code(r"""
rows = []
for name in dict.fromkeys(SAVED):            # once each, also after a part's cell was re-run
    pdf = OUT / f"{name}.pdf"
    x0, y0, x1, y1 = (float(v) for v in re.search(rb"MediaBox \[([^\]]*)\]", pdf.read_bytes()).group(1).split())
    rows.append(dict(file=pdf.name, width_in=round((x1 - x0) / 72, 2), height_in=round((y1 - y0) / 72, 2)))
pd.DataFrame(rows)
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
