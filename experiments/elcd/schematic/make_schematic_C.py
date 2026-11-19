#!/usr/bin/env python3
"""
Panel A of the ELCD figure, drawn from REAL data: schematic_data.npz exported by export_chain8z.py (chain8z, the
model of panel A, one rollout from a mid-chain start, the LASER iterates from a stale warm start; see that
file's docstring for the npz layout). The desk with the robot on top, the legend under it, then the stack of
iterate cards, k = 0 under the desk and the iterations going DOWN. The training demonstrations are not in that file
(the LASA dataset may not be redistributed): training_demos() rebuilds them from the dataset, which
experiments/elcd/download_data.sh fetches. The C in the file name is a variant letter of the drawing, not a panel.

What is drawn
  desk    the learned data-space field (gray streamlines integrated from the gridded field with
          bilinear interpolation), demonstration dots (demos[0], subsampled), x*(t) in blue, the
          node dots t_0..t_3 and ONE robot: outline ghosts at t_1 and t_2, one solid exposure at t_3.
  cards   for k in K_SHOW: x*(t) in blue under the k-th accepted LASER iterate x^(k)(t) in
          orange; node markers at iterates[k][idx_i]; the sequential solver's progress after k iterations as a
          dark segment along x* ending in a hollow ring (n_k = round(seq_steps_per_iter * k) samples).
State-space mapping: desk width u <- data x, desk depth v <- data y, the bbox of x* (and of the first FIT_ITERS
iterates) fitted to U_RANGE x V_RANGE of the unit square; desk, cards and field share this mapping and ONE oblique
projection (make_schematic.P with the geometry override below), so the parallelograms are congruent.

Output: OUT_STEM.svg (physical width PRINT_W_IN, so Illustrator / LaTeX place it at print size without
rescaling), OUT_STEM.pdf (vector, same size, fonts embedded), OUT_STEM_outlined.pdf (the same with the text as paths,
ghostscript), OUT_STEM.png (2x) and _1x.png; OUT_STEM is results/elcd/check/figs/panel_a by default.
Command line: NAME=VALUE overrides of module parameters, e.g. python make_schematic_C.py PART=desk (the figure
notebook, build_figure_elcd_notebook.py, draws PART=desk and PART=cards with its own overrides)
"""
import math
import os
import subprocess
import tempfile
import sys

import numpy as np
from PIL import Image
import make_schematic as A

# One projection for everything (desk and cards are the SAME parallelogram). Set BEFORE arm3d is
# used: arm3d reads A.X0/W/H/SHEAR at call time.
A.X0, A.W, A.H = 210, 500, 150
A.SHEAR = round(A.H * 100.0 / 140.0)              # the desk's original edge angle (100/140)

from make_schematic import P, fmt, path_d, FONT, C_TRUE, C_GUESS, C_STREAM, C_DEMO, C_TXT  # noqa: E402
import arm3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DATA = os.path.join(HERE, "schematic_data.npz")   # alpha = 2
OUT_STEM = os.path.join(REPO, "results", "elcd", "check", "figs", "panel_a")
PART = "all"                      # all: the whole figure; desk: the robot and the desk with its t_i labels; cards: the
                                  # legend, the k axis and the cards (a part is cropped to its content plus PART_PAD px)
PART_PAD = 12
K_SHOW = [0, 1, 2, -1]            # cards, top -> bottom; -1 = K (the last iterate)
N_NODES = 4
NODE_ARC = [0.0, 0.25, 0.50, 0.90]   # time nodes t_i as fractions of the ARC LENGTH of x*: t_0 = the start (the
                                  # fixed initial condition, never corrected);
                                  # t_1 / t_2 inside the ranges where the first damped LASER step moves the
                                  # node TOWARD x* (0.33 / 0.66 fell in the 19 % of samples where it
                                  # overshoots); t_3 before the samples bunch at the goal. (0.06 for t_0
                                  # shows the exact-prefix effect instead: converged after one iteration.)
FIT_ITERS = 2                     # the unit square is fitted to the bbox of x* AND iterates[:FIT_ITERS]
                                  # (the warm start overhangs the card otherwise; 0 = x* only)
MARGIN = 0.06                     # bbox -> unit square margin (fraction of the square)
U_RANGE, V_RANGE = (0.06, 0.94), (0.06, 0.90)   # where the bbox lands on the desk (v: far -> near; the near
                                  # band is left free so the base can stand further back without stretching t_3)
POLY_N = 550                      # polylines (x*, iterates) downsampled to about this many points
DEMO_EVERY = 14                   # demonstration dots: every DEMO_EVERY-th sample of demos[0] (~every 10th;
                                  # denser reads as a second dotted trajectory where the demo leaves x*)
TIME_ARROW_ARC = 0.50             # the time-direction arrowhead on x* (desk) sits at this arc fraction

# canvas / layout (px)
TOP_MARGIN = 14                   # above the highest point of the arm
DESK_W, DESK_H = 600, 180         # desk / card parallelogram (A.W, A.H); SHEAR follows H (see apply_geometry)
SLAB, CARD_SLAB = 14, 9           # desk / card thickness
CARD_GAP = 38                     # vertical gap between cards
LAYOUT = "top"                    # desk + robot on top, k = 0 under it, iterations go DOWN (the only layout)
TOP_GAP = 44                      # desk slab bottom -> far edge of the k = 0 card (+ the legend)
DESK_DX = 0                       # desk (and robot) shifted right by this
LABEL_CARD_NODES = True           # also label the nodes on the first card (k = 0)
NODE_LABEL_DESK = (12, 26)        # label offset from the desk's node dot (below the gripper)
NODE_LABEL_CARD = (12, -12)       # label offset from the first card's node marker
NODE_LABEL_CARD_NUDGE = {}        # extra (dx, dy) px for single labels on the first card, by node: {1: (-12, -3)}
K_AXIS_X = 46                     # the "LASER iterations k" arrow
K_LABEL_X = A.X0 + A.SHEAR * 0.5 - 16   # right-aligned k labels, 16 px off the card's edge at mid-height
ELLIPSIS_DY = 0                   # px: the ⋮ of the skipped iterations sits this far below the centre of its gap

# sizes (px at FONT_SCALE = 1; apply_geometry multiplies the font sizes, SCRIPT_MIN and the legend spacing by
# FONT_SCALE). The canvas prints PRINT_W_IN wide, so 1 px = PRINT_W_IN * 72 / canvas width pt (0.238 pt at 3.4 in):
# FONT_SCALE 1.35 gives labels 9.0 pt, legend 7.7 pt, sub/superscripts >= 7.1 pt (FONT_SCALE 1: 6.7 / 5.7 / 5.2 pt).
PRINT_W_IN = 3.4                  # printed width (in): about 2/5 of a letter page (8.5 in)
FONT_SCALE = 1.35
FS_LABEL, FS_SMALL, FS_AXIS = 28, 24, 26
SCRIPT_RATIO, SCRIPT_MIN = 0.78, 22   # sub/superscript glyph size: base * ratio, never below SCRIPT_MIN px
SW_XSTAR, SW_ITER, SW_STREAM, SW_SEQ = 4.0, 3.5, 2.0, 6.0
SW_SLAB, SW_CARD_SLAB = 1.4, 1.2  # outline of the desk slab and of a card slab
HALO_XSTAR_K = 13.0               # wide blue halo under the orange on the k = K card
R_NODE, R_DEMO, R_SEQ, R_GOAL = 6.5, 3.0, 8.5, 9.0
STREAM_HEAD = 9.0
STREAM_MIN_STEPS = 60             # shorter integrated lines are dropped (no arrowed stubs)

# colours (blue / orange from make_schematic)
C_SEQ = "#111111"
STREAM_OPACITY = 0.8
DEMO_OPACITY = 0.30

# streamlines of the learned field (uv units; the unit square is the desk)
SEED_NU, SEED_NV = 14, 9          # evenly spaced seed grid
SEED_MIN_DIST = 0.045             # no seeds this close to x*
STREAM_DS = 0.004                 # unit-speed step
STREAM_MAX_STEPS = 900
MERGE_DIST, MERGE_STEPS = 0.016, 30   # stop when this close to x* for this many consecutive steps
SEP_DIST = 0.032                  # stop when this close to an already drawn streamline (even spacing)
GOAL_R = 0.03                     # stop at the goal
CARD_STREAM_W, CARD_STREAM_OPACITY = 1.4, 0.5   # streamline stroke of the desk's streamlines on the cards
CARD_STREAM_HEAD = 7.0            # direction arrowhead on every card streamline (0 = none)

# robot
ARM_SCALE = 1.25                  # arm3d is tuned for W = 400 / H = 120; the desk here is 1.25x
BASE_U, BASE_BACK = 0.80, 90.0    # yaw axis: fraction of W along the far edge, px behind it (post-scale).
                                  # Right of and behind the nodes: the arm reaches ACROSS the desk (reach
                                  # 0.78 / 0.90 / 0.90 at t_1 / t_2 / t_3) instead of folding above them
                                  # (BASE_U 0.27: reach 0.5, elbows stacked 260 px above the desk)
GHOST_OPACITY, SOLID_OPACITY = 0.35, 1.0
GHOST_OUTLINE_W = 1.4             # arm3d.GHOST_OUTLINE_W (ghost link outline)
SOLID_NODE = -1                   # the solid exposure sits at this node (-1 = t_3, the last)
GHOST_NODES = [1, 2]              # outline ghosts at these nodes (t_0 at the far edge is a folded pose, reach
                                  # 0.55, and sets the arm's top; [2] or [1, 2] gives a 120 px shorter canvas
                                  # at the cost of a ghost at every node t_0..t_2); the others keep their dot

# the canvas ends RIGHT_MARGIN right of the widest element
ARM_OVERHANG = 60                 # the robot base's reach right of the desk's near-right corner
RIGHT_MARGIN = 30

# legend (between the desk and the k = 0 card): rows of entries (xstar, iter, field, demo, seq)
LEGEND = [["xstar", "iter", "seq"], ["field", "demo"]]
LEGEND_ROW = 40                   # row pitch (px)

# direct labels: (u, v) anchor on the plane + screen offset (dx, dy)
LABEL_XSTAR = ((0.11, 0.74), (0.0, 0.0))        # "x*(t)" under the left end of the S-loop (desk)
LABEL_ITER = ((0.86, 0.36), (0.0, -14.0))       # "x^(k)(t)" on the k = 0 card;
                                                # -14 px clears t_3 at FONT_SCALE 1.35


def _script(s, rest, dy, size):
    """a raised/lowered tspan; the baseline reset sits on the tspan that carries the FOLLOWING text
    (`rest`), so no glyphless tspan is needed (Inkscape / Illustrator may drop one). With nothing
    following, nothing needs resetting."""
    out = f'<tspan dy="{dy}" font-size="{size}">{s}</tspan>'
    if rest:
        out += f'<tspan dy="{-dy}">{rest}</tspan>'
    return out


def _script_size(base):
    return max(int(round(base * SCRIPT_RATIO)), SCRIPT_MIN)


def sup(s, rest="", base=None):
    base = base or FS_LABEL
    return _script(s, rest, -int(base * 0.42), _script_size(base))


def sub(s, rest="", base=None):
    base = base or FS_LABEL
    return _script(s, rest, int(base * 0.22), _script_size(base))


def ital(s):
    return f'<tspan font-style="italic">{s}</tspan>'


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def training_demos(start):
    """The training demonstrations of chain8z from sample `start` on, (n_train, N+1, 2): what export_chain8z.py
    read, rebuilt from the LASA dataset (ELCD_LASA_DIR, by default data/lasa/DataSet at the repository root)."""
    sys.path.insert(0, REPO)
    from experiments.elcd.train import data as DATA
    cfg = DATA.CHAIN_Z["chain8z"]                  # load_chain_z("chain8z") with the dataset folder made absolute
    demos = DATA.load_chain(cfg["shapes"], name="chain8z", gap=float(cfg["gap"]), offsets=cfg["offsets"],
                            lasa_dir=os.path.join(REPO, DATA.LASA_DIR))
    return demos.pos[:demos.n_train, start:]


class Data:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.X = d["X_star"]                       # (N+1, 2)
        self.demos = training_demos(int(d["start"]))   # (n_train, N+1, 2)
        self.It = d["iterates"]                    # (K+1, N+1, 2)
        self.err = d["err"]
        self.K, self.N = int(d["K"]), int(d["N"])
        self.alpha = float(d["alpha"])
        self.spi = float(d["seq_steps_per_iter"])
        self.gx, self.gy = d["grid_x"], d["grid_y"]
        self.fu, self.fv = d["field_u"], d["field_v"]   # (G, G), indexed [iy, ix] (meshgrid 'xy')
        self.x_eq = d["x_eq"]
        fit = np.vstack([self.X] + [self.It[k] for k in range(min(FIT_ITERS, self.K + 1))])
        self.lo, self.hi = fit.min(0), fit.max(0)
        seg = np.linalg.norm(np.diff(self.X, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        self.arc = s / s[-1]
        self.idx = np.array([int(np.searchsorted(self.arc, f)) for f in NODE_ARC])
        self.idx[-1] = min(self.idx[-1], self.N)
        assert len(self.idx) == N_NODES

    # --- state space -> unit square: u <- x, v <- y ---
    @staticmethod
    def axes():
        """(data axis shown along u, data axis shown along v)."""
        return (0, 1)

    def to_uv(self, pts):
        pts = np.asarray(pts, float)
        a, b = self.axes()
        u = (pts[..., a] - self.lo[a]) / (self.hi[a] - self.lo[a])
        v = (pts[..., b] - self.lo[b]) / (self.hi[b] - self.lo[b])
        return np.stack([U_RANGE[0] + (U_RANGE[1] - U_RANGE[0]) * u, V_RANGE[0] + (V_RANGE[1] - V_RANGE[0]) * v], axis=-1)

    def from_uv(self, uv):
        uv = np.asarray(uv, float)
        u = (uv[..., 0] - U_RANGE[0]) / (U_RANGE[1] - U_RANGE[0])
        v = (uv[..., 1] - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0])
        a, b = self.axes()
        ca = self.lo[a] + u * (self.hi[a] - self.lo[a])
        cb = self.lo[b] + v * (self.hi[b] - self.lo[b])
        return np.stack([ca, cb], axis=-1)

    def vec_uv(self, f):
        """a data-space vector (fx, fy) -> its (fu, fv) image under the (linear part of the) mapping."""
        a, b = self.axes()
        su = (U_RANGE[1] - U_RANGE[0]) / (self.hi[a] - self.lo[a])
        sv = (V_RANGE[1] - V_RANGE[0]) / (self.hi[b] - self.lo[b])
        return np.array([f[a] * su, f[b] * sv])

    def field_at(self, xy):
        """bilinear interpolation of the gridded field at data-space xy; None outside the grid."""
        x, y = float(xy[0]), float(xy[1])
        if not (self.gx[0] <= x <= self.gx[-1] and self.gy[0] <= y <= self.gy[-1]):
            return None
        ix = min(int(np.searchsorted(self.gx, x) - 1), len(self.gx) - 2)
        iy = min(int(np.searchsorted(self.gy, y) - 1), len(self.gy) - 2)
        ix, iy = max(ix, 0), max(iy, 0)
        tx = (x - self.gx[ix]) / (self.gx[ix + 1] - self.gx[ix])
        ty = (y - self.gy[iy]) / (self.gy[iy + 1] - self.gy[iy])
        w = np.array([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        fu = w @ np.array([self.fu[iy, ix], self.fu[iy, ix + 1], self.fu[iy + 1, ix], self.fu[iy + 1, ix + 1]])
        fv = w @ np.array([self.fv[iy, ix], self.fv[iy, ix + 1], self.fv[iy + 1, ix], self.fv[iy + 1, ix + 1]])
        return np.array([fu, fv])

    def seq_n(self, k):
        """samples the sequential solver has completed after k parallel iterations' worth of time."""
        return int(min(round(self.spi * k), self.N))


def thin(pts, n=POLY_N):
    """downsample a polyline to about n points, keeping the first and last."""
    step = max(1, int(round(len(pts) / n)))
    out = pts[::step]
    if (len(pts) - 1) % step:
        out = np.vstack([out, pts[-1]])
    return out


# ----------------------------------------------------------------------------
# Streamlines of the learned field (uv space, unit speed, even spacing)
# ----------------------------------------------------------------------------
def make_streamlines(D):
    xs_uv = D.to_uv(D.X)[::3]
    goal_uv = D.to_uv(D.x_eq)
    drawn = []                                    # (M, 2) points of accepted lines, for spacing
    lines = []

    def near_drawn(p):
        return bool(drawn) and np.min(np.linalg.norm(np.vstack(drawn) - p, axis=1)) < SEP_DIST

    def step_dir(p):
        f = D.field_at(D.from_uv(p))
        if f is None:
            return None
        g = D.vec_uv(f)
        n = np.linalg.norm(g)
        return None if n < 1e-9 else g / n

    seeds = [(u, v) for v in np.linspace(MARGIN, 1 - MARGIN, SEED_NV) for u in np.linspace(MARGIN, 1 - MARGIN, SEED_NU)]
    for seed in seeds:
        p = np.array(seed)
        if np.min(np.linalg.norm(xs_uv - p, axis=1)) < SEED_MIN_DIST or near_drawn(p):
            continue
        pts, merged = [p.copy()], 0
        for _ in range(STREAM_MAX_STEPS):
            d1 = step_dir(p)
            if d1 is None:
                break
            d2 = step_dir(p + 0.5 * STREAM_DS * d1)      # midpoint rule
            if d2 is None:
                break
            p = p + STREAM_DS * d2
            if not (0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0):
                break
            pts.append(p.copy())
            if np.linalg.norm(p - goal_uv) < GOAL_R or near_drawn(p):
                break
            if np.min(np.linalg.norm(xs_uv - p, axis=1)) < MERGE_DIST:
                merged += 1
                if merged > MERGE_STEPS:
                    break
            else:
                merged = 0
        if len(pts) > STREAM_MIN_STEPS:
            arr = np.array(pts)
            lines.append(arr)
            drawn.append(arr[::4])
    return lines


# ----------------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------------
def head(p, d, size, color):
    """filled arrowhead with its tip at p pointing along d."""
    d = d / (np.linalg.norm(d) + 1e-12)
    n = np.array([-d[1], d[0]])
    b1 = p - d * size + n * size * 0.42
    b2 = p - d * size - n * size * 0.42
    return (f'<polygon points="{fmt(p[0])},{fmt(p[1])} {fmt(b1[0])},{fmt(b1[1])} {fmt(b2[0])},{fmt(b2[1])}" '
            f'fill="{color}" stroke="none"/>')



def arrow(p, q, color, sw, size, halo=True):
    """shaft p -> q (shortened by the head) + head at q; the head is never smaller than `size`."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    if L < 1e-9:
        return ""
    u = d / L
    out = []
    if L > size:
        q2 = q - u * (size * 0.9)
        if halo:
            out.append(f'<line x1="{fmt(p[0])}" y1="{fmt(p[1])}" x2="{fmt(q2[0])}" y2="{fmt(q2[1])}" '
                       f'stroke="#ffffff" stroke-width="{sw + 3.5:.1f}" stroke-linecap="round"/>')
        out.append(f'<line x1="{fmt(p[0])}" y1="{fmt(p[1])}" x2="{fmt(q2[0])}" y2="{fmt(q2[1])}" '
                   f'stroke="{color}" stroke-width="{sw:.1f}" stroke-linecap="round"/>')
    if halo:
        out.append(head(q + u * 1.5, u, size + 2.5, "#ffffff"))
    out.append(head(q, u, size, color))
    return "\n".join(out)


def text(x, y, s, size=None, fill=C_TXT, anchor="start", italic=False, bold=False, extra="", halo=False):
    size = size or FS_LABEL           # read at call time: apply_geometry rescales FS_LABEL
    st = f'font-family="{FONT}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
    if italic:
        st += ' font-style="italic"'
    if bold:
        st += ' font-weight="bold"'
    out = []
    if halo:                          # the white stroke alone (no fill): the PDF then holds each label as text once
        halo_st = st.replace(f'fill="{fill}"', 'fill="none"')
        out.append(f'<text x="{fmt(x)}" y="{fmt(y)}" {halo_st} stroke="#ffffff" stroke-width="5" '
                   f'stroke-linejoin="round" {extra}>{s}</text>')
    out.append(f'<text x="{fmt(x)}" y="{fmt(y)}" {st} {extra}>{s}</text>')
    return "\n".join(out)


def seq_marker(x, y, r=R_SEQ):
    """the sequential solver's position: a hollow black ring on a white under-ring, so an orange
    node marker drawn later shows INSIDE it when the two coincide (k = 0 .. 2 cards)."""
    return (f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r + 2.0:.1f}" fill="none" stroke="#ffffff" stroke-width="2.4"/>'
            f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r:.1f}" fill="none" stroke="{C_SEQ}" stroke-width="2.6"/>')


def node_marker(x, y, color, r=R_NODE):
    return f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{r:.1f}" fill="{color}" stroke="#ffffff" stroke-width="2"/>'


def slab(y0, h, thick, fill_top, fills, stroke, sw):
    X0, W, SH = A.X0, A.W, A.SHEAR
    fl, fr = (X0 + SH, y0), (X0 + W + SH, y0)
    nl, nr = (X0, y0 + h), (X0 + W, y0 + h)
    quad = f"{fl[0]},{fl[1]} {fr[0]},{fr[1]} {nr[0]},{nr[1]} {nl[0]},{nl[1]}"
    out = [f'<polygon points="{nl[0]},{nl[1]} {nr[0]},{nr[1]} {nr[0]},{nr[1] + thick} {nl[0]},{nl[1] + thick}" fill="{fills[0]}"/>',
           f'<polygon points="{nr[0]},{nr[1]} {fr[0]},{fr[1]} {fr[0]},{fr[1] + thick} {nr[0]},{nr[1] + thick}" fill="{fills[1]}"/>',
           f'<polygon points="{quad}" fill="{fill_top}" stroke="{stroke}" stroke-width="{sw}"/>']
    return "\n".join(out), quad


def scr(D, pts, y0):
    uv = D.to_uv(pts)
    return np.array([P(u, v, y0) for u, v in np.atleast_2d(uv)])


# ----------------------------------------------------------------------------
# Robot setup
# ----------------------------------------------------------------------------
_ARM_SCALED = [False]


def scale_arm():
    """arm3d is tuned for a 400 x 120 desk; scale every length/radius with the desk (once)."""
    if _ARM_SCALED[0]:
        return
    for nm in ("D", "PED_W", "PED_D", "PED_H", "TURRET_R", "TURRET_H", "COL_R", "COLUMN", "L1", "L2",
               "REACH_MARGIN", "WRIST_LINK", "FLANGE_H", "HAND_H", "FINGER_H", "PEN_EXTRA", "HAND_W",
               "FINGER_W", "FINGER_GAP", "R_WLINK", "R_SHOULDER", "R_ELBOW", "R_WRIST", "HALO", "PEN_W"):
        setattr(arm3d, nm, getattr(arm3d, nm) * ARM_SCALE)
    arm3d.W_UPPER = tuple(w * ARM_SCALE for w in arm3d.W_UPPER)
    arm3d.W_FORE = tuple(w * ARM_SCALE for w in arm3d.W_FORE)
    arm3d.OUTLINE_W = arm3d.OUTLINE_W * min(ARM_SCALE, 1.4)
    arm3d.GHOST_OUTLINE_W = GHOST_OUTLINE_W
    arm3d.BASE_U, arm3d.BASE_BACK = BASE_U, BASE_BACK
    _ARM_SCALED[0] = True


def drawn_nodes():
    """nodes that get an arm exposure: the ghosts (far -> near order is applied in desk()) + the solid one."""
    solid = list(range(N_NODES))[SOLID_NODE]
    return [i for i in GHOST_NODES if i != solid], solid


def desk_y0(D):
    """far-edge y of the desk so that the arm's highest point sits TOP_MARGIN below the canvas top."""
    _, S = arm3d.robot_base_3d(0.0)
    ghosts, solid = drawn_nodes()
    targets = [tuple(p) for p in scr(D, D.X[D.idx[ghosts + [solid]]], 0.0)]
    for t in targets:
        arm3d.reach_check(S, *t)
    top = arm3d.arm_top_y(S, targets)
    return float(math.ceil(TOP_MARGIN - top))


# ----------------------------------------------------------------------------
# Desk with the (one) robot
# ----------------------------------------------------------------------------
def desk(D, y0, streams):
    out = []
    base_svg, S = arm3d.robot_base_3d(y0)          # behind the desk: before the slab
    out.append(base_svg)
    sl, quad = slab(y0, A.H, SLAB, "#fbfbfb", ("#cfcfcf", "#b9b9b9"), "#8a8a8a", SW_SLAB)
    out.append(sl)
    out.append(f'<clipPath id="clipDesk"><polygon points="{quad}"/></clipPath>')
    # field
    out.append(f'<g clip-path="url(#clipDesk)" stroke="{C_STREAM}" stroke-width="{SW_STREAM}" fill="none" '
               f'stroke-opacity="{STREAM_OPACITY}" stroke-linecap="round">')
    heads = []
    for ln in streams:
        pts = np.array([P(u, v, y0) for u, v in ln])
        out.append(f'<path d="{path_d(pts)}"/>')
        i = int(0.5 * (len(pts) - 1))
        heads.append(head(pts[i], pts[min(i + 2, len(pts) - 1)] - pts[i], STREAM_HEAD, C_STREAM))
    out.append('</g>')
    # demonstration dots (under the field's arrowheads and x*), then the arrowheads, then x*
    out.append(f'<g fill="{C_DEMO}" fill-opacity="{DEMO_OPACITY}">')
    for x, y in scr(D, D.demos[0][::DEMO_EVERY], y0):
        out.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{R_DEMO}"/>')
    out.append('</g>')
    out.append(f'<g clip-path="url(#clipDesk)" fill-opacity="{STREAM_OPACITY}">' + "\n".join(heads) + '</g>')
    xs = scr(D, thin(D.X), y0)
    out.append(f'<path d="{path_d(xs)}" fill="none" stroke="{C_TRUE}" stroke-width="{SW_XSTAR}" '
               f'stroke-linecap="round" stroke-linejoin="round"/>')
    gx, gy = scr(D, D.x_eq, y0)[0]
    out.append(f'<circle cx="{fmt(gx)}" cy="{fmt(gy)}" r="{R_GOAL}" fill="#ffffff" stroke="{C_TRUE}" stroke-width="3"/>')
    # time direction: an arrowhead on a clean stretch of x* (not under the hand / beside the goal)
    j = int(np.searchsorted(D.arc, TIME_ARROW_ARC))
    xs_full = scr(D, D.X[[j - 6, j]], y0)
    out.append(head(xs_full[1], xs_full[1] - xs_full[0], 10.0, C_TRUE))
    # nodes
    nodes = scr(D, D.X[D.idx], y0)
    # robot: ghosts (GHOST_NODES) far -> near, the solid exposure last; the other nodes keep their dot
    targets = [tuple(p) for p in nodes]
    ghosts, solid = drawn_nodes()
    for i in ghosts + [solid]:
        arm3d.reach_check(S, *targets[i])
    order = sorted(ghosts, key=lambda i: D.to_uv(D.X[D.idx[i]])[1]) + [solid]
    for i in order:
        if i == solid:
            out.append(arm3d.arm_pose_3d(S, *targets[i], SOLID_OPACITY, style="solid"))
        else:
            out.append(arm3d.arm_pose_3d(S, *targets[i], GHOST_OPACITY, style="ghost"))
    for x, y in nodes:
        out.append(node_marker(x, y, C_TRUE))
    out.append(arm3d.shoulder_cap_3d(S))
    # label x*(t)
    (lu, lv), (dx, dy) = LABEL_XSTAR
    lx, ly = P(lu, lv, y0)
    out.append(text(lx + dx, ly + dy + FS_LABEL * 0.35, "x*(t)", fill=C_TRUE, italic=True, halo=True))
    print("arm3d poses (drawn nodes):\n" + arm3d.describe(S, [targets[i] for i in ghosts + [solid]],
                                                           names=[f"t{i}" for i in ghosts + [solid]]))
    return "\n".join(out), nodes


# ----------------------------------------------------------------------------
# One card of the stack
# ----------------------------------------------------------------------------
def card(D, k, y0, label, streams):
    """one card, the desk's parallelogram (scr() reads A.W / A.H / A.SHEAR at call time)."""
    return _card(D, k, y0, label, streams)


def _card(D, k, y0, label, streams):
    out = []
    sl, quad = slab(y0, A.H, CARD_SLAB, "#ffffff", ("#d6d6d6", "#c4c4c4"), "#9a9a9a", SW_CARD_SLAB)
    out.append(sl)
    out.append(f'<clipPath id="clipCard{k}"><polygon points="{quad}"/></clipPath>')
    # the desk's streamlines, lighter, with their arrowheads
    out.append(f'<g clip-path="url(#clipCard{k})" stroke="{C_STREAM}" stroke-width="{CARD_STREAM_W}" fill="none" '
               f'stroke-opacity="{CARD_STREAM_OPACITY}" stroke-linecap="round">')
    heads = []
    for ln in streams:
        pts = np.array([P(u, v, y0) for u, v in ln])
        out.append(f'<path d="{path_d(pts)}"/>')
        i = int(0.5 * (len(pts) - 1))
        if CARD_STREAM_HEAD > 0:
            heads.append(head(pts[i], pts[min(i + 2, len(pts) - 1)] - pts[i], CARD_STREAM_HEAD, C_STREAM))
    out.append('</g>')
    out.append(f'<g clip-path="url(#clipCard{k})" fill-opacity="{CARD_STREAM_OPACITY}">' + "\n".join(heads) + '</g>')
    converged = k == D.K
    xs = scr(D, thin(D.X), y0)
    if converged:                                  # wide blue halo so the coincidence shows
        out.append(f'<path d="{path_d(xs)}" fill="none" stroke="{C_TRUE}" stroke-width="{HALO_XSTAR_K}" '
                   f'stroke-opacity="0.35" stroke-linecap="round" stroke-linejoin="round"/>')
    out.append(f'<path d="{path_d(xs)}" fill="none" stroke="{C_TRUE}" stroke-width="{SW_XSTAR}" '
               f'stroke-linecap="round" stroke-linejoin="round"/>')
    it = scr(D, thin(D.It[k]), y0)
    out.append(f'<path d="{path_d(it)}" fill="none" stroke="{C_GUESS}" stroke-width="{SW_ITER}" '
               f'stroke-linecap="round" stroke-linejoin="round"/>')
    # sequential progress: dark segment along x* up to sample n_k, then its ring marker (drawn
    # BEFORE the node markers so a coincident orange node shows inside the ring)
    n = D.seq_n(k)
    if n >= 1:
        sq = scr(D, D.X[:n + 1], y0)
        out.append(f'<path d="{path_d(sq)}" fill="none" stroke="#ffffff" stroke-width="{SW_SEQ + 3}" stroke-linecap="round" stroke-linejoin="round"/>')
        out.append(f'<path d="{path_d(sq)}" fill="none" stroke="{C_SEQ}" stroke-width="{SW_SEQ}" stroke-linecap="round" stroke-linejoin="round"/>')
    sx, sy = scr(D, D.X[n], y0)[0]
    out.append(seq_marker(sx, sy))
    # the node markers, in a separate layer drawn after the cards
    nodes = scr(D, D.It[k][D.idx], y0)
    markers = [node_marker(x, y, C_GUESS) for x, y in nodes]
    # label x^(k)(t) on the k = 0 card
    if k == 0:
        (lu, lv), (dx, dy) = LABEL_ITER
        lx, ly = P(lu, lv, y0)
        lx, ly, anchor = lx + dx, ly + dy, "middle"
        out.append(text(lx, ly + FS_LABEL * 0.35, "x" + sup("(k)", "(t)"), fill=C_GUESS, italic=True, anchor=anchor, halo=True))
    # k label (left, right-aligned off the card's edge)
    ym = y0 + A.H * 0.5
    out.append(text(K_LABEL_X, ym + FS_LABEL * 0.35, label, anchor="end"))
    return "\n".join(out), nodes, "\n".join(markers)


# ----------------------------------------------------------------------------
# Legend (one row) and the bottom block
# ----------------------------------------------------------------------------
def legend_entry(e, x, y):
    """one legend entry (swatch + label) at x; returns (svg, advance to the next entry)."""
    out = []
    if e == "xstar":
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 40}" y2="{y}" stroke="{C_TRUE}" stroke-width="{SW_XSTAR}" stroke-linecap="round"/>')
        out.append(text(x + 52, y + FS_SMALL * 0.35, "x*(" + ital("t") + ")", size=FS_SMALL, fill=C_TRUE, italic=True))
        return "\n".join(out), 160
    if e == "iter":
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 40}" y2="{y}" stroke="{C_GUESS}" stroke-width="{SW_ITER}" stroke-linecap="round"/>')
        out.append(text(x + 52, y + FS_SMALL * 0.35, "x" + sup("(k)", "(t)", base=FS_SMALL), size=FS_SMALL, fill=C_GUESS, italic=True))
        return "\n".join(out), 190
    if e == "field":
        pts = np.array([[x, y + 2], [x + 12, y - 5], [x + 26, y + 5], [x + 40, y - 1]])
        out.append(f'<path d="{path_d(pts)}" fill="none" stroke="{C_STREAM}" stroke-width="{SW_STREAM}" stroke-linecap="round"/>')
        out.append(head(np.array([x + 44, y - 2]), np.array([1.0, -0.2]), STREAM_HEAD, C_STREAM))
        out.append(text(x + 56, y + FS_SMALL * 0.35, "learned field", size=FS_SMALL))
        return "\n".join(out), 215
    if e == "demo":
        for dx in (0, 11, 22, 33):
            out.append(f'<circle cx="{x + dx}" cy="{y + (2 if dx % 22 else -2)}" r="{R_DEMO}" fill="{C_DEMO}" fill-opacity="{DEMO_OPACITY}"/>')
        out.append(text(x + 46, y + FS_SMALL * 0.35, "demonstration", size=FS_SMALL))
        return "\n".join(out), 215
    if e == "seq":
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 30}" y2="{y}" stroke="{C_SEQ}" stroke-width="{SW_SEQ}" stroke-linecap="round"/>')
        out.append(seq_marker(x + 32, y))
        out.append(text(x + 50, y + FS_SMALL * 0.35, "sequential solver", size=FS_SMALL))
        return "\n".join(out), 250
    raise SystemExit(f"unknown legend entry {e!r}")


def legend(y):
    """the LEGEND rows, the first at y (swatch centre line), starting 70 px left of the cards' far-left corner."""
    out = []
    for r, row in enumerate(LEGEND):
        x = A.X0 - 70
        for e in row:
            svg, adv = legend_entry(e, x, y + r * LEGEND_ROW)
            out.append(svg)
            x += adv * FONT_SCALE
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def layout(D, ks):
    """far-edge y of the desk, {k: far-edge y of card k}, the desk's X0 and the y of the lowest content
    (see LAYOUT). The stack pitch is one card + CARD_GAP; the desk's y keeps the arm inside the canvas."""
    card_h = A.H
    pitch = card_h + CARD_GAP
    y0 = desk_y0(D)
    if LAYOUT == "top":
        y_desk, x_desk = y0, A.X0 + DESK_DX
        mid = len(LEGEND) * LEGEND_ROW + 8         # the legend sits in the gap
        stack_top = y_desk + A.H + SLAB + TOP_GAP + mid
        order = ks                                 # k = 0 under the desk
    else:
        raise SystemExit(f"LAYOUT must be top, got {LAYOUT!r}")
    card_y = {k: stack_top + slot * pitch for slot, k in enumerate(order)}
    y_cards_bottom = max(card_y.values()) + card_h + CARD_SLAB
    y_bottom = y_cards_bottom
    return y_desk, card_y, x_desk, y_bottom


def build(D):
    scale_arm()
    streams = make_streamlines(D)
    print(f"{len(streams)} streamlines")
    ks = [k if k >= 0 else D.K for k in K_SHOW]
    labels = [f"{ital('k')} = {k}" for k in ks]
    y_desk, card_y, x_desk, y_bottom = layout(D, ks)
    h_canvas = int(y_bottom + 30)
    # the canvas ends after the widest element
    w_canvas = int(max(x_desk + A.W + A.SHEAR + ARM_OVERHANG, A.X0 + (A.W + A.SHEAR)) + RIGHT_MARGIN)

    desk_g, cards_g = [], []                       # the SVG pieces of the two parts (PART); wrap() frames them
    x0_cards = A.X0
    A.X0 = x_desk                                  # the desk (and the robot: arm3d reads A.X0) may sit elsewhere
    desk_svg, desk_nodes = desk(D, y_desk, streams)
    A.X0 = x0_cards
    desk_g.append(f'<g id="desk">\n{desk_svg}\n</g>')
    cards_svg, card_nodes, marker_svg = [], {}, []
    for k, lab in zip(ks, labels):
        c, nodes, markers = card(D, k, card_y[k], lab, streams)
        cards_svg.append(f'<g id="card_k{k}">\n{c}\n</g>')
        card_nodes[k] = nodes
        marker_svg.append(f'<g id="nodes_k{k}">\n{markers}\n</g>')
    cards_g.extend(cards_svg)

    # the node markers, then t_i beside the desk's node dots (and the first card's markers)
    cards_g.extend(marker_svg)
    dx, dy = NODE_LABEL_DESK
    for i in range(N_NODES):
        x, y = desk_nodes[i]
        desk_g.append(text(x + dx, y + dy, "t" + sub(str(i)), anchor="start", italic=True, halo=True))
    if LABEL_CARD_NODES:
        dx, dy = NODE_LABEL_CARD
        for i in range(N_NODES):
            x, y = card_nodes[ks[0]][i]
            nx, ny = NODE_LABEL_CARD_NUDGE.get(i, (0, 0))
            cards_g.append(text(x + dx + nx, y + dy + ny, "t" + sub(str(i)), anchor="start", italic=True, halo=True))

    # the k axis (from the k = 0 card down to the k = K card)
    card_h = A.H
    y_lo = card_y[ks[0]] + card_h * 0.5
    y_hi = card_y[ks[-1]] + card_h * 0.5
    cards_g.append('<g id="k_axis">')
    cards_g.append(arrow(np.array([K_AXIS_X, y_lo + 10]), np.array([K_AXIS_X, y_hi - 10]), C_TXT, 2.6, 13, halo=False))
    cards_g.append(text(K_AXIS_X - 14, 0.5 * (y_lo + y_hi), "LASER iterations " + ital("k"), size=FS_AXIS, anchor="middle",
                    extra=f'transform="rotate(-90 {K_AXIS_X - 14} {0.5 * (y_lo + y_hi):.1f})"'))
    # skipped iterations: an ellipsis in the gap between the last two cards
    kk = sorted(ks)
    if len(kk) >= 2 and kk[-1] - kk[-2] > 1:
        y_gap = min(card_y[kk[-1]], card_y[kk[-2]]) + card_h + CARD_SLAB + 0.5 * (CARD_GAP - CARD_SLAB) + ELLIPSIS_DY
        cards_g.append(text(K_LABEL_X, y_gap + FS_LABEL * 0.35, "⋮", anchor="end"))
    cards_g.append('</g>')

    # the legend: rows centred in the desk -> k = 0 gap
    y_leg = y_desk + A.H + SLAB + TOP_GAP * 0.5 + 4 + LEGEND_ROW * 0.5
    cards_g.append(f'<g id="legend">\n{legend(y_leg)}\n</g>')
    print(f"layout {LAYOUT}; canvas {w_canvas} x {h_canvas}; desk y0 {y_desk:.0f}; cards at {card_y}; "
          f"seq samples per card {[D.seq_n(k) for k in ks]}; err {[f'{D.err[k]:.1e}' for k in ks]}")
    return desk_g, cards_g, w_canvas, h_canvas


def wrap(body, box):
    """the SVG of body (a list of pieces) showing the canvas region box = (x, y, w, h), PRINT_W_IN wide in print."""
    x, y, w, h = box
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{PRINT_W_IN:.4f}in" '
                      f'height="{PRINT_W_IN * h / w:.4f}in" viewBox="{x} {y} {w} {h}">',
                      f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff"/>', *body, "</svg>"])


def content_box(svg_text, w_canvas, h_canvas):
    """(x, y, w, h) of the non-white pixels of svg_text rendered at 1 px per canvas px, padded by PART_PAD and
    kept inside the canvas: the crop of a part to its content."""
    with tempfile.TemporaryDirectory() as tmp:
        src, png = os.path.join(tmp, "part.svg"), os.path.join(tmp, "part.png")
        # the root sized in px: rsvg-convert rounds an inch size to whole px at 90 dpi, which at some PRINT_W_IN
        # leaves transparent edge columns that read as ink, so the part came out uncropped (seen at 2.45 in)
        head, rest = svg_text.split(">", 1)
        head = head[:head.index(" width=")] + f' width="{w_canvas}" height="{h_canvas}"' + head[head.index(" viewBox="):]
        with open(src, "w", encoding="utf-8") as f:
            f.write(head + ">" + rest)
        subprocess.run(["rsvg-convert", "-w", str(w_canvas), "-o", png, src], check=True)
        img = Image.open(png)
        assert img.size == (w_canvas, h_canvas), (img.size, w_canvas, h_canvas)
        ink = np.asarray(img.convert("RGB")).min(axis=2) < 250
    ys, xs = np.nonzero(ink)
    x0, x1 = max(int(xs.min()) - PART_PAD, 0), min(int(xs.max()) + 1 + PART_PAD, w_canvas)
    y0, y1 = max(int(ys.min()) - PART_PAD, 0), min(int(ys.max()) + 1 + PART_PAD, h_canvas)
    return x0, y0, x1 - x0, y1 - y0


def apply_geometry():
    """re-derive the projection (A.W / A.H / A.SHEAR) and the x positions that depend on it from
    DESK_W / DESK_H, after the NAME=VALUE overrides (a K_LABEL_X override is recomputed)."""
    global K_LABEL_X, FS_LABEL, FS_SMALL, FS_AXIS, SCRIPT_MIN, LEGEND_ROW
    FS_LABEL, FS_SMALL, FS_AXIS, SCRIPT_MIN = (round(v * FONT_SCALE, 1) for v in (FS_LABEL, FS_SMALL, FS_AXIS, SCRIPT_MIN))
    LEGEND_ROW = round(LEGEND_ROW * FONT_SCALE, 1)
    A.W, A.H = DESK_W, DESK_H
    A.SHEAR = round(A.H * 100.0 / 140.0)
    K_LABEL_X = A.X0 + A.SHEAR * 0.5 - 16


def apply_overrides(argv):
    g = globals()
    for a in argv:
        if "=" not in a:
            raise SystemExit(f"expected NAME=VALUE, got {a!r}")
        k, v = a.split("=", 1)
        if k not in g:
            raise SystemExit(f"unknown parameter {k!r}")
        cur = g[k]
        if isinstance(cur, bool):
            g[k] = v.strip().lower() in ("1", "true", "yes")
        elif isinstance(cur, (int, float)):
            g[k] = type(cur)(float(v)) if isinstance(cur, int) and float(v).is_integer() else float(v)
        elif isinstance(cur, str):
            g[k] = v
        else:
            g[k] = eval(v)  # noqa: S307 (lists / tuples from the command line)
        print(f"override {k} = {g[k]!r}")


def rasterise(stem, w_px):
    """PNGs at 2x / 1x the canvas's px width (the svg's own size is in inches), and the vector PDF."""
    for zoom, suffix in ((2, ".png"), (1, "_1x.png")):
        subprocess.run(["rsvg-convert", "-w", str(zoom * w_px), "-o", stem + suffix, stem + ".svg"], check=True)
    # rsvg-convert maps inches to px at --dpi and cairo writes px as pt, so 72 dpi keeps the size exact
    subprocess.run(["rsvg-convert", "-f", "pdf", "--dpi-x", "72", "--dpi-y", "72", "-o", stem + ".pdf", stem + ".svg"],
                   check=True)
    # the same PDF with every glyph turned into a path: opens in Illustrator without the fonts installed
    subprocess.run(["gs", "-q", "-dNoOutputFonts", "-sDEVICE=pdfwrite", "-o", stem + "_outlined.pdf", stem + ".pdf"],
                   check=True)


if __name__ == "__main__":
    apply_overrides(sys.argv[1:])
    apply_geometry()
    D = Data(DATA)
    print(f"data: N={D.N} K={D.K} alpha={D.alpha:g} seq_steps_per_iter={D.spi:.1f} nodes idx={D.idx.tolist()}")
    desk_g, cards_g, w_canvas, h_canvas = build(D)
    parts = {"all": desk_g + cards_g, "desk": desk_g, "cards": cards_g}
    if PART not in parts:
        raise SystemExit(f"PART must be all / desk / cards, got {PART!r}")
    canvas = (0, 0, w_canvas, h_canvas)
    box = canvas if PART == "all" else content_box(wrap(parts[PART], canvas), w_canvas, h_canvas)
    s = wrap(parts[PART], box)
    pt = PRINT_W_IN * 72 / box[2]
    print(f"print {PRINT_W_IN:.2f} x {PRINT_W_IN * box[3] / box[2]:.2f} in; text {FS_LABEL * pt:.1f} / {FS_SMALL * pt:.1f} / "
          f"{FS_AXIS * pt:.1f} pt, scripts >= {SCRIPT_MIN * pt:.1f} pt"
          + (f"; part {PART}: canvas region x {box[0]} y {box[1]} w {box[2]} h {box[3]}" if PART != "all" else ""))
    os.makedirs(os.path.dirname(OUT_STEM) or ".", exist_ok=True)
    with open(OUT_STEM + ".svg", "w", encoding="utf-8") as f:
        f.write(s)
    rasterise(OUT_STEM, box[2])
    print("wrote", OUT_STEM + ".svg / .pdf / _outlined.pdf / .png / _1x.png", f"({os.path.getsize(OUT_STEM + '.svg') // 1024} KB svg)")
