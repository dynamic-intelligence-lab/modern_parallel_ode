"""ELCD network schematic, "spaces" layout: a data-space panel and a latent-space panel joined by encoder / decoder
(the encoder / decoder part of panel A of the ELCD figure). The network variant: blue x*(t) from x_0 to the goal
x_eq in data space; z* with gray flow lines and the equation dz/dt = A(z)(z - z*) in latent space; the encoder carries
a small net that outputs a monotone spline, and the decoder carries the same glyph mirrored (same weights,
inverted). The cards are drawn at NETWORK_SCALE; the latent panel moves right so each card keeps CARD_STUB px of
arrow on both sides.

The decoder drawn here is illustrative, not the trained flow (on chain8z the trained flow squeezes the data box into
a thin creased band, so its grid is unreadable): psi(w) = swirl(C + K R(ALPHA) (w - M)). swirl rotates the circle
of radius r about C by theta(r), a monotone cubic through TWIST_DEG that is zero beyond R2. It maps every circle about
C onto itself, so it is a diffeomorphism. C, ALPHA, L1, L2, R2 and TWIST_DEG were fitted to an S-shaped curve
(mean Chamfer distance 2.1 px, with penalties on the end points, on turns tighter than 16 px and on curvature
wiggles) so that the straight latent line maps onto it.

    python make_nn_spaces.py network OUT_STEM=/path/stem [PRINT_W_IN=<in>] [NAME=VALUE ...]

writes the network variant at print size to <stem>.svg/.pdf/_outlined.pdf/.png (2x)/_1x.png (ghostscript for the
outlined copy): PRINT_W_IN wide, or at PT_PER_PX (the 9 / 7.7 / 7.1 pt text of make_schematic_C.py) when
PRINT_W_IN is 0. Without OUT_STEM it writes results/elcd/check/figs/elcd_nn_spaces_network.svg/.png/_1x.png. network
is the only variant (the figure notebook passes the word). numpy, scipy and rsvg-convert only.
"""
import os
import subprocess
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---- style --------------------------------------------------------------------------------------------------------
FONT = "'DejaVu Sans', sans-serif"
FS_LABEL, FS_SECOND, FS_SCRIPT = 28, 24, 22
C_TEXT, C_BLUE, C_GRAY, C_STROKE = "#222222", "#2F6DB5", "#B9B9B9", "#9a9a9a"
ARROW_W, HEAD_LEN, HEAD_HALF = 2.6, 13.0, 5.5
FLOW_W, FLOW_HEAD_LEN, FLOW_HEAD_HALF = 2.0, 8.5, 3.8
CURVE_W = 4.0
FRAME_W = 1.2                             # outline of the two panels and of the network cards
# print scale: 28 px labels print at 9 pt (24 px at 7.7 pt, 22 px scripts at 7.1 pt), the text sizes of
# make_schematic_C.py's default drawing at 3.4 in (FONT_SCALE 1.35); a 638 px canvas prints 2.85 in wide
PT_PER_PX = 9.0 / FS_LABEL

# ---- layout (1x px) -----------------------------------------------------------------------------------------------
W, H = 620, 290
PANEL_Y, PANEL_W, PANEL_H = 44, 232, 228
DATA_X, LAT_X = 8, 380
TITLE_Y = 31
GAP_X = (DATA_X + PANEL_W, LAT_X)         # the arrows run between the panels
ENC_Y, DEC_Y = 100, 216
Z_STAR = np.array([495.0, 132.0])
FLOW_ANGLES = (-126.6, -55.7, 15.7, 87.6, 159.5)   # gray flow lines into z* (degrees, screen coordinates)
FLOW_R_OUT, FLOW_R_IN, FLOW_BEND = 72.0, 14.0, -14.0
CARD_W, CARD_H, CARD_STUB = 92, 46, 24    # network cards at scale 1; CARD_STUB = arrow left on each side of a card
NETWORK_SCALE = 1.2
OUT_STEM = ""                             # NAME=VALUE on the command line (OUT_STEM=/path/stem): write the network
                                          # variant at NETWORK_SCALE to that stem (.svg, .pdf, _outlined.pdf, .png, _1x.png)
PRINT_W_IN = 0.0                          # > 0: that SVG prints this wide and its text scales with it (PT_PER_PX follows);
                                          # 0: PT_PER_PX stays 9 pt per 28 px label

# ---- the illustrative decoder -------------------------------------------------------------------------------------
C = np.array([135.302, 164.185])          # twist centre, data px
ALPHA = np.radians(34.287)                # direction of the pre-twist line
L1, L2 = 95.676, 84.828                   # pre-twist line from C - L1 u to C + L2 u
R2 = 111.386                              # theta(r) = 0 for r >= R2
TWIST_DEG = (136.341, 133.624, 129.642, 122.306, 105.274, 38.500, 13.422, 0.0)   # theta at linspace(0, R2, 8)
K = 1.9                                   # data px per latent px
Z0 = Z_STAR - np.array([(L1 + L2) / K, 0.0])   # z(t) runs horizontally into z*
M = Z0 + np.array([L1 / K, 0.0])          # the latent point that maps to C

_rk = np.linspace(0.0, R2, len(TWIST_DEG))
_th = np.radians(TWIST_DEG)
_slope = PchipInterpolator(_rk, _th).derivative()(_rk)
_slope[0] = _slope[-1] = 0.0              # flat at the centre and at R2: the twist is C^1 everywhere
_twist = CubicHermiteSpline(_rk, _th, _slope)


def twist(r):
    return np.where(r < R2, _twist(np.minimum(r, R2)), 0.0)


def decode(w):
    """Latent px (..., 2) -> data px (..., 2)."""
    ca, sa = np.cos(ALPHA), np.sin(ALPHA)
    d = w - M
    p = K * np.stack([ca * d[..., 0] - sa * d[..., 1], sa * d[..., 0] + ca * d[..., 1]], -1)
    th = twist(np.hypot(p[..., 0], p[..., 1]))
    c, s = np.cos(th), np.sin(th)
    return C + np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], -1)


def latent_line(n=500):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return Z0 + t * (Z_STAR - Z0)


X_CURVE = decode(latent_line())           # x*(t) in data px
X0, X_STAR = X_CURVE[0], X_CURVE[-1]

# ---- SVG helpers ------------------------------------------------------------------------------------------------
SUB = '<tspan dy="7" font-size="{s}">{t}</tspan><tspan dy="-7"></tspan>'
SUP = '<tspan dy="-11" font-size="{s}">{t}</tspan><tspan dy="11"></tspan>'


def sub(t, s=FS_SCRIPT):
    return SUB.format(s=s, t=t)


def sup(t, s=FS_SCRIPT):
    return SUP.format(s=s, t=t)


def m(s):
    """Math text: letters (Latin, Greek, ż, ℓ) italic, digits, brackets and operators upright."""
    out, run = [], ""
    for ch in s + "\0":
        if ch.isalpha():
            run += ch
            continue
        if run:
            out.append(f'<tspan font-style="italic">{run}</tspan>')
            run = ""
        if ch != "\0":
            out.append(ch)
    return "".join(out)


def text(x, y, inner, size, color=C_TEXT, anchor="middle", italic=False, halo=False):
    style = ' font-style="italic"' if italic else ""
    out = (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}"{style} '
           f'fill="{color}" text-anchor="{anchor}" xml:space="preserve">{inner}</text>')
    if halo:   # a white outlined copy underneath (librsvg 2.50 ignores paint-order on text), so lines do not cut it
        under = out.replace(f'fill="{color}"', 'fill="#ffffff" stroke="#ffffff" stroke-width="6" '
                                               'stroke-linejoin="round"', 1)
        out = under + out
    return out


def head(tip, ux, uy, length, half, color):
    bx, by = tip[0] - length * ux, tip[1] - length * uy
    return (f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {bx - half * uy:.1f},{by + half * ux:.1f} '
            f'{bx + half * uy:.1f},{by - half * ux:.1f}" fill="{color}"/>')


def harrow(x1, x2, y, color=C_TEXT, dash=None, width=ARROW_W):
    """Horizontal arrow from x1 to x2 (tip exactly at x2)."""
    ux = 1.0 if x2 > x1 else -1.0
    bx = x2 - HEAD_LEN * ux
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{bx + ux:.1f}" y2="{y:.1f}" stroke="{color}" '
            f'stroke-width="{width}"{d}/>' + head((x2, y), ux, 0.0, HEAD_LEN, HEAD_HALF, color))


def hline(x1, x2, y, color=C_TEXT, width=ARROW_W):
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"/>'


def polyline(P, color, width, opacity=None, extra=""):
    d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in P)
    op = f' stroke-opacity="{opacity}"' if opacity is not None else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round"{op}{extra}/>')


def panel(x, title):
    return (f'<rect x="{x + 1.5}" y="{PANEL_Y + 2.5}" width="{PANEL_W}" height="{PANEL_H}" rx="8" '
            f'fill="#000000" fill-opacity="0.07"/>'
            f'<rect x="{x}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="8" fill="#ffffff" '
            f'stroke="{C_STROKE}" stroke-width="{FRAME_W}"/>'
            + text(x + PANEL_W / 2, TITLE_Y, title, FS_LABEL))


def flow_line(ang):
    """Gray flow line from radius FLOW_R_OUT into z* (quadratic, twisted by FLOW_BEND degrees), arrowhead at the tip."""
    a, am = np.radians(ang), np.radians(ang + FLOW_BEND)
    rm = (FLOW_R_OUT + FLOW_R_IN) / 2
    p0 = Z_STAR + FLOW_R_OUT * np.array([np.cos(a), np.sin(a)])
    tip = Z_STAR + FLOW_R_IN * np.array([np.cos(a), np.sin(a)])
    p1 = Z_STAR + rm * np.array([np.cos(am), np.sin(am)])
    u = (tip - p1) / np.linalg.norm(tip - p1)
    e = tip - (FLOW_HEAD_LEN - 1.5) * u
    return (f'<path d="M{p0[0]:.1f},{p0[1]:.1f} Q{p1[0]:.1f},{p1[1]:.1f} {e[0]:.1f},{e[1]:.1f}" fill="none" '
            f'stroke="{C_GRAY}" stroke-width="{FLOW_W}" stroke-linecap="round"/>'
            + head(tip, u[0], u[1], FLOW_HEAD_LEN, FLOW_HEAD_HALF, C_GRAY))


def dot(p, r, fill, stroke=None, sw=2.5):
    s = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{fill}"{s}/>'


# ---- glyphs (network variant) -------------------------------------------------------------------------------------
def mlp_glyph(cx, cy, layers=(2, 3, 2), dx=11.0, dy=10.0, r=2.4, color=C_TEXT):
    """Small fully connected net: columns of nodes, every edge drawn."""
    x0 = cx - dx * (len(layers) - 1) / 2
    cols = [[(x0 + i * dx, cy + (k - (n - 1) / 2) * dy) for k in range(n)] for i, n in enumerate(layers)]
    g = [f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{C_STROKE}" '
         f'stroke-width="0.9"/>' for c1, c2 in zip(cols, cols[1:]) for a in c1 for b in c2]
    g += [dot(p, r, color) for col in cols for p in col]
    return "".join(g)


def spline_fn(u, a=0.8):
    """A smooth monotone S on [0, 1]: v' = 1 - a cos(2 pi u) > 0 for a < 1."""
    return u - a * np.sin(2 * np.pi * u) / (2 * np.pi)


SPLINE_KNOTS = np.stack([np.linspace(0, 1, 5), spline_fn(np.linspace(0, 1, 5))], 1)


def spline_glyph(x, y, size, inverse=False, color=C_TEXT):
    """Monotone spline y = s(x) in a square frame, knots as dots; inverse=True draws s^{-1} (mirror in the diagonal)."""
    u = np.linspace(0, 1, 60)
    v = spline_fn(u)
    pts, kn = np.stack([u, v], 1), SPLINE_KNOTS.copy()
    if inverse:
        pts, kn = pts[:, ::-1], kn[:, ::-1]
    to_px = lambda q: np.stack([x + q[:, 0] * size, y + size - q[:, 1] * size], 1)
    g = [f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="none" stroke="{C_STROKE}" stroke-width="0.9"/>',
         polyline(to_px(pts), color, 1.8)]
    g += [dot(p, 2.0, color) for p in to_px(kn)]
    return "".join(g)


def flow_card(cx, cy, mirrored, scale=1.0):
    """Encoder card: net -> spline (left to right). Decoder card: the same, mirrored, with the inverse spline.
    Drawn about the origin and placed with one transform, so `scale` enlarges everything in it (strokes too)."""
    w, h, s = CARD_W, CARD_H, 32
    g = [f'<g transform="translate({cx:.1f},{cy:.1f}) scale({scale})">',
         f'<rect x="{-w / 2}" y="{-h / 2}" width="{w}" height="{h}" rx="6" fill="#ffffff" '
         f'stroke="{C_STROKE}" stroke-width="{FRAME_W / scale:.2f}"/>']   # FRAME_W as printed: the group is scaled
    sg = -1 if mirrored else 1
    g.append(mlp_glyph(-26 * sg, 0))
    g.append(f'<line x1="{-11 * sg}" y1="0" x2="{-4 * sg}" y2="0" stroke="{C_STROKE}" stroke-width="1.2"/>'
             + head((-1 * sg, 0), sg, 0, 5, 2.6, C_STROKE))
    g.append(spline_glyph(2 if not mirrored else -2 - s, -s / 2, s, inverse=mirrored))
    g.append('</g>')
    return "".join(g), w * scale


def gap_shift(scale):
    """How far the latent panel moves right so that a card of this scale keeps CARD_STUB px of arrow on each side."""
    return max(0.0, CARD_W * scale + 2 * CARD_STUB - (GAP_X[1] - GAP_X[0]))


# ---- the figure -------------------------------------------------------------------------------------------------
def build(scale=NETWORK_SCALE):
    """(SVG body, canvas width) of the network variant. `scale` sizes the network cards; the latent panel and
    everything in it move right by gap_shift(scale)."""
    shift = gap_shift(scale)
    g = [f'<rect width="{W + shift}" height="{H}" fill="#ffffff"/>']
    g.append(panel(DATA_X, "data space"))
    g.append(f'<g transform="translate({shift:.1f},0)">' + panel(LAT_X, "latent space") + '</g>')
    clip_d, clip_l = "network-data", "network-lat"
    g.append(f'<defs><clipPath id="{clip_d}"><rect x="{DATA_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" '
             f'rx="8"/></clipPath>')
    lo = M + (np.array([DATA_X, PANEL_Y]) - C) / K           # the latent box that maps onto the data panel
    hi = M + (np.array([DATA_X + PANEL_W, PANEL_Y + PANEL_H]) - C) / K
    g.append(f'<clipPath id="{clip_l}"><rect x="{lo[0]:.1f}" y="{lo[1]:.1f}" width="{hi[0] - lo[0]:.1f}" '
             f'height="{hi[1] - lo[1]:.1f}"/></clipPath></defs>')

    # -- data space: x*(t)
    g.append('<g id="data-trajectory">')
    g.append(polyline(X_CURVE, C_BLUE, CURVE_W))
    g.append(dot(X0, 5, "#ffffff", C_BLUE))
    g.append(dot(X_STAR, 6, C_BLUE))
    g.append(text(32, 101, m('x') + sub("0"), FS_LABEL, C_BLUE, "start"))
    g.append(text(43, 143, m('x*(t)'), FS_LABEL, C_BLUE, "start"))
    # the goal in data space: x_eq with z* = phi(x_eq), as in the paper's text. The star stays on the solution x*(t)
    g.append(text(185, 248, m('x') + sub("eq"), FS_LABEL, C_BLUE, "start"))
    g.append('</g>')

    # -- latent space: flow lines, z*, the equation
    g.append(f'<g id="latent" transform="translate({shift:.1f},0)">')
    g += [flow_line(a) for a in FLOW_ANGLES]
    g.append(dot(Z_STAR, 6, C_BLUE))
    # z* sits in the gap between the flow lines at -55.7 and 15.7 degrees: at (516, 141) the lower one ran through it
    g.append(text(518, 127, m('z*'), FS_LABEL, C_BLUE, "start"))
    a_sym = f'<tspan fill="{C_TEXT}">' + m('A') + '</tspan>'
    g.append(text(LAT_X + PANEL_W / 2, 240, m('ż') + ' = ' + a_sym + m('(z)(z − z*)'), FS_LABEL))
    g.append('</g>')

    # -- encoder / decoder
    phi = f'<tspan font-size="{FS_LABEL}" fill="{C_TEXT}">' + m('φ')
    enc_lab = f'<tspan font-size="{FS_SECOND}">encoder </tspan>' + phi + '</tspan>'
    dec_lab = (f'<tspan font-size="{FS_SECOND}">decoder </tspan>' + phi
               + sup("−1") + '</tspan>')
    gx0, gx1 = GAP_X[0], GAP_X[1] + shift
    mid = (gx0 + gx1) / 2
    g.append('<g id="maps">')
    half = CARD_H * scale / 2
    card, cw = flow_card(mid, ENC_Y, mirrored=False, scale=scale)
    g += [hline(gx0, mid - cw / 2, ENC_Y), card, harrow(mid + cw / 2, gx1, ENC_Y)]
    dy = DEC_Y - 8
    card, cw = flow_card(mid, dy, mirrored=True, scale=scale)
    g += [hline(gx1, mid + cw / 2, dy), card, harrow(mid - cw / 2, gx0, dy)]
    g.append(text(mid, ENC_Y - half - 9, enc_lab, FS_SECOND))
    g.append(text(mid, dy + half + 26, dec_lab, FS_SECOND))
    g.append('</g>')
    return "\n".join(g), W + shift


def svg(body, w, h):
    """width / height in inches at PT_PER_PX (Illustrator and LaTeX place it at print size)."""
    size = f'width="{w * PT_PER_PX / 72:.4f}in" height="{h * PT_PER_PX / 72:.4f}in"'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" {size} viewBox="0 0 {w} {h}">\n'
            f'{body}\n</svg>\n')


def write(name, content, w, zooms=((2, ""), (1, "_1x")), pdf=False):
    """SVG, PNGs at zoom x the px width w, and with pdf=True <name>.pdf (vector, print size, fonts embedded)
    plus <name>_outlined.pdf (text as paths, opens in Illustrator without the fonts installed)."""
    base = name if os.path.isabs(name) else os.path.join(HERE, name)
    path = base + ".svg"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    for z, suffix in zooms:
        subprocess.run(["rsvg-convert", "-w", str(round(z * w)), "-o", f"{base}{suffix}.png", path],
                       check=True)
    if pdf:   # rsvg-convert maps inches to px at --dpi and cairo writes px as pt: 72 dpi keeps the size exact
        out = base
        subprocess.run(["rsvg-convert", "-f", "pdf", "--dpi-x", "72", "--dpi-y", "72", "-o", out + ".pdf", path],
                       check=True)
        subprocess.run(["gs", "-q", "-dNoOutputFonts", "-sDEVICE=pdfwrite", "-o", out + "_outlined.pdf", out + ".pdf"],
                       check=True)
    print("wrote", path, "+ .pdf / _outlined.pdf" if pdf else "")


def apply_overrides(argv):
    """NAME=VALUE arguments set the module's parameters (as in make_schematic_C.py)."""
    g = globals()
    for a in argv:
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


def main(argv):
    global PT_PER_PX
    apply_overrides([a for a in argv if "=" in a])
    if OUT_STEM:                          # the paper's panel: the network variant alone, at print size
        body, w = build(scale=NETWORK_SCALE)
        if PRINT_W_IN > 0:
            PT_PER_PX = PRINT_W_IN * 72 / w
        write(OUT_STEM, svg(body, w, H), w, pdf=True)
        print(f"print {w * PT_PER_PX / 72:.2f} x {H * PT_PER_PX / 72:.2f} in; text {FS_LABEL * PT_PER_PX:.1f} / "
              f"{FS_SECOND * PT_PER_PX:.1f} / {FS_SCRIPT * PT_PER_PX:.1f} pt")
        return
    body, w = build()
    stem = os.path.join(REPO, "results", "elcd", "check", "figs", "elcd_nn_spaces_network")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    write(stem, svg(body, w, H), w)


if __name__ == "__main__":
    # the fitted decoder must send the latent line's ends to the two end dots (x_0 ~ (70.4, 93.6), x* ~ (174.9, 240))
    assert np.allclose(X0, [70.6, 93.7], atol=1.5) and np.allclose(X_STAR, [174.8, 239.3], atol=1.5), (X0, X_STAR)
    main(sys.argv[1:])
