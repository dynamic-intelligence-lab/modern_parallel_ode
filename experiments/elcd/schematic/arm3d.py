#!/usr/bin/env python3
"""
arm3d: the 3-DoF robot of the schematic (base yaw, shoulder pitch, elbow pitch; the
wrist pitch is slaved so the tool hangs vertically), modelled in a physical 3-D frame
and drawn in the SAME oblique projection as the desk of make_schematic / make_schematic_C.

Physical frame (all lengths in screen px of the desk's width axis):
    x : along the desk width,  x = u * W            (u = 0 is the left edge)
    d : into the desk depth,   d = v * D            (v = 0 far edge; d grows toward the viewer)
    z : height above the desk plane (z = 0 is the desk top)
Projection:  proj(x, d, z, y0) = P(x / W, d / D, y0) - (0, z)
i.e. the desk's own make_schematic.P for z = 0, so desk, cards and robot share ONE projection.
X0 / W / H / SHEAR are read from make_schematic AT CALL TIME, so make_schematic_C's geometry
override block is honoured.  unproj_desk inverts the projection for points on the desk plane,
so a pen tip lands exactly on its screen node (asserted in arm_pose_3d).

Kinematics (ik3): yaw = atan2 in the (x, d) plane from the base to the target; then planar
2-link IK in the vertical (radial, z) plane, elbow-up branch; wrist pitch = -(shoulder + elbow)
so the tool is vertical.  Nothing is clamped: ik3 / reach_check raise ValueError when the
shoulder-wrist distance leaves (|L1 - L2| + REACH_MARGIN, L1 + L2 - REACH_MARGIN).

Look (Franka-like, reduced): white links (light gradient across the link + dark outline),
dark pitch-joint "pills" drawn along the projected joint axis (they tilt with the base yaw),
a small hand (wrist cylinder, flange, hand block, two fingers) holding a pen whose tip is
exactly on the node, and ONE solid base: a floor pedestal behind the desk whose top is flush
with the desk plane (its front face is cut by the desk slab, so draw it BEFORE the slab),
a yaw turret, the column and the shoulder housing.

Use (make_schematic_C.py):
    base_svg, S = robot_base_3d(y0)          # draw BEFORE the desk slab
    arm_pose_3d(S, tx, ty, opacity, label)   # one exposure; pen tip at screen (tx, ty)
    arm_pose_3d(..., style="ghost")          # outline-only exposure (the ghosts of panel A); default "solid" = old look
    shoulder_cap_3d(S)                       # draw AFTER the exposures: the one physical shoulder stays crisp
    arm_top_y(S, targets)                    # highest screen point of the arm (title-clearance check)
"""
import math
from typing import NamedTuple

import numpy as np

import make_schematic as A
from make_schematic import P, path_d, FONT, C_TXT, C_ROBOT

# ----------------------------------------------------------------------------
# Parameters (physical px unless noted; see the docstring for the frame)
# ----------------------------------------------------------------------------
# Depth of the desk.  The oblique projection draws the full depth as the screen vector
# (-SHEAR, +H) ~ 148 px long, so D sets the depth scale: D = 296 would be a textbook cabinet
# projection (0.5), D = 148 cavalier (1.0).  D = 240 (~0.62) keeps the near-left node t_0
# reachable without pushing the elbows above the panel title.
D = 240.0

# base: position along the far edge (fraction of W) and distance BEHIND the far edge.
# u = 0.44 is midway between t_0 and t_3 in x: the t_2 / t_3 forearms then pass either side of
# the column instead of in front of it (column clearance 26 px, elbows 118 px
# apart, reach 0.55..0.92, no upper arm leaning back past vertical).  A base further left
# (u ~ 0.30) composes more dynamically but the t_2 forearm washes over the solid column.
BASE_U = 0.44
BASE_BACK = 36.0                       # d of the yaw axis is -BASE_BACK
PED_W, PED_D, PED_H = 84.0, 52.0, 46.0 # floor pedestal: width, depth, height BELOW the desk plane
                                       # (front face at d = -BASE_BACK + PED_D/2 = -10: just behind the far edge)
TURRET_R, TURRET_H = 19.0, 12.0        # yaw turret (cylinder on the pedestal top)
COL_R = 12.5                           # column radius
COLUMN = 44.0                          # desk plane -> shoulder pitch axis (squat: every px here is a px of
                                       # headroom above the far edge; the arm needs ~192 px in total)
L1, L2 = 130.0, 180.0                  # upper arm, forearm (Franka-like ratio 0.72; sum 310 vs max wrist
                                       # distance 286 at t_0)
REACH_MARGIN = 6.0                     # px kept away from the singular configurations

# tool (vertical, wrist centre -> pen tip): wrist cylinder, flange ring, hand block, fingers, pen
WRIST_LINK, FLANGE_H, HAND_H, FINGER_H, PEN_EXTRA = 22.0, 5.0, 11.0, 8.0, 14.0
HAND_W, FINGER_W, FINGER_GAP = 22.0, 4.0, 5.0
R_WLINK = 6.5                          # wrist cylinder radius (screen px)

# drawing (screen px)
W_UPPER = (24.0, 18.0)                 # upper arm width at shoulder -> elbow (tapered capsule)
W_FORE = (16.0, 11.0)                  # forearm width at elbow -> wrist
R_SHOULDER, R_ELBOW, R_WRIST = 11.0, 9.0, 6.5   # joint radii (pill radius; shoulder = solid cap)
JOINT_LEN = 2.3                        # pill length along its axis, as a multiple of the radius
HALO = 3.0                             # white halo under each exposure, px per side
OUTLINE_W = 1.2                        # link outline
PEN_W = 2.6
LIGHT_DIR = np.array([-0.5, -0.85])    # screen direction the light comes from (upper-left)
# ghost style (arm_pose_3d(..., style="ghost")): outline-only exposure, no halo, no gradient, no
# seams; a faint flat fill so overlapping ghost links still occlude each other a little
GHOST_FILL, GHOST_FILL_OPACITY = "#ffffff", 0.55
GHOST_OUTLINE_W = 1.8                  # link outline in the ghost style (solid style: OUTLINE_W)

# colours (C_ROBOT is the schematic's robot grey; outlines use it so the legend matches)
C_OUT = C_ROBOT
C_LINK = ("#ffffff", "#ededed", "#bfbfbf")   # white body: lit, mid, shaded
C_DARK = ("#7d7d7d", "#474747", "#262626")   # dark joint pills / flange / fingers
C_DARK_OUT = "#1e1e1e"
C_HAND, C_SEAM, C_PEN = "#f4f4f4", "#9a9a9a", "#262626"
C_PED = ("#e4e4e4", "#c3c3c3", "#a2a2a2")    # pedestal top, front, right side
C_PED_OUT = "#6f6f6f"
C_COL = ("#f6f6f6", "#d8d8d8", "#9c9c9c")    # column / turret cylinder shading

_gid = [0]


class Shoulder3D(NamedTuple):
    """shoulder pivot in physical coordinates + the panel's y0 (needed to unproject targets)."""
    x: float
    d: float
    z: float
    y0: float


def tool_len():
    """wrist height above the desk plane (the whole tool stack is vertical)."""
    return WRIST_LINK + FLANGE_H + HAND_H + FINGER_H + PEN_EXTRA


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------
def proj(x, d, z, y0):
    """physical (x, d, z) -> screen (sx, sy); identical to make_schematic.P for z = 0."""
    sx, sy = P(x / A.W, d / D, y0)
    return np.array([sx, sy - z])


def unproj_desk(sx, sy, y0):
    """screen point ON the desk plane -> physical (x, d) with z = 0 (exact inverse of proj)."""
    v = (sy - y0) / A.H
    return sx - A.X0 - (1.0 - v) * A.SHEAR, v * D


def base_xyz():
    return BASE_U * A.W, -BASE_BACK, 0.0


# ----------------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------------
def _reach_bounds():
    return abs(L1 - L2) + REACH_MARGIN, L1 + L2 - REACH_MARGIN


def reach_check(S: Shoulder3D, tx, ty):
    """shoulder -> wrist distance for a pen tip at screen (tx, ty); raises if unreachable."""
    xt, dt = unproj_desk(tx, ty, S.y0)
    dist = math.hypot(math.hypot(xt - S.x, dt - S.d), tool_len() - S.z)
    lo, hi = _reach_bounds()
    if not (lo < dist < hi):
        raise ValueError(f"target ({tx:.1f},{ty:.1f}) at wrist distance {dist:.1f} px not in ({lo:.1f}, {hi:.1f}); "
                         f"move BASE_U/BASE_BACK, change L1/L2 or COLUMN")
    return dist


def ik3(S: Shoulder3D, tx, ty):
    """base yaw + planar 2-link IK (elbow up) for a pen tip at screen (tx, ty) on the desk.

    Returns a dict: pivot, elbow, wrist, tip (physical (x, d, z)), l_hat (unit pitch axis,
    horizontal, normal to the arm plane), yaw, q = (shoulder, elbow, wrist) in radians, dist."""
    xt, dt = unproj_desk(tx, ty, S.y0)
    yaw = math.atan2(dt - S.d, xt - S.x)
    cy, sy_ = math.cos(yaw), math.sin(yaw)
    pivot = np.array([S.x, S.d, S.z])
    r = math.hypot(xt - S.x, dt - S.d)
    dz = tool_len() - S.z
    dist = math.hypot(r, dz)
    lo, hi = _reach_bounds()
    if not (lo < dist < hi):
        raise ValueError(f"unreachable: wrist distance {dist:.1f} not in ({lo:.1f}, {hi:.1f}) "
                         f"for target screen ({tx:.1f}, {ty:.1f}) = phys ({xt:.1f}, {dt:.1f})")
    # planar (radial, z): shoulder at (0, 0), wrist at (r, dz)
    a = (L1 ** 2 + dist ** 2 - L2 ** 2) / (2.0 * dist)
    h = math.sqrt(max(L1 ** 2 - a ** 2, 0.0))
    ur, uz = r / dist, dz / dist
    nr, nz = -uz, ur                                  # perpendicular, "up" when u points outward
    er, ez = a * ur + h * nr, a * uz + h * nz         # elbow (planar), elbow-up branch
    elbow = pivot + np.array([er * cy, er * sy_, ez])
    wrist = np.array([xt, dt, tool_len()])
    tip = np.array([xt, dt, 0.0])
    assert np.allclose(np.linalg.norm(elbow - pivot), L1, atol=1e-6)
    assert np.allclose(np.linalg.norm(wrist - elbow), L2, atol=1e-6), "IK closure failed"
    q_sh = math.atan2(ez, er)                                  # shoulder pitch from horizontal
    q_el = math.atan2(dz - ez, r - er) - q_sh                  # elbow pitch (relative)
    q_wr = -(q_sh + q_el)                                      # wrist keeps the tool vertical
    l_hat = np.array([-sy_, cy, 0.0])                          # pitch axis (horizontal, normal to the arm plane)
    return dict(pivot=pivot, elbow=elbow, wrist=wrist, tip=tip, yaw=yaw, l_hat=l_hat,
                q=(q_sh, q_el, q_wr), dist=dist)


# ----------------------------------------------------------------------------
# SVG primitives
# ----------------------------------------------------------------------------
def fmt(x):
    return f"{x:.1f}"


def _uid(prefix):
    _gid[0] += 1
    return f"{prefix}{_gid[0]}"


def _widths(width):
    return (float(width), float(width)) if np.isscalar(width) else (float(width[0]), float(width[1]))


def _grad_across(p, q, width, cols):
    """linear gradient across a screen segment p->q, lit edge toward LIGHT_DIR. Returns (id, def)."""
    d = q - p
    n = np.array([-d[1], d[0]])
    n = n / (np.linalg.norm(n) + 1e-9)
    if np.dot(n, LIGHT_DIR) < 0:
        n = -n
    m = 0.5 * (p + q)
    a, b = m + n * width * 0.5, m - n * width * 0.5
    gid = _uid("lk")
    g = (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
         f'x1="{fmt(a[0])}" y1="{fmt(a[1])}" x2="{fmt(b[0])}" y2="{fmt(b[1])}">'
         f'<stop offset="0" stop-color="{cols[0]}"/><stop offset="0.45" stop-color="{cols[1]}"/>'
         f'<stop offset="1" stop-color="{cols[2]}"/></linearGradient>')
    return gid, g


def capsule_path(p, q, width, shorten_p=0.0, shorten_q=0.0, n_arc=14):
    """outline of a (possibly tapered) capsule from p to q; width is a scalar or (w_p, w_q)."""
    w1, w2 = _widths(width)
    d = q - p
    L = np.linalg.norm(d) + 1e-9
    u = d / L
    p2, q2 = p + u * shorten_p, q - u * shorten_q
    ang = math.atan2(u[1], u[0])
    th_q = np.linspace(ang - math.pi / 2, ang + math.pi / 2, n_arc)
    th_p = np.linspace(ang + math.pi / 2, ang + 3 * math.pi / 2, n_arc)
    cap_q = [q2 + 0.5 * w2 * np.array([math.cos(t), math.sin(t)]) for t in th_q]
    cap_p = [p2 + 0.5 * w1 * np.array([math.cos(t), math.sin(t)]) for t in th_p]
    return path_d(cap_q + cap_p) + " Z"


def link(p, q, width, defs, cols=C_LINK, outline=C_OUT, sw=OUTLINE_W, shorten_p=0.0, shorten_q=0.0, ghost=False):
    """a link (or any cylinder seen from the side) as a tapered capsule with an across-link gradient
    (ghost=True: flat faint fill, thicker outline, no gradient def)."""
    if ghost:
        return (f'<path d="{capsule_path(p, q, width, shorten_p, shorten_q)}" fill="{GHOST_FILL}" '
                f'fill-opacity="{GHOST_FILL_OPACITY}" stroke="{outline}" stroke-width="{GHOST_OUTLINE_W:.1f}" stroke-linejoin="round"/>')
    w1, w2 = _widths(width)
    gid, g = _grad_across(p, q, max(w1, w2), cols)
    defs.append(g)
    return (f'<path d="{capsule_path(p, q, width, shorten_p, shorten_q)}" fill="url(#{gid})" '
            f'stroke="{outline}" stroke-width="{sw:.1f}" stroke-linejoin="round"/>')


def halo(p, q, width, shorten_p=0.0, shorten_q=0.0):
    """white halo under a capsule: the same shape, HALO px wider on each side."""
    w1, w2 = _widths(width)
    return f'<path d="{capsule_path(p, q, (w1 + 2 * HALO, w2 + 2 * HALO), shorten_p, shorten_q)}" fill="#ffffff"/>'


def seam(p, q, width, s, color=C_SEAM):
    """thin panel seam across a link at fraction s (Franka body panel line)."""
    w1, w2 = _widths(width)
    d = q - p
    u = d / (np.linalg.norm(d) + 1e-9)
    n = np.array([-u[1], u[0]])
    c = p + d * s
    r = 0.5 * (w1 + (w2 - w1) * s) - 0.8
    a, b = c + n * r, c - n * r
    return f'<line x1="{fmt(a[0])}" y1="{fmt(a[1])}" x2="{fmt(b[0])}" y2="{fmt(b[1])}" stroke="{color}" stroke-width="0.9"/>'


def pill_ends(c3, l_hat, r, y0):
    """screen end points of a joint pill centred at physical c3 along the unit pitch axis l_hat."""
    half = 0.5 * JOINT_LEN * r
    return proj(*(c3 - half * l_hat), y0), proj(*(c3 + half * l_hat), y0)


def pill(c3, l_hat, r, y0, defs, ghost=False):
    """dark pitch-joint cylinder: a capsule along the PROJECTED joint axis, so it tilts and
    foreshortens with the base yaw (the 3-D cue of the hinge).  ghost: outline only."""
    a, b = pill_ends(c3, l_hat, r, y0)
    return link(a, b, 2 * r, defs, cols=C_DARK, outline=C_DARK_OUT, sw=1.0, ghost=ghost)


def rect(x, y, w, h, fill, rx=2.5, outline=C_OUT, sw=1.0, ghost=False):
    if ghost:
        fill, sw = f'{GHOST_FILL}" fill-opacity="{GHOST_FILL_OPACITY}', GHOST_OUTLINE_W
    return (f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" rx="{rx}" '
            f'fill="{fill}" stroke="{outline}" stroke-width="{sw}"/>')


def rect_halo(x, y, w, h, rx=2.5):
    return (f'<rect x="{fmt(x - HALO)}" y="{fmt(y - HALO)}" width="{fmt(w + 2 * HALO)}" height="{fmt(h + 2 * HALO)}" '
            f'rx="{rx + HALO}" fill="#ffffff"/>')


def _ellipse_pts(cx, cd, cz, R, y0, n=56):
    """a horizontal circle of radius R at (cx, cd, cz), projected (a tilted ellipse in the desk's shear)."""
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.array([proj(cx + R * math.cos(t), cd + R * math.sin(t), cz, y0) for t in th])


def _front_arc(pts):
    """indices of the arc of a projected horizontal circle that faces the viewer: between the two
    silhouette points (extreme screen x), the half that contains the lowest point (max screen y)."""
    n = len(pts)
    iL, iR, iF = int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0])), int(np.argmax(pts[:, 1]))
    arc1 = [(iL + k) % n for k in range((iR - iL) % n + 1)]
    arc2 = [(iR + k) % n for k in range((iL - iR) % n + 1)]
    return arc1 if iF in arc1 else arc2[::-1]


def cylinder(cx, cd, z0, z1, R, y0, defs, cols=C_COL, top_fill=None, stroke=C_OUT, sw=1.1, top=True):
    """vertical cylinder in the oblique projection: shaded front half + top ellipse, both built
    from projected horizontal circles so the rims carry the desk's shear."""
    bot, tp = _ellipse_pts(cx, cd, z0, R, y0), _ellipse_pts(cx, cd, z1, R, y0)
    arc = _front_arc(tp)
    side = [tp[i] for i in arc] + [bot[i] for i in arc[::-1]]
    gid = _uid("cyl")
    xl, xr = float(np.min(tp[:, 0])), float(np.max(tp[:, 0]))
    defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{fmt(xl)}" y1="0" x2="{fmt(xr)}" y2="0">'
                f'<stop offset="0" stop-color="{cols[0]}"/><stop offset="0.4" stop-color="{cols[1]}"/>'
                f'<stop offset="1" stop-color="{cols[2]}"/></linearGradient>')
    out = [f'<path d="{path_d(side)} Z" fill="url(#{gid})" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>']
    if top:
        out.append(f'<path d="{path_d(tp)} Z" fill="{top_fill or cols[0]}" stroke="{stroke}" stroke-width="{sw}"/>')
    return "\n".join(out)


# ----------------------------------------------------------------------------
# The one physical robot: pedestal, turret, column, shoulder
# ----------------------------------------------------------------------------
def pedestal_3d(y0):
    """floor stand behind the desk, top flush with the desk plane (z = 0); its front and right
    faces run below desk level, so the desk slab drawn afterwards cuts them."""
    bx, bd, _ = base_xyz()
    x0, x1 = bx - PED_W / 2, bx + PED_W / 2
    d0, d1 = bd - PED_D / 2, bd + PED_D / 2         # d1 = front (nearer) face
    top = [proj(x0, d0, 0, y0), proj(x1, d0, 0, y0), proj(x1, d1, 0, y0), proj(x0, d1, 0, y0)]
    front = [proj(x0, d1, 0, y0), proj(x1, d1, 0, y0), proj(x1, d1, -PED_H, y0), proj(x0, d1, -PED_H, y0)]
    side = [proj(x1, d0, 0, y0), proj(x1, d1, 0, y0), proj(x1, d1, -PED_H, y0), proj(x1, d0, -PED_H, y0)]
    g = [f'<g class="pedestal" stroke="{C_PED_OUT}" stroke-width="1.0" stroke-linejoin="round">',
         f'<path d="{path_d(side)} Z" fill="{C_PED[2]}"/>',
         f'<path d="{path_d(front)} Z" fill="{C_PED[1]}"/>',
         f'<path d="{path_d(top)} Z" fill="{C_PED[0]}"/>',
         '</g>']
    return "\n".join(g)


def robot_base_3d(y0):
    """pedestal + contact shadow + yaw turret + column + white shoulder housing, all solid.
    Draw BEFORE the desk slab.  Returns (svg, Shoulder3D)."""
    bx, bd, bz = base_xyz()
    S = Shoulder3D(bx, bd, bz + COLUMN, y0)
    defs, g = [], [pedestal_3d(y0)]
    sh = _ellipse_pts(bx + 2.0, bd + 2.5, 0.0, TURRET_R * 1.25, y0)       # small contact shadow
    g.append(f'<path d="{path_d(sh)} Z" fill="#000" opacity="0.09"/>')
    g.append(cylinder(bx, bd, 0.0, TURRET_H, TURRET_R, y0, defs, cols=("#ededed", "#c9c9c9", "#8e8e8e")))
    # dark yaw ring at the foot of the column (Franka-style band)
    ring = _ellipse_pts(bx, bd, TURRET_H, COL_R + 1.5, y0)
    arc = _front_arc(ring)
    g.append(f'<path d="{path_d([ring[i] for i in arc])}" fill="none" stroke="{C_DARK[1]}" stroke-width="3.2" stroke-linecap="round"/>')
    g.append(cylinder(bx, bd, TURRET_H, COLUMN - 4.0, COL_R, y0, defs, top=False))
    # shoulder housing: white rounded head on the column; the dark pitch bulb goes on top (shoulder_cap_3d)
    Ssc = proj(S.x, S.d, S.z, y0)
    g.append(link(Ssc + np.array([0.0, 7.0]), Ssc + np.array([0.0, -7.0]), 2 * (R_SHOULDER + 2.5), defs))
    svg = ['<g class="robot-base" stroke-linecap="round" stroke-linejoin="round">', '<defs>', *defs, '</defs>', *g, '</g>']
    return "\n".join(svg), S


def shoulder_cap_3d(S: Shoulder3D):
    """the solid shoulder pitch joint (dark bulb with a light rim): draw AFTER the exposures so the
    one physical shoulder is the crispest joint instead of being washed out by four upper arms."""
    c = proj(S.x, S.d, S.z, S.y0)
    gid = _uid("rg")
    return (f'<defs><radialGradient id="{gid}" cx="0.38" cy="0.35" r="0.72">'
            f'<stop offset="0" stop-color="{C_DARK[0]}"/><stop offset="1" stop-color="{C_DARK[2]}"/></radialGradient></defs>'
            f'<g class="shoulder-cap"><circle cx="{fmt(c[0])}" cy="{fmt(c[1])}" r="{R_SHOULDER:.1f}" fill="url(#{gid})" '
            f'stroke="{C_DARK_OUT}" stroke-width="1.2"/>'
            f'<circle cx="{fmt(c[0])}" cy="{fmt(c[1])}" r="{R_SHOULDER * 0.42:.1f}" fill="none" stroke="#ffffff" '
            f'stroke-opacity="0.55" stroke-width="1.2"/></g>')


# ----------------------------------------------------------------------------
# One exposure
# ----------------------------------------------------------------------------
def arm_pose_3d(S: Shoulder3D, tx, ty, opacity, label=None, label_dx=6.0, style="solid"):
    """one exposure of the arm (draw order far -> near: upper arm, elbow, forearm, wrist, tool);
    the pen tip lands EXACTLY at screen (tx, ty) on the desk plane.
    style: "solid" (white shaded body, halo, seams: the original look) or "ghost" (outline-only
    exposure with a faint flat fill, no halo, no seams; combine with opacity ~0.35-0.5)."""
    if style not in ("solid", "ghost"):
        raise ValueError(f"style must be 'solid' or 'ghost', got {style!r}")
    gh = style == "ghost"
    k = ik3(S, tx, ty)
    y0 = S.y0
    Psc, Esc, Wsc = (proj(*k[n], y0) for n in ("pivot", "elbow", "wrist"))
    Tsc = np.array([tx, ty])
    assert np.allclose(proj(*k["tip"], y0), Tsc, atol=1e-6), "pen tip must land on the target"
    assert abs(Wsc[0] - tx) < 1e-6, "the tool is vertical, so wrist and tip share screen x"
    x, y = Wsc
    y_fl = y + WRIST_LINK                    # flange top
    y_hd = y_fl + FLANGE_H                   # hand top
    y_fg = y_hd + HAND_H                     # finger top
    y_pen = y_fg + FINGER_H                  # finger tip = pen start
    ea, eb = pill_ends(k["elbow"], k["l_hat"], R_ELBOW, y0)
    wa, wb = pill_ends(k["wrist"], k["l_hat"], R_WRIST, y0)
    defs, hl, body = [], [], []
    if not gh:
        # --- white halo under everything (kept clear of the solid shoulder cap) ---
        hl.append(halo(Psc, Esc, W_UPPER, shorten_p=R_SHOULDER + 1.0))
        hl.append(halo(ea, eb, 2 * R_ELBOW))
        hl.append(halo(Esc, Wsc, W_FORE))
        hl.append(halo(wa, wb, 2 * R_WRIST))
        hl.append(halo(np.array([x, y]), np.array([x, y_fl]), 2 * R_WLINK))
        hl.append(rect_halo(x - HAND_W / 2, y_fl - 1, HAND_W, FLANGE_H + HAND_H + FINGER_H + 1))
        hl.append(halo(np.array([x, y_pen]), Tsc, PEN_W, shorten_q=2.5))
    # --- body, far -> near ---
    body.append(link(Psc, Esc, W_UPPER, defs, ghost=gh, shorten_p=(R_SHOULDER + 1.0) if gh else 0.0))
    if not gh:
        body.append(seam(Psc, Esc, W_UPPER, 0.18))
        body.append(seam(Psc, Esc, W_UPPER, 0.84))
    body.append(pill(k["elbow"], k["l_hat"], R_ELBOW, y0, defs, ghost=gh))
    body.append(link(Esc, Wsc, W_FORE, defs, ghost=gh))
    if not gh:
        body.append(seam(Esc, Wsc, W_FORE, 0.14))
        body.append(seam(Esc, Wsc, W_FORE, 0.86))
    body.append(pill(k["wrist"], k["l_hat"], R_WRIST, y0, defs, ghost=gh))
    # tool stack (vertical): wrist cylinder, dark flange, hand block, two dark fingers, pen
    body.append(link(np.array([x, y]), np.array([x, y_fl]), 2 * R_WLINK, defs, shorten_p=R_WRIST * 0.5, ghost=gh))
    body.append(rect(x - R_WLINK - 1.0, y_fl - 1.0, 2 * R_WLINK + 2.0, FLANGE_H + 1.0, C_DARK[1], rx=2.0, outline=C_DARK_OUT, ghost=gh))
    body.append(rect(x - HAND_W / 2, y_hd, HAND_W, HAND_H, C_HAND, rx=3.0, ghost=gh))
    if not gh:
        body.append(f'<line x1="{fmt(x - HAND_W / 2 + 3)}" y1="{fmt(y_hd + 3.5)}" x2="{fmt(x + HAND_W / 2 - 3)}" '
                    f'y2="{fmt(y_hd + 3.5)}" stroke="{C_SEAM}" stroke-width="0.9"/>')
    for dx in (-FINGER_GAP, FINGER_GAP):
        body.append(rect(x + dx - FINGER_W / 2, y_fg - 1.0, FINGER_W, FINGER_H + 1.0, C_DARK[1], rx=1.2, outline=C_DARK_OUT, ghost=gh))
    body.append(f'<line x1="{fmt(x)}" y1="{fmt(y_pen - 1.0)}" x2="{fmt(tx)}" y2="{fmt(ty)}" '
                f'stroke="{C_PEN}" stroke-width="{PEN_W}" stroke-linecap="round"/>')
    svg = [f'<g class="arm-pose" opacity="{opacity:.2f}" stroke-linecap="round" stroke-linejoin="round">',
           '<defs>', *defs, '</defs>', *hl, *body, '</g>']
    if label:                                 # beside the hand, with a white stroke halo under the text
        t = (f'x="{fmt(x + HAND_W / 2 + label_dx)}" y="{fmt(y_hd + HAND_H / 2 + 5)}" font-family="{FONT}" '
             f'font-size="14" font-style="italic"')
        svg.append(f'<text {t} fill="none" stroke="#ffffff" stroke-width="3.5" stroke-linejoin="round">{label}</text>')
        svg.append(f'<text {t} fill="{C_TXT}">{label}</text>')
    return "\n".join(svg)


# ----------------------------------------------------------------------------
# Layout helpers / diagnostics
# ----------------------------------------------------------------------------
def arm_top_y(S: Shoulder3D, targets):
    """highest (min) screen y of any drawn arm part over the given pen-tip targets, halo included."""
    ys = [proj(S.x, S.d, S.z, S.y0)[1] - R_SHOULDER - 2.5 - HALO]
    for tx, ty in targets:
        k = ik3(S, tx, ty)
        for n, r in (("elbow", R_ELBOW), ("wrist", R_WRIST)):
            for e in pill_ends(k[n], k["l_hat"], r, S.y0):
                ys.append(e[1] - r - HALO - 1.0)
        ys.append(proj(*k["elbow"], S.y0)[1] - 0.5 * max(W_UPPER) - HALO - 1.0)
    return float(min(ys))


def describe(S: Shoulder3D, targets, names=None):
    """one line per node: reach used and joint angles in degrees (console diagnostics).
    `names` labels the lines (default t0, t1, ... in the order given)."""
    names = names or [f"t{i}" for i in range(len(targets))]
    lines = []
    for name, (tx, ty) in zip(names, targets):
        k = ik3(S, tx, ty)
        q = np.degrees(k["q"])
        e = proj(*k["elbow"], S.y0)
        lines.append(f"{name}: tip phys=({k['tip'][0]:.0f},{k['tip'][1]:.0f})  wrist dist={k['dist']:.0f}/{L1 + L2:.0f} "
                     f"({k['dist'] / (L1 + L2):.2f})  yaw={math.degrees(k['yaw']):.0f}  q_sh={q[0]:.0f} q_el={q[1]:.0f} "
                     f"q_wr={q[2]:.0f}  elbow z={k['elbow'][2]:.0f} scr=({e[0]:.0f},{e[1]:.0f})")
    return "\n".join(lines)
