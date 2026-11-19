"""The oblique projection, the SVG path helpers, the font and the colours that make_schematic_C.py and arm3d.py
share for panel A of the ELCD figure. make_schematic_C.py sets X0, W, H and SHEAR (and moves X0 while it draws the
desk); P, and arm3d through it, read them at call time, so the desk, the cards and the robot share one projection.
"""

# panel ("table") geometry, oblique projection of the unit square (u, v).
X0, W, SHEAR, H = 250, 400, 100, 140   # near-left x, width, shear of far edge, height

# colours
C_TRUE, C_GUESS = "#2F6DB5", "#F08A24"
C_STREAM, C_DEMO, C_TXT = "#B9B9B9", "#1f1f1f", "#222222"
C_ROBOT = "#4d4d4d"
FONT = "'DejaVu Sans', sans-serif"


def P(u, v, y0):
    """plane (u,v) -> screen (x,y) for a panel whose far edge is at y0."""
    return X0 + u * W + (1.0 - v) * SHEAR, y0 + v * H


def fmt(x):
    return f"{x:.1f}"


def path_d(pts_screen):
    return "M " + " L ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts_screen)
