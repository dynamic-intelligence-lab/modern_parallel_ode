"""The forward-Euler step-size example of the appendix: figure fig:step-size and the numbers quoted with it.

x' = -20 x + 20 sin(2t), x(0) = 10, T = 5, discretized by forward Euler, g_k(s) = s + h f(s, t_{k-1}), and solved
by M in {50, 100, 200, 400} Jacobi sweeps started from the constant path x0. For every M and every step size
h = T/K the total error E(M, h) is the L1 norm on [0, T] of the piecewise-linear interpolant of the iterate minus
the exact solution. The step-size estimates h_asy, h_num and their midpoint h_mid are those of the appendix.

  python experiments/appendix/euler_step_size/figure_step_size.py [--out PDF]   # numpy and matplotlib; 30 s on a CPU

Prints every number the appendix quotes next to the recomputed value, then draws the figure (by default into
figs/euler_step_size/Euler_Error_Analysis_Plot_one_panel.pdf). The paper's PDF of this figure was drawn outside this
repository; this script recomputes its data and redraws it in the same layout.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
T, X0 = 5.0, 10.0
MS = (50, 100, 200, 400)
C = LX = 20.0                   # f is affine in x with slope -20: one-sided Lipschitz constant -c, Lipschitz l_x
LT, VBAR = 40.0, 220.0          # sup |df/dt| and sup |f| on X = [-10, 10] x [0, T]
D = LX * VBAR + LT
H_GRID = np.linspace(1e-4, 0.02, 500)   # the step sizes of the curves, each rounded to the nearest h = T/K


def f(x, t):
    return -20.0 * x + 20.0 * np.sin(2.0 * t)


def exact(t):
    """Solution of x' = f(x, t), x(0) = X0."""
    a, b = 100.0 / 101.0, -10.0 / 101.0
    return (X0 - b) * np.exp(-20.0 * t) + a * np.sin(2.0 * t) + b * np.cos(2.0 * t)


def jacobi(M, K):
    """Grid t_0..t_K and the iterate after M Jacobi sweeps p_k <- p_{k-1} + h f(p_{k-1}, t_{k-1}), p^(0) = x0."""
    h = T / K
    t = h * np.arange(K + 1)
    p = np.full(K + 1, X0)
    for _ in range(M):
        p[1:] = p[:-1] + h * f(p[:-1], t[:-1])      # the right-hand side is evaluated before the assignment
    return t, p


GX, GW = np.polynomial.legendre.leggauss(6)
TAU = np.concatenate([(j + (GX + 1) / 2) / 8 for j in range(8)])     # 8 panels x 6 Gauss nodes per step, in [0, 1]
W = np.tile(GW / 16, 8)                                               # their weights, summing to 1


def total_error(M, K):
    """E(M, T/K) = int_0^T |p_hat(t) - x(t)| dt, p_hat the piecewise-linear interpolant of the iterate."""
    t, p = jacobi(M, K)
    h = T / K
    tq = t[:-1, None] + h * TAU
    pq = p[:-1, None] * (1 - TAU) + p[1:, None] * TAU
    return float(h * np.sum(np.abs(pq - exact(tq)) * W))


def bisect(fn, a, b):
    """Root of fn in [a, b], fn(a) and fn(b) of opposite signs, to the last bit."""
    fa = fn(a)
    assert (fa > 0) != (fn(b) > 0), (a, b)
    while True:
        m = 0.5 * (a + b)
        if m in (a, b):
            return m
        if (fn(m) > 0) == (fa > 0):
            a = m
        else:
            b = m


def rho_fe(h):
    return np.exp(-C * h) + np.exp(LX * h) - 1.0 - LX * h


def drho_fe(h):
    return -C * np.exp(-C * h) + LX * np.exp(LX * h) - LX


H_STABLE = bisect(lambda h: rho_fe(h) - 1.0, 1e-6, 0.1)    # rho_FE(h) < 1 exactly for 0 < h < H_STABLE


def h_asy(M):
    return np.log(2 * C**2 * M * T * VBAR / D) / (C * M)


def h_num(M):
    """Minimizer of max(T - Mh, 0) rho_FE(h)^M + d/(2c vbar) h over I, the stable h < T/M: located on a grid of
    10^4 points, then the root of the derivative between the two neighbours of the smallest grid value."""
    g = lambda h: np.maximum(T - M * h, 0.0) * rho_fe(h) ** M + D / (2 * C * VBAR) * h
    dg = lambda h: (-M * rho_fe(h) ** M + (T - M * h) * M * rho_fe(h) ** (M - 1) * drho_fe(h)
                    + D / (2 * C * VBAR))
    hs = np.linspace(0.0, min(T / M, H_STABLE), 10001)[1:-1]
    i = int(np.argmin(g(hs)))
    assert 0 < i < len(hs) - 1
    return bisect(dg, hs[i - 1], hs[i + 1])


def h_stat(M):
    """Stationary point of psi(h) = T vbar (T - Mh) e^{-cMh} + Td/(2c) h on (0, T/M)."""
    return bisect(lambda h: M * VBAR * np.exp(-C * M * h) * (1 + C * (T - M * h)) - D / (2 * C), 1e-9, T / M)


def curves_and_marks(M):
    """E on the grid h = T/K, K = round(T/H_GRID); the smallest E over every integer K between the grid points two
    places either side of the grid minimum; the step-size estimates."""
    Ks = np.unique(np.round(T / H_GRID).astype(int))
    E = np.array([total_error(M, K) for K in Ks])
    i = int(np.argmin(E))
    Kr = np.arange(Ks[max(i - 2, 0)], Ks[min(i + 2, len(Ks) - 1)] + 1)
    Er = np.array([total_error(M, K) for K in Kr])
    j = int(np.argmin(Er))
    assert 0 < j < len(Kr) - 1, (M, Kr[j])
    ha, hn = h_asy(M), h_num(M)
    hm = (ha + hn) / 2
    K_mid = int(round(T / hm))
    return dict(K=Ks, h=T / Ks, E=E, K_min=int(Kr[j]), h_min=T / Kr[j], E_min=float(Er[j]), h_asy=ha, h_num=hn,
                h_stat=h_stat(M), h_mid=hm, K_mid=K_mid, E_mid=total_error(M, K_mid))


def report(res):
    """Prints every number the appendix quotes next to its recomputed value, and the marks of the figure."""
    r = [res[M] for M in MS]
    pct = lambda v: 100 * np.array(v)
    rows = [
        ("c = l_x (affine slope)", "20", [C, LX]),
        ("l_t = max |40 cos 2t|", "40", [np.max(np.abs(40 * np.cos(2 * np.linspace(0, T, 100001))))]),
        ("vbar = max |f| on [-10, 10] x [0, T]", "220",
         [np.max(np.abs(f(np.array([-10.0, 10.0])[:, None], np.linspace(0, T, 100001)[None])))]),
        ("d = l_x vbar + l_t", "4,440", [D]),
        ("largest h with 10|1 - 20h| + 20h <= 10", "1/11",
         [bisect(lambda h: 10 * abs(1 - 20 * h) + 20 * h - 10, 0.06, 0.2)]),
        ("rho_FE(h) < 1 for h below", "(not quoted)", [H_STABLE]),
        ("2 c^2 M T vbar > d", "holds", [2 * C**2 * M * T * VBAR > D for M in MS]),
        ("rho_FE(h_asy) < 1", "holds", [rho_fe(x["h_asy"]) < 1 for x in r]),
        ("empirical minimizer K (grid K in [250, 50000])", "interior minimum", [x["K_min"] for x in r]),
        ("h_mid / h_min - 1, %", "2.6 to 14.2, shrinking with M", pct([x["h_mid"] / x["h_min"] - 1 for x in r])),
        ("E(T/round(T/h_mid)) / E_min - 1, %", "at most 9", pct([x["E_mid"] / x["E_min"] - 1 for x in r])),
        ("h_asy / h_min - 1, %", "at or above, by at most 20", pct([x["h_asy"] / x["h_min"] - 1 for x in r])),
        ("h_num / h_min - 1, %", "at or above, by at most 20", pct([x["h_num"] / x["h_min"] - 1 for x in r])),
        ("h_num / h_asy - 1, %", "at most 10", pct([x["h_num"] / x["h_asy"] - 1 for x in r])),
        ("c M h_asy", "about 9 to 11", [C * M * x["h_asy"] for M, x in zip(MS, r)]),
        ("h_asy / h_stat - 1, %", "within 1", pct([x["h_asy"] / x["h_stat"] - 1 for x in r])),
        ("M h_asy / T", "about 0.1", [M * x["h_asy"] / T for M, x in zip(MS, r)]),
        ("beta M h_asy^2", "0.85 (M = 50) to 0.16 (M = 400)", [LX**2 * M * x["h_asy"] ** 2 / 2 for M, x in zip(MS, r)]),
        ("rho_FE(h_asy)^M / exp(-c M h_asy)", "2.9, 1.7, 1.35, 1.18",
         [rho_fe(x["h_asy"]) ** M / np.exp(-C * M * x["h_asy"]) for M, x in zip(MS, r)]),
        ("beta h_asy / c", "0.09, 0.05, 0.03, 0.01", [LX**2 * x["h_asy"] / (2 * C) for x in r]),
        ("h_num / h_stat - 1, %", "11, 5, 2, 1", pct([x["h_num"] / x["h_stat"] - 1 for x in r])),
    ]
    print(f"{'quantity':48s} {'paper':32s} recomputed (M = {', '.join(map(str, MS))} where four values)")
    for name, quoted, vals in rows:
        print(f"{name:48s} {quoted:32s} " + "  ".join(f"{v:.6g}" if not isinstance(v, (bool, np.bool_)) else str(v)
                                                       for v in vals))
    print("\nmarks of the figure:")
    for M in MS:
        x = res[M]
        print(f"  M = {M:3d}: star at h = T/{x['K_min']} = {x['h_min']:.9g}, E = {x['E_min']:.9g}; "
              f"dashed line at h_mid = {x['h_mid']:.9g} (h_asy = {x['h_asy']:.9g}, h_num = {x['h_num']:.9g})")


def draw(res, out):
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "STIXGeneral", "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(MS)))
    for M, col in zip(MS, colors):
        ax.plot(res[M]["h"], res[M]["E"], color=col, lw=2.2, label=f"Error ($M={M}$)")
        ax.axvline(res[M]["h_mid"], color=col, ls="--", lw=1.8, alpha=0.85, label=f"Crit $h$ ($M={M}$)")
    for M, col in zip(MS, colors):
        ax.scatter(res[M]["h_min"], res[M]["E_min"], s=196, marker="*", facecolors=[col], edgecolors="k",
                   linewidths=0.7, zorder=3)
    ax.scatter([], [], s=121, marker="*", facecolors="white", edgecolors="k", linewidths=0.7,
               label="Minimum of $E$")
    ax.set_yscale("log")
    ax.set_xlim(0, 0.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.0025))
    ax.set_xlabel("Time Step ($h$)", fontsize=13)
    ax.set_ylabel("Integrated Error ($E$)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    # the axes box of the paper's PDF, in points on its 403.2 x 331.2 pt page
    fig.subplots_adjust(left=56.76 / 403.2, right=375.84 / 403.2, bottom=46.88 / 331.2, top=319.32 / 331.2)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print("wrote", os.path.relpath(out, ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "figs", "euler_step_size",
                                                  "Euler_Error_Analysis_Plot_one_panel.pdf"))
    a = ap.parse_args()
    res = {M: curves_and_marks(M) for M in MS}
    report(res)
    draw(res, a.out)


if __name__ == "__main__":
    main()
