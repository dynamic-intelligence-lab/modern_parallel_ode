"""The paper's figure fig:slotine-li-fixedpoint-rollout (figs/slotine_li_jacobi_3panel.pdf in the paper's source;
here figs/robot_control/slotine_li_jacobi_3panel.pdf) from the result file of run_slotine_li.py (numpy and
matplotlib only).

For the iterates of the Jacobi iteration after 0, 500 and 2000 sweeps it draws the end-effector path (blue) with the
reference path (dashed gray), the arm at the final time and the true and estimated payload mass. The plotting code
is copied verbatim from a notebook of the authors (not in this repository) into plot_3panel (rcParams, fk_np and the
body between the marker comments); its panel titles say "Optimization step", and each step is one Jacobi sweep.
The script also prints after how many sweeps the iterate coincides with the sequential rollout.
    python experiments/appendix/robot_control/adaptive/figure_slotine_li.py [RESULTS_DIR [FIGURE_DIR]]
RESULTS_DIR defaults to results/robot_control (the shipped file), FIGURE_DIR to figs/robot_control.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results/robot_control"
FIG = sys.argv[2] if len(sys.argv) > 2 else "figs/robot_control"
l1, l2 = 1.0, 0.85
blue_color = "#00B0F0"


# ===== verbatim from the notebook cell: fk_np =====
def fk_np(q):
    q1, q2 = q

    p0 = np.array([0.0, 0.0])

    p1 = np.array([
        l1 * np.cos(q1),
        l1 * np.sin(q1),
    ])

    p2 = p1 + np.array([
        l2 * np.cos(q1 + q2),
        l2 * np.sin(q1 + q2),
    ])

    return p0, p1, p2



# ===== verbatim from the notebook cell: plot style =====


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,

    "axes.linewidth": 1.0,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def plot_3panel(S_checkpoints, checkpoint_iters, plot_checkpoint_iters, t, xy_d, mass_true, out_prefix):
    """The notebook's three-panel figure for the iterates saved at plot_checkpoint_iters."""
    checkpoint_iters = tuple(int(k) for k in checkpoint_iters)
    plot_checkpoint_iters = tuple(int(k) for k in plot_checkpoint_iters)
    N = len(t)
    # ===== verbatim from the notebook cell (except the two lines above and mass_true) =====
    S_checkpoints_np = np.array(S_checkpoints)
    t_np = np.array(t)
    desired_path = np.array(xy_d)

    mass_true_log = np.asarray(mass_true)


    plot_indices = [
        checkpoint_iters.index(k)
        for k in plot_checkpoint_iters
    ]

    qs_list = [
        S_checkpoints_np[i, :, :2]
        for i in plot_indices
    ]

    mass_hat_list = [
        S_checkpoints_np[i, :, 4]
        for i in plot_indices
    ]

    ee_list = []

    for qs_i in qs_list:
        ee_i = np.array([
            fk_np(q)[2]
            for q in qs_i
        ])

        ee_list.append(ee_i)

    time_idx = len(t_np) - 1


    # ============================================================
    # Plot
    # ============================================================
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 4.8),
        sharex=True,
        sharey=True,
    )

    for ax, k_stop, qs_i, ee_i, mhat_i in zip(
        axes,
        plot_checkpoint_iters,
        qs_list,
        ee_list,
        mass_hat_list,
    ):
        q_now = qs_i[time_idx]

        p0, p1, p2 = fk_np(q_now)

        ax.set_aspect("equal")

        ax.set_xlim(-0.4, 1.9)
        ax.set_ylim(-0.4, 1.6)

        ax.grid(
            True,
            alpha=0.45,
            linewidth=0.8,
        )

        optimization_fraction = k_stop / N

        ax.set_title(
            f"Optimization step = {optimization_fraction:.4f}N",
            pad=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(
            r"$x_1$",
            fontsize=24,
        )

        ax.plot(
            desired_path[:, 0],
            desired_path[:, 1],
            "--",
            color="0.55",
            linewidth=1.8,
            label="desired path",
            zorder=1,
        )

        ax.plot(
            ee_i[: time_idx + 1, 0],
            ee_i[: time_idx + 1, 1],
            color=blue_color,
            linewidth=3.0,
            alpha=0.95,
            zorder=2,
        )

        ax.plot(
            [p0[0], p1[0], p2[0]],
            [p0[1], p1[1], p2[1]],
            "o-",
            color="0.35",
            linewidth=6.0,
            markersize=9.0,
            markerfacecolor="0.55",
            markeredgecolor="0.25",
            markeredgewidth=1.2,
            solid_capstyle="round",
            zorder=4,
        )

        ax.plot(
            [p2[0]],
            [p2[1]],
            "o",
            color="0.55",
            markersize=15,
            alpha=0.35,
            zorder=5,
        )

        ax.text(
            0.03,
            0.95,
            f"time: {t_np[time_idx]:.2f} s\n"
            f"true mass: {mass_true_log[time_idx]:.2f} kg\n"
            f"est. mass: {mhat_i[time_idx]:.2f} kg",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
        )

    axes[0].legend(
        loc="lower right",
        frameon=True,
    )

    fig.supylabel(
        r"$x_2$",
        fontsize=24,
    )

    fig.tight_layout(w_pad=2.0)
    # ===== end of the verbatim block =====
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def main():
    """The paper's figure: fp64, plain Jacobi (the r1 of the keys), guess reference + noise (ref_noise), sweeps 0,
    500 and 2000."""
    os.makedirs(FIG, exist_ok=True)
    d = np.load(os.path.join(RES, "fixed_point_check_fp64.npz"))
    plot_3panel(d["ckpt_ref_noise_r1"], d["ckpt_iters"], (0, 500, 2000),
                d["t"], d["xy_d"], d["mass_true"], os.path.join(FIG, "slotine_li_jacobi_3panel"))
    print("wrote", os.path.join(FIG, "slotine_li_jacobi_3panel.pdf"))
    below = d["hist_ref_noise_r1"][:, 0] < 1e-6                 # distance of all five states to the sequential rollout
    k = int(np.argmax(below)) if below.any() else -1
    print(f"the iterate is within 1e-6 of the sequential rollout from sweep {k} = {k / int(d['N']):.3f} N on "
          f"(N = {int(d['N'])}; -1: not within the {len(below) - 1} sweeps of the file)")


if __name__ == "__main__":
    main()
