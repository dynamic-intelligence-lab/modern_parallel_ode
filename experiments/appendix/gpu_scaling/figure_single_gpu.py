"""fig:singlegpu-jacobi: draws figs/gpu_scaling/gpu_scaling.pdf from the speedups of speedups.py. Every cell without
a finite speedup (out of memory, or never run because the sweep stopped at a smaller N) is drawn as an
out-of-memory mark.

    python experiments/appendix/gpu_scaling/figure_single_gpu.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.appendix.gpu_scaling.speedups import (  # noqa: E402
    ALG_LABELS, ALG_ORDER, GPU_LABELS, GPU_ORDER, ROOT, T_LABELS, T_TARGETS, _gpu_cmap, _lookup_cell,
)

def plot_main_speedup(figsize=(8.4, 2.8)):
    fig, axes = plt.subplots(1, len(T_TARGETS), figsize=figsize, sharey=True)
    cmap = _gpu_cmap(len(GPU_ORDER))
    width = 0.13
    x = np.arange(len(ALG_ORDER))
    for ax, T in zip(axes, T_TARGETS):
        for i, gpu in enumerate(GPU_ORDER):
            means, stds, ooms = [], [], []
            for alg in ALG_ORDER:
                cell = _lookup_cell(gpu, 1, alg, T)
                if cell is None or bool(cell['all_oom']) or np.isnan(cell['speedup_mean']):
                    means.append(np.nan)
                    stds.append(np.nan)
                    ooms.append(True)
                else:
                    means.append(float(cell['speedup_mean']))
                    stds.append(float(cell['speedup_std']))
                    ooms.append(False)
            offset = (i - (len(GPU_ORDER) - 1) / 2) * width
            bars = ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=2,
                color=cmap[i], edgecolor='black', linewidth=0.4,
                error_kw=dict(elinewidth=0.8, ecolor='0.25'),
                label=GPU_LABELS[gpu],
            )
            for b, oom in zip(bars, ooms):
                if oom:
                    ax.plot(b.get_x() + b.get_width() / 2, 1.0,
                            marker='x', color=cmap[i],
                            markersize=5, markeredgewidth=1.2,
                            linestyle='None', zorder=10)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [ALG_LABELS[a] for a in ALG_ORDER],
            rotation=20,
            ha='right',
            rotation_mode='anchor'
        )
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.grid(True, axis='y', which='both', alpha=0.3)
        ax.set_title(T_LABELS[T])
    axes[0].set_ylabel('Speedup over sequential rollout')

    # GPU bars and the OOM marker share one legend
    handles, labels = axes[-1].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='x', color='0.25',
                          markersize=5, markeredgewidth=1.2,
                          linestyle='None'))
    labels.append('OOM')
    axes[-1].legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    fontsize=7.5, title='GPU', title_fontsize=8,
                    frameon=True)
    fig.tight_layout()
    return fig

fig_main = plot_main_speedup()
os.makedirs(os.path.join(ROOT, 'figs', 'gpu_scaling'), exist_ok=True)
fig_main.savefig(os.path.join(ROOT, 'figs', 'gpu_scaling', 'gpu_scaling.pdf'), bbox_inches='tight')
