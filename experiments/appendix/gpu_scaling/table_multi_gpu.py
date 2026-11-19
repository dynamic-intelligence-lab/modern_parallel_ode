"""tab:multigpu-jacobi from the speedups of speedups.py: prints the cells with two decimals, then a LaTeX tabular of
the same cells rounded once to the one decimal of the paper. It writes no file; the paper's table was typed from
these numbers.

    python experiments/appendix/gpu_scaling/table_multi_gpu.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from experiments.appendix.gpu_scaling.speedups import MG_DEVICES, T_LABELS, T_TARGETS, _lookup_cell  # noqa: E402

def make_multigpu_table():
    rows = []
    for n in MG_DEVICES:
        row = {}
        for T in T_TARGETS:
            cell = _lookup_cell('b200', n, 'jacobi', T)
            if cell is None or np.isnan(cell['speedup_mean']):
                row[T_LABELS[T]] = '—'
            else:
                row[T_LABELS[T]] = (
                    f"{cell['speedup_mean']:.2f} ± {cell['speedup_std']:.2f}"
                )
        rows.append(row)
    df = pd.DataFrame(rows, index=[f'{n}× B200' for n in MG_DEVICES])
    df.index.name = 'GPUs'
    return df


def multigpu_table_latex(precision=2, dash='---'):
    cols = [T_LABELS[T] for T in T_TARGETS]
    lines = [
        r'\begin{tabular}{l' + 'c' * len(T_TARGETS) + '}',
        r'\toprule',
        'GPUs & ' + ' & '.join(cols) + r' \\',
        r'\midrule',
    ]
    for n in MG_DEVICES:
        cells = [rf'{n}$\times$ B200']
        for T in T_TARGETS:
            cell = _lookup_cell('b200', n, 'jacobi', T)
            if cell is None or np.isnan(cell['speedup_mean']):
                cells.append(dash)
            else:
                m, s = cell['speedup_mean'], cell['speedup_std']
                cells.append(f'${m:.{precision}f} \\pm {s:.{precision}f}$')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


df_mg = make_multigpu_table()
print(df_mg.to_string())
print()
print(multigpu_table_latex(precision=1))
