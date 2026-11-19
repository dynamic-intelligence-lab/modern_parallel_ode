"""tab:multigpu-jacobi and the iteration counts of the GPU-scaling appendix, recomputed from the shipped result files.

speedups.py turns results/gpu_scaling/*.npz into per-seed speedups and their mean and sample standard deviation over
the 5 seeds. Every cell of the paper's table, rounded once to one decimal, must equal the printed value; the ranges
of iteration counts that the appendix quotes must be those of the stored rows.
"""
import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
with contextlib.redirect_stdout(io.StringIO()):
    from experiments.appendix.gpu_scaling import speedups as S  # noqa: E402

# (GPUs, N): (mean, std) as printed in the paper
TABLE = {
    (1, 100_000): ("492.7", "0.2"), (1, 1_000_000): ("591.0", "0.1"), (1, 10_000_000): ("547.6", "13.6"),
    (2, 100_000): ("419.3", "0.7"), (2, 1_000_000): ("1003.9", "0.6"), (2, 10_000_000): ("1058.1", "25.5"),
    (4, 100_000): ("342.3", "2.9"), (4, 1_000_000): ("1428.9", "1.8"), (4, 10_000_000): ("2053.2", "49.3"),
    (8, 100_000): ("168.4", "5.9"), (8, 1_000_000): ("1457.7", "21.7"), (8, 10_000_000): ("3209.1", "76.7"),
}


@pytest.mark.parametrize("cell", list(TABLE), ids=lambda c: f"{c[0]}x-N{c[1]}")
def test_multi_gpu_table_cell(cell):
    c = S._lookup_cell("b200", cell[0], "jacobi", cell[1])
    assert int(c["n_valid"]) == 5
    assert (f"{c['speedup_mean']:.1f}", f"{c['speedup_std']:.1f}") == TABLE[cell]


def test_iteration_count_ranges():
    counts = {}
    for df in S.single_dfs.values():
        ok = df[~df["oom"].astype(bool)]
        for alg, its in ok.groupby("algorithm")["iters_to_tol"]:
            counts.setdefault(alg, []).extend(int(i) for i in its)
    ranges = {alg: (min(v), max(v)) for alg, v in counts.items()}
    assert ranges == {"full_deer": (6, 6), "quasi_deer": (86, 106), "jacobi": (20, 22), "jacobi_scan": (22, 23)}
    mg = {int(i) for df in S.mg_dfs.values() for i in df["iters_to_tol"]}
    assert mg <= {22, 23}
