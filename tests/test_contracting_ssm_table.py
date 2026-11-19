"""Table tab:eigenworms_gamma_min recomputed from the shipped runs table.

experiments/appendix/contracting_ssm/table_gamma_min.py on results/contracting_ssm/runs.csv must give the 65 cells of
the table as printed in the paper's LaTeX source, and the runs table must give the paper's "test accuracies coincide
in 23 of the 25 (gamma_min, seed) pairs".
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.contracting_ssm.table_gamma_min import RUNS, table

# The rows of tab:eigenworms_gamma_min in the paper, cell by cell.
X = "$\\times$"
PAPER_ROWS = [
    ["0.01", "84.8", "5.2", "51.2", "5.9", "0.56", "0.04", "12.12", "1.02", "91" + X, "8.7", "4.2" + X, "0.4"],
    ["0.02", "85.4", "6.8", "45.8", "1.5", "0.52", "0.03", "7.83", "0.38", "89" + X, "6.0", "5.9" + X, "0.3"],
    ["0.05", "85.0", "6.6", "49.5", "3.6", "0.46", "0.02", "4.06", "0.07", "108" + X, "9.2", "12.2" + X, "1.0"],
    ["0.10", "84.8", "5.5", "50.2", "6.0", "0.42", "0.02", "2.25", "0.06", "121" + X, "17.2", "22.3" + X, "3.1"],
    ["0.20", "83.1", "8.2", "45.0", "1.0", "0.39", "0.01", "1.25", "0.02", "117" + X, "4.2", "36.0" + X, "1.0"],
]


def test_table_equals_paper():
    rows, notes = table(pd.read_csv(RUNS))
    assert rows == PAPER_ROWS
    assert notes == ["gamma_min 0.02: mean accuracies seq 85.71, quasi 85.71, jacobi 84.71"]


def test_accuracies_coincide_in_23_of_25_pairs():
    runs = pd.read_csv(RUNS)
    sweep = runs[runs["group"] == "gamma_min_sweep"]
    acc = sweep.pivot_table(index=["gamma_min", "seed"], columns="solver", values="test_acc", aggfunc="first")
    assert len(acc) == 25
    same = (acc["seq"] == acc["quasi"]) & (acc["seq"] == acc["jacobi"])
    assert int(same.sum()) == 23
    assert sorted(acc.index[~same]) == [(0.02, 3), (0.05, 3)]
