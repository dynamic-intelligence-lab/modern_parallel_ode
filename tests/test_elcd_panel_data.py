"""The pick of the best seed of the paper's figure (panel_data.best_seed) on a synthetic frame, and the
method's tick label of the main figures (table_training_time.x_label): no records, no JAX."""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.elcd.figures import panel_data as PF  # noqa: E402
from experiments.elcd.figures import table_training_time as MF  # noqa: E402


def test_best_seed_takes_the_lowest_final_value_and_ignores_non_finite_ones():
    rows = []
    for seed, final in ((0, 0.5), (1, 0.2), (2, np.nan), (3, 0.3)):
        rows += [dict(seed=seed, item="laser", step=5000, train_chamfer=0.1),      # an early evaluation, lower, not final
                 dict(seed=seed, item="laser", step=10000, train_chamfer=final),
                 dict(seed=seed, item="deer", step=10000, train_chamfer=0.01)]
    assert PF.best_seed(pd.DataFrame(rows), "laser") == (1, 0.2)
    assert PF.best_seed(pd.DataFrame(rows), "laser", seeds=(0, 3)) == (3, 0.3)


def test_x_label_with_and_without_the_rate():
    assert MF.x_label("scale-elk0.03", rate=False) == "scale-ELK" and MF.x_label("scale-elk0.03") == "scale-ELK\ng = 0.03"
