"""Tests of the step-size example of the appendix (experiments/appendix/euler_step_size/figure_step_size.py).

The marks of the paper's figure, read from its PDF (experiments/appendix/euler_step_size/paper_figure_marks.json),
against the recomputation: the four stars (step size and error), the four dashed lines (h_mid) and the 156 curve
vertices (error at the vertex's step size T/K, which is a point of the drawn grid) agree to a relative 5e-8, the
precision of the coordinates in the PDF. The Jacobi iterate against sequential forward Euler, and the bounds the
appendix states about the example.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.appendix.euler_step_size import figure_step_size as es  # noqa: E402

MARKS = json.load(open(os.path.join(ROOT, "experiments", "appendix", "euler_step_size",
                                     "paper_figure_marks.json")))["marks"]
RTOL = 5e-8


@pytest.fixture(scope="module")
def res():
    return {M: es.curves_and_marks(M) for M in es.MS}


@pytest.mark.parametrize("M", es.MS)
def test_marks_equal_the_paper_figure(res, M):
    r, ref = res[M], MARKS[str(M)]
    assert round(es.T / ref["star"][0]) == r["K_min"]
    assert abs(ref["star"][0] / r["h_min"] - 1) < RTOL
    assert abs(ref["star"][1] / r["E_min"] - 1) < RTOL
    assert abs(ref["dashed_h"] / r["h_mid"] - 1) < RTOL
    curve = dict(zip(r["K"].tolist(), r["E"]))
    for h, E in ref["curve"]:
        K = round(es.T / h)
        assert abs(es.T / K / h - 1) < 1e-6
        assert abs(curve[K] / E - 1) < RTOL


def test_jacobi_equals_sequential_forward_euler():
    K = 300
    h = es.T / K
    s = np.empty(K + 1)
    s[0] = es.X0
    for k in range(1, K + 1):
        s[k] = s[k - 1] + h * es.f(s[k - 1], (k - 1) * h)
    t, p = es.jacobi(K, K)
    np.testing.assert_allclose(t, h * np.arange(K + 1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(p, s, rtol=0, atol=1e-12)
    _, p = es.jacobi(5, K)                      # M sweeps make the first M + 1 states exact, not more
    np.testing.assert_allclose(p[:6], s[:6], rtol=0, atol=1e-12)
    assert np.all(np.abs(p[6:] - s[6:]) > 1e-3)


def test_bounds_stated_in_the_appendix(res):
    for M in es.MS:
        r = res[M]
        assert 250 < r["K_min"] < 50000                                  # interior minimum
        assert r["E_mid"] / r["E_min"] - 1 <= 0.09
        assert 0 <= r["h_asy"] / r["h_min"] - 1 <= 0.20
        assert 0 <= r["h_num"] / r["h_min"] - 1 <= 0.20
        assert r["h_num"] / r["h_asy"] - 1 <= 0.10
        assert abs(r["h_asy"] / r["h_stat"] - 1) <= 0.01
        assert es.rho_fe(r["h_asy"]) < 1
    gaps = [res[M]["h_mid"] / res[M]["h_min"] - 1 for M in es.MS]
    assert all(a > b > 0 for a, b in zip(gaps, gaps[1:]))               # the gap shrinks as M grows
