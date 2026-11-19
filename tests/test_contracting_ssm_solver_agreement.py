"""Solver agreement on small fp64 problems.

The three solvers solve the same Euler recurrence; converged solvers agree
to ~tol/||I-J|| (not float epsilon).
"""
import jax
import jax.numpy as jnp
import jax.random as jr

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.shared.contracting_ssm_cell import init_cell_params
from experiments.appendix.contracting_ssm.cell_solvers import solve

jax.config.update("jax_enable_x64", True)

D = 4
NINP = 3
T = 16
DT = 0.1                   # smaller dt -> tighter Newton convergence
NEWTON_TOL = 1e-8
COMPARE_TOL = 1e-4


def _make_setup():
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP), dtype=jnp.float64)
    return p, x_seq


def test_euler_family_agree():
    """seq, jacobi, quasi-DEER all use Euler -> agree at ~Newton tol."""
    p, x_seq = _make_setup()
    h_seq    = solve(x_seq, p, ("seq",    DT, NEWTON_TOL, 500, 100, "exp", 1.5, 0.0, "off", 0.0, 0.0))
    h_jacobi = solve(x_seq, p, ("jacobi", DT, NEWTON_TOL, 500, 100, "exp", 1.5, 0.0, "off", 0.0, 0.0))
    h_quasi  = solve(x_seq, p, ("quasi",  DT, NEWTON_TOL, 500, 100, "exp", 1.5, 0.0, "off", 0.0, 0.0))

    diff_jac = float(jnp.max(jnp.abs(h_seq - h_jacobi)))
    diff_qua = float(jnp.max(jnp.abs(h_seq - h_quasi)))
    assert diff_jac < COMPARE_TOL, f"seq vs jacobi: {diff_jac}"
    assert diff_qua < COMPARE_TOL, f"seq vs quasi:  {diff_qua}"
