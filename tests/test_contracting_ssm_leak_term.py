"""Tests for the input-dependent leak term -gamma(u)*h.

The leak term gives a state-independent contraction floor (gamma_min). With
both A bounded and gamma > 0, the per-step Euler Jacobian
    G'(h) = 1 + dt*(-gamma + A*sech²(h))
is uniformly bounded below 1 on the saturation side, regardless of |h|.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.shared.contracting_ssm_cell import (
    init_cell_params, cell_ode_fn, _gamma_diag,
)
from experiments.appendix.contracting_ssm.cell_solvers import solve, solve_diagnostics

jax.config.update("jax_enable_x64", True)

D = 4
NINP = 3
T = 32
ALPHA_MAX = 1.5
GAMMA_MIN = 0.01
GAMMA_MAX = 0.4


def test_gamma_off_reproduces_legacy_cell():
    """gamma_param='off': the RHS equals, bit for bit, the RHS through the leak term with gamma fixed at zero
    (sigmoid, gamma_min = gamma_max = 0), and gamma(u) is exactly zero."""
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    h = jr.normal(jr.PRNGKey(1), (D,), dtype=jnp.float64)
    u = jr.normal(jr.PRNGKey(2), (NINP,), dtype=jnp.float64)

    rhs_default = cell_ode_fn(0.0, h, u, p, ("exp", 1.5, 0.0), ("sigmoid", 0.0, 0.0))  # the leak path at gamma = 0
    rhs_explicit_off = cell_ode_fn(0.0, h, u, p, ("exp", 1.5, 0.0), ("off", 0.0, 0.0))
    diff = float(jnp.max(jnp.abs(rhs_default - rhs_explicit_off)))
    assert diff == 0.0, f"gamma=off should be bit-exact: {diff}"

    # And gamma diag is exactly zero in off mode.
    g = _gamma_diag(u, p, ("off", 0.0, 0.0))
    assert bool(jnp.all(g == 0.0)), f"gamma_diag(off) != 0: {g}"


def test_gamma_sigmoid_bounds():
    """gamma(u) ∈ [gamma_min, gamma_max] for any u, regardless of pre-MLP magnitude."""
    p = init_cell_params(
        jr.PRNGKey(0), NINP, D,
        A_param="sigmoid", gamma_param="sigmoid",
    )
    # Inflate gamma head to drive sigmoid to extremes.
    p = p._replace(
        gamma_W2=p.gamma_W2 * 100.0,
        gamma_b2=p.gamma_b2 * 100.0,
    )
    gamma_static = ("sigmoid", GAMMA_MIN, GAMMA_MAX)
    for u in [jnp.zeros((NINP,)), 5.0 * jnp.ones((NINP,)), -5.0 * jnp.ones((NINP,))]:
        g = _gamma_diag(u, p, gamma_static)
        assert bool(jnp.all(g >= GAMMA_MIN)), f"gamma below floor: {g}"
        assert bool(jnp.all(g <= GAMMA_MAX)), f"gamma above ceiling: {g}"


def test_leak_gives_strict_contraction_under_saturation():
    """With sigmoid A + sigmoid gamma, jac_max < 1 even when h saturates.

    Sets up the same saturation regime as test_a_param_sigmoid (large negative
    g_b2 -> A near -alpha_min, large b_b2 -> forcing pushes h into tanh
    saturation), but with gamma_param='sigmoid' active. The expected jac_max
    is bounded above by 1 - dt*gamma_min, strictly less than 1.
    """
    p = init_cell_params(
        jr.PRNGKey(0), NINP, D,
        A_param="sigmoid", gamma_param="sigmoid",
    )
    # Force A toward saturation regime, exactly mirroring the test that
    # demonstrated boundary-1 behavior without the leak term.
    p = p._replace(
        g_W1=jnp.zeros_like(p.g_W1),
        g_b1=jnp.zeros_like(p.g_b1),
        g_W2=jnp.zeros_like(p.g_W2),
        g_b2=-100.0 * jnp.ones_like(p.g_b2),  # A ~ -alpha_min
        b_W1=jnp.zeros_like(p.b_W1),
        b_b1=jnp.zeros_like(p.b_b1),
        b_W2=jnp.zeros_like(p.b_W2),
        b_b2=5.0 * jnp.ones_like(p.b_b2),     # forcing ~= +1, drives saturation
        # gamma_W2 stays as Xavier init -> some spread, but always >= gamma_min.
    )
    x_seq = jnp.zeros((T, NINP), dtype=p.g_b2.dtype)
    solver_static = ("seq", 1.0, 1e-5, 200, 50,
                     "sigmoid", ALPHA_MAX, 0.0,
                     "sigmoid", GAMMA_MIN, GAMMA_MAX)

    _, _, _, jac_max, _, _, jac_frac_above_1 = solve_diagnostics(
        x_seq, p, solver_static,
    )
    # Strict upper bound: 1 - dt*gamma_min = 0.99.
    assert float(jac_max) <= 1.0 - 1.0 * GAMMA_MIN + 1e-12, (
        f"leak failed to enforce contraction floor: jac_max={jac_max}"
    )
    assert float(jac_frac_above_1) == 0.0, "no entries should exceed 1"


def test_stability_assertion_fires():
    """Construction should reject dt*(gamma_max + alpha_max) >= 2 to fail-fast."""
    p = init_cell_params(
        jr.PRNGKey(0), NINP, D,
        A_param="sigmoid", gamma_param="sigmoid",
    )
    x_seq = jnp.zeros((T, NINP), dtype=p.g_b2.dtype)
    # dt * (gamma_max + alpha_max) = 1.0 * (0.6 + 1.5) = 2.1 >= 2 -> reject.
    bad = ("seq", 1.0, 1e-5, 200, 50,
           "sigmoid", 1.5, 0.0,
           "sigmoid", 0.01, 0.6)
    with pytest.raises(ValueError, match="Euler stability"):
        solve(x_seq, p, bad)


def test_leak_solvers_agree():
    """seq, jacobi, quasi all use the same Euler discretization with leak."""
    p = init_cell_params(
        jr.PRNGKey(0), NINP, D,
        A_param="sigmoid", gamma_param="sigmoid",
    )
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP), dtype=jnp.float64)
    # Use small dt so Newton converges cleanly with leak active.
    base = (0.1, 1e-8, 500, 100, "sigmoid", ALPHA_MAX, 0.0,
            "sigmoid", GAMMA_MIN, GAMMA_MAX)
    h_seq    = solve(x_seq, p, ("seq",   ) + base)
    h_jacobi = solve(x_seq, p, ("jacobi",) + base)
    h_quasi  = solve(x_seq, p, ("quasi", ) + base)

    diff_jac = float(jnp.max(jnp.abs(h_seq - h_jacobi)))
    diff_qua = float(jnp.max(jnp.abs(h_seq - h_quasi)))
    assert diff_jac < 1e-4, f"seq vs jacobi (with leak): {diff_jac}"
    assert diff_qua < 1e-4, f"seq vs quasi (with leak):  {diff_qua}"
