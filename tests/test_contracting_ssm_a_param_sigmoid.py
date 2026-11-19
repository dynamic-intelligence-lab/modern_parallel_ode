"""Tests for A_param='sigmoid' parametrization.

The whole point of sigmoid mode is to keep |A_ii(x)| bounded so Euler-family
solvers stay stable even when training drives g(x) to extreme values. We test
that:
  (a) The bound is exact: -alpha_max < A_ii(x) <= 0 for any x.
  (b) Solvers run finite forward + grad even with aggressively-perturbed g
      (simulating after-training drift).
"""
import jax
import jax.numpy as jnp
import jax.random as jr

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.shared.contracting_ssm_cell import init_cell_params, _A_diag
from experiments.appendix.contracting_ssm.cell_solvers import solve, solve_diagnostics

jax.config.update("jax_enable_x64", True)

D = 4
NINP = 3
T = 64
ALPHA_MAX = 1.5


def _perturbed_params(key, scale: float):
    """Build a cell with g_W2/g_b2 inflated by `scale` to simulate drift."""
    p = init_cell_params(key, NINP, D, A_param="sigmoid", gamma_param="off",
                         init_alpha_range=(0.1, ALPHA_MAX))
    return p._replace(
        g_W2=p.g_W2 * scale,
        g_b2=p.g_b2 * scale,
    )


def test_sigmoid_bound_holds():
    """A_ii(x) ∈ [-alpha_max, 0] for any x, regardless of g magnitude.

    (Mathematically the open interval; in fp64 sigmoid saturates to 1.0
    exactly for very large g, hitting -alpha_max — still bounded.)
    """
    p = _perturbed_params(jr.PRNGKey(0), scale=100.0)   # extreme drift
    for u in [jnp.zeros((NINP,)), 5.0 * jnp.ones((NINP,)), -5.0 * jnp.ones((NINP,))]:
        A = _A_diag(u, p, A_static=("sigmoid", ALPHA_MAX, 0.0))
        assert bool(jnp.all(A >= -ALPHA_MAX)), f"A out of lower bound: {A}"
        assert bool(jnp.all(A <= 0.0)), f"A out of upper bound: {A}"


def test_sigmoid_keeps_seq_euler_stable_under_drift():
    """With sigmoid bound and inflated g, seq Euler stays finite."""
    p = _perturbed_params(jr.PRNGKey(0), scale=100.0)
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP), dtype=jnp.float64)
    solver_static = ("seq", 1.0, 1e-5, 200, 50, "sigmoid", ALPHA_MAX, 0.0, "off", 0.0, 0.0)

    h_traj = solve(x_seq, p, solver_static)
    assert bool(jnp.all(jnp.isfinite(h_traj))), "seq Euler diverged under drift"

    g = jax.grad(lambda pp: jnp.sum(solve(x_seq, pp, solver_static) ** 2))(p)
    finite = all(bool(jnp.all(jnp.isfinite(l))) for l in jax.tree_util.tree_leaves(g))
    assert finite, "seq grads NaN under drift"


def test_sigmoid_tanh_cell_can_hit_nonexpansive_boundary():
    """alpha_min bounds A, but A*tanh(h) is not uniformly strictly contractive.

    If forcing exceeds the leak magnitude, h drifts into tanh saturation. Then
    sech²(h) rounds to zero and the Euler propagator is exactly 1.0. That is
    a boundary/non-strict case, not true |J| > 1 expansion.
    """
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="sigmoid", gamma_param="off")
    p = p._replace(
        g_W1=jnp.zeros_like(p.g_W1),
        g_b1=jnp.zeros_like(p.g_b1),
        g_W2=jnp.zeros_like(p.g_W2),
        g_b2=-100.0 * jnp.ones_like(p.g_b2),  # A ~= -alpha_min
        b_W1=jnp.zeros_like(p.b_W1),
        b_b1=jnp.zeros_like(p.b_b1),
        b_W2=jnp.zeros_like(p.b_W2),
        b_b2=5.0 * jnp.ones_like(p.b_b2),     # forcing ~= +1
    )
    x_seq = jnp.zeros((T, NINP), dtype=p.g_b2.dtype)
    solver_static = ("seq", 1.0, 1e-5, 200, 50, "sigmoid", ALPHA_MAX, 0.1, "off", 0.0, 0.0)

    _, _, _, jac_max, _, _, jac_frac_above_1 = solve_diagnostics(
        x_seq, p, solver_static,
    )
    assert float(jac_max) <= 1.0, f"sigmoid Euler expanded: {jac_max}"
    assert float(jac_max) == 1.0, "expected tanh saturation to hit |J|=1"
    assert float(jac_frac_above_1) == 0.0, "boundary equality should not count as above 1"


def test_exp_param_unchanged():
    """A_param="exp" gives A_ii = -exp(-g)."""
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    u = jnp.ones((NINP,))
    A = _A_diag(u, p, ("exp", 1.5, 0.0))
    expected = -jnp.exp(-(p.g_W2 @ jnp.tanh(p.g_W1 @ u + p.g_b1) + p.g_b2))
    assert bool(jnp.allclose(A, expected, atol=1e-12)), "exp parametrization drifted"
