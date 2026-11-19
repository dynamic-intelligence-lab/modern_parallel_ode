"""Verify custom_vjp gradients match a JAX-autodiff reference.

Reference (Path A): a fixed-iter lax.scan implementation of Jacobi
(Euler step) for the quasi solver. Slow but correct.

Custom (Path B): @jax.custom_vjp + parallel-scan diagonal adjoint
(cell_solvers.py:solve). Fast, must match reference within COMPARE_TOL.

test_custom_vjp_paper_configuration checks the configuration of the paper's
runs (sigmoid A, sigmoid gamma, Euler with dt=1) for the three solvers
against autodiff through a sequential lax.scan of the same recurrence.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.shared.contracting_ssm_cell import init_cell_params, cell_ode_fn
from experiments.appendix.contracting_ssm.cell_solvers import solve

jax.config.update("jax_enable_x64", True)

D = 4
NINP = 3
T = 16
DT = 0.1
NEWTON_TOL = 1e-8
COMPARE_TOL = 1e-4
N_REF_ITERS = 200          # enough to converge at this dt and problem size


def _scan_jacobi_reference(x_seq, p, dt, n_iters):
    """Fixed-iter Jacobi (Euler step), differentiable by JAX autodiff."""
    T_ = x_seq.shape[0]
    D_ = p.g_b2.shape[0]
    h0 = jnp.zeros((D_,), dtype=x_seq.dtype)
    H0 = jnp.zeros((T_, D_), dtype=x_seq.dtype)

    def body(H, _):
        H_prev = jnp.concatenate([h0[None], H[:-1]], axis=0)
        def step(hp, xc):
            return hp + dt * cell_ode_fn(0.0, hp, xc, p, ("exp", 1.5, 0.0), ("off", 0.0, 0.0))
        H_new = jax.vmap(step)(H_prev, x_seq)
        return H_new, None

    H_final, _ = jax.lax.scan(body, H0, None, length=n_iters)
    return H_final


def test_custom_vjp_matches_autodiff_reference_params():
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP), dtype=jnp.float64)

    def loss_custom(p_):
        return jnp.sum(solve(x_seq, p_, ("quasi", DT, NEWTON_TOL, 500, 100, "exp", 1.5, 0.0, "off", 0.0, 0.0)) ** 2)

    def loss_ref(p_):
        return jnp.sum(_scan_jacobi_reference(x_seq, p_, DT, N_REF_ITERS) ** 2)

    g_custom = jax.grad(loss_custom)(p)
    g_ref = jax.grad(loss_ref)(p)

    for name in p._fields:
        gc = getattr(g_custom, name)
        gr = getattr(g_ref, name)
        diff = float(jnp.max(jnp.abs(gc - gr)))
        assert diff < COMPARE_TOL, f"param grad mismatch on {name}: {diff}"


def test_custom_vjp_matches_autodiff_reference_x_seq():
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP), dtype=jnp.float64)

    def loss_custom(x_):
        return jnp.sum(solve(x_, p, ("quasi", DT, NEWTON_TOL, 500, 100, "exp", 1.5, 0.0, "off", 0.0, 0.0)) ** 2)

    def loss_ref(x_):
        return jnp.sum(_scan_jacobi_reference(x_, p, DT, N_REF_ITERS) ** 2)

    g_custom = jax.grad(loss_custom)(x_seq)
    g_ref = jax.grad(loss_ref)(x_seq)
    diff = float(jnp.max(jnp.abs(g_custom - g_ref)))
    assert diff < COMPARE_TOL, f"x_seq grad mismatch: {diff}"


@pytest.mark.parametrize("solver", ["seq", "quasi", "jacobi"])
def test_custom_vjp_paper_configuration(solver):
    """Forward solve and custom_vjp gradient (parameters and inputs) in the
    configuration of the paper's runs, against autodiff through a sequential
    lax.scan of the Euler recurrence. Solver tolerance 1e-12, float64."""
    D_, T_, gamma_min = 8, 100, 0.01
    A_static, gamma_static = ("sigmoid", 1.5, 0.0), ("sigmoid", gamma_min, 0.4)
    p = init_cell_params(jr.PRNGKey(0), D_, D_, A_param="sigmoid", gamma_param="sigmoid")
    x_seq = jr.normal(jr.PRNGKey(1), (T_, D_), dtype=jnp.float64)
    w = jr.normal(jr.PRNGKey(2), (T_, D_), dtype=jnp.float64)
    st = (solver, 1.0, 1e-12, 5000, 500, *A_static, *gamma_static)

    def h_ref(p_, x_):
        def step(h, u):
            h_next = h + cell_ode_fn(0.0, h, u, p_, A_static, gamma_static)
            return h_next, h_next
        return jax.lax.scan(step, jnp.zeros(D_), x_)[1]

    diff_h = float(jnp.max(jnp.abs(solve(x_seq, p, st) - h_ref(p, x_seq))))
    assert diff_h < 1e-8, f"{solver}: forward mismatch {diff_h}"
    g_custom = jax.grad(lambda p_, x_: jnp.sum(w * solve(x_, p_, st)), argnums=(0, 1))(p, x_seq)
    g_ref = jax.grad(lambda p_, x_: jnp.sum(w * h_ref(p_, x_)), argnums=(0, 1))(p, x_seq)
    for gc, gr in zip(jax.tree_util.tree_leaves(g_custom), jax.tree_util.tree_leaves(g_ref)):
        diff = float(jnp.max(jnp.abs(gc - gr)))
        assert diff < 1e-8, f"{solver}: gradient mismatch {diff}"
