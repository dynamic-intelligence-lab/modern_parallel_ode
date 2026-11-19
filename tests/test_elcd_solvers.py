"""Tests of experiments/elcd/solvers.py (CPU, fp64).

The exact linearizer where A is constant (W1 = 0), against the closed-form stability polynomial. One instance: D 2,
dt 0.01, gain 4, rate 1, seed 0.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.elcd.model import A_matrix, cast, init_synthetic, make_ode  # noqa: E402
from experiments.elcd.solvers import exact_step_linearizer  # noqa: E402

D, DT = 2, 0.01
GAIN, RATE = 4.0, 1.0
PARAMS = cast(init_synthetic(0, D, gain=GAIN, rate=RATE), jnp.float64)


# ----------------------------------------------------------------------------- linearizers

def _stability_polynomial(A, dt, step):
    """The exact step Jacobian of the linear field z' = A (z - z*): I + dt A (Euler) or p_4(dt A) (RK4)."""
    Z = dt * A
    if step == "euler":
        return np.eye(A.shape[0]) + Z
    return np.eye(A.shape[0]) + Z + Z @ Z / 2 + Z @ Z @ Z / 6 + Z @ Z @ Z @ Z / 24


@pytest.mark.parametrize("step", ["euler", "rk4"])
def test_exact_linearizer_is_the_stability_polynomial_when_A_is_constant(step):
    """With W1 = 0 in both MLPs A is constant, and the exact step Jacobian equals the stability polynomial of dt A
    at every state (tolerance 1e-12)."""
    params = PARAMS._replace(W1s=jnp.zeros_like(PARAMS.W1s), W1a=jnp.zeros_like(PARAMS.W1a))
    X_prev = jnp.asarray(np.random.default_rng(5).normal(size=(7, D)) * 3.0)
    exact = np.asarray(jax.jit(exact_step_linearizer(make_ode(params), DT, step))(X_prev))
    A = np.asarray(A_matrix(params, X_prev[0]))
    assert np.max(np.abs(np.asarray(A_matrix(params, X_prev[3])) - A)) <= 1e-15    # A really is constant
    poly = _stability_polynomial(A, DT, step)
    assert np.max(np.abs(exact - poly[None])) <= 1e-12
