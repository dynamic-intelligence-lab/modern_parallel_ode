"""The fp32 path of parode/implicit.py on the cubic double well (CPU, x64 off).

x' = a x - x^3, a = 1, x0 = 0.3, N = 400, dt = 0.05, zero guess, tol 1e-5. Plain DEER returns a non-finite
trajectory (the failing example). The iterate guard at C = 10 does not rescue this polynomial system: as measured,
it hits the cap of 200 iterations with a finite iterate and residual 39.5.
LASER (both trial orders) converges, and the gradient of loss = sum x_t^2 through the IFT adjoint matches jax.grad
through the fp32 sequential rollout. Every output stays float32. This module never imports the fp64 test modules;
each test runs under jax.experimental.disable_x64 so it is independent of the process's x64 flag.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.implicit import implicit_solve, seq_solve  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402

A, X0, N, DT = 1.0, 0.3, 400, 0.05
TOL, MAX_ITER = 1e-5, 200
GUARD = 10.0
FIXED_POINT_ATOL = 1e-4                 # |x_N - sqrt(a)| after T = 20 time units
GRAD_RTOL = 1e-5


@pytest.fixture(autouse=True)
def _x64_off():
    with jax.experimental.disable_x64():
        yield


def well_ode(p):
    return lambda t, x: p["a"] * x - x ** 3


def _instance():
    params = {"a": jnp.asarray(np.float32(A))}
    x0 = jnp.asarray(np.array([X0], np.float32))
    return params, x0, jnp.zeros((N, 1), jnp.float32)


def _solve(**kw):
    params, x0, guess = _instance()
    return implicit_solve(well_ode, params, x0, guess, DT, tol=TOL, max_iter=MAX_ITER, **kw)


def test_t4_plain_deer_is_not_finite():
    traj, res, it = _solve()
    assert traj.dtype == jnp.float32 and res.dtype == jnp.float32
    assert not bool(jnp.all(jnp.isfinite(traj)))
    assert not bool(res <= TOL)


def test_t4_guard_does_not_converge():
    traj, res, it = _solve(guard=GUARD)
    assert traj.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(traj))) and float(jnp.max(jnp.abs(traj))) <= GUARD
    assert int(it) == MAX_ITER and float(res) > 1.0


@pytest.mark.parametrize("cfg", [LaserConfig(propose_first="plain"), LaserConfig()],
                         ids=["laser-plain", "laser"])
def test_t4_laser_converges_and_matches_bptt(cfg):
    params, x0, guess = _instance()

    def par_loss(p):
        traj, res, it = implicit_solve(well_ode, p, x0, guess, DT, tol=TOL, max_iter=MAX_ITER, laser=cfg)
        return jnp.sum(traj[1:] ** 2), (res, it, traj)

    def seq_loss(p):
        return jnp.sum(seq_solve(well_ode, p, x0, DT, N)[1:] ** 2)

    (loss, (res, it, traj)), g = jax.jit(jax.value_and_grad(par_loss, has_aux=True))(params)
    g_ref = jax.jit(jax.grad(seq_loss))(params)
    assert traj.dtype == res.dtype == loss.dtype == g["a"].dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(traj))) and float(res) < TOL and int(it) < MAX_ITER
    assert abs(float(traj[-1, 0]) - np.sqrt(A)) < FIXED_POINT_ATOL
    assert abs(float(g["a"] - g_ref["a"])) / abs(float(g_ref["a"])) < GRAD_RTOL
