"""Tests of the ELK baseline port (parode/elk_baseline.py)."""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.elk_baseline import (elk_filter_dense, elk_filter_diag, elk_integrate, elk_step,  # noqa: E402
                                           kalman_filter_sequential)
from parode.explicit_parallel import explicit_par_integrate, newton_step, euler_step  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402


def _random_model(key, N, D, scale=0.9):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    F = scale * jax.random.normal(k1, (N - 1, D, D)) / np.sqrt(D)
    b = jax.random.normal(k2, (N - 1, D))
    m0 = jax.random.normal(k3, (D,))
    y = jax.random.normal(k4, (N, D))
    return F, b, m0, y


@pytest.mark.parametrize("sigmasq", [1e-2, 1.0, 1e3])
def test_parallel_dense_matches_sequential(sigmasq):
    F, b, m0, y = _random_model(jax.random.PRNGKey(0), N=60, D=3)
    par = elk_filter_dense(F, b, m0, y, sigmasq)
    seq = kalman_filter_sequential(F, b, m0, y, sigmasq)
    assert np.max(np.abs(np.asarray(par - seq))) < 1e-9 * max(1.0, float(np.max(np.abs(seq))))


def test_diag_matches_dense_on_diagonal_dynamics():
    F, b, m0, y = _random_model(jax.random.PRNGKey(1), N=40, D=4)
    Fd = jax.vmap(jnp.diag)(F)                              # (N-1, D)
    Fdense = jax.vmap(jnp.diag)(Fd)                         # (N-1, D, D) diagonal
    dense = elk_filter_dense(Fdense, b, m0, y, 0.7)
    diag = elk_filter_diag(Fd, b, m0, y, 0.7)
    assert np.max(np.abs(np.asarray(dense - diag))) < 1e-10


def test_large_sigmasq_is_the_deer_step():
    ode = prepare_ode_function(lambda t, x, u=None: -x + jnp.tanh(x) * 0.5)
    x0 = jnp.array([1.0, -0.5]); dt = 0.05; N = 50
    times = dt * jnp.arange(N + 1)
    X = jnp.zeros((N, 2)) + 0.1
    X_deer = X + newton_step(X, x0, ode, dt, euler_step, times, None)[0]
    X_elk, _ = elk_step(X, x0, ode, dt, times, sigmasq=1e12)
    assert np.max(np.abs(np.asarray(X_elk - X_deer))) < 1e-8


@pytest.mark.parametrize("sigmasq", [1e-2, 1.0, 1e6])
def test_true_trajectory_is_a_fixed_point(sigmasq):
    ode = prepare_ode_function(lambda t, x, u=None: x - x ** 3)
    x0 = jnp.array([1.5]); dt = 0.05; N = 100
    times = dt * jnp.arange(N + 1)
    truth = seq_integrate(ode, x0, dt, N, "euler")[1:]
    X_new, res = elk_step(truth, x0, ode, dt, times, sigmasq)
    assert np.max(np.abs(np.asarray(X_new - truth))) < 1e-11
    assert float(res) < 1e-11


@pytest.mark.parametrize("sigmasq,quasi,iters", [(1.0, False, 600), (1e2, False, 80), (1e8, False, 12), (1e2, True, 80)])
def test_converges_on_a_contractive_problem(sigmasq, quasi, iters):
    """Heavier damping (smaller sigma^2) converges linearly and more slowly: measured about 500
    iterations at sigma^2 = 1, 50 at 1e2, 5 at 1e8 (the DEER limit); sigma^2 = 1e-2 needs thousands."""
    ode = lambda t, x, u=None: -x + jnp.sin(x) * 0.3  # noqa: E731
    x0 = jnp.array([1.0, -2.0, 0.5]); dt = 0.05; N = 200
    truth = seq_integrate(prepare_ode_function(ode), x0, dt, N, "euler")
    sol, res, it = elk_integrate(ode, x0, jnp.zeros((N, 3)), dt, sigmasq, tol=1e-10, max_iter=iters, quasi=quasi)
    assert float(res) < 1e-10 and int(it) < iters
    assert np.max(np.abs(np.asarray(sol - truth))) < 1e-8


def test_heavy_damping_decreases_monotonically():
    ode = prepare_ode_function(lambda t, x, u=None: -x + jnp.sin(x) * 0.3)
    x0 = jnp.array([1.0, -2.0, 0.5]); dt = 0.05; N = 200
    times = dt * jnp.arange(N + 1)
    X = jnp.zeros((N, 3)); prev = np.inf
    for _ in range(60):
        X, res = elk_step(X, x0, ode, dt, times, sigmasq=1e-2)
        assert float(res) <= prev; prev = float(res)
    assert prev < 1.5


def test_deer_limit_matches_parode_solver_on_cubic_from_wrong_basin():
    ode = lambda t, x, u=None: x - x ** 3  # noqa: E731
    x0 = jnp.array([1.5]); dt = 0.05; N = 200
    guess = -jnp.ones((N, 1))
    sol_p, _, it_p = explicit_par_integrate(ode, x0, guess, dt, 100, "euler", tol=1e-9)
    sol_e, res_e, it_e = elk_integrate(ode, x0, guess, dt, 1e12, tol=1e-9, max_iter=100)
    assert float(res_e) < 1e-9
    assert np.max(np.abs(np.asarray(sol_e - sol_p))) < 1e-7
