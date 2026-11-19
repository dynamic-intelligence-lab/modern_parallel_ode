"""Solver-level tests for LASER in explicit_par_integrate (CPU, fp64, about a minute)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.explicit_parallel import explicit_par_integrate  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402
from ode_problems import earlier_laser  # noqa: E402

TOL = 1e-9


def _truth(ode, x0, dt, N):
    return seq_integrate(ode, x0, dt, N, "euler")


def _plain(ode, x0, guess, dt, max_iter=60):
    sol, res, it = explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=TOL)
    return np.asarray(sol), float(res), int(it)


def _damped(ode, x0, guess, dt, max_iter=60, **kw):
    cfg = earlier_laser(return_history=True, **kw)
    sol, res, it, hist = explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=TOL, laser=cfg)
    return np.asarray(sol), float(res), int(it), {k: np.asarray(v)[:int(it)] for k, v in hist.items()}


def _err(sol, truth):
    return float(np.max(np.abs(sol - np.asarray(truth))))


def cubic(t, x, u=None):
    return x - x ** 3


def test_cubic_double_well_from_zero_guess_recovers_where_plain_deer_nans():
    for dt, N in ((0.05, 400), (0.1, 128)):
        x0 = jnp.array([1.5])
        truth = _truth(cubic, x0, dt, N)
        guess = jnp.zeros((N, 1))
        sol_p, res_p, _ = _plain(cubic, x0, guess, dt)
        assert not np.all(np.isfinite(sol_p)) or not np.isfinite(res_p)
        sol, res, it, hist = _damped(cubic, x0, guess, dt)
        assert np.all(np.isfinite(sol)) and res < TOL
        assert it <= 12, it
        assert _err(sol, truth) < 1e-8
        assert hist["gamma"][0] > 0          # damping engaged on the first accepted step
        assert hist["gamma"][-1] == 0.0      # exact Newton at the end


def test_wrong_basin_guess_costs_no_more_than_plain_deer():
    dt, N = 0.05, 400
    x0 = jnp.array([1.5])
    truth = _truth(cubic, x0, dt, N)
    guess = -jnp.ones((N, 1))
    sol_p, res_p, it_p = _plain(cubic, x0, guess, dt, max_iter=80)
    assert np.isfinite(res_p) and res_p < 1e-6
    sol, res, it, hist = _damped(cubic, x0, guess, dt, max_iter=80)
    assert res < TOL and _err(sol, truth) < 1e-8
    assert it <= it_p + 3, (it, it_p)
    assert np.sum(hist["n_trials"] > 1) <= 3, hist["n_trials"]


def test_saddle_transit_truth_converges_from_zero_guess():
    dt, N = 0.05, 400
    guess = jnp.zeros((N, 1))
    for x0v in (1e-2, 1e-3):
        x0 = jnp.array([x0v])
        truth = _truth(cubic, x0, dt, N)
        sol_p, res_p, _ = _plain(cubic, x0, guess, dt)
        assert not np.all(np.isfinite(sol_p)) or not np.isfinite(res_p)
        sol, res, it, _ = _damped(cubic, x0, guess, dt, max_iter=80)
        assert res < TOL and _err(sol, truth) < 1e-7
        assert it <= 25, it


def test_nonnormal_contractive_system_is_not_damped():
    dt, N = 0.05, 200
    A = jnp.array([[-1.0, 10.0], [0.0, -1.0]])

    def ode(t, x, u=None):
        return A @ x

    x0 = jnp.array([1.0, 1.0])
    truth = _truth(ode, x0, dt, N)
    guess = jnp.zeros((N, 2))
    _, _, it_p = _plain(ode, x0, guess, dt)
    sol, res, it, hist = _damped(ode, x0, guess, dt)
    assert res < TOL and _err(sol, truth) < 1e-8
    assert it <= it_p + 1
    assert np.all(hist["gamma"] == 0.0)


def test_alternating_shear_converges_in_plain_count():
    dt, N = 0.05, 200

    def ode(t, x, u=None):
        s = jnp.where(jnp.mod(jnp.round(t / dt), 2) == 0, 10.0, -10.0)
        return jnp.array([-0.5 * x[0] + s * x[1], -0.5 * x[1]]) + jnp.array([jnp.sin(t), jnp.cos(t)])

    x0 = jnp.array([0.3, -0.2])
    truth = _truth(ode, x0, dt, N)
    guess = jnp.zeros((N, 2))
    _, _, it_p = _plain(ode, x0, guess, dt)
    sol, res, it, _ = _damped(ode, x0, guess, dt)
    assert res < TOL and _err(sol, truth) < 1e-8
    assert it <= it_p + 1


def test_tanh_rnn_bounded_iterates():
    dt, N, D = 0.1, 128, 8
    W = 2.5 / np.sqrt(D) * jax.random.normal(jax.random.PRNGKey(3), (D, D))

    def ode(t, x, u=None):
        return -x + jnp.tanh(W @ x)

    x0 = 2.0 * jax.random.normal(jax.random.PRNGKey(9), (D,))
    truth = _truth(ode, x0, dt, N)
    guess = jnp.zeros((N, D))
    _, _, it_p = _plain(ode, x0, guess, dt)
    sol, res, it, hist = _damped(ode, x0, guess, dt)
    assert res < TOL and _err(sol, truth) < 1e-7
    assert it <= it_p + 2
    assert np.max(hist["step_inf"]) < 50.0


def test_contractive_problem_is_bit_for_bit_plain_newton():
    dt, N = 0.1, 100

    def ode(t, x, u=None):
        return -x + 0.3 * jnp.tanh(x) + jnp.sin(t)

    x0 = jnp.array([0.7, -0.4])
    guess = jnp.zeros((N, 2))
    sol_p, _, it_p = _plain(ode, x0, guess, dt)
    sol, res, it, hist = _damped(ode, x0, guess, dt)
    assert np.all(hist["gamma"] == 0.0) and np.all(hist["n_trials"] == 1)
    assert abs(it - it_p) <= 1
    assert np.max(np.abs(sol - sol_p)) < 1e-10



def test_lle_first_mode_damps_from_the_first_iteration():
    dt, N = 0.05, 400
    x0 = jnp.array([1.5])
    truth = _truth(cubic, x0, dt, N)
    guess = jnp.zeros((N, 1))
    sol, res, it, hist = _damped(cubic, x0, guess, dt, propose_first="lle")
    assert res < TOL and _err(sol, truth) < 1e-8 and it <= 12
    assert hist["gamma"][0] > 0.9 and hist["n_trials"][0] == 1   # ReLU(0.976 + eps), accepted at once
    assert hist["gamma"][-1] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
