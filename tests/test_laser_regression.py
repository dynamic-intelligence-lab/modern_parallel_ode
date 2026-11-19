"""Regression tests of parode.laser in explicit_par_integrate, over the earlier defaults (CPU)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.explicit_parallel import explicit_par_integrate  # noqa: E402
from ode_problems import earlier_laser  # noqa: E402

TOL = 1e-9


def cubic(t, x, u=None):
    return x - x ** 3


def _hist(hist, it):
    return {k: np.asarray(v)[: int(it)] for k, v in hist.items()}


def test_linear_growth_exponent_read_exactly():
    def ode(t, x, u=None):
        return x

    cfg = earlier_laser(return_history=True)
    _, res, it, hist = explicit_par_integrate(ode, jnp.array([1.0]), jnp.zeros((100, 1)), 0.05, 20, "euler",
                                               tol=TOL, laser=cfg)
    assert int(it) <= 2                                     # linear problem: one Newton step plus rounding cleanup
    assert abs(float(hist["lambda_h"][0]) - np.log(1.05) / 0.05) < 1e-9


def test_lle_first_on_the_cubic_zero_guess():
    """propose_first="lle" on the cubic zero guess in explicit_par_integrate's Euler step, the library's one caller
    of run_damped_newton: converged within 8 iterations, every first trial accepted, and the first damping is
    ReLU(lambda_h(0) + eps) = log(1.05) / dt + 1 / (N dt)."""
    N, dt = 400, 0.05
    x0, guess = jnp.array([1.5]), jnp.zeros((400, 1))
    cfg = earlier_laser(propose_first="lle", return_history=True)
    g_euler = np.log(1.05) / dt + 1.0 / (N * dt)            # ReLU(lambda_h(0) + eps) for Euler
    sol, res, it, hist = explicit_par_integrate(cubic, x0, guess, dt, 60, "euler", tol=TOL, laser=cfg)
    h = _hist(hist, it)
    assert float(res) < TOL and int(it) <= 8, int(it)
    assert np.all(h["n_trials"] == 1)                       # LLE-informed first trials all accepted
    assert abs(h["gamma"][0] - g_euler) < 1e-6, (h["gamma"][0], g_euler)


def test_growth_cap_bounds_the_steps():
    """A tanh RNN from the zero guess: the growth cap rejects at least one trial and every accepted step stays
    below 50 in max norm."""
    D, N, dt = 8, 128, 0.1
    W = 2.5 / np.sqrt(D) * jax.random.normal(jax.random.PRNGKey(3), (D, D))

    def ode(t, x, u=None):
        return -x + jnp.tanh(W @ x)

    x0 = 2.0 * jax.random.normal(jax.random.PRNGKey(9), (D,))
    cfg = earlier_laser(return_history=True)
    sol, res, it, hist = explicit_par_integrate(ode, x0, jnp.zeros((N, D)), dt, 60, "euler", tol=TOL, laser=cfg)
    assert float(res) < TOL and np.all(np.isfinite(np.asarray(sol)))
    h = _hist(hist, it)
    assert np.max(h["step_inf"]) < 50.0 and np.sum(h["n_trials"] - 1) >= 1


def test_contractive_high_dimensional_linear_is_not_damped_in_lle_mode():
    D, N, dt = 8, 128, 0.1

    def ode(t, x, u=None):
        return -0.05 * x

    x0 = jnp.ones(D)
    lam_true = np.log(1.0 - 0.05 * dt) / dt                  # -0.0501; a Frobenius by-product reported +0.032
    cfg = earlier_laser(propose_first="lle", return_history=True)
    _, res, it, hist = explicit_par_integrate(ode, x0, jnp.zeros((N, D)), dt, 60, "euler", tol=TOL, laser=cfg)
    h = _hist(hist, it)
    assert np.all(np.abs(h["lambda_h"] - lam_true) < 1e-6)   # exponent reported without dimension bias
    # eps = 1/T = 0.078 exceeds |lambda| here, so the lle rule damps by design; plain-first does not
    assert np.all(np.abs(h["gamma"] - max(lam_true + 1.0 / (N * dt), 0.0)) < 1e-6)
    _, res_p, it_p, hist_p = explicit_par_integrate(ode, x0, jnp.zeros((N, D)), dt, 60, "euler", tol=TOL,
                                                    laser=earlier_laser(return_history=True))
    assert int(it_p) <= 2 and np.all(_hist(hist_p, it_p)["gamma"] == 0.0)


def test_jacobi_fallback_does_not_trap_the_loop():
    x0, dt = jnp.array([1.5]), 0.1
    cfg = earlier_laser(n_retry=0, return_history=True)         # Jacobi only
    sol, res, it, hist = explicit_par_integrate(cubic, x0, jnp.zeros((128, 1)), dt, 300, "euler", tol=TOL, laser=cfg)
    h = _hist(hist, it)
    assert int(it) <= 129 and float(res) < TOL
    assert not np.any(np.isinf(h["lambda_h"]))
    # monotone growth cap forces fallbacks; must still converge quickly and not repeat gamma = inf trials
    cfg2 = earlier_laser(growth_cap=1.0, n_retry=3, propose_first="lle", return_history=True)
    _, res2, it2, hist2 = explicit_par_integrate(cubic, jnp.array([1e-2]), jnp.zeros((400, 1)), 0.05, 300, "euler",
                                                  tol=TOL, laser=cfg2)
    h2 = _hist(hist2, it2)
    assert int(it2) < 300 and float(res2) < TOL               # a monotone cap degenerates toward Jacobi (about 90 it)
    assert np.all(h2["n_trials"] <= 4)                        # at most n_retry damped trials, then one Jacobi step
    assert not np.any(np.isinf(h2["lambda_h"]))


def test_max_iter_zero_and_rounding_floor_exit():
    x0, guess, dt = jnp.array([1.5]), jnp.zeros((400, 1)), 0.05
    sol, res, it = explicit_par_integrate(cubic, x0, guess, dt, 0, "euler", tol=TOL, laser=earlier_laser())
    assert int(it) == 0 and np.array_equal(np.asarray(sol)[1:], np.asarray(guess))
    # unreachable tolerance: stop at the rounding floor instead of burning max_iter
    cfg = earlier_laser(return_history=True)
    sol, res, it, hist = explicit_par_integrate(cubic, x0, guess, dt, 60, "euler", tol=1e-30, laser=cfg)
    assert int(it) < 20 and float(res) < 1e-13



def test_power_iteration_spectral_norm():
    from parode.laser import spectral_norm
    key = jax.random.PRNGKey(5)
    P = jax.random.normal(key, (16, 16))
    assert abs(float(spectral_norm(P)) / np.linalg.norm(np.asarray(P), 2) - 1.0) < 1e-2      # 16 steps, random 16x16
    Q = np.linalg.qr(np.asarray(jax.random.normal(key, (6, 6))))[0] * 3.0     # isometry: all singular values equal
    assert abs(float(spectral_norm(jnp.asarray(Q))) - 3.0) < 1e-9
    Fb = jnp.eye(4)[None] + 0.3 * jax.random.normal(key, (10, 4, 4))
    ref = np.array([np.linalg.norm(np.asarray(f), 2) for f in Fb])
    assert np.max(np.abs(np.asarray(spectral_norm(Fb)) / ref - 1.0)) < 1e-2


def test_plain_first_recovers_at_long_horizon_from_overflowed_scan():
    # lambda N dt = 0.976 * 1000 > log(FLT_MAX64): the plain trial overflows and its by-product is inf
    x0, dt, N = jnp.array([1.5]), 0.05, 20000
    cfg = earlier_laser(return_history=True)
    sol, res, it, hist = explicit_par_integrate(cubic, x0, jnp.zeros((N, 1)), dt, 40, "euler", tol=TOL, laser=cfg)
    h = _hist(hist, it)
    assert float(res) < TOL and int(it) <= 12, int(it)
    assert h["n_trials"][0] <= 3                            # exact exponent recovered on the first rejection
    assert abs(h["gamma"][0] - (np.log(1.05) / dt + 1.0 / (N * dt))) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
