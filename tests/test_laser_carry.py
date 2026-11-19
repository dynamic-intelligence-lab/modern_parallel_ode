"""LaserConfig(exponent="carry") and the retry path's lam_known (CPU, fp64, under a minute).

The tests call run_damped_newton directly, on the explicit Euler terms of tests/ode_problems.euler_terms, so that its
iterate X (without x0), n_retries and history are seen as the loop returns them.
"""
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode import laser as laser_module  # noqa: E402
from parode.laser import LaserConfig, run_damped_newton  # noqa: E402
from ode_problems import earlier_laser, euler_terms  # noqa: E402

TOL = 1e-9
MAX_ITER = 60
# the damping proposed first, with the exact and with the carried exponent; the rest as in the earlier defaults
EXACT = earlier_laser(propose_first="lle", exponent="exact")
CARRY = earlier_laser(propose_first="lle", exponent="carry")


def _solve(ode, x0, guess, dt, cfg):
    """run_damped_newton on the explicit Euler terms with the configuration cfg (its history set here); returns
    (X, residual, iterations, n_retries, history)."""
    terms_fn, residual_fn, scan_fn = euler_terms(ode, x0, dt, guess.shape[0])
    cfg = replace(cfg, return_history=True)
    return run_damped_newton(guess, terms_fn, scan_fn, dt, cfg, TOL, MAX_ITER, residual_fn=residual_fn)


def _solve_np(ode, x0, guess, dt, cfg):
    X, res, it, nr, hist = _solve(ode, x0, guess, dt, cfg)
    it = int(it)
    return np.asarray(X), float(res), it, int(nr), {k: np.asarray(v)[:it] for k, v in hist.items()}


def cubic(t, x, u=None):
    return x - x ** 3


def contractive(t, x, u=None):
    return -x + 0.3 * jnp.tanh(x) + jnp.sin(t)


A_EXP = jnp.array([[0.5, 1.0], [0.0, 0.3]])        # expansive linear system: every iterate has the same exponent


def expansive_linear(t, x, u=None):
    return A_EXP @ x


def test_carry_is_bit_for_bit_exact_on_a_contractive_problem():
    """T1: gamma 0 on every iteration, so the carried exponent never enters the iterates."""
    dt, N = 0.1, 100
    x0, guess = jnp.array([0.7, -0.4]), jnp.zeros((N, 2))
    X_e, res_e, it_e, nr_e, h_e = _solve_np(contractive, x0, guess, dt, EXACT)
    X_c, res_c, it_c, nr_c, h_c = _solve_np(contractive, x0, guess, dt, CARRY)
    assert res_e < TOL and np.all(h_e["gamma"] == 0.0) and np.all(h_e["lambda_h"] + 1.0 / (N * dt) < 0)
    assert (it_e, nr_e) == (it_c, nr_c)
    assert res_e == res_c and np.array_equal(X_e, X_c)
    assert np.array_equal(h_e["residual"], h_c["residual"]) and np.array_equal(h_e["n_trials"], h_c["n_trials"])
    assert np.all(h_c["lambda_h"] == h_c["lambda_h"][0])   # the guess's exponent, carried


def test_carry_converges_like_exact_on_an_expansive_linear_guess():
    """T2: the exact mode damps on every iteration (constant Jacobians); carry matches it within one."""
    dt, N = 0.05, 200
    x0, guess = jnp.array([1.0, 1.0]), jnp.zeros((N, 2))
    X_e, res_e, it_e, nr_e, h_e = _solve_np(expansive_linear, x0, guess, dt, EXACT)
    X_c, res_c, it_c, nr_c, h_c = _solve_np(expansive_linear, x0, guess, dt, CARRY)
    assert res_e < TOL and res_c < TOL
    assert np.max(h_e["gamma"]) > 0 and np.max(h_c["gamma"]) > 0
    assert abs(it_e - it_c) <= 1 and abs(nr_e - nr_c) <= 1
    assert np.max(np.abs(X_e - X_c)) < 1e-9


def test_carry_on_a_cold_cubic_guess_converges_but_keeps_the_guess_damping():
    """T2 on the cubic double well from the zero guess (the case the damping tests use): nothing is rejected,
    so carry never refreshes the guess's expansive exponent (0.976) and damps with gamma 1.03 on every
    iteration; it converges, at 16 iterations against exact's 7 (CPU fp64). Recorded as measured."""
    dt, N = 0.05, 400
    x0, guess = jnp.array([1.5]), jnp.zeros((N, 1))
    X_e, res_e, it_e, nr_e, h_e = _solve_np(cubic, x0, guess, dt, EXACT)
    X_c, res_c, it_c, nr_c, h_c = _solve_np(cubic, x0, guess, dt, CARRY)
    assert res_e < TOL and res_c < TOL and np.max(np.abs(X_e - X_c)) < 1e-7
    assert np.max(h_e["gamma"]) > 0 and np.max(h_c["gamma"]) > 0
    assert nr_e == 0 and nr_c == 0
    assert np.all(h_c["lambda_h"] == h_c["lambda_h"][0]) and np.all(h_c["gamma"] == h_c["gamma"][0])
    assert h_e["gamma"][-1] == 0.0 and h_c["gamma"][-1] > 0
    assert it_e <= 8 and it_c <= 3 * it_e, (it_e, it_c)


def test_lam_known_changes_no_number_in_exact_mode(monkeypatch):
    """T3: the paper's configuration (the defaults) on the cubic zero guess has one retry; with the retry's guarded
    recomputation forced back on (lam_known False) every output is bit for bit the same."""
    dt, N = 0.05, 400
    x0, guess = jnp.array([1.5]), jnp.zeros((N, 1))
    ref = _solve_np(cubic, x0, guess, dt, LaserConfig())
    assert ref[3] >= 1 and np.max(ref[4]["n_trials"]) >= 2        # the retry body runs
    orig = laser_module._retry_gamma
    calls = []

    def forced(*args, **kw):
        calls.append(kw.get("lam_known", args[11] if len(args) > 11 else False))
        args = args[:11]
        kw = {k: v for k, v in kw.items() if k != "lam_known"}
        return orig(*args, lam_known=False, **kw)

    monkeypatch.setattr(laser_module, "_retry_gamma", forced)
    out = _solve_np(cubic, x0, guess, dt, LaserConfig())
    assert calls and all(calls)                                   # the body asked for lam_known=True
    assert np.array_equal(ref[0], out[0]) and ref[1] == out[1] and ref[2:4] == out[2:4]
    for k in ref[4]:
        assert np.array_equal(ref[4][k], out[4][k], equal_nan=True), k


def test_carry_under_vmap_converges():
    """T4: the training path vmaps the solve over a chunk of rollouts."""
    dt, N = 0.05, 200
    x0s = jnp.array([[1.5], [1.2], [-1.4], [0.8]])
    guess = jnp.zeros((N, 1))
    X, res, it, nr, hist = jax.vmap(lambda x0: _solve(cubic, x0, guess, dt, CARRY))(x0s)
    X_e, res_e, it_e, _, _ = jax.vmap(lambda x0: _solve(cubic, x0, guess, dt, EXACT))(x0s)
    assert X.shape == (4, N, 1) and np.all(np.isfinite(np.asarray(X)))
    assert np.all(np.asarray(res) < TOL) and np.all(np.asarray(it) < MAX_ITER)
    assert np.max(np.abs(np.asarray(X) - np.asarray(X_e))) < 1e-7


def test_carry_needs_lle_first_and_uniform_damping():
    """T5"""
    with pytest.raises(ValueError):
        LaserConfig(exponent="carry", propose_first="plain")
    with pytest.raises(ValueError):
        LaserConfig(exponent="stale")
    LaserConfig(exponent="carry", propose_first="lle")
