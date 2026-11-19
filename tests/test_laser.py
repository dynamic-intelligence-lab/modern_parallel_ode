"""Tests for parode.laser primitives (CPU, seconds)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.laser import (  # noqa: E402
    lle_dense, safe_lognorm, scale_jacobians,
)
from ode_problems import estimate_lle_from_vector  # noqa: E402


def _random_jacobians(key, T, D, scale=0.6, dtype=jnp.float64):
    return (jnp.eye(D)[None] + scale * jax.random.normal(key, (T, D, D))).astype(dtype)


def _brute_lle(F, dt):
    P = np.eye(F.shape[-1])
    for Ft in np.asarray(F):
        P = Ft @ P
    return np.log(np.linalg.norm(P, 2)) / (F.shape[0] * dt)


def test_lle_dense_agrees_with_sequential_estimator():
    dt = 0.05
    F = _random_jacobians(jax.random.PRNGKey(1), 512, 6)
    lam_scan = float(lle_dense(F, dt))
    lam_brute = _brute_lle(F, dt)
    v = jax.random.normal(jax.random.PRNGKey(2), (6,))
    lam_seq = float(estimate_lle_from_vector(F, v, dt))
    assert abs(lam_scan - lam_brute) < 1e-8
    # a single random start vector under-estimates the top exponent by log(alignment)/(N dt)
    assert lam_seq <= lam_scan + 1e-8 and abs(lam_scan - lam_seq) < 0.2


def test_lle_dense_is_overflow_free_in_float32():
    F = (3.0 * jnp.eye(3)[None] + 0.01 * jax.random.normal(jax.random.PRNGKey(3), (512, 3, 3))).astype(jnp.float32)
    lam = float(lle_dense(F, 1.0))
    assert np.isfinite(lam)
    assert abs(lam - np.log(3.0)) < 0.01


def test_lle_dense_gradient_matches_finite_differences():
    dt = 0.1
    F = _random_jacobians(jax.random.PRNGKey(4), 32, 3)
    g = jax.grad(lambda F_: lle_dense(F_, dt))(F)
    v = jax.random.normal(jax.random.PRNGKey(5), F.shape)
    v = v / jnp.linalg.norm(v)
    h = 1e-6
    fd = (float(lle_dense(F + h * v, dt)) - float(lle_dense(F - h * v, dt))) / (2 * h)
    assert abs(float(jnp.sum(g * v)) - fd) < 1e-6


def test_exact_shift_under_scaling():
    dt = 0.05
    F = _random_jacobians(jax.random.PRNGKey(6), 128, 4)
    for gamma in (0.5, 2.0):
        lam = float(lle_dense(F, dt))
        lam_scaled = float(lle_dense(scale_jacobians(F, jnp.array(gamma), dt), dt))
        assert abs(lam_scaled - (lam - gamma)) < 1e-9


def test_safe_lognorm_handles_overflow_and_zero():
    assert float(safe_lognorm(jnp.zeros((3, 3)))) == -np.inf
    assert float(safe_lognorm(jnp.full((3, 3), jnp.inf))) == np.inf
    P = jax.random.normal(jax.random.PRNGKey(10), (4, 4))
    # 16 power-iteration steps: relative error (sigma_2/sigma_1)^32 on sigma_max, about 1e-3 for a random
    # 4x4; the exponent needs only log(1+err)/(N dt), far below the 0.5 log(D)/(N dt) Frobenius bias
    assert abs(float(safe_lognorm(P)) - np.log(np.linalg.norm(np.asarray(P), 2))) < 1e-2
    assert abs(float(safe_lognorm(jnp.array([1.0, -3.0, 2.0]))) - np.log(3.0)) < 1e-12



def test_scaled_product_matches_scan_and_brute_force():
    from parode.laser import scaled_product
    for T in (1, 2, 7, 64, 129):
        F = _random_jacobians(jax.random.PRNGKey(T), T, 3)
        M_hat, log_s = scaled_product(F)
        P = np.eye(3)
        for Ft in np.asarray(F):
            P = Ft @ P
        assert np.allclose(np.asarray(M_hat) * np.exp(float(log_s)), P, rtol=1e-10, atol=1e-10), T
    F32 = (3.0 * jnp.eye(3)[None] + 0.01 * jax.random.normal(jax.random.PRNGKey(3), (600, 3, 3))).astype(jnp.float32)
    M_hat, log_s = scaled_product(F32)
    assert np.isfinite(float(log_s)) and abs(float(log_s) / 600 - np.log(3.0)) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
