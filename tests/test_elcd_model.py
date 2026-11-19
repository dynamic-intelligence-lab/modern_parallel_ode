"""Tests of experiments/elcd/model.py (CPU, fp64).

The contraction identity sym A(z) <= -alpha I at random points for several gains and dimensions, f(z*) = 0 and
J(z*) = A(z*), J = A + M against forward-mode AD and central differences, the settling rate of `init_synthetic`
and `cast`. Every assert states its tolerance.
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

from experiments.elcd.model import (  # noqa: E402
    ALPHA, HIDDEN, A_matrix, cast, curvature_matrix, field, init_synthetic, jacobian,
)


def _params(seed, D, gain, rate=1.0, **kw):
    return cast(init_synthetic(seed, D, gain=gain, rate=rate, **kw), jnp.float64)


def _np(x):
    return np.asarray(x, dtype=np.float64)


# ----------------------------------------------------------------------------- the field

@pytest.mark.parametrize("D", [2, 5])
@pytest.mark.parametrize("gain", [1.0, 4.0, 8.0])
def test_sym_A_is_at_most_minus_alpha_at_random_points(D, gain):
    """A + A^T <= -2 alpha I (the ELCD constraint of Jaffe et al. 2024): the largest eigenvalue of sym A(z) is
    at most -alpha at 100 random points of scale 3 per coordinate. Tolerance 1e-12, the rounding of P_s^T P_s."""
    params = _params(0, D, gain)
    zs = jnp.asarray(np.random.default_rng(D * 10 + int(gain)).normal(size=(100, D)) * 3.0)
    A = _np(jax.jit(jax.vmap(lambda z: A_matrix(params, z)))(zs))
    lam_max = np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, 1, 2)))[:, -1]
    assert np.all(lam_max <= -ALPHA + 1e-12), f"max eig sym A = {lam_max.max():.3e} above -alpha (gain {gain}, D {D})"


@pytest.mark.parametrize("z_star", [None, np.array([0.3, -1.2, 0.5])])
def test_field_vanishes_and_jacobian_equals_A_at_the_equilibrium(z_star):
    """f(z*) = A(z*) (z* - z*) = 0 exactly; J(z*) = A(z*) and M(z*) = 0 to 1e-12."""
    params = _params(1, 3, 4.0, z_star=z_star)
    f_star = _np(field(params, params.z_star))
    assert np.max(np.abs(f_star)) <= 1e-15
    J, A, M = (_np(g(params, params.z_star)) for g in (jacobian, A_matrix, curvature_matrix))
    assert np.max(np.abs(J - A)) <= 1e-12
    assert np.max(np.abs(M)) <= 1e-12


@pytest.mark.parametrize("gain", [1.0, 4.0])
def test_jacobian_is_A_plus_curvature_and_matches_central_differences(gain):
    """J = A + M to 1e-12 (M is defined as J - A, so this is the rounding of the subtraction); J against central
    differences of f with h = 1e-5 to 1e-6 (truncation h^2 f''' / 6 of order 1e-10 at gain 4, rounding 1e-10);
    the columns of M against central differences of A applied to z - z*, the same tolerance."""
    D = 3
    params = _params(2, D, gain)
    z = jnp.asarray(np.array([0.7, -0.4, 0.9]))
    J, A, M = (_np(g(params, z)) for g in (jacobian, A_matrix, curvature_matrix))
    assert np.max(np.abs(J - (A + M))) <= 1e-12
    h = 1e-5
    f = jax.jit(lambda y: field(params, y))
    Af = jax.jit(lambda y: A_matrix(params, y))
    v = _np(z - params.z_star)
    J_fd = np.stack([_np(f(z + h * e) - f(z - h * e)) / (2 * h) for e in np.eye(D)], axis=1)
    M_fd = np.stack([(_np(Af(z + h * e) - Af(z - h * e)) / (2 * h)) @ v for e in np.eye(D)], axis=1)
    assert np.max(np.abs(J - J_fd)) <= 1e-6, f"max |J - J_fd| = {np.max(np.abs(J - J_fd)):.3e}"
    assert np.max(np.abs(M - M_fd)) <= 1e-6, f"max |M - M_fd| = {np.max(np.abs(M - M_fd)):.3e}"
    assert np.max(np.abs(M)) > 1e-3                      # the curvature is not trivially zero away from z*


# ----------------------------------------------------------------------------- instances

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_init_synthetic_rate_sets_the_settling_rate(seed):
    """-max Re eig A(z*) is within 30 percent of alpha + rate^2 at rate 1, D 4, for gains 1 and 4 (the gain scales
    W1, which does not enter at z* = 0): P_s(z*) is rate I plus xavier-scale cross terms of about 0.02, so
    G = P_s^T P_s is within a few percent of I and Bendixson puts every Re eig A in [-alpha - lambda_max G,
    -alpha - lambda_min G]. The same bound gives settling >= alpha for every instance (tolerance 1e-12) and, at
    rate 0, settling <= alpha + 0.05 (G is then of order 1e-3)."""
    D, rate = 4, 1.0
    target = ALPHA + rate ** 2
    for gain in (1.0, 4.0):
        params = _params(seed, D, gain, rate=rate)
        settling = -np.max(np.linalg.eigvals(_np(A_matrix(params, params.z_star))).real)
        assert abs(settling - target) <= 0.3 * target, f"settling rate {settling:.4f} vs alpha + r^2 = {target}"
        assert settling >= ALPHA - 1e-12
    rate0 = _params(seed, D, 1.0, rate=0.0)
    settling0 = -np.max(np.linalg.eigvals(_np(A_matrix(rate0, rate0.z_star))).real)
    assert ALPHA - 1e-12 <= settling0 <= ALPHA + 0.05, f"rate-0 settling rate {settling0:.5f}"


def test_init_synthetic_gain_scales_W1_only_and_is_deterministic():
    p1, p4 = init_synthetic(0, 3, gain=1.0, rate=1.0), init_synthetic(0, 3, gain=4.0, rate=1.0)
    assert np.max(np.abs(p4.W1s - 4.0 * p1.W1s)) <= 1e-15 and np.max(np.abs(p4.W1a - 4.0 * p1.W1a)) <= 1e-15
    assert np.array_equal(p4.W2s, p1.W2s) and np.array_equal(p4.W2a, p1.W2a)
    assert np.array_equal(p1.b2s, np.full(9, 0.01) + np.eye(3).ravel())       # rate vec(I) on b2 of P_s
    assert np.array_equal(p1.b2a, np.full(9, 0.01))
    assert all(np.array_equal(a, b) for a, b in zip(init_synthetic(0, 3, gain=1.0, rate=1.0), p1))
    assert p1.W1s.shape == (HIDDEN, 3) and p1.W2s.shape == (9, HIDDEN) and p1.z_star.shape == (3,)


def test_cast_sets_every_leaf_dtype_and_keeps_shapes_and_values():
    p = init_synthetic(0, 3, gain=1.0)
    for dtype in (jnp.float32, jnp.float64):
        q = cast(p, dtype)
        assert all(leaf.dtype == dtype for leaf in q), [leaf.dtype for leaf in q]
        assert all(leaf.shape == np.shape(orig) for leaf, orig in zip(q, p))
        assert q.D == 3
    assert np.array_equal(_np(cast(p, jnp.float64).W2s), p.W2s)
    assert np.max(np.abs(_np(cast(p, jnp.float32).W2s) - p.W2s)) <= 1e-6      # fp32 rounding of entries <= 0.5
