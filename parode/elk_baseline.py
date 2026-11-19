# Third-party origin: the ELK update and the diagonal filter follow elk/algs/elk.py and
# elk/utils/parallel_kalman_scalar.py of https://github.com/lindermanlab/elk
#     Copyright (c) 2024, Linderman Lab
#     BSD 3-Clause License
# and the dense filter elements and their combine follow
# dynamax/linear_gaussian_ssm/parallel_inference.py of https://github.com/probml/dynamax
#     Copyright (c) 2022 Probabilistic machine learning
#     MIT License
# Licence texts and list of changes: THIRD_PARTY_NOTICES.md at the repository root, sections "ELK"
# and "dynamax".
"""ELK baseline (Gonzalez, Warrington, Smith, Linderman, NeurIPS 2024) for parode's Euler DEER.

ELK realizes Levenberg-Marquardt damping of the DEER Newton step as a parallel Kalman
filter on the linearized recurrence. This module follows the released code
(elk/algs/elk.py, function elk_alg) exactly: the linear-Gaussian state-space model has
dynamics x_{t+1} = F_t x_t + b_t + w_t with w_t ~ N(0, I), emissions y_t = x_t + v_t with
v_t ~ N(0, sigma^2 I), observations y = the current iterate, initial mean g(x_0) with
covariance I, and the new iterate is the vector of FILTERED means (the released code runs
no smoother pass). lambda_LM = 1 / sigma^2; sigma^2 -> infinity recovers the DEER step,
sigma^2 -> 0 pins the iterate to itself. The filter elements and their associative combine
are those of Sarkka and Garcia-Fernandez (2021) as implemented in dynamax's
parallel_inference.py (MIT); the diagonal version is the scalar filter shipped with ELK,
vectorized over coordinates. Process noise is the identity regardless of dt, as in ELK.

In parode's conventions (X excludes x_0, c = -r, F_t = dg/dx at X_{t-1}):
    m_0 = g(x_0) = c_0 + X_0,   b_t = g(X_t) - F_t X_t = c_{t+1} + X_{t+1} - F_t X_t,   y = X.
"""
import jax.numpy as jnp
from jax import lax

from parode.explicit_parallel import affine_terms, euler_step, linear_terms
from parode.ode_utils import prepare_ode_function
from parode.quasi_parallel import linear_terms as linear_terms_diag


# ----------------------------------------------------------------------------- dense filter

def _dense_elements(F, b, m0, y, sigmasq):
    """Filtering elements (A, b, C, J, eta) for the LGSSM with Q = I, H = I, R = sigma^2 I, P0 = I."""
    N, D = y.shape
    eye = jnp.eye(D, dtype=y.dtype)
    kappa = 1.0 / (1.0 + sigmasq)                     # Kalman gain of a unit-covariance prior
    # first element: prior N(m0, I) updated with y_0
    A0 = jnp.zeros((1, D, D), y.dtype)
    b0 = (m0 + kappa * (y[0] - m0))[None]
    C0 = ((1.0 - kappa) * eye)[None]
    J0 = jnp.zeros((1, D, D), y.dtype)
    eta0 = jnp.zeros((1, D), y.dtype)
    # generic elements: dynamics t-1 -> t, observation y_t
    At = (1.0 - kappa) * F
    bt = b + kappa * (y[1:] - b)
    Ct = jnp.broadcast_to((1.0 - kappa) * eye, (N - 1, D, D))
    Jt = kappa * jnp.einsum("tji,tjk->tik", F, F)                       # F^T F
    etat = kappa * jnp.einsum("tji,tj->ti", F, y[1:] - b)               # F^T (y - b)
    return (jnp.concatenate([A0, At]), jnp.concatenate([b0, bt]), jnp.concatenate([C0, Ct]),
            jnp.concatenate([J0, Jt]), jnp.concatenate([eta0, etat]))


def _dense_combine(e1, e2):
    """Associative combine of filtering elements, earlier element first (dynamax _filtering_operator)."""
    A1, b1, C1, J1, eta1 = e1
    A2, b2, C2, J2, eta2 = e2
    D = A1.shape[-1]
    eye = jnp.eye(D, dtype=A1.dtype)
    temp = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(eye + C1 @ J2, -1, -2), jnp.swapaxes(A2, -1, -2)), -1, -2)
    A = temp @ A1
    b = jnp.einsum("...ij,...j->...i", temp, b1 + jnp.einsum("...ij,...j->...i", C1, eta2)) + b2
    C = temp @ C1 @ jnp.swapaxes(A2, -1, -2) + C2
    temp = jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(eye + J2 @ C1, -1, -2), A1), -1, -2)
    eta = jnp.einsum("...ij,...j->...i", temp, eta2 - jnp.einsum("...ij,...j->...i", J2, b1)) + eta1
    J = temp @ J2 @ A1 + J1
    return A, b, C, J, eta


def elk_filter_dense(F, b, m0, y, sigmasq):
    """Filtered means (N, D) of the ELK state-space model; F (N-1, D, D), b (N-1, D), m0 (D,), y (N, D)."""
    elems = _dense_elements(F, b, m0, y, sigmasq)
    _, means, _, _, _ = lax.associative_scan(_dense_combine, elems)
    return means


# ----------------------------------------------------------------------------- diagonal filter

def _diag_elements(F, b, m0, y, sigmasq):
    N, D = y.shape
    kappa = 1.0 / (1.0 + sigmasq)
    A0 = jnp.zeros((1, D), y.dtype)
    b0 = (m0 + kappa * (y[0] - m0))[None]
    C0 = jnp.full((1, D), 1.0 - kappa, y.dtype)
    J0 = jnp.zeros((1, D), y.dtype)
    eta0 = jnp.zeros((1, D), y.dtype)
    At = (1.0 - kappa) * F
    bt = b + kappa * (y[1:] - b)
    Ct = jnp.full((N - 1, D), 1.0 - kappa, y.dtype)
    Jt = kappa * F * F
    etat = kappa * F * (y[1:] - b)
    return (jnp.concatenate([A0, At]), jnp.concatenate([b0, bt]), jnp.concatenate([C0, Ct]),
            jnp.concatenate([J0, Jt]), jnp.concatenate([eta0, etat]))


def _diag_combine(e1, e2):
    A1, b1, C1, J1, eta1 = e1
    A2, b2, C2, J2, eta2 = e2
    temp = A2 / (1.0 + C1 * J2)
    A = temp * A1
    b = temp * (b1 + C1 * eta2) + b2
    C = temp * C1 * A2 + C2
    temp = A1 / (1.0 + J2 * C1)
    eta = temp * (eta2 - J2 * b1) + eta1
    J = temp * J2 * A1 + J1
    return A, b, C, J, eta


def elk_filter_diag(F, b, m0, y, sigmasq):
    """Filtered means (N, D) with diagonal dynamics F (N-1, D): ELK's scalar filter per coordinate."""
    elems = _diag_elements(F, b, m0, y, sigmasq)
    _, means, _, _, _ = lax.associative_scan(_diag_combine, elems)
    return means


# ----------------------------------------------------------------------------- reference

def kalman_filter_sequential(F, b, m0, y, sigmasq):
    """Textbook sequential Kalman filter for the same model (dense), for testing."""
    D = y.shape[1]
    eye = jnp.eye(D, dtype=y.dtype)
    R = sigmasq * eye

    def update(m_pred, P_pred, yt):
        S = P_pred + R
        K = jnp.linalg.solve(S.T, P_pred.T).T
        return m_pred + K @ (yt - m_pred), P_pred - K @ P_pred

    m, P = update(m0, eye, y[0])
    means = [m]
    for t in range(1, y.shape[0]):
        m_pred = F[t - 1] @ m + b[t - 1]
        P_pred = F[t - 1] @ P @ F[t - 1].T + eye
        m, P = update(m_pred, P_pred, y[t])
        means.append(m)
    return jnp.stack(means)


# ----------------------------------------------------------------------------- solver

def elk_step(X, x0, ode_fn, dt, times, sigmasq, quasi=False):
    """One ELK iteration on parode's explicit Euler recurrence. Returns (X_new, max|r(X_new)|).

    quasi: the diagonal of the step Jacobians (parode.quasi_parallel.linear_terms) and the diagonal filter.
    """
    c = affine_terms(X, x0, ode_fn, dt, euler_step, times, None)
    m0 = c[0] + X[0]
    if quasi:
        F = linear_terms_diag(X, ode_fn, dt, euler_step, times, None)
        b = c[1:] + X[1:] - F * X[:-1]
        X_new = elk_filter_diag(F, b, m0, X, sigmasq)
    else:
        F = linear_terms(X, ode_fn, dt, euler_step, times, None)
        b = c[1:] + X[1:] - jnp.einsum("tij,tj->ti", F, X[:-1])
        X_new = elk_filter_dense(F, b, m0, X, sigmasq)
    c_new = affine_terms(X_new, x0, ode_fn, dt, euler_step, times, None)
    return X_new, jnp.max(jnp.abs(c_new))


def elk_integrate(ode, x0, guess, dt, sigmasq, tol, max_iter, quasi=False):
    """ELK with a fixed sigma^2 (lambda_LM = 1/sigma^2), stopping on the residual like the other
    baselines of the benchmark. Returns (trajectory with x0, residual, iterations)."""
    ode_fn = prepare_ode_function(ode)
    N = guess.shape[0]
    times = dt * jnp.arange(N + 1)

    def body(val):
        X, _, it = val
        X_new, res = elk_step(X, x0, ode_fn, dt, times, sigmasq, quasi)
        return X_new, res, it + 1

    def cond(val):
        _, res, it = val
        return (res > tol) & (it < max_iter)

    X, res, it = lax.while_loop(cond, body, (guess, jnp.array(jnp.inf, guess.dtype), 0))
    return jnp.vstack((x0, X)), res, it
