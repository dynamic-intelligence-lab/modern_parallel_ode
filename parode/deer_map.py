# Third-party origin: adapted from src/qdeer.py and src/deer.py of
# https://github.com/lindermanlab/parallel-mcmc
#     Copyright (c) 2025, Linderman Lab
#     BSD 3-Clause License
# which are adapted from https://github.com/machine-discovery/deer (commit 17b0b62)
#     Copyright (c) 2023, Machine Discovery Ltd
#     BSD 3-Clause License
# Licence texts and list of changes: THIRD_PARTY_NOTICES.md at the repository root, sections
# "parallel-mcmc" and "DEER".
"""
DEER solver for general discrete maps.

Implements quasi-Newton DEER iteration for arbitrary discrete maps
    y[n+1] = func(y[n], driver[n], params)
using diagonal Jacobian approximation via Hutchinson JVP and parallel
prefix scan.

Adapted from parallel-mcmc's qdeer.py (Xavier Gonzalez, 2024) to reuse
parode's existing scan primitives.
"""

from typing import Callable, Any, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from parode.quasi_parallel import combine_fc
from parode.explicit_parallel import combine_fc as combine_fc_dense


# ---------------------------------------------------------------------------
# 1. Fused f(x) + diag(J) via Hutchinson JVP
# ---------------------------------------------------------------------------

def quasi_diag_estimator(
    state: jnp.ndarray,
    driver: Any,
    params: Any,
    func: Callable,
    key: jnp.ndarray,
    num_probes: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate func value and diagonal of Jacobian w.r.t. state in one pass.

    Uses Rademacher random vectors and JVP to fuse the primal evaluation
    with a Hutchinson diagonal estimator.

    Parameters
    ----------
    state : (D,) current state
    driver : per-step driver (e.g. PRNG key)
    params : arbitrary params forwarded to func
    func : (state, driver, params) -> next_state
    key : PRNG key for Rademacher vectors
    num_probes : number of Hutchinson probes (default 1)

    Returns
    -------
    f_val : (D,) func(state, driver, params)
    jac_diag : (D,) approximate diagonal of d func / d state
    """
    z_rad = jr.rademacher(key, (num_probes, state.shape[0])).astype(state.dtype)

    def deer_jvp(v):
        return jax.jvp(lambda s: func(s, driver, params), (state,), (v,))

    f_vals, jac_vals = jax.vmap(deer_jvp)(z_rad)
    jac_diag = jnp.mean(z_rad * jac_vals, axis=0)
    return f_vals[0], jac_diag


# ---------------------------------------------------------------------------
# 2. Parallel solve via associative scan (reuses parode's combine_fc)
# ---------------------------------------------------------------------------

def _par_solve(
    gts: jnp.ndarray,
    rhs: jnp.ndarray,
    y0: jnp.ndarray,
) -> jnp.ndarray:
    """Solve the diagonal recurrence y[n+1] = -gts[n]*y[n] + rhs[n].

    Uses ``lax.associative_scan`` with parode's ``combine_fc`` operator.

    Parameters
    ----------
    gts : (T, D) diagonal "matrices" (negative sign already applied)
    rhs : (T, D) right-hand side vectors
    y0 : (D,) initial condition

    Returns
    -------
    yt : (T, D) solution (excludes y0)
    """
    eye = jnp.ones(gts.shape[-1], dtype=gts.dtype)[None]  # (1, D)
    F = jnp.concatenate((eye, -gts), axis=0)               # (T+1, D)
    c = jnp.concatenate((y0[None], rhs), axis=0)            # (T+1, D)
    _, yt = jax.lax.associative_scan(jax.vmap(combine_fc), (F, c))
    return yt[1:]  # drop the y0 row


# ---------------------------------------------------------------------------
# 3. Quasi-DEER (diagonal Jacobian) Newton iteration for discrete maps
# ---------------------------------------------------------------------------

def deer_map_solve(
    func: Callable,
    y0: jnp.ndarray,
    drivers: jnp.ndarray,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 100,
    num_probes: int = 1,
    damp_factor: float = 1.0,
    clip_val: float = 1e8,
) -> Tuple[jnp.ndarray, int]:
    """Solve a discrete map y[n+1] = func(y[n], driver[n], params) via DEER.

    Parameters
    ----------
    func : (state, driver, params) -> next_state
        The discrete map. state is (D,), driver is per-step data.
    y0 : (D,) initial state
    drivers : (T, ...) per-step drivers (e.g. PRNG keys)
    params : dict; must contain 'key' for Hutchinson probes
    yinit_guess : (T, D) or None. Initial guess for all T states.
        Defaults to zeros.
    max_iter : maximum Newton iterations
    num_probes : number of Hutchinson probes per step
    damp_factor : damping on the diagonal Jacobian
    clip_val : clipping bound for the Jacobian diagonals (the solution is clipped to +-1e8)

    Returns
    -------
    yt : (T, D) solved states
    n_iters : number of Newton iterations used
    """
    T = drivers.shape[0]
    D = y0.shape[0]
    dtype = y0.dtype

    solve_func = func
    y0_solve = y0

    if yinit_guess is None:
        yinit_guess = jnp.zeros((T, D), dtype=dtype)

    # Tolerances
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    # Split keys for Hutchinson probes — one per time step
    keys = jr.split(params['key'], T)

    def iter_func(iter_inp):
        err, yt, iiter = iter_inp

        # Shift: prepend y0, drop last
        ytparams = jnp.concatenate((y0_solve[None], yt[:-1]), axis=0)

        # Fused eval + diag Jacobian
        rhs, gts = jax.vmap(
            quasi_diag_estimator, in_axes=(0, 0, None, None, 0, None)
        )(ytparams, drivers, params, solve_func, keys, num_probes)

        # Clip and negate Jacobian diagonals
        gts = -jnp.clip(damp_factor * gts, -clip_val, clip_val)

        # Form RHS for linear solve
        rhs = rhs + gts * ytparams

        # Parallel solve
        yt_next = _par_solve(gts, rhs, y0_solve)

        # Clip solution and handle NaNs
        yt_next = jnp.clip(yt_next, -1e8, 1e8)
        yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        # Convergence check (relative)
        err = jnp.max(jnp.abs(yt_next - yt) - rtol * jnp.abs(yt))

        return err, yt_next, iiter + 1

    def cond_func(iter_inp):
        err, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)
    iiter = jnp.array(0, dtype=jnp.int32)

    _, yt, n_iters = jax.lax.while_loop(
        cond_func, iter_func, (err, yinit_guess, iiter)
    )

    return yt, n_iters


# ---------------------------------------------------------------------------
# 4. Full DEER (dense Jacobian) for discrete maps
# ---------------------------------------------------------------------------

def _par_solve_full(
    Js: jnp.ndarray,
    rhs: jnp.ndarray,
    y0: jnp.ndarray,
) -> jnp.ndarray:
    """Solve the dense linear recurrence y[n+1] = Js[n] @ y[n] + rhs[n].

    Uses ``lax.associative_scan`` with the dense ``combine_fc`` operator
    from ``parode.explicit_parallel``.

    Parameters
    ----------
    Js : (T, D, D) dense step Jacobians
    rhs : (T, D) right-hand side vectors
    y0 : (D,) initial condition

    Returns
    -------
    yt : (T, D) solution (excludes y0)
    """
    D = y0.shape[0]
    dtype = Js.dtype
    eye = jnp.broadcast_to(jnp.eye(D, dtype=dtype), (1, D, D))  # (1, D, D)
    F = jnp.concatenate((eye, Js), axis=0)                     # (T+1, D, D)
    c = jnp.concatenate((y0[None], rhs), axis=0)               # (T+1, D)
    _, yt = jax.lax.associative_scan(jax.vmap(combine_fc_dense), (F, c))
    return yt[1:]  # drop the y0 row


def deer_map_solve_full(
    func: Callable,
    y0: jnp.ndarray,
    drivers: jnp.ndarray,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 100,
    damp_factor: float = 1.0,
    clip_val: float = 1e8,
) -> Tuple[jnp.ndarray, int]:
    """Full DEER (dense Jacobian) Newton iteration for discrete maps.

    Solves y[n+1] = func(y[n], driver[n], params) in parallel via
    Newton linearization + dense-matrix associative scan. Memory scales
    as O(T * D^2); prefer the diagonal ``deer_map_solve`` when D is large.

    Parameters
    ----------
    func : (state, driver, params) -> next_state
    y0 : (D,) initial state
    drivers : (T, ...) per-step drivers (can be a dummy zeros array for
        autonomous maps)
    params : arbitrary params forwarded to func; ``params['key']`` is *not*
        required (Full DEER is deterministic).
    yinit_guess : (T, D) or None. Initial guess for all T states.
        Defaults to zeros.
    max_iter : maximum Newton iterations
    damp_factor : damping on the Jacobians
    clip_val : clipping bound for the Jacobians (the solution is clipped to +-1e8)

    Returns
    -------
    yt : (T, D) solved states
    n_iters : number of Newton iterations used
    """
    T = drivers.shape[0]
    D = y0.shape[0]
    dtype = y0.dtype

    solve_func = func
    y0_solve = y0

    if yinit_guess is None:
        yinit_guess = jnp.zeros((T, D), dtype=dtype)

    # Tolerances
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def _step_and_jac(state, driver):
        f_val = solve_func(state, driver, params)
        J = jax.jacfwd(lambda s: solve_func(s, driver, params))(state)
        return f_val, J

    def iter_func(iter_inp):
        err, yt, iiter = iter_inp

        # Shift: prepend y0, drop last
        ytparams = jnp.concatenate((y0_solve[None], yt[:-1]), axis=0)  # (T, D)

        # Dense Jacobian + function value at each shifted state
        f_vals, Js = jax.vmap(_step_and_jac)(ytparams, drivers)  # (T, D), (T, D, D)

        # Damp and clip
        Js = jnp.clip(damp_factor * Js, -clip_val, clip_val)

        # Affine residual: rhs_k = f_k - J_k @ ytparams_k
        rhs = f_vals - jnp.einsum("tij,tj->ti", Js, ytparams)

        # Parallel solve
        yt_next = _par_solve_full(Js, rhs, y0_solve)

        # Clip solution and handle NaNs
        yt_next = jnp.clip(yt_next, -1e8, 1e8)
        yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        # Convergence check (relative)
        err = jnp.max(jnp.abs(yt_next - yt) - rtol * jnp.abs(yt))

        return err, yt_next, iiter + 1

    def cond_func(iter_inp):
        err, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)
    iiter = jnp.array(0, dtype=jnp.int32)

    _, yt, n_iters = jax.lax.while_loop(
        cond_func, iter_func, (err, yinit_guess, iiter)
    )

    return yt, n_iters
