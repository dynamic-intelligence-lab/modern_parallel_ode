"""The sequential rollout and a fixed-length Jacobi iteration for contractive discrete maps.

``sequential_rollout`` is the reference trajectory and the timed baseline.
``jacobi_scan`` runs a fixed number of Jacobi sweeps in a ``lax.scan`` and
records, after every sweep, the error to the sequential trajectory,
max_t ||s^k_t - x*_t||_2, and the residual. For a map with Lipschitz constant
rho in the l2 norm (the contractive RNN of contractive_rnn.py) the error obeys

    max_t ||s^k_t - x*_t||_2 <= rho^k max_t ||s^0_t - x*_t||_2,

so the number of sweeps needed for a tolerance does not grow with the sequence
length ``T`` (N in the paper).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Sequential rollout (ground truth)
# ---------------------------------------------------------------------------


def sequential_rollout(
    f: Callable, x0: jnp.ndarray, drivers: jnp.ndarray
) -> jnp.ndarray:
    """Compute (x_1, ..., x_T) sequentially via ``lax.scan``.

    Parameters
    ----------
    f : callable
        Discrete map with signature ``f(x, driver) -> next_x``.
    x0 : (D,)
        Initial state.
    drivers : (T, ...)
        Drivers ``c_t`` for ``t = 1, ..., T``.  ``drivers[0]`` produces x_1
        from x_0, ``drivers[1]`` produces x_2 from x_1, and so on.

    Returns
    -------
    states : (T, D)
        Full sequential trajectory (excludes x_0).
    """

    def body(carry: jnp.ndarray, c: jnp.ndarray):
        x_next = f(carry, c)
        return x_next, x_next

    _, states = jax.lax.scan(body, x0, drivers)
    return states


# ---------------------------------------------------------------------------
# Jacobi iteration with per-iteration error tracking
# ---------------------------------------------------------------------------


def jacobi_scan(
    f: Callable,
    x0: jnp.ndarray,
    initial_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    x_true: jnp.ndarray,
    max_iters: int,
) -> "tuple[jnp.ndarray, jnp.ndarray]":
    """Run ``max_iters`` Jacobi iterations, recording error at each step.

    The Jacobi shift map is
        s^{k+1}_t = f(s^k_{t-1}, driver_t),
    applied in parallel over ``t = 1, ..., T`` (with s^k_0 := x0 fixed).

    Parameters
    ----------
    f : callable
        Discrete map ``(state, driver) -> next_state``.
    x0 : (D,)
        Fixed initial state x_0.
    initial_guess : (T, D)
        Initial guess for (s_1, ..., s_T).
    drivers : (T, ...)
        Non-autonomous drivers.
    x_true : (T, D)
        Ground truth sequential trajectory for error measurement.
    max_iters : int
        Number of Jacobi iterations to run (fixed for jit).

    Returns
    -------
    final_states : (T, D)
        States after ``max_iters`` iterations.
    err_history : (max_iters, 2) with columns ``[err, resid]``.

        ``err_history[k-1, 0]`` = ``max_t ||s^k_t - x^*_t||_2`` is the error of
        the iterate *after* k Jacobi sweeps.

        ``err_history[k-1, 1]`` = ``max_t || s^{k-1}_t - f_t(s^{k-1}_{t-1}) ||_2``
        is the self-consistency residual of the iterate *fed into* the k-th
        Jacobi sweep.  Equivalently, it equals the update-size
        ``max_t ||s^k_t - s^{k-1}_t||_2``.

        Both quantities use the max-over-time of the per-step L2 norm — the
        norm in which the contraction theorem is stated.
    """

    def body(states: jnp.ndarray, _):
        # prev[t] = s^k_{t-1}, with prev[0] = x0 fixed.
        prev = jnp.concatenate([x0[None], states[:-1]], axis=0)
        # Apply f in parallel: new[t] = f(s^k_{t-1}, driver_t) = s^{k+1}_t.
        new_states = jax.vmap(f)(prev, drivers)

        # err of the *updated* iterate s^{k+1}
        err = jnp.max(jnp.linalg.norm(new_states - x_true, axis=-1))

        # residual of the *current* iterate s^k  (= update size ||s^k - s^{k+1}||)
        resid = jnp.max(jnp.linalg.norm(states - new_states, axis=-1))

        return new_states, jnp.stack([err, resid])

    final_states, err_history = jax.lax.scan(
        body, initial_guess, None, length=max_iters
    )
    return final_states, err_history

