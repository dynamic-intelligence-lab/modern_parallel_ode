"""Strongly contractive non-autonomous discrete RNN for Jacobi convergence tests.

The map is
    f_t(x) = W @ tanh(x) + c_t,
where W has exact operator norm ``rho`` (all singular values forced to rho via
SVD rescaling).  Because tanh is 1-Lipschitz in the l2 norm, this gives a
globally Lipschitz map with constant exactly ``rho``:

    ||f_t(x) - f_t(y)||_2
        = ||W (tanh(x) - tanh(y))||_2
        <= ||W||_2 * ||tanh(x) - tanh(y)||_2
        <= rho * ||x - y||_2.

So for rho < 1 the system is incrementally contractive with rate ``rho``, which
is the setting of the paper's convergence theorem for Jacobi iteration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr


def init_contractive_rnn(
    D: int, rho: float, key: jax.Array
) -> jnp.ndarray:
    """Return a (D, D) weight matrix with operator norm exactly ``rho``.

    Parameters
    ----------
    D : int
        State dimension.
    rho : float
        Desired spectral norm / Lipschitz constant of ``W``.  Must satisfy
        0 <= rho < 1 for the map to be contractive.
    key : jax.Array
        PRNG key.

    Returns
    -------
    W : (D, D) jax.Array
        Matrix with all singular values equal to ``rho``.  This is the "most
        isotropic" matrix of a given operator norm, useful for stress testing
        the uniform contraction assumption of the theory.
    """
    W_raw = jr.normal(key, (D, D))
    U, _, Vh = jnp.linalg.svd(W_raw, full_matrices=False)
    # All singular values set to rho, so ||W||_2 = rho exactly.
    W = rho * (U @ Vh)
    return W


def gen_drivers(
    T: int, D: int, key: jax.Array, scale: float = 1.0
) -> jnp.ndarray:
    """Sample non-autonomous drivers c_t ~ N(0, scale^2 I) of shape (T, D)."""
    return scale * jr.normal(key, (T, D))


def rnn_step(x: jnp.ndarray, c: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """Single RNN step: f_t(x) = W @ tanh(x) + c_t."""
    return W @ jnp.tanh(x) + c


def make_step_fn(W: jnp.ndarray):
    """Return a closed-over step function f(x, c) bound to ``W``.

    The returned callable has signature ``f(x: (D,), c: (D,)) -> (D,)`` and is
    suitable for ``vmap`` and ``lax.scan``.
    """

    def f(x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        return rnn_step(x, c, W)

    return f

