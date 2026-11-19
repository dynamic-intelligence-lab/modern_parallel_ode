"""Solver pieces of the ELCD experiment on parode's primitives.

`seq_scan` is the sequential rollout (lax.scan with the step function the residual uses) of every evaluation and
figure. The step function is parode's (`rk4_step`, `euler_step`). The other functions serve
schematic/export_chain8z.py, the solve of panel A of the ELCD figure: the exact step Jacobians by forward-mode AD
(`exact_step_linearizer`) and the (terms_fn, residual_fn, scan_fn) triple of a dense DEER step on parode's
`affine_terms` and `par_scan` with dtype-safe elements (`make_triple`, `deer_scan`, `_par_init`).
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from parode.explicit_parallel import affine_terms, euler_step, par_scan, rk4_step

STEP_FUNCTIONS = {"rk4": rk4_step, "euler": euler_step}


# ----------------------------------------------------------------------------- pieces

def _times(N: int, dt: float, dtype) -> jnp.ndarray:
    return dt * jnp.arange(N + 1, dtype=dtype)


def seq_scan(ode: Callable, x0: jnp.ndarray, dt: float, N: int, step: str) -> jnp.ndarray:
    """N explicit steps by lax.scan with the step function the residual uses; (N + 1, D) including x0."""
    step_fn = STEP_FUNCTIONS[step]

    def body(x, t):
        xn = x + step_fn(ode, x, t, dt, None)
        return xn, xn

    _, traj = lax.scan(body, x0, _times(N, dt, x0.dtype)[:-1])
    return jnp.concatenate((x0[None], traj))


def exact_step_linearizer(ode: Callable, dt: float, step: str) -> Callable:
    """X_prev (M, D) -> (M, D, D): I + d step / dx by forward-mode AD, the exact Jacobian of the discrete map
    (parode's `linear_terms` with the identity in the dtype of X)."""
    step_fn = STEP_FUNCTIONS[step]

    def linearize(X_prev):
        t = jnp.zeros((), X_prev.dtype)
        J = jax.vmap(lambda x: jax.jacfwd(lambda z: step_fn(ode, z, t, dt, None))(x))(X_prev)
        return jnp.eye(X_prev.shape[1], dtype=X_prev.dtype) + J

    return linearize


def _par_init(F: jnp.ndarray, c: jnp.ndarray):
    """Scan elements of the DEER update from F (N-1, D, D) and c = -r (N, D), built with zeros_like so an fp32 scan
    never promotes when x64 is on (parode's par_init allocates in the default dtype)."""
    first = F[0] @ c[0]
    tF = jnp.concatenate((jnp.zeros_like(F[:1]), F[1:]))
    tc = jnp.concatenate(((first + c[1])[None], c[2:]))
    return tF, tc


def deer_scan(F: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """delta (N, D): the dense DEER update by parode's associative scan."""
    _, cs = par_scan(_par_init(F, c))
    return jnp.concatenate((c[:1], cs))


def make_triple(ode: Callable, x0: jnp.ndarray, dt: float, N: int, step: str, linearizer: Callable,
                scan_fn: Callable = deer_scan):
    """(terms_fn, residual_fn, scan_fn) on X (N, D) excluding x0: terms_fn returns the step Jacobians of the maps
    X_t -> X_{t+1} for t = 1..N-1 (linearizer at X[:-1]) and c = -r; residual_fn returns c alone."""
    times = _times(N, dt, x0.dtype)
    step_fn = STEP_FUNCTIONS[step]

    def residual_fn(X):
        return affine_terms(X, x0, ode, dt, step_fn, times, None)

    def terms_fn(X):
        return linearizer(X[:-1]), residual_fn(X)

    return terms_fn, residual_fn, scan_fn
