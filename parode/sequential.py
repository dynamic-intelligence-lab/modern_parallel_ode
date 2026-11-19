# Third-party origin: derived from parode/sequential.py of parallel-ode by Casian Iacob,
# https://github.com/casiacob/parallel-ode
# What was taken and changed: THIRD_PARTY_NOTICES.md at the repository root, section "parallel-ode".
import jax.numpy as jnp
from typing import Callable, Optional
from jax import lax
from parode.sequential_integrators import integrators
from parode.ode_utils import prepare_ode_function


def seq_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    dt: float,
    nb_steps: int,
    method: str,
    t0: float = 0.0,
    input_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Sequential rollout of x' = ode(t, x), or ode(t, x, u) with u = input_fn(t), from x0 of shape (D,):
    nb_steps steps of dt from t0 with method "euler" or "rk4". Returns (nb_steps + 1, D), x0 first."""
    if method not in integrators:
        raise ValueError(f"method must be one of {tuple(integrators)}, got {method!r}")
    integrator = integrators[method]
    ode_fn = prepare_ode_function(ode)
    times = t0 + dt * jnp.arange(nb_steps)

    def integrate_body(carry, current_t):
        carry = integrator(ode_fn, carry, current_t, dt, input_fn)
        return carry, carry

    _, x = lax.scan(integrate_body, x0, xs=times, length=nb_steps)
    return jnp.vstack((x0, x))
