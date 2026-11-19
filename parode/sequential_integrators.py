# Third-party origin: derived from parode/sequential_integrators.py of parallel-ode by Casian Iacob,
# https://github.com/casiacob/parallel-ode
# What was taken and changed: THIRD_PARTY_NOTICES.md at the repository root, section "parallel-ode".
import jax.numpy as jnp
from typing import Callable, Optional

from parode.ode_utils import evaluate_input


def euler(
    ode: Callable, x: jnp.ndarray, t: float, dt: float, input_fn: Optional[Callable]
) -> jnp.ndarray:
    u_t = evaluate_input(input_fn, t)
    return x + ode(t, x, u_t) * dt


def rk4(
    ode: Callable, x: jnp.ndarray, t: float, dt: float, input_fn: Optional[Callable]
) -> jnp.ndarray:
    def evaluate(t_local, x_local):
        u_local = evaluate_input(input_fn, t_local)
        return ode(t_local, x_local, u_local)

    k1 = evaluate(t, x) * dt
    k2 = evaluate(t + 0.5 * dt, x + 0.5 * k1) * dt
    k3 = evaluate(t + 0.5 * dt, x + 0.5 * k2) * dt
    k4 = evaluate(t + dt, x + k3) * dt
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


integrators = {
    "euler": euler,
    "rk4": rk4,
}
