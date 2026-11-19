"""
Sequential ODE integrators that match the two parallel solvers in
implicit_deer.py — for fair speedup measurements.

`parode.sequential.seq_integrate` covers euler and rk4, but not the
exponential integrator. This module fills that gap with `seq_exp_integrate`
and offers a unified `seq_integrate_dispatch(solver=...)` that mirrors the
solver name space used in implicit_deer.py.
"""
from typing import Callable

import jax.numpy as jnp
from jax import lax

from parode.ode_utils import prepare_ode_function
from parode.exp_parallel import _exp_step_and_phi
from parode.sequential import seq_integrate as _seq_integrate_explicit


def seq_exp_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    dt: float,
    nb_steps: int,
    t0: float = 0.0,
):
    """Sequential exponential integrator — counterpart of the parallel
    exponential solver of implicit_deer.py, using lax.scan instead of associative_scan.

    Each step linearizes around the previous state (A = df/dx, b = f - A*x)
    and applies Phi=expm(dt*A), x_next = Phi*x + dt*phi_1(dt*A)*b via the
    block-matrix expm trick (delegated to parode.exp_parallel._exp_step_and_phi).
    """
    ode_fn = prepare_ode_function(ode)
    times = t0 + dt * jnp.arange(nb_steps)

    def body(carry, t):
        _, x_next = _exp_step_and_phi(ode_fn, carry, t, dt, None)
        return x_next, x_next

    _, x = lax.scan(body, x0, xs=times, length=nb_steps)
    return jnp.vstack((x0, x))


def seq_integrate_dispatch(
    ode: Callable,
    x0: jnp.ndarray,
    dt: float,
    nb_steps: int,
    solver: str,
    t0: float = 0.0,
):
    """Sequential one-shot dispatch over the two solver names used by
    implicit_deer.py. Returns trajectory of shape (N+1, D)."""
    if solver == "exp":
        return seq_exp_integrate(ode, x0, dt, nb_steps, t0=t0)
    if solver == "euler":
        return _seq_integrate_explicit(ode, x0, dt, nb_steps, method="euler", t0=t0)
    raise ValueError(f"Unknown solver: {solver!r}")
