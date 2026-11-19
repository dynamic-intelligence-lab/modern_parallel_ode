# Third-party origin: the implicit-function adjoint follows deer_implicit, and the NaN guard of the Newton loop
# follows micro_deer_alg, of src/micro_deer/algs/micro_deer.py of https://github.com/lindermanlab/micro_deer
#     Copyright (c) 2026 Linderman Lab
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section "micro_deer".
"""
Parallel ODE integration with implicit-function-theorem backward pass.

Forward: lax.while_loop on Newton iterations (real early stop at tolerance).
Backward: implicit differentiation at the converged fixed point — a single
reverse-time lax.scan over the per-step Jacobians of the converged trajectory.

Two discretization schemes via the `solver` argument, dispatched inside
the same custom_vjp so the IFT-adjoint structure is shared (both are DEER,
Newton's method on the discretized trajectory: the paper's DEER-exp and Euler-DEER):
  - 'exp'   — exponential integrator (exp-Euler / phi-method, via expm trick)
  - 'euler' — explicit Euler:  x_{n+1} = x_n + dt * f(t_n, x_n)
Returns (trajectory (N+1, D), update, iterations): update is the largest entry
of the last Newton update, the quantity the loop stops on.

Compared to a solver that runs a lax.scan of fixed length K over the Newton
iterations and differentiates through every iteration:
  - O(1) memory in K (no unrolled tape across Newton iterations)
  - True early stop instead of running max_iter every time
  - Backward cost is one reverse-time scan regardless of K
  - Exact gradients at convergence (up to fixed-point tolerance)

API: pass `make_ode(params) -> ode(t, x) [or ode(t, x, u)]` as a factory and
`params` as a separate (differentiable) pytree.

Public wrappers (preset `solver`):
  exp_par_integrate_implicit
  euler_par_integrate_implicit
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import jacfwd, lax, vmap

from parode.ode_utils import prepare_ode_function
from parode.exp_parallel import _exp_step_and_phi, newton_step as _newton_step_exp
from parode.explicit_parallel import (
    newton_step as _newton_step_explicit,
    euler_step,
)


# ─── Per-solver one-step maps ───────────────────────────────────────
# Both return x_next as a function of x_prev. We need both the forward
# Newton solve (via the matching parode newton_step) and the per-step Jacobian
# d(x_next)/d(x_prev) at the converged trajectory in the backward pass.

def _step_map(solver, ode_fn, x_prev, t, dt):
    """One-step ODE integrator: x_prev -> x_next."""
    if solver == "exp":
        _, x_next = _exp_step_and_phi(ode_fn, x_prev, t, dt, None)
        return x_next
    if solver == "euler":
        return x_prev + euler_step(ode_fn, x_prev, t, dt, None)
    raise ValueError(f"Unknown solver: {solver!r}")


def _newton_step_dispatch(solver, X, x0, ode_fn, dt, times):
    """Dispatch one Newton step to the matching parode kernel."""
    if solver == "exp":
        return _newton_step_exp(X, x0, ode_fn, dt, times, None)
    if solver == "euler":
        return _newton_step_explicit(X, x0, ode_fn, dt, euler_step, times, None)
    raise ValueError(f"Unknown solver: {solver!r}")


# ─── Unified custom_vjp (solver is a non-diff static arg) ──────────

@partial(jax.custom_vjp, nondiff_argnums=(0, 4, 5, 6, 7, 8))
def _par_integrate_implicit(
    make_ode,
    params,
    x0,
    initial_guess,
    dt,
    max_iter,
    t0,
    tol,
    solver,
):
    """Internal: parallel-in-time DEER + IFT adjoint, parameterized by solver.

    Use the public wrappers ({exp,euler}_par_integrate_implicit) instead.
    """
    ode_fn = prepare_ode_function(make_ode(params))
    nb_steps = initial_guess.shape[0]
    times = t0 + dt * jnp.arange(nb_steps + 1)

    def while_body(val):
        X, _, iteration = val
        X_N, err, _ = _newton_step_dispatch(solver, X, x0, ode_fn, dt, times)
        X_new = X + X_N
        # NaN guard: fall back to initial_guess on NaN, mirroring micro_deer.
        X_new = jnp.where(jnp.isnan(X_new), initial_guess, X_new)
        err = jnp.where(jnp.isnan(err), jnp.array(1e3), err)
        return X_new, err, iteration + 1

    def while_cond(val):
        _, err, iteration = val
        return (err > tol) & (iteration < max_iter)

    sol, update, nb_iterations = lax.while_loop(
        while_cond, while_body, (initial_guess, jnp.array(1e3), 0)
    )
    return jnp.vstack((x0, sol)), update, nb_iterations


def _fwd(make_ode, params, x0, initial_guess, dt, max_iter, t0, tol, solver):
    sol, update, nb_iterations = _par_integrate_implicit(
        make_ode, params, x0, initial_guess, dt, max_iter, t0, tol, solver
    )
    return (sol, update, nb_iterations), (sol, params, x0)


def _bwd(make_ode, dt, max_iter, t0, tol, solver, res, g):
    """Adjoint via implicit function theorem at the converged fixed point.

    Recursion (full Jacobian, reverse-time):
        lambda[N-1] = g_X[N-1]
        lambda[k]   = g_X[k] + J[k]^T @ lambda[k+1]   for k = N-2, ..., 0
    where J[k] = d step_{k+1} / d X[k] evaluated at the converged trajectory.
    """
    sol, params, x0 = res
    g_sol, _, _ = g

    X = sol[1:]               # (N, D)
    g_X = g_sol[1:]
    g_x0_out = g_sol[0]
    N = X.shape[0]

    X_prev = jnp.vstack((x0[None, :], X[:-1]))     # (N, D)
    times = t0 + dt * jnp.arange(N + 1)
    t_steps = times[:-1]

    def F(x_prev, t, p):
        ode_fn = prepare_ode_function(make_ode(p))
        return _step_map(solver, ode_fn, x_prev, t, dt)

    def jac_F_x(xp, t):
        return jacfwd(lambda x: F(x, t, params))(xp)

    Js = vmap(jac_F_x)(X[:-1], t_steps[1:])         # (N-1, D, D)

    def adjoint_step(lam_next, inputs):
        g_k, J_k = inputs
        lam_k = g_k + J_k.T @ lam_next
        return lam_k, lam_k

    _, lambdas_rev = lax.scan(
        adjoint_step, g_X[-1], (g_X[:-1][::-1], Js[::-1])
    )
    lambdas = jnp.vstack((lambdas_rev[::-1], g_X[-1][None]))  # (N, D)

    # Gradient w.r.t. x0
    J0 = jacfwd(lambda x: F(x, t_steps[0], params))(x0)
    g_x0 = g_x0_out + J0.T @ lambdas[0]

    # Gradient w.r.t. params: single VJP through the batched one-step map.
    def all_steps_F(p):
        return vmap(lambda xp, t: F(xp, t, p))(X_prev, t_steps)

    _, vjp_fn = jax.vjp(all_steps_F, params)
    g_params = vjp_fn(lambdas)[0]

    # initial_guess does not affect the converged solution (gradient is zero).
    g_guess = jnp.zeros_like(X)

    return (g_params, g_x0, g_guess)


_par_integrate_implicit.defvjp(_fwd, _bwd)


# ─── Public convenience wrappers ────────────────────────────────────
# Each preset closes over `solver`; same return signature and call args.

def exp_par_integrate_implicit(make_ode, params, x0, initial_guess,
                                dt, max_iter, t0, tol):
    """Exp-Euler (phi-method) DEER + IFT adjoint. See module docstring."""
    return _par_integrate_implicit(
        make_ode, params, x0, initial_guess, dt, max_iter, t0, tol, "exp"
    )


def euler_par_integrate_implicit(make_ode, params, x0, initial_guess,
                                  dt, max_iter, t0, tol):
    """Explicit Euler DEER + IFT adjoint. Conditionally stable: needs
    dt * |max_eig(df/dx)| < 2 for the linearized one-step map to contract.
    """
    return _par_integrate_implicit(
        make_ode, params, x0, initial_guess, dt, max_iter, t0, tol, "euler"
    )
