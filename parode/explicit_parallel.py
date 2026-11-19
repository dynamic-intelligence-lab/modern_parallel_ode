# Third-party origin: derived from parode/explicit_parallel.py of parallel-ode by Casian Iacob,
# https://github.com/casiacob/parallel-ode
# What was taken and changed: THIRD_PARTY_NOTICES.md at the repository root, section "parallel-ode".
import jax.numpy as jnp
from jax import vmap, jacfwd, lax
from typing import Callable, Optional

from parode.ode_utils import prepare_ode_function, evaluate_input
from parode.laser import LaserConfig, run_damped_newton


def rk4_step(ode, x, t, dt, input_fn):
    def evaluate(t_local, x_local):
        u_local = evaluate_input(input_fn, t_local)
        return ode(t_local, x_local, u_local)

    k1 = evaluate(t, x) * dt
    k2 = evaluate(t + 0.5 * dt, x + 0.5 * k1) * dt
    k3 = evaluate(t + 0.5 * dt, x + 0.5 * k2) * dt
    k4 = evaluate(t + dt, x + k3) * dt
    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def euler_step(ode, x, t, dt, input_fn):
    u_t = evaluate_input(input_fn, t)
    return ode(t, x, u_t) * dt


step_functions = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def affine_terms(X, x0, ode, dt, step, times, input_fn): 
    '''
    Compute the affine terms for the explicit parallel integration. (residual)
    '''
    X_prev = jnp.vstack((x0, X[:-1]))
    t_prev = times[:-1]
    return -X + X_prev + vmap(step, in_axes=(None, 0, 0, None, None))(
        ode, X_prev, t_prev, dt, input_fn
    )


def linear_terms(X, ode, dt, step, times, input_fn):
    J = vmap(jacfwd(step, 1), in_axes=(None, 0, 0, None, None))(
        ode, X[:-1], times[1:-1], dt, input_fn
    )
    I = jnp.eye(X.shape[1])
    return I + J


def combine_fc(elem1, elem2):
    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=False)


def par_init(F, c, x0):
    """Build the scan elements. The first element's F never enters the c-part of
    any composition, so it is free and set to zero."""
    tF0 = jnp.zeros_like(F[0])
    tc0 = F[0] @ x0 + c[0]
    tF = jnp.concatenate((tF0[None], F[1:]))
    tc = jnp.concatenate((tc0[None], c[1:]))
    elems = (tF, tc)
    return elems


def newton_step(X, x0, ode, dt, step, times, input_fn):
    c = affine_terms(X, x0, ode, dt, step, times, input_fn) #residual
    F = linear_terms(X, ode, dt, step, times, input_fn)
    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    # Infinite norm of the update
    X_N = jnp.vstack((c[0], elems[1]))
    norm_res = jnp.max(jnp.abs(X_N))
    norm_c = jnp.max(jnp.abs(c))
    return X_N, norm_res, norm_c


def explicit_par_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    initial_guess: jnp.ndarray,
    dt: float,
    max_iter: int,
    method: str,
    t0: float = 0.0,
    input_fn: Optional[Callable] = None,
    *,
    tol: float,
    laser: Optional[LaserConfig] = None,
    return_retries: bool = False,
):
    """Newton (DEER) parallel integration with an explicit step.

    ode is f(t, x), or f(t, x, u) with u = input_fn(t); x0 has shape (D,) and initial_guess
    (N, D) holds the N states after x0; dt is the step, max_iter the cap on the iterations,
    method "euler" or "rk4" and t0 the time of x0. tol, passed by keyword, bounds the size
    the loop stops on (below).

    laser: optional LASER configuration (parode.laser.LaserConfig). When
    given, the Newton loop scales the step Jacobians by exp(-gamma dt) with
    gamma driven by the finite-time Lyapunov exponent of the iterate, guards
    each step with a non-monotone acceptance test, and stops on the residual
    max|r| < tol. Returns (sol (N+1, D) with x0, a size, nb_iterations), plus the history dict when
    laser.return_history is set. The size is the one the loop stops on: for the plain
    loop max|delta| of the last Newton update, for LASER max|r| at the returned iterate.
    return_retries appends n_retries, the rejected trials of the LASER loop (int32; 0
    for the plain loop, which has no retries), right after nb_iterations and before the
    history.
    """
    ode_fn = prepare_ode_function(ode)
    if method not in step_functions:
        raise ValueError(f"method must be one of {tuple(step_functions)}, got {method!r}")
    step = step_functions[method]
    nb_steps = initial_guess.shape[0]
    times = t0 + dt * jnp.arange(nb_steps + 1)

    if laser is not None:
        return _explicit_par_integrate_laser(ode_fn, x0, initial_guess, dt, step, times, input_fn, laser, tol,
                                             max_iter, return_retries)

    def while_body(val):
        X, _, iteration = val
        X_N, err, _ = newton_step(X, x0, ode_fn, dt, step, times, input_fn) #second output is the update, third is the residual
        X_new = X + X_N
        iteration += 1
        return X_new, err, iteration

    def while_cond(val):
        _, err, iteration = val
        return (err > tol) & (iteration < max_iter)

    sol, residual, nb_iterations = lax.while_loop(
        while_cond, while_body, (initial_guess, jnp.array(1e3), 0)
    )
    out = (jnp.vstack((x0, sol)), residual, nb_iterations)
    return out + (jnp.int32(0),) if return_retries else out


def _explicit_par_integrate_laser(ode_fn, x0, initial_guess, dt, step, times, input_fn, cfg, tol, max_iter,
                                  return_retries=False):
    """explicit_par_integrate with LASER's loop (parode.laser.run_damped_newton)."""

    def residual_fn(X):
        return affine_terms(X, x0, ode_fn, dt, step, times, input_fn)

    def terms_fn(X):
        return linear_terms(X, ode_fn, dt, step, times, input_fn), residual_fn(X)

    def scan_fn(F, c):
        _, cs = par_scan(par_init(F, c[1:], c[0]))
        return jnp.vstack((c[0], cs))

    sol, residual, nb_iterations, n_retries, history = run_damped_newton(
        initial_guess, terms_fn, scan_fn, dt, cfg, tol, max_iter, residual_fn=residual_fn)
    out = (jnp.vstack((x0, sol)), residual, nb_iterations)
    if return_retries:
        out = out + (n_retries,)
    return out + (history,) if cfg.return_history else out
