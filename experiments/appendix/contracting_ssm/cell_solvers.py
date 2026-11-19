"""Solver dispatch and custom_vjp parallel-scan adjoint.

Forward: dispatches to one of three parode solvers (seq / jacobi / quasi),
all solving the explicit-Euler discretization of the same continuous-time ODE.

Backward: implicit-function-theorem at the converged fixed point. The one-step
Jacobian -gamma + A·diag(tanh'(h)) is exactly diagonal (A and gamma are
diagonal, tanh acts elementwise), so the adjoint scan runs on per-step
diagonal propagators in O(log T) parallel time.
"""
from __future__ import annotations
import functools

import jax
import jax.numpy as jnp
from jax import lax, vmap

from parode.sequential import seq_integrate
from parode.jacobi import jacobi_alg
from parode.quasi_parallel import quasi_par_integrate, combine_fc as combine_fc_diag

from experiments.appendix.shared.contracting_ssm_cell import cell_ode_fn


# ============================================================
# Input wrapper: discrete x_seq -> continuous input_fn for parode
# ============================================================

def _make_step_indexed_input_fn(x_seq, t0, dt):
    """Adapter from step-indexed inputs to parode's input_fn(t) API.

    x_seq has no timestamps. Solver step k, evaluated at t0 + k * dt,
    consumes x_seq[k]. The time argument is used only to recover k because
    parode requests inputs through input_fn(t).
    """
    T = x_seq.shape[0]

    def input_fn(t):
        k_float = (t - t0) / dt
        k = jnp.clip(jnp.round(k_float).astype(jnp.int32), 0, T - 1)
        return x_seq[k]

    return input_fn


# ============================================================
# Forward dispatch (no custom_vjp here)
# ============================================================

def _forward_solve(x_seq, cell_params, solver, dt, tol,
                   max_iter_jacobi, max_iter_newton, A_static, gamma_static):
    """Run the chosen solver.

    Returns (h_traj, iters, residual):
        h_traj   — (T, D) trajectory h_1..h_T
        iters    — int32 scalar: Newton iterations (quasi) or Jacobi sweeps
                   (jacobi) until the residual is below tol or the cap is
                   reached; 1 for seq (non-iterative). The number tells the
                   caller how much parallel compute the solver needed.
        residual — float scalar: max-abs residual at termination
                   (0.0 for seq, which doesn't compute it).
    """
    T = x_seq.shape[0]
    D = cell_params.g_b2.shape[0]
    h0 = jnp.zeros((D,), dtype=x_seq.dtype)
    initial_guess = jnp.zeros((T, D), dtype=x_seq.dtype)
    input_fn = _make_step_indexed_input_fn(x_seq, t0=0.0, dt=dt)
    zero_res = jnp.array(0.0, dtype=x_seq.dtype)

    def ode_fn(t, h, u):
        return cell_ode_fn(t, h, u, cell_params, A_static, gamma_static)

    if solver == "seq":
        # seq_integrate returns (T+1, D) including x0; drop x0
        full = seq_integrate(ode_fn, h0, dt=dt, nb_steps=T, method="euler",
                             t0=0.0, input_fn=input_fn)
        return full[1:], jnp.int32(1), zero_res
    elif solver == "jacobi":
        # parode/jacobi.py's jacobi_alg has true early-stop via while_loop and
        # an explicit merit_fxn for the convergence metric. We define the
        # discrete map (Euler step) and the recurrence-residual merit so its
        # diagnostics match quasi: same metric, real iter counts.
        def discrete_map(h, x_curr):
            return h + dt * cell_ode_fn(0.0, h, x_curr, cell_params, A_static, gamma_static)

        def merit_fn(f, init_state, states, drvrs):
            prev = jnp.concatenate([init_state[None], states[:-1]], axis=0)
            return jnp.max(jnp.abs(states - vmap(f)(prev, drvrs)))

        _, H, iters, _is_nan, residual, _, _ = jacobi_alg(
            f=discrete_map,
            initial_state=h0,
            states_guess=initial_guess,
            drivers=x_seq,
            num_iters=max_iter_jacobi,
            merit_fxn=merit_fn,
            tol=tol,
        )
        return H, iters, residual
    elif solver == "quasi":
        full, residual, iters = quasi_par_integrate(
            ode=ode_fn, x0=h0, initial_guess=initial_guess,
            dt=dt, max_iter=max_iter_newton,
            t0=0.0, input_fn=input_fn, tol=tol,
        )
        return full[1:], iters, residual
    else:
        raise ValueError(f"unknown solver: {solver!r}")


# ============================================================
# Per-step diagonal propagator (used by adjoint scan)
# ============================================================

def _step_propagator_diag(h_prev, x_curr, cell_params, dt, A_static, gamma_static):
    """∂h_next/∂h_prev for a single step, returned as a (D,) diagonal.

    Computed as a JVP of _discrete_step with v = 1: since the Jacobian is
    diagonal (A and gamma diagonal, tanh elementwise), the JVP with all-ones
    tangent equals the diagonal entries. This makes the propagator a strict
    consequence of the discrete step, ruling out math drift between the two.
    """
    step_fn = lambda hp: _discrete_step(hp, x_curr, cell_params, dt, A_static, gamma_static)
    _, diag = jax.jvp(step_fn, (h_prev,), (jnp.ones_like(h_prev),))
    return diag


# ============================================================
# Discrete step (used by VJP for parameter / x_seq gradients)
# ============================================================

def _discrete_step(h_prev, x_curr, cell_params, dt, A_static, gamma_static):
    """Discrete h_next from h_prev given input x_curr (explicit Euler). Must
    match what the forward solver's discrete recurrence converges to."""
    return h_prev + dt * cell_ode_fn(0.0, h_prev, x_curr, cell_params, A_static, gamma_static)


# ============================================================
# Adjoint scan (parallel, diagonal Jacobian)
# ============================================================

def _par_solve_diag(Fs, rhs, y0):
    """Solve y_{n+1} = Fs[n] * y_n + rhs[n] (elementwise) via associative scan.
    Fs: (T, D), rhs: (T, D), y0: (D,) -> (T, D) excluding y0.
    """
    _, D = Fs.shape
    eye = jnp.ones((1, D), dtype=Fs.dtype)
    F = jnp.concatenate((eye, Fs), axis=0)
    c = jnp.concatenate((y0[None], rhs), axis=0)
    _, yt = lax.associative_scan(vmap(combine_fc_diag), (F, c))
    return yt[1:]


def _adjoint_scan_diag(propagators, g):
    """Solve the adjoint recurrence:
        lambdas[T-1] = g[T-1]
        lambdas[t-1] = propagators[t] · lambdas[t] + g[t-1]   for t = T-1..1
    Diagonal propagators -> elementwise multiply, no transpose needed.

    propagators: (T, D), g: (T, D) -> lambdas: (T, D)
    """
    T = g.shape[0]
    if T == 1:
        return g
    # Time-reverse to get a forward recurrence:
    #   mu_0 = g[T-1]
    #   mu_s = propagators[T-s] · mu_{s-1} + g[T-1-s]   for s = 1..T-1
    F_rev = jnp.flip(propagators[1:], axis=0)         # (T-1, D)
    g_rev = jnp.flip(g[:-1], axis=0)                  # (T-1, D)
    mu_init = g[-1]
    mu = _par_solve_diag(F_rev, g_rev, mu_init)       # (T-1, D)
    lambdas_interior = jnp.flip(mu, axis=0)
    return jnp.concatenate([lambdas_interior, g[-1:]], axis=0)


# ============================================================
# Public entry: custom_vjp wrapper
# ============================================================

def _unpack_solver_static(solver_static):
    """Unpack solver_static into (core, A_static, gamma_static).

    solver_static: 11-tuple
        (solver, dt, tol, max_iter_jacobi, max_iter_newton,
         A_param, A_alpha_max, A_alpha_min,
         gamma_param, gamma_min, gamma_max)

    Validates Euler stability when both A and gamma are bounded:
    ``dt * (gamma_max + alpha_max) < 2`` is required for ``|G'(h)| < 1``
    on the negative side. The check fires once at trace time (solver_static
    is a static argument).
    """
    (solver, dt, tol, mij, min_,
     A_param, alpha_max, alpha_min,
     gamma_param, gamma_min, gamma_max) = solver_static
    if gamma_param != "off" and A_param == "sigmoid":
        bound = dt * (gamma_max + alpha_max)
        if bound >= 2.0:
            raise ValueError(
                f"Euler stability violated: dt*(gamma_max + alpha_max) = "
                f"{bound:.3f} >= 2.0. Reduce gamma_max, alpha_max, or dt."
            )
    A_static = (A_param, alpha_max, alpha_min)
    gamma_static = (gamma_param, gamma_min, gamma_max)
    return (solver, dt, tol, mij, min_), A_static, gamma_static


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve(x_seq, cell_params, solver_static):
    """Solve the cell ODE driven by x_seq, returning (T, D) trajectory.

    solver_static: 11-tuple
        (solver, dt, tol, max_iter_jacobi, max_iter_newton,
         A_param, A_alpha_max, A_alpha_min,
         gamma_param, gamma_min, gamma_max)
    Non-differentiable; must be a fixed Python tuple at call time. The A and
    gamma parametrizations are selected at solve time without rebuilding the
    parameter pytree. ``gamma_param="off"`` is the cell without the leak term.
    """
    (solver, dt, tol, mij, min_), A_static, gamma_static = _unpack_solver_static(solver_static)
    h, _iters, _res = _forward_solve(
        x_seq, cell_params, solver, dt, tol, mij, min_, A_static, gamma_static,
    )
    return h


def solve_diagnostics(x_seq, cell_params, solver_static):
    """Forward-only solve plus solver convergence + per-step Jacobian stats.

    Returns 7-tuple (h, iters, residual, jac_max, jac_mean, jac_p99,
    jac_frac_above_1). Jacobian stats are aggregated over the converged
    (T, D) one-step propagator diagonals. ``jac_frac_above_1`` is strict
    expansion, not equality at the nonexpansive boundary. No custom_vjp;
    eval-only.
    """
    (solver, dt, tol, mij, min_), A_static, gamma_static = _unpack_solver_static(solver_static)
    h, iters, residual = _forward_solve(
        x_seq, cell_params, solver, dt, tol, mij, min_, A_static, gamma_static,
    )

    # One-step Jacobian diagonal at the converged trajectory.
    h0 = jnp.zeros_like(h[0])
    h_prev_seq = jnp.concatenate([h0[None], h[:-1]], axis=0)
    propagators = vmap(
        _step_propagator_diag, in_axes=(0, 0, None, None, None, None),
    )(h_prev_seq, x_seq, cell_params, dt, A_static, gamma_static)   # (T, D)
    abs_p = jnp.abs(propagators)
    jac_max  = abs_p.max()
    jac_mean = abs_p.mean()
    jac_p99  = jnp.percentile(abs_p, 99.0)
    # Count true expansion only. Equality at 1.0 is common when tanh saturates:
    # the sech^2 factor rounds to zero and the Euler propagator becomes exactly
    # the identity in that coordinate. That is non-strict/nonexpansive, but not
    # growth above 1.
    above_one_tol = 16.0 * jnp.finfo(abs_p.dtype).eps
    jac_frac_above_1 = (abs_p > 1.0 + above_one_tol).mean()
    return h, iters, residual, jac_max, jac_mean, jac_p99, jac_frac_above_1


def _solve_fwd(x_seq, cell_params, solver_static):
    h_traj = solve(x_seq, cell_params, solver_static)
    return h_traj, (x_seq, cell_params, h_traj)


def _solve_bwd(solver_static, residuals, g):
    x_seq, cell_params, h_traj = residuals
    (_solver, dt, _tol, _mij, _min), A_static, gamma_static = _unpack_solver_static(solver_static)

    D = h_traj.shape[1]
    h0 = jnp.zeros((D,), dtype=h_traj.dtype)
    h_prev_seq = jnp.concatenate([h0[None], h_traj[:-1]], axis=0)   # (T, D)
    # x_seq[k] drives step k (h_k -> h_{k+1}) via ZOH; same indexing used here.

    propagators = vmap(
        _step_propagator_diag, in_axes=(0, 0, None, None, None, None)
    )(h_prev_seq, x_seq, cell_params, dt, A_static, gamma_static)   # (T, D)

    lambdas = _adjoint_scan_diag(propagators, g)                    # (T, D)

    # Parameter / input grads via VJP of the discrete step, vmapped over time
    def all_steps(cell_params_, x_seq_):
        # h_prev_seq is held fixed (treated as constant w.r.t. params/x_seq);
        # this is correct because the propagator scan already accounted for
        # the h-chain dependency.
        def step_one(hp, xc):
            return _discrete_step(hp, xc, cell_params_, dt, A_static, gamma_static)
        return vmap(step_one)(h_prev_seq, x_seq_)

    _, vjp_fn = jax.vjp(all_steps, cell_params, x_seq)
    d_cell_params, d_x_seq = vjp_fn(lambdas)
    return d_x_seq, d_cell_params


solve.defvjp(_solve_fwd, _solve_bwd)
