"""Parallel-in-time forward (plain DEER, guarded DEER or LASER) with an implicit-function-theorem adjoint.

implicit_solve returns the fixed point of the recurrence x_t = S(x_{t-1}, t_{t-1}, u_{t-1}; params) of the
chosen stepper: parode's public integrator with laser passed through (LASER changes the Newton iteration, not
the fixed point) or the iterate-guard baseline. The backward pass is the adjoint recursion at that trajectory,

    lambda_N = g_N,   lambda_t = g_t + J_{t+1}^T lambda_{t+1},   J_t = dS/dx at (x_{t-1}, t_{t-1}),

one associative scan on the time-reversed sequence (combine_fc of explicit_parallel on J_t^T), then one VJP of
the time-batched one-step map for the params and inputs
cotangents, and g_x0 = g_traj[0] + J_1^T lambda_1. The adjoint is exact only where the forward converged, so
the residual is max|r| recomputed from the returned trajectory (plain explicit_par_integrate reports the update
norm, which is not a gate) and `unconverged` decides what a residual above tol returns. All Python-level
configuration lives in the frozen dataclass _Static, the single nondiff argument of the custom_vjp.

Options of the ELCD paper runs: max_iter 0 caps each solve at its own length N; the others are off by default,
so that the default program is unchanged. stop="laser" gives every Newton
solver LASER's stopping rule (max|r| <= tol at the current iterate, the guess included, or a non-finite or stalled
step, or the cap) and gates on the solver's own final residual instead of a recomputed one; plain DEER then runs on
_shared_loop, LASER's loop without the damping, and scale_rate g > 0 is constant scale-ELK on the same loop (every
step Jacobian scaled by exp(-g dt)), and elk_sigma2 s2 > 0 is ELK on the same loop (the update is the filtered mean
of the released elk_alg's linear-Gaussian model, parode/elk_baseline.py, in delta form). implicit_forward is
the forward alone (no gradient flows) and attach_adjoint ties a path found elsewhere (the sequential fallback's, or
the parallel solver's) to its parameters with this adjoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jacfwd, lax, vmap

from parode.explicit_parallel import (affine_terms, euler_step, explicit_par_integrate, linear_terms, newton_step,
                                      par_init, par_scan, rk4_step, step_functions, combine_fc as _combine_dense)
from parode.laser import LaserConfig
from parode.ode_utils import prepare_ode_function
from parode.sequential import seq_integrate

from parode.elk_baseline import elk_filter_dense

SOLVERS = ("euler", "rk4")      # the one-step maps that the argument `solver` selects
UNCONVERGED = ("zero", "nan", "pass")
STOPS = ("parode", "laser")     # parode's integrators as they are, or LASER's stopping rule for every Newton solver
SOLVER_NAMES = ("seq", "deer", "guard", "laser-plain", "laser", "laser-carry")
ZOH_ROUNDING_MARGIN = 0.25   # floor((t - t0)/dt + margin): RK4 half steps map to k, fp32 time drift is absorbed


def solver_kwargs(name: str, guard_value: float) -> dict:
    """Keyword arguments of implicit_solve for a solver name of SOLVER_NAMES ("seq" uses seq_solve); the tests of the
    library use these names. The ELCD experiment has its own names and table: experiments/elcd/train/losses.py."""
    table = {"deer": {}, "guard": {"guard": guard_value},
             "laser-plain": {"laser": LaserConfig(propose_first="plain")}, "laser": {"laser": LaserConfig()},
             # the paper runs' LASER with the exponent of the guess carried between iterations instead of recomputed
             # after every accepted one: one scan per iteration
             "laser-carry": {"laser": LaserConfig(exponent="carry")}}
    if name == "seq":
        raise ValueError("'seq' is the sequential baseline; call seq_solve instead of implicit_solve")
    if name not in table:
        raise ValueError(f"solver name must be one of {SOLVER_NAMES}, got {name!r}")
    return dict(table[name])


def zoh_input_fn(seq: jnp.ndarray, t0: float, dt: float) -> Callable:
    """Zero-order hold: u(t) = seq[k], k = floor((t - t0)/dt + 0.25) clipped to [0, N-1].

    round() would send the RK4 half-step times t + dt/2 to k or k + 1 by the parity of k.
    """
    n_steps = seq.shape[0]

    def input_fn(t):
        k = jnp.floor((t - t0) / dt + ZOH_ROUNDING_MARGIN).astype(jnp.int32)
        return seq[jnp.clip(k, 0, n_steps - 1)]

    return input_fn


def step_map(solver: str, ode_fn: Callable, x_prev: jnp.ndarray, t, dt: float, input_fn) -> jnp.ndarray:
    """The one-step map S(x_{t-1}, t_{t-1}) whose fixed point the forward returns."""
    if solver == "euler":
        return x_prev + euler_step(ode_fn, x_prev, t, dt, input_fn)
    return x_prev + rk4_step(ode_fn, x_prev, t, dt, input_fn)


@dataclass(frozen=True)
class _Static:
    """Every Python-level setting of one solve; hashable, the nondiff argument of the custom_vjp."""
    make_ode: Callable
    solver: str
    dt: float
    t0: float
    tol: float
    max_iter: int
    laser: Optional[LaserConfig]
    guard: Optional[float]
    unconverged: str
    input_fn: Optional[Callable]     # non-differentiable input path (closes over data)
    has_inputs: bool                 # differentiable input path (inputs array argument)
    stop: str = "parode"             # STOPS
    scale_rate: float = 0.0          # constant scale-ELK's rate g (per unit time): J -> exp(-g dt) J; 0: plain DEER
                                     # (the Lorenz driver's scale-elk:<k> scales J -> (1 - k) J instead)
    elk_sigma2: float = 0.0          # ELK's observation variance sigma^2 (lambda_LM = 1 / sigma^2); 0: not ELK


def _make_static(make_ode, x0, initial_guess, dt, solver, inputs, input_fn, t0, tol, max_iter, laser, guard,
                 unconverged, stop="parode", scale_rate=0.0, elk_sigma2=0.0) -> _Static:
    """Validate every argument at the API boundary and freeze the configuration. max_iter 0 is the solve's own
    length, initial_guess.shape[0] (the paper runs' cap: N iterations make plain DEER's whole path exact)."""
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if stop not in STOPS:
        raise ValueError(f"stop must be one of {STOPS}, got {stop!r}")
    if stop == "laser" and guard is not None:
        raise ValueError(f"stop 'laser' runs plain DEER, scale-ELK and LASER for {SOLVERS}, without the guard; "
                         f"got solver {solver!r}, guard {guard!r}")
    if not (math.isfinite(float(scale_rate)) and float(scale_rate) >= 0.0):
        raise ValueError(f"scale_rate must be a finite rate >= 0, got {scale_rate!r}")
    if float(scale_rate) > 0.0 and (stop != "laser" or laser is not None):
        raise ValueError("constant scale-ELK (scale_rate > 0) runs on the shared loop: needs stop 'laser' and no laser")
    if not (math.isfinite(float(elk_sigma2)) and float(elk_sigma2) >= 0.0):
        raise ValueError(f"elk_sigma2 must be a finite variance >= 0 (0: not ELK), got {elk_sigma2!r}")
    if float(elk_sigma2) > 0.0 and (stop != "laser" or laser is not None or float(scale_rate) > 0.0):
        raise ValueError("ELK (elk_sigma2 > 0) runs on the shared loop: needs stop 'laser', no laser, no scale_rate")
    if unconverged not in UNCONVERGED:
        raise ValueError(f"unconverged must be one of {UNCONVERGED}, got {unconverged!r}")
    if inputs is not None and input_fn is not None:
        raise ValueError("pass either inputs (differentiable) or input_fn (not differentiable), not both")
    if laser is not None and laser.return_history:
        raise ValueError("laser.return_history is not supported")
    if guard is not None:
        if laser is not None:
            raise ValueError("guard and laser are alternative Newton loops; pass one of them")
        if not float(guard) > 0.0:
            raise ValueError(f"guard must be a positive bound on |X|, got {guard!r}")
    if not (float(dt) > 0.0 and float(tol) > 0.0 and int(max_iter) >= 0):
        raise ValueError(f"need dt > 0, tol > 0 and max_iter >= 0 (0: the solve's length), got dt={dt}, tol={tol}, "
                         f"max_iter={max_iter}")
    if x0.ndim != 1 or initial_guess.ndim != 2 or initial_guess.shape[1] != x0.shape[0]:
        raise ValueError(f"x0 must be (D,) and initial_guess (N, D), got {x0.shape} and {initial_guess.shape}")
    if inputs is not None and (inputs.ndim != 2 or inputs.shape[0] != initial_guess.shape[0]):
        raise ValueError(f"inputs must be (N, d_in) with N = {initial_guess.shape[0]}, got {inputs.shape}")
    cap = int(initial_guess.shape[0]) if int(max_iter) == 0 else int(max_iter)
    return _Static(make_ode, solver, float(dt), float(t0), float(tol), cap, laser,
                   None if guard is None else float(guard), unconverged, input_fn,
                   inputs is not None, stop, float(scale_rate), float(elk_sigma2))


def _resolve_input_fn(st: _Static, inputs):
    return zoh_input_fn(inputs, st.t0, st.dt) if st.has_inputs else st.input_fn


def _times(st: _Static, n_steps: int, dtype) -> jnp.ndarray:
    """t_0..t_N in the trajectory's dtype."""
    return st.t0 + st.dt * jnp.arange(n_steps + 1, dtype=dtype)


def _residual(st: _Static, ode_fn, traj, input_fn) -> jnp.ndarray:
    """max|r| of the trajectory, r_t = x_t - S(x_{t-1}, t_{t-1}); NaN or inf where the trajectory is."""
    t_prev = _times(st, traj.shape[0] - 1, traj.dtype)[:-1]
    pred = vmap(lambda x, t: step_map(st.solver, ode_fn, x, t, st.dt, input_fn))(traj[:-1], t_prev)
    return jnp.max(jnp.abs(traj[1:] - pred))


def _step_jacobians(st: _Static, ode_fn, X, input_fn) -> jnp.ndarray:
    """J_t = dS/dx at (x_{t-1}, t_{t-1}) for t = 2..N along X (N, D), shape (N-1, D, D).

    Equals parode.explicit_parallel.linear_terms for euler and rk4.
    """
    t_prev = _times(st, X.shape[0], X.dtype)[1:-1]
    S = lambda x, t: step_map(st.solver, ode_fn, x, t, st.dt, input_fn)
    return vmap(lambda x, t: jacfwd(S)(x, t))(X[:-1], t_prev)


def _apply_guard(X: jnp.ndarray, bound: float) -> jnp.ndarray:
    """Iterate guard of the guarded-DEER baseline: NaN -> 0, then clip to [-C, C] (infinite entries -> +-C)."""
    return jnp.clip(jnp.where(jnp.isnan(X), 0.0, X), -bound, bound)


def _newton_loop(st: _Static, ode_fn, x0, guess, input_fn):
    """Guarded DEER on parode's newton_step: the iterate is clipped after every update and the loop stops on
    max|r| < tol. Returns (X, n_iter).
    """
    step = step_functions[st.solver]
    times = _times(st, guess.shape[0], guess.dtype)

    def body(val):
        X, _, it = val
        upd, _, _ = newton_step(X, x0, ode_fn, st.dt, step, times, input_fn)
        X_new = _apply_guard(X + upd, st.guard)
        res = jnp.max(jnp.abs(affine_terms(X_new, x0, ode_fn, st.dt, step, times, input_fn)))
        return X_new, res, it + 1

    def cond(val):
        _, stat, it = val
        return (stat > st.tol) & (it < st.max_iter)

    init = (guess, jnp.array(jnp.inf, guess.dtype), jnp.int32(0))
    X, _, n_iter = lax.while_loop(cond, body, init)
    return X, n_iter


def _shared_loop(st: _Static, ode_fn, x0, guess, input_fn):
    """Plain DEER (scale_rate 0) or constant scale-ELK with LASER's stopping rule (stop 'laser').

    The stopping rule of parode's run_damped_newton: the guess is tested first (a converged guess costs 0
    iterations), and the loop runs while max|c| > tol, k < max_iter and not done, where done is a non-finite step
    (the last finite iterate is returned with residual inf, as LASER returns its last accepted one) or a stalled step
    (max|delta| <= 8 eps max(1, max|X_new|): the dtype's rounding floor). The iteration is parode's own plain DEER
    body, newton_step: the Jacobians F = linear_terms(X) at the top of the iteration, the plain DEER scan on s F with
    s = exp(-g dt) (a Python float, so that fp32 stays fp32 and g = 0 traces no multiplication), and the residual
    c = affine_terms at the new iterate, carried to the next test. Per iteration one Jacobian evaluation, one scan
    and one residual evaluation; no Jacobian of the final iterate or of a passing guess (LASER forms those for its
    exponent, its own cost). No acceptance test.

    ELK (elk_sigma2 s2 > 0) replaces the scan by the released elk_alg's update: the filtered means of the linear-Gaussian
    model x_{t+1} = F_t x_t + b_t + N(0, I), y_t = x_t + N(0, s2 I), prior N(S(x0), I), observed at the current iterate.
    Shifted by the iterate (the filter is affine-equivariant) its means are the update itself: delta =
    filter(F, c[1:], c[0], 0, s2), the same fixed points as DEER's and DEER's step as s2 -> inf. Returns (traj (N+1, D),
    max|r| of the returned iterate, iterations)."""
    step = step_functions[st.solver]
    times = st.t0 + st.dt * jnp.arange(guess.shape[0] + 1)          # parode's own grid (explicit_par_integrate)
    scale = math.exp(-st.scale_rate * st.dt)
    eps_mach = jnp.finfo(guess.dtype).eps
    residual = lambda X: affine_terms(X, x0, ode_fn, st.dt, step, times, input_fn)

    def body(carry):
        X, c, _, k, _ = carry
        F = linear_terms(X, ode_fn, st.dt, step, times, input_fn)
        if st.elk_sigma2 > 0.0:
            delta = elk_filter_dense(F, c[1:], c[0], jnp.zeros_like(c), st.elk_sigma2)
        else:
            _, cs = par_scan(par_init(F if scale == 1.0 else F * scale, c[1:], c[0]))
            delta = jnp.vstack((c[0], cs))
        X_new = X + delta
        c_new = residual(X_new)
        ok = jnp.all(jnp.isfinite(X_new)) & jnp.all(jnp.isfinite(c_new))
        res_new = jnp.where(ok, jnp.max(jnp.abs(c_new)), jnp.inf)
        stalled = ok & (jnp.max(jnp.abs(delta)) <= 8.0 * eps_mach * jnp.maximum(1.0, jnp.max(jnp.abs(X_new))))
        return jnp.where(ok, X_new, X), c_new, res_new, k + 1, (~ok) | stalled

    def cond(carry):
        _, _, res, k, done = carry
        return (res > st.tol) & (k < st.max_iter) & (~done)

    c0 = residual(guess)
    init = (guess, c0, jnp.max(jnp.abs(c0)), jnp.int32(0), jnp.bool_(False))
    X, _, res, n_iter, _ = lax.while_loop(cond, body, init)
    return jnp.vstack((x0[None], X)), res, n_iter


def _parode_forward(st: _Static, ode_fn, x0, guess, input_fn, cfg, retries: bool = False):
    """parode's integrator for the stepper, plain (cfg None) or LASER (cfg a LaserConfig). retries: the output gains
    LASER's n_retries after the iterations."""
    kw = dict(t0=st.t0, input_fn=input_fn, tol=st.tol, laser=cfg)
    if retries:
        kw["return_retries"] = True
    return explicit_par_integrate(ode_fn, x0, guess, st.dt, st.max_iter, st.solver, **kw)


def _forward(st: _Static, params, x0, guess, inputs, retries: bool = False):
    """(traj, residual, n_iter), plus n_retries (int32: LASER's rejected trials; 0 for the loops without retries)
    when retries is set."""
    ode_fn = prepare_ode_function(st.make_ode(params))
    input_fn = _resolve_input_fn(st, inputs)
    laser_retries = retries and st.laser is not None     # the only loop with retries; the others report 0
    n_retries = jnp.int32(0)
    if st.stop == "laser":                    # every solver reports its own final max|r| (LASER's: inf on failure)
        if st.laser is None:
            traj, res, n_iter = _shared_loop(st, ode_fn, x0, guess, input_fn)
        else:
            out = _parode_forward(st, ode_fn, x0, guess, input_fn, st.laser, laser_retries)
            traj, res, n_iter = out[:3]
            n_retries = out[3] if laser_retries else n_retries
        return (traj, res, n_iter, n_retries) if retries else (traj, res, n_iter)
    if st.guard is not None:
        X, n_iter = _newton_loop(st, ode_fn, x0, guess, input_fn)
        traj = jnp.vstack((x0[None], X))
    else:
        out = _parode_forward(st, ode_fn, x0, guess, input_fn, st.laser, laser_retries)
        traj, n_iter = out[0], out[2]
        n_retries = out[3] if laser_retries else n_retries
    res = _residual(st, ode_fn, traj, input_fn)
    return (traj, res, n_iter, n_retries) if retries else (traj, res, n_iter)


def _adjoint_scan_dense(Js, g):
    """lambda_N = g_N, lambda_t = g_t + Js[t]^T lambda_{t+1} with Js[t] = J_{t+1}, shape (N-1, D, D); g (N, D)."""
    D = g.shape[1]
    As = jnp.concatenate([jnp.zeros((1, D, D), g.dtype), jnp.swapaxes(Js, 1, 2)[::-1]])
    _, lam_rev = lax.associative_scan(vmap(_combine_dense), (As, g[::-1]))
    return lam_rev[::-1]


def _propagate(st: _Static, ode_fn, input_fn, x0, X, g_traj):
    """Adjoint states lambda_1..lambda_N and the x0 cotangent g_traj[0] + J_1^T lambda_1."""
    t_first = _times(st, X.shape[0], X.dtype)[0]
    S1 = lambda z: step_map(st.solver, ode_fn, z, t_first, st.dt, input_fn)
    lambdas = _adjoint_scan_dense(_step_jacobians(st, ode_fn, X, input_fn), g_traj[1:])
    return lambdas, g_traj[0] + jacfwd(S1)(x0).T @ lambdas[0]


def _gate(st: _Static, residual, grads):
    """Zero or poison the cotangents of an unconverged forward (residual > tol)."""
    if st.unconverged == "pass":
        return grads
    ok = residual <= st.tol
    fill = jnp.nan if st.unconverged == "nan" else 0.0
    return jax.tree_util.tree_map(lambda a: jnp.where(ok, a, jnp.asarray(fill, a.dtype)), grads)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _implicit_core(st: _Static, params, x0, initial_guess, inputs):
    return _forward(st, params, x0, initial_guess, inputs)


def _fwd(st, params, x0, initial_guess, inputs):
    traj, residual, n_iter = _forward(st, params, x0, initial_guess, inputs)
    return (traj, residual, n_iter), (params, x0, traj, inputs, residual)


def _adjoint_cotangents(st: _Static, params, x0, traj, inputs, g_traj):
    """The IFT adjoint at the trajectory traj (N+1, D): the (params, x0, inputs) cotangents of g_traj, ungated."""
    X = traj[1:]
    ode_fn = prepare_ode_function(st.make_ode(params))
    lambdas, g_x0 = _propagate(st, ode_fn, _resolve_input_fn(st, inputs), x0, X, g_traj)
    X_prev = jnp.vstack((x0[None], X[:-1]))
    t_prev = _times(st, X.shape[0], X.dtype)[:-1]

    def all_steps(p, u_all):                        # the time-batched one-step map as a function of (params, inputs)
        ode_p = prepare_ode_function(st.make_ode(p))
        input_fn = zoh_input_fn(u_all, st.t0, st.dt) if st.has_inputs else st.input_fn
        return vmap(lambda xp, t: step_map(st.solver, ode_p, xp, t, st.dt, input_fn))(X_prev, t_prev)

    _, pullback = jax.vjp(all_steps, params, inputs)
    g_params, g_inputs = pullback(lambdas)
    return g_params, g_x0, g_inputs


def _bwd(st, res, g):
    params, x0, traj, inputs, residual = res
    g_traj = g[0]                                   # cotangents on residual and n_iter are dropped
    g_params, g_x0, g_inputs = _adjoint_cotangents(st, params, x0, traj, inputs, g_traj)
    g_params, g_x0, g_inputs = _gate(st, residual, (g_params, g_x0, g_inputs))
    return g_params, g_x0, jnp.zeros_like(traj[1:]), (g_inputs if st.has_inputs else None)


_implicit_core.defvjp(_fwd, _bwd)


def implicit_solve(make_ode: Callable, params, x0: jnp.ndarray, initial_guess: jnp.ndarray, dt: float, *,
                   solver: str = "euler", inputs: Optional[jnp.ndarray] = None,
                   input_fn: Optional[Callable] = None, t0: float = 0.0, tol: float, max_iter: int,
                   laser: Optional[LaserConfig] = None, guard: Optional[float] = None, unconverged: str = "zero",
                   stop: str = "parode", scale_rate: float = 0.0, elk_sigma2: float = 0.0):
    """Parallel-in-time solve of dx/dt = make_ode(params)(t, x[, u]) with the IFT adjoint.

    Differentiable in params, x0 and inputs, an (N, d_in) array read by zero-order hold (zoh_input_fn): at each
    stage time t the input is inputs[floor((t - t0)/dt + 1/4)], clipped to the array. Step k, from x_k to x_{k+1},
    reads inputs[k] with euler; with rk4 it reads inputs[k] in its first three stages and inputs[k+1] in its last.
    seq_solve of this module is the sequential rollout with the same reading. input_fn(t) is the
    non-differentiable alternative. tol and max_iter are required; max_iter 0 caps the solve at N iterations.
    The default stop="parode" keeps each loop's own stop (plain DEER when its last update max|delta| <= tol,
    LASER and the guard when max|r| <= tol, every loop at max_iter) and gates on max|r| recomputed from the
    returned trajectory. A residual above tol, which includes every solve with a tol below the rounding level of
    the dtype, gives a zero gradient (unconverged="nan": NaN, "pass": the ungated adjoint).
    guard=C runs plain DEER with the iterate guard (euler and rk4); the paper's runs do not use it.
    stop="laser": LASER's stopping rule for every solver and the gate on the solver's own residual, the stop of
    every solver of the paper's ELCD runs; scale_rate g > 0 (with stop "laser"): constant scale-ELK; elk_sigma2
    s2 > 0 (with stop "laser"): ELK; these two are the scale-ELK and ELK baselines of those runs.
    Returns (traj (N+1, D) including x0, residual max|r|, n_iter).
    """
    st = _make_static(make_ode, x0, initial_guess, dt, solver, inputs, input_fn, t0, tol, max_iter, laser,
                      guard, unconverged, stop, scale_rate, elk_sigma2)
    return _implicit_core(st, params, x0, initial_guess, inputs)


def implicit_forward(make_ode: Callable, params, x0: jnp.ndarray, initial_guess: jnp.ndarray, dt: float, *,
                     solver: str = "euler", inputs: Optional[jnp.ndarray] = None,
                     input_fn: Optional[Callable] = None, t0: float = 0.0, tol: float, max_iter: int,
                     laser: Optional[LaserConfig] = None, guard: Optional[float] = None, stop: str = "parode",
                     scale_rate: float = 0.0, elk_sigma2: float = 0.0, retries: bool = False):
    """implicit_solve's forward alone, (traj, residual, n_iter), with every input behind lax.stop_gradient: no
    gradient flows through it and its while loops are never differentiated (the sequential fallback's Phase A,
    losses.solve_starts_fallback; attach_adjoint gives the final path its gradient). retries: the output gains
    n_retries (int32), LASER's rejected trials over the solve (its hidden scans: under vmap the retry body runs for
    the whole batch when one solve retries), 0 for the loops without retries (DEER, scale-ELK, ELK, guard)."""
    st = _make_static(make_ode, x0, initial_guess, dt, solver, inputs, input_fn, t0, tol, max_iter, laser,
                      guard, "zero", stop, scale_rate, elk_sigma2)
    sg = lax.stop_gradient
    return _forward(st, sg(params), sg(x0), sg(initial_guess), None if inputs is None else sg(inputs), retries)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _attach_core(st: _Static, params, x0, traj, inputs, valid):
    return traj


def _attach_fwd(st, params, x0, traj, inputs, valid):
    return traj, (params, x0, traj, inputs, valid)


def _attach_bwd(st, res, g_traj):
    params, x0, traj, inputs, valid = res
    grads = _adjoint_cotangents(st, params, x0, traj, inputs, g_traj)
    g_params, g_x0, g_inputs = jax.tree_util.tree_map(lambda a: jnp.where(valid, a, jnp.zeros_like(a)), grads)
    return g_params, g_x0, None, (g_inputs if st.has_inputs else None), None   # None: zero for the path and the flag


_attach_core.defvjp(_attach_fwd, _attach_bwd)


def attach_adjoint(make_ode: Callable, params, x0: jnp.ndarray, traj: jnp.ndarray, dt: float, *, valid,
                   solver: str = "euler", inputs: Optional[jnp.ndarray] = None, input_fn: Optional[Callable] = None,
                   t0: float = 0.0):
    """traj (N+1, D), a fixed point of the stepper's recurrence from x0 found without gradient (the parallel solve
    of implicit_forward or the sequential fallback's loop), returned as it is, with implicit_solve's IFT adjoint as
    its derivative in (params, x0, inputs). valid False (a scalar bool; batched under vmap) zeroes the cotangents of
    that path. At a sequential loop's path the adjoint equals backpropagation through the loop up to rounding."""
    st = _make_static(make_ode, x0, traj[1:], dt, solver, inputs, input_fn, t0, 1.0, 1, None, None,
                      "zero")                          # tol and max_iter: placeholders, the adjoint reads neither
    return _attach_core(st, params, x0, traj, inputs, valid)


def seq_solve(make_ode: Callable, params, x0: jnp.ndarray, dt: float, N: int, *, solver: str = "euler",
              inputs: Optional[jnp.ndarray] = None, input_fn: Optional[Callable] = None, t0: float = 0.0):
    """Sequential rollout of the same one-step map, differentiable by ordinary autodiff (BPTT), by
    parode.sequential.seq_integrate. Returns traj (N+1, D) including x0.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if inputs is not None and input_fn is not None:
        raise ValueError("pass either inputs or input_fn, not both")
    if inputs is not None:
        input_fn = zoh_input_fn(inputs, t0, dt)
    ode = make_ode(params)
    return seq_integrate(ode, x0, dt, int(N), solver, t0=t0, input_fn=input_fn)
