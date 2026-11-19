"""fp64 tests of parode/implicit.py (CPU, about a minute on six cores).

T1 IFT gradient against BPTT through seq_solve, T2 LASER and plain DEER share the fixed point, T3 zero cotangents
and the unconverged gate, T5 jit(vmap) batch consistency, T6 input indexing and an input given to an ODE f(t, x)
(explicit_par_integrate and seq_integrate too). T4, the fp32 path, is
tests/test_implicit_fp32.py (a module of its own, since jax_enable_x64 is process-wide).
Random instances are drawn with numpy.random.default_rng and cast at the end (rnn_instance's dtype);
tests/test_implicit_fp32.py solves a double-well instance of its own.
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from dataclasses import replace  # noqa: E402

from parode.explicit_parallel import explicit_par_integrate  # noqa: E402
from parode.implicit import implicit_solve, seq_solve, solver_kwargs, zoh_input_fn  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402

TOL, MAX_ITER = 1e-10, 100
GRAD_RTOL = 1e-9
DT, N, D, D_IN, D_OUT, RADIUS = 0.1, 64, 6, 2, 1, 2.5
WELL_DT, WELL_N, WELL_X0 = 0.05, 80, 0.3          # the largest horizon where plain DEER converges from the tiled guess
WELL3_N = 100                                      # T5 horizon
LASER, LASER_PLAIN, LASER_CARRY = LaserConfig(), LaserConfig(propose_first="plain"), LaserConfig(exponent="carry")
CONFIGS = {"deer": None, "laser-plain": LASER_PLAIN, "laser": LASER, "laser-carry": LASER_CARRY}


def rnn_ode(p):
    return lambda t, h, u: -h + p["W"] @ jnp.tanh(h) + p["B"] @ u


def well_ode(p):
    return lambda t, x: p["a"] * x - x ** 3


def rnn_instance(seed=0, n_steps=N, dtype=jnp.float64):
    """W with spectral radius RADIUS, smooth sine inputs and a sine target; all drawn in numpy, cast at the end."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D, D)) / np.sqrt(D)
    W = W * (RADIUS / np.max(np.abs(np.linalg.eigvals(W))))
    params = {"W": W, "B": rng.standard_normal((D, D_IN)) / np.sqrt(D_IN),
              "C": rng.standard_normal((D_OUT, D)) / np.sqrt(D)}
    h0 = 0.1 * rng.standard_normal(D)
    t = np.arange(n_steps) * DT
    phases = t[:, None, None] * rng.uniform(0.1, 1.0, (D_IN, 3)) + rng.uniform(0.0, 2 * np.pi, (D_IN, 3))
    inputs = np.sum(np.sin(phases), -1)
    targets = np.sin(0.3 * t)[:, None] * np.ones((1, D_OUT))
    cast = lambda a: jnp.asarray(a, dtype)
    return jax.tree_util.tree_map(cast, params), cast(h0), cast(inputs), cast(targets)


PARAMS, H0, INPUTS, TARGETS = rnn_instance()
ZERO_GUESS = jnp.zeros((N, D))
TILED_GUESS = jnp.tile(H0, (N, 1))


def readout_loss(traj, p, targets):
    return jnp.mean((traj[1:] @ p["C"].T - targets) ** 2)


def leaf_rel_errors(a, b):
    """Relative L2 error per leaf of two pytrees with the same structure."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return [float(jnp.linalg.norm(x - y) / jnp.linalg.norm(y)) for x, y in zip(la, lb)]


def rel_err(a, b):
    return max(leaf_rel_errors(a, b))


def par_loss(p, h0, u, guess, solver, cfg, max_iter=MAX_ITER, **kw):
    traj, res, it = implicit_solve(rnn_ode, p, h0, guess, DT, solver=solver, inputs=u, tol=TOL, max_iter=max_iter,
                                   laser=cfg, **kw)
    return readout_loss(traj, p, TARGETS), (res, it, traj)


def seq_loss(p, h0, u, solver):
    return readout_loss(seq_solve(rnn_ode, p, h0, DT, N, solver=solver, inputs=u), p, TARGETS)


_REF = {}


def seq_reference(solver):
    if solver not in _REF:
        _REF[solver] = jax.jit(jax.grad(seq_loss, argnums=(0, 1, 2)), static_argnums=3)(PARAMS, H0, INPUTS, solver)
    return _REF[solver]


# ----------------------------------------------------------------------------- T1

@pytest.mark.parametrize("solver,cfg_name", [(s, c) for s in ("euler", "rk4") for c in CONFIGS])
def test_t1_ift_gradient_matches_bptt(solver, cfg_name):
    ref = seq_reference(solver)
    fn = jax.jit(jax.value_and_grad(par_loss, argnums=(0, 1, 2), has_aux=True), static_argnums=(4, 5))
    (_, (res, it, traj)), grads = fn(PARAMS, H0, INPUTS, ZERO_GUESS, solver, CONFIGS[cfg_name])
    assert bool(jnp.all(jnp.isfinite(traj))) and float(res) <= TOL and int(it) < MAX_ITER
    errs = leaf_rel_errors(grads, ref)             # B, C, W, h0, inputs
    assert max(errs) < GRAD_RTOL, errs


# ----------------------------------------------------------------------------- T2

def _solve_all(make_ode, p, x0, guess, dt, **kw):
    out = {name: implicit_solve(make_ode, p, x0, guess, dt, tol=TOL, max_iter=MAX_ITER, laser=cfg, **kw)
           for name, cfg in CONFIGS.items()}
    for name, (traj, res, it) in out.items():
        assert bool(jnp.all(jnp.isfinite(traj))), name
        assert float(res) <= TOL and int(it) < MAX_ITER, (name, float(res), int(it))
    return out


def test_t2_laser_fixed_point_equals_plain_rnn():
    out = _solve_all(rnn_ode, PARAMS, H0, TILED_GUESS, DT, inputs=INPUTS)
    for name in ("laser-plain", "laser", "laser-carry"):
        assert float(jnp.max(jnp.abs(out[name][0] - out["deer"][0]))) < 1e-8, name


def test_t2_laser_fixed_point_equals_plain_double_well():
    p, x0 = {"a": jnp.asarray(1.0)}, jnp.asarray([WELL_X0])
    out = _solve_all(well_ode, p, x0, jnp.tile(x0, (WELL_N, 1)), WELL_DT)
    truth = seq_solve(well_ode, p, x0, WELL_DT, WELL_N)
    for name in ("laser-plain", "laser", "laser-carry"):
        assert float(jnp.max(jnp.abs(out[name][0] - out["deer"][0]))) < 1e-8, name
    assert float(jnp.max(jnp.abs(out["deer"][0] - truth))) < 1e-8


# ----------------------------------------------------------------------------- T3

def _params_grad(guess, cfg=LASER_PLAIN, **kw):
    def loss(p):
        return par_loss(p, H0, INPUTS, guess, "euler", cfg, **kw)[0]
    return jax.jit(jax.grad(loss))(PARAMS)


_G_ZERO = {}


def laser_grad_from_zero():
    """dL/dparams of the LASER_PLAIN Euler solve from the zero guess, with the initial-guess cotangent (one compile)."""
    if not _G_ZERO:
        loss = lambda p, guess: par_loss(p, H0, INPUTS, guess, "euler", LASER_PLAIN)[0]
        _G_ZERO["params"], _G_ZERO["guess"] = jax.jit(jax.grad(loss, argnums=(0, 1)))(PARAMS, ZERO_GUESS)
    return _G_ZERO["params"], _G_ZERO["guess"]


def test_t3_initial_guess_gets_zero_cotangent():
    _, g_guess = laser_grad_from_zero()
    assert g_guess.shape == ZERO_GUESS.shape and bool(jnp.all(g_guess == 0.0))


def test_t3_gradient_independent_of_guess_and_of_aux_outputs():
    g_zero, _ = laser_grad_from_zero()
    assert rel_err(_params_grad(TILED_GUESS), g_zero) < 1e-12

    def loss_with_aux(p):
        loss, (res, it, _) = par_loss(p, H0, INPUTS, ZERO_GUESS, "euler", LASER_PLAIN)
        return loss + res + it.astype(loss.dtype)
    assert rel_err(jax.jit(jax.grad(loss_with_aux))(PARAMS), g_zero) < 1e-12


def test_t3_unconverged_gate():
    def loss(p, mode):
        return par_loss(p, H0, INPUTS, ZERO_GUESS, "euler", LASER_PLAIN, max_iter=2, unconverged=mode)
    grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True), static_argnums=1)
    (_, (res, it, _)), g_zero = grad_fn(PARAMS, "zero")
    _, g_nan = grad_fn(PARAMS, "nan")
    _, g_pass = grad_fn(PARAMS, "pass")
    assert float(res) > TOL and int(it) == 2
    # The gate acts on what flows through the solve (W, B, h0, inputs). dL/dC is the readout's own gradient and
    # stays finite, so a training loop must skip the whole step on converged == False, not rely on zeros.
    ode_leaves = lambda g: [g["W"], g["B"]]
    assert all(bool(jnp.all(g == 0.0)) for g in ode_leaves(g_zero))
    assert all(bool(jnp.all(jnp.isnan(g))) for g in ode_leaves(g_nan))
    assert all(bool(jnp.all(jnp.isfinite(g))) and bool(jnp.any(g != 0.0)) for g in ode_leaves(g_pass))
    assert bool(jnp.all(jnp.isfinite(g_zero["C"]))) and bool(jnp.all(g_zero["C"] == g_pass["C"]))


def test_t3_argument_validation():
    with pytest.raises(ValueError):
        implicit_solve(rnn_ode, PARAMS, H0, ZERO_GUESS, DT, inputs=INPUTS, tol=TOL, max_iter=MAX_ITER, guard=20.0,
                       laser=LASER_PLAIN)
    with pytest.raises(ValueError):
        implicit_solve(rnn_ode, PARAMS, H0, ZERO_GUESS, DT, inputs=INPUTS, tol=TOL, max_iter=MAX_ITER,
                       laser=replace(LASER_PLAIN, return_history=True))
    with pytest.raises(TypeError):                  # tol and max_iter are the solver's arguments, not LASER's
        replace(LASER_PLAIN, tol=1e-6)
    with pytest.raises(TypeError):                  # and they have no default
        implicit_solve(rnn_ode, PARAMS, H0, ZERO_GUESS, DT, inputs=INPUTS, laser=LASER_PLAIN)
    with pytest.raises(ValueError):
        implicit_solve(rnn_ode, PARAMS, H0, ZERO_GUESS, DT, inputs=INPUTS, tol=TOL, max_iter=MAX_ITER,
                       input_fn=zoh_input_fn(INPUTS, 0.0, DT))
    with pytest.raises(ValueError):
        solver_kwargs("seq", 20.0)
    assert solver_kwargs("guard", 20.0) == {"guard": 20.0} and solver_kwargs("laser", 20.0) == {"laser": LASER}
    assert solver_kwargs("laser-carry", 20.0) == {"laser": LASER_CARRY} and LASER_CARRY.exponent == "carry"


# ----------------------------------------------------------------------------- T5

def test_t5_jit_vmap_batch_consistency():
    p, dt = {"a": jnp.asarray(1.0)}, WELL_DT
    x0s = jnp.asarray([[0.3], [0.2], [-0.25], [0.35]])
    guess = jnp.zeros((WELL3_N, 1))

    def loss_one(p, x0):
        traj, res, it = implicit_solve(well_ode, p, x0, guess, dt, tol=TOL, max_iter=MAX_ITER, laser=LASER_PLAIN)
        return jnp.sum(traj ** 2), (res, it)

    def batch_loss(p):
        losses, aux = jax.vmap(lambda x0: loss_one(p, x0))(x0s)
        return jnp.mean(losses), aux

    (_, (res, it)), g_batch = jax.jit(jax.value_and_grad(batch_loss, has_aux=True))(p)
    g_single = jax.jit(jax.grad(lambda p, x0: loss_one(p, x0)[0]))
    g_mean = jnp.mean(jnp.stack([g_single(p, x0)["a"] for x0 in x0s]))
    assert res.shape == (4,) and it.shape == (4,) and bool(jnp.all(res <= TOL))
    assert abs(float(g_batch["a"] - g_mean)) / abs(float(g_mean)) < 1e-12


# ----------------------------------------------------------------------------- T6

def test_t6_input_indexing():
    k = 10
    one_hot = jnp.zeros((N, D_IN)).at[k, 0].set(1.0)
    base = seq_solve(rnn_ode, PARAMS, H0, DT, N, inputs=jnp.zeros((N, D_IN)))
    kicked = seq_solve(rnn_ode, PARAMS, H0, DT, N, inputs=one_hot)
    traj, res, _ = implicit_solve(rnn_ode, PARAMS, H0, ZERO_GUESS, DT, inputs=one_hot, tol=TOL, max_iter=200,
                                 laser=LASER_PLAIN)
    assert bool(jnp.all(kicked[: k + 1] == base[: k + 1]))               # inputs[k] drives step k + 1 only
    assert float(jnp.max(jnp.abs(kicked[k + 1] - base[k + 1]))) > 1e-3
    assert float(res) <= TOL and float(jnp.max(jnp.abs(traj - kicked))) < 1e-8
    u = zoh_input_fn(INPUTS, 0.0, DT)
    times = DT * jnp.arange(N)
    assert bool(jnp.all(jax.vmap(u)(times + 0.5 * DT) == INPUTS))         # the RK4 half step reads seq[k]
    assert bool(jnp.all(jax.vmap(u)(times) == INPUTS))              # the gradient w.r.t. inputs is asserted per case in T1


def test_t6_an_input_given_to_an_ode_of_t_x_raises():
    """An ODE f(t, x) given an input (input_fn= or inputs=) raises ValueError; f(t, x, u) with an input and f(t, x)
    without one still solve."""
    n, x0, p = 50, jnp.asarray([0.5]), jnp.asarray(1.0)
    guess, u = jnp.zeros((n, 1)), jnp.ones((n, 1))
    u_fn = lambda t: u[0]
    decay = lambda p: (lambda t, x: -p * x)
    driven = lambda p: (lambda t, x, v: -p * x + v)
    for call in (lambda: explicit_par_integrate(decay(p), x0, guess, DT, MAX_ITER, "rk4", input_fn=u_fn, tol=TOL),
                 lambda: seq_integrate(decay(p), x0, DT, n, "rk4", input_fn=u_fn),
                 lambda: implicit_solve(decay, p, x0, guess, DT, solver="rk4", inputs=u, tol=TOL, max_iter=MAX_ITER)):
        with pytest.raises(ValueError, match=r"f\(t, x, u\)"):
            call()
    for make_ode, kw, kw_implicit in ((driven, {"input_fn": u_fn}, {"inputs": u}), (decay, {}, {})):
        ref = seq_integrate(make_ode(p), x0, DT, n, "rk4", **kw)
        sol, _, _ = explicit_par_integrate(make_ode(p), x0, guess, DT, MAX_ITER, "rk4", tol=TOL, **kw)
        traj, res, _ = implicit_solve(make_ode, p, x0, guess, DT, solver="rk4", tol=TOL, max_iter=MAX_ITER,
                                      **kw_implicit)
        assert float(res) <= TOL
        assert float(jnp.max(jnp.abs(sol - ref))) < 1e-10 and float(jnp.max(jnp.abs(traj - ref))) < 1e-10
