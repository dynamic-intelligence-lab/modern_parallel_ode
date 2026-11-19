"""The paper runs' solve options in parode/implicit.py (CPU, about two minutes).

stop "laser": plain DEER and constant scale-ELK on _shared_loop with LASER's stopping rule, LASER with its own
residual; max_iter 0, the solve's own length; implicit_forward (no gradient) and attach_adjoint (the IFT adjoint at a
given path). fp64, except the fp32 divergence test, which runs under disable_x64 as tests/test_implicit_fp32.py does.
"""
import math
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from parode import implicit as I  # noqa: E402
from parode.implicit import attach_adjoint, implicit_forward, implicit_solve, seq_solve  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402
from parode.explicit_parallel import affine_terms, linear_terms, par_init, par_scan, rk4_step  # noqa: E402

TOL, MAX_ITER = 1e-10, 100
LASER = LaserConfig()
DT, N, D, D_IN = 0.1, 64, 6, 2
WELL_DT, WELL_X0 = 0.05, 0.3


def rnn_ode(p):
    return lambda t, h, u: -h + p["W"] @ jnp.tanh(h) + p["B"] @ u


def well_ode(p):
    return lambda t, x: p["a"] * x - x ** 3


def rnn_instance(seed=0):
    """W with spectral radius 2.5 and B as in tests/test_implicit.py's instance; it draws no readout C from the same
    generator, so h0 and the smooth sine inputs are other draws than that module's."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D, D)) / np.sqrt(D)
    W = W * (2.5 / np.max(np.abs(np.linalg.eigvals(W))))
    params = {"W": jnp.asarray(W), "B": jnp.asarray(rng.standard_normal((D, D_IN)) / np.sqrt(D_IN))}
    h0 = jnp.asarray(0.1 * rng.standard_normal(D))
    t = np.arange(N) * DT
    phases = t[:, None, None] * rng.uniform(0.1, 1.0, (D_IN, 3)) + rng.uniform(0.0, 2 * np.pi, (D_IN, 3))
    return params, h0, jnp.asarray(np.sum(np.sin(phases), -1))


PARAMS, H0, INPUTS = rnn_instance()
ZERO = jnp.zeros((N, D))


def _solve(guess=ZERO, **kw):
    kw = dict(dict(solver="rk4", inputs=INPUTS, tol=TOL, max_iter=MAX_ITER), **kw)
    return jax.jit(lambda p, h, g: implicit_solve(rnn_ode, p, h, g, DT, **kw))(PARAMS, H0, guess)


def _loop_path():
    return jax.jit(lambda p, h: seq_solve(rnn_ode, p, h, DT, N, solver="rk4", inputs=INPUTS))(PARAMS, H0)


# ----------------------------------------------------------------------------- the shared loop

@pytest.mark.parametrize("scale_rate", [0.0, 1.0])
def test_shared_loop_converges_to_the_loop_path(scale_rate):
    traj, res, it = _solve(stop="laser", scale_rate=scale_rate)
    assert float(res) <= TOL and 0 < int(it) < MAX_ITER
    assert float(jnp.max(jnp.abs(traj - _loop_path()))) <= 1e-8
    # its own residual is the recomputed one up to rounding
    recomputed = I._residual(I._make_static(rnn_ode, H0, ZERO, DT, "rk4", INPUTS, None, 0.0, TOL, MAX_ITER, None, None,
                                            "zero"), I.prepare_ode_function(rnn_ode(PARAMS)), traj,
                             I.zoh_input_fn(INPUTS, 0.0, DT))
    assert abs(float(res) - float(recomputed)) <= 1e-13


def test_shared_deer_stops_at_the_first_iterate_that_passes():
    """LASER's rule: the loop stops at the first iterate with max|r| <= tol, so every smaller cap leaves the residual
    above tol (the guess itself does not pass: the loop made n >= 2 iterations)."""
    _, res, it = _solve(stop="laser")
    n = int(it)
    assert float(res) <= TOL and n >= 2
    for k in range(1, n):
        _, res_k, it_k = _solve(stop="laser", max_iter=k)
        assert int(it_k) == k and float(res_k) > TOL, k


@pytest.mark.parametrize("cfg", ["deer", "scale-elk", "laser"])
def test_a_converged_guess_costs_no_iteration(cfg):
    kw = dict(deer={}, **{"scale-elk": dict(scale_rate=1.0), "laser": dict(laser=LASER)})[cfg]
    truth = _loop_path()
    traj, res, it = _solve(guess=truth[1:], stop="laser", **kw)
    assert int(it) == 0 and float(res) <= TOL and bool(jnp.all(traj == truth))


def test_scale_elk_step_is_the_scan_on_the_damped_jacobians():
    """One iteration of scale-ELK at rate g from the zero guess: X_1 = X_0 + scan(exp(-g dt) F(X_0), c(X_0))."""
    g = 2.0
    traj, _, it = _solve(stop="laser", scale_rate=g, max_iter=1)
    ode = I.prepare_ode_function(rnn_ode(PARAMS))
    input_fn = I.zoh_input_fn(INPUTS, 0.0, DT)
    times = DT * jnp.arange(N + 1)

    @jax.jit
    def by_hand(X):
        F = linear_terms(X, ode, DT, rk4_step, times, input_fn)
        c = affine_terms(X, H0, ode, DT, rk4_step, times, input_fn)
        _, cs = par_scan(par_init(F * math.exp(-g * DT), c[1:], c[0]))
        return X + jnp.vstack((c[0], cs))
    assert int(it) == 1
    assert float(jnp.max(jnp.abs(traj[1:] - by_hand(ZERO)))) <= 1e-13


def test_scale_elk_at_rate_zero_is_plain_deer():
    a, b = _solve(stop="laser"), _solve(stop="laser", scale_rate=0.0)
    assert all(bool(jnp.all(x == y)) for x, y in zip(a, b))


def test_laser_reports_its_own_residual_on_the_same_path():
    traj_p, res_p, it_p = _solve(laser=LASER)
    traj_l, res_l, it_l = _solve(laser=LASER, stop="laser")
    assert bool(jnp.all(traj_p == traj_l)) and int(it_p) == int(it_l)
    assert float(res_l) <= TOL and abs(float(res_l) - float(res_p)) <= 1e-13


# ----------------------------------------------------------------------------- the cap

def test_max_iter_zero_is_the_solve_length():
    st = I._make_static(rnn_ode, H0, ZERO, DT, "rk4", INPUTS, None, 0.0, TOL, 0, None, None, "zero")
    assert st.max_iter == N


def test_a_solve_that_needs_more_than_200_iterations_converges_under_cap_n():
    """Near-Jacobi scale-ELK (exp(-g dt) about 2e-22) from the unstable equilibrium x = 0 of the double well moves its
    exact front one step per sweep, so it needs about N sweeps: cap 200 stops it unconverged, cap 0 (N) does not."""
    n = 300
    params, x0, guess = {"a": jnp.asarray(1.0)}, jnp.asarray([WELL_X0]), jnp.zeros((n, 1))
    run = lambda cap: jax.jit(lambda p: implicit_solve(well_ode, p, x0, guess, WELL_DT, solver="rk4", tol=TOL,
                                                       max_iter=cap, stop="laser", scale_rate=1e3))(params)
    _, res200, it200 = run(200)
    _, res_n, it_n = run(0)
    assert int(it200) == 200 and float(res200) > TOL
    assert 200 < int(it_n) <= n and float(res_n) <= TOL


# ----------------------------------------------------------------------------- forward and attach

def test_implicit_forward_passes_no_gradient():
    def f(p, h):
        traj, _, _ = implicit_forward(rnn_ode, p, h, ZERO, DT, solver="rk4", inputs=INPUTS, tol=TOL,
                                      max_iter=MAX_ITER, stop="laser")
        return jnp.sum(traj ** 2)
    gp, gh = jax.jit(jax.grad(f, argnums=(0, 1)))(PARAMS, H0)
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree_util.tree_leaves((gp, gh)))


@pytest.mark.parametrize("kw", [dict(), dict(laser=LASER), dict(laser=LaserConfig(exponent="carry")),
                                dict(stop="laser", scale_rate=1.0)])
def test_implicit_forward_reports_the_retries_as_an_extra_output(kw):
    """retries=True appends n_retries (int32, >= 0; 0 for the loops without retries) and leaves (traj, residual,
    n_iter) as they are, under jit and under vmap."""
    run = lambda retries: jax.jit(lambda p, h: implicit_forward(rnn_ode, p, h, ZERO, DT, solver="rk4", inputs=INPUTS,
                                                                tol=TOL, max_iter=MAX_ITER, retries=retries, **kw))
    out3 = run(False)(PARAMS, H0)
    traj, res, it, nr = run(True)(PARAMS, H0)
    assert len(out3) == 3 and float(res) <= TOL
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out3, (traj, res, it)))
    assert nr.dtype == jnp.int32 and nr.shape == () and int(nr) >= 0
    assert "laser" in kw or int(nr) == 0
    batched = jax.jit(jax.vmap(lambda h: implicit_forward(rnn_ode, PARAMS, h, ZERO, DT, solver="rk4", inputs=INPUTS,
                                                          tol=TOL, max_iter=MAX_ITER, retries=True, **kw)[3]))
    assert np.array_equal(np.asarray(batched(jnp.stack([H0, H0]))), np.array([int(nr)] * 2))


def test_attach_at_the_loop_path_is_backpropagation_through_the_loop():
    def loss_of(traj):
        return jnp.sum(jnp.sin(traj[1:]) * jnp.arange(1, N + 1)[:, None] / N)

    def bptt(p, h, u):
        return loss_of(seq_solve(rnn_ode, p, h, DT, N, solver="rk4", inputs=u))

    def attached(p, h, u, valid):
        path = jax.lax.stop_gradient(seq_solve(rnn_ode, p, h, DT, N, solver="rk4", inputs=u))
        return loss_of(attach_adjoint(rnn_ode, p, h, path, DT, valid=valid, solver="rk4", inputs=u))

    want = jax.jit(jax.grad(bptt, argnums=(0, 1, 2)))(PARAMS, H0, INPUTS)
    got = jax.jit(jax.grad(attached, argnums=(0, 1, 2)))(PARAMS, H0, INPUTS, jnp.bool_(True))
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        assert float(jnp.linalg.norm(g - w) / jnp.linalg.norm(w)) <= 1e-9
    off = jax.jit(jax.grad(attached, argnums=(0, 1, 2)))(PARAMS, H0, INPUTS, jnp.bool_(False))
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree_util.tree_leaves(off))


def test_attach_is_the_identity_forward():
    path = _loop_path()
    out = jax.jit(lambda p, h: attach_adjoint(rnn_ode, p, h, path, DT, valid=jnp.bool_(True), solver="rk4",
                                              inputs=INPUTS))(PARAMS, H0)
    assert bool(jnp.all(out == path))


# ----------------------------------------------------------------------------- validation

@pytest.mark.parametrize("kw", [dict(stop="newton"), dict(stop="laser", guard=10.0), dict(scale_rate=1.0),
                                dict(elk_sigma2=1.0), dict(stop="laser", elk_sigma2=-1.0),
                                dict(stop="laser", elk_sigma2=1.0, scale_rate=1.0),
                                dict(stop="laser", elk_sigma2=1.0, laser=LASER),
                                dict(stop="laser", scale_rate=-1.0), dict(stop="laser", scale_rate=math.inf),
                                dict(stop="laser", scale_rate=1.0, laser=LASER), dict(max_iter=-1)])
def test_invalid_options_are_refused(kw):
    with pytest.raises(ValueError):
        implicit_solve(rnn_ode, PARAMS, H0, ZERO, DT,
                       **dict(dict(solver="rk4", inputs=INPUTS, tol=TOL, max_iter=MAX_ITER), **kw))


# ----------------------------------------------------------------------------- fp32: a diverging solve

def test_fp32_shared_deer_stops_at_the_first_non_finite_step():
    """tests/test_implicit_fp32.py's failing example (double well, N 400, zero guess, tol 1e-5): parode's plain DEER
    returns a non-finite path. Under stop "laser" the shared loop stops at the first non-finite step and returns the
    last finite iterate with residual inf (as LASER does when no trial is finite), and the gate zeroes its gradient."""
    with jax.experimental.disable_x64():
        params = {"a": jnp.asarray(np.float32(1.0))}
        x0, guess = jnp.asarray(np.array([WELL_X0], np.float32)), jnp.zeros((400, 1), jnp.float32)
        solve = lambda p: implicit_solve(well_ode, p, x0, guess, WELL_DT, solver="rk4", tol=1e-5, max_iter=200,
                                         stop="laser")
        traj, res, it = jax.jit(solve)(params)
        assert traj.dtype == jnp.float32 and res.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(traj))) and bool(jnp.isinf(res)) and 1 <= int(it) <= 200
        grad = jax.jit(jax.grad(lambda p: jnp.sum(solve(p)[0] ** 2)))(params)
        assert float(grad["a"]) == 0.0


# ----------------------------------------------------------------------------- the stall exit

def contracting_ode(p):
    return lambda t, x: -p["k"] * x - x ** 3


@pytest.mark.parametrize("dtype,tol", [("fp64", 1e-20), ("fp32", 1e-12)])
@pytest.mark.parametrize("cfg", ["deer", "scale-elk", "laser"])
def test_a_tolerance_below_the_rounding_floor_ends_in_a_stall_not_at_the_cap(dtype, tol, cfg):
    """x' = -5 x - x^3 contracts by about 0.78 per step, so near its fixed point the Newton update is rounding noise
    of a few ulp: with tol below the dtype's floor every solver must leave by LASER's stall test (an update at most
    8 eps max(1, max|X|)), early, with a finite residual above tol, instead of iterating to the cap."""
    import contextlib
    kw = dict(deer={}, **{"scale-elk": dict(scale_rate=1.0), "laser": dict(laser=LASER)})[cfg]
    ctx = jax.experimental.disable_x64() if dtype == "fp32" else contextlib.nullcontext()
    with ctx:
        f = np.float32 if dtype == "fp32" else np.float64
        params, x0 = {"k": jnp.asarray(f(5.0))}, jnp.asarray(np.array([1.0], f))
        guess = jnp.zeros((60, 1), x0.dtype)
        traj, res, it = jax.jit(lambda p: implicit_solve(contracting_ode, p, x0, guess, 0.05, solver="rk4", tol=tol,
                                                         max_iter=1000, stop="laser", **kw))(params)
        assert traj.dtype == x0.dtype and bool(jnp.all(jnp.isfinite(traj)))
        assert bool(jnp.isfinite(res)) and float(res) > tol and int(it) < 50
        assert float(res) <= 100 * float(np.finfo(f).eps)               # it did reach the floor


def test_stop_laser_gates_on_the_solvers_own_residual_without_recomputing(monkeypatch):
    """Under stop 'laser' no solver recomputes the residual (the gate reads the one the stop test used); the default
    path still does."""
    def boom(*args, **kw):
        raise AssertionError("_residual was called")
    monkeypatch.setattr(I, "_residual", boom)
    for kw in (dict(stop="laser"), dict(stop="laser", scale_rate=1.0), dict(stop="laser", laser=LASER)):
        _, res, _ = _solve(**kw)
        assert float(res) <= TOL
    with pytest.raises(AssertionError):
        _solve()


# ----------------------------------------------------------------------------- ELK

def _first_terms(X):
    ode = I.prepare_ode_function(rnn_ode(PARAMS))
    input_fn = I.zoh_input_fn(INPUTS, 0.0, DT)
    times = DT * jnp.arange(N + 1)
    return (linear_terms(X, ode, DT, rk4_step, times, input_fn),
            affine_terms(X, H0, ode, DT, rk4_step, times, input_fn))


def test_elk_with_a_huge_variance_takes_the_deer_step():
    """lambda_LM = 1 / s2 -> 0: one ELK iteration is the plain DEER step (the elk_baseline test, on the RK4 map). The
    instance expands, so the filter's covariance grows to about 1e8 over the 64 steps: s2 must dwarf that (at 1e12 the
    observations still pull with weight about 1e-4)."""
    deer, _, _ = _solve(stop="laser", max_iter=1)
    elk, _, it = _solve(stop="laser", max_iter=1, elk_sigma2=1e24)
    assert int(it) == 1 and float(jnp.max(jnp.abs(elk - deer))) <= 1e-8 * float(jnp.max(jnp.abs(deer)))


def test_elk_delta_form_is_the_released_absolute_form():
    """The loop's update, the filtered means of the model shifted by the iterate (delta form), is the released
    elk_alg's absolute form: filter(F, b = c[1:] + X[1:] - F X[:-1], m0 = c[0] + X[0], y = X) - X."""
    from parode.elk_baseline import elk_filter_dense
    s2 = 0.5
    X = jnp.asarray(np.random.default_rng(3).standard_normal((N, D)) * 0.1)
    F, c = jax.jit(_first_terms)(X)
    absolute = jax.jit(lambda F, c, X: elk_filter_dense(F, c[1:] + X[1:] - jnp.einsum("tij,tj->ti", F, X[:-1]),
                                                        c[0] + X[0], X, s2))(F, c, X)
    delta = jax.jit(lambda F, c: elk_filter_dense(F, c[1:], c[0], jnp.zeros_like(c), s2))(F, c)
    assert float(jnp.max(jnp.abs((X + delta) - absolute))) <= 1e-12 * max(1.0, float(jnp.max(jnp.abs(absolute))))
    traj, _, _ = _solve(guess=X, stop="laser", max_iter=1, elk_sigma2=s2)
    assert float(jnp.max(jnp.abs(traj[1:] - absolute))) <= 1e-12 * max(1.0, float(jnp.max(jnp.abs(absolute))))


def test_elk_converges_to_the_loop_path():
    traj, res, it = _solve(stop="laser", elk_sigma2=100.0, max_iter=500)
    assert float(res) <= TOL and int(it) < 500
    assert float(jnp.max(jnp.abs(traj - _loop_path()))) <= 1e-8
