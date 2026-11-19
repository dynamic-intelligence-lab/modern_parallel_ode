"""Tests of two of LASER's safeguards, the acceptance-rule options accept_ref="best" and step_cap (parode.laser).

Background: with the growth test of the earlier defaults (accept_ref="prev"), which measures growth against the
previous residual, the loop can accept a 100x growth every iteration and drift to overflow on
systems with fast expansive phases (pre-chaotic Lorenz, cart-pole). Measuring against the best
accepted residual bounds the total excursion; a per-step cap of 2 keeps single steps from jumping
into regions the scaled family cannot leave.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ode_problems import earlier_laser, make_guess, make_system  # noqa: E402
from parode.explicit_parallel import explicit_par_integrate  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402

SAFEGUARDS = dict(accept_ref="best", step_cap=2.0)


def _solve(name, D, N, dt, guess_kind, max_iter=100, **cfg):
    ode, x0, _ = make_system(name, D, jax.random.PRNGKey(0))
    truth = seq_integrate(prepare_ode_function(ode), x0, dt, N, "euler")
    guess = make_guess(guess_kind, x0, truth, N, jax.random.PRNGKey(1))
    sol, res, it, h = explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=1e-9,
                                             laser=earlier_laser(return_history=True, **cfg))
    it = int(it)
    hist = {k: np.asarray(v)[:it] for k, v in h.items()}
    return np.asarray(sol), float(res), it, hist, np.asarray(truth)


def test_config_validation():
    with pytest.raises(ValueError):
        LaserConfig(accept_ref="mean")
    with pytest.raises(ValueError):
        LaserConfig(step_cap=0.5)
    LaserConfig(accept_ref="best", step_cap=2.0)


def test_lorenz_earlier_default_drifts_and_safeguards_converge():
    sol, res, it, hist, truth = _solve("lorenz", 3, 1000, 0.01, "zeros")
    assert not (np.all(np.isfinite(sol)) and res < 1e-9)              # the earlier default rule fails here
    sol, res, it, hist, truth = _solve("lorenz", 3, 1000, 0.01, "zeros", **SAFEGUARDS)
    assert np.all(np.isfinite(sol)) and res < 1e-9 and it <= 40
    assert np.max(np.abs(sol - truth)) < 1e-6


def test_best_reference_bounds_the_total_excursion():
    """Every accepted residual is at most growth_cap times the smallest accepted residual before it."""
    _, res, it, hist, _ = _solve("lorenz", 3, 1000, 0.01, "x0_tiled", accept_ref="best", growth_cap=10.0, step_cap=2.0)
    r = hist["residual"]
    assert res < 1e-9
    running_best = np.minimum.accumulate(np.r_[np.inf, r[:-1]])
    ok = np.isfinite(running_best)
    assert np.all(r[ok] <= 10.0 * running_best[ok] * (1 + 1e-12))


def test_step_cap_bounds_growth_per_step():
    _, res, it, hist, _ = _solve("cartpole", 4, 2000, 0.01, "upright", **SAFEGUARDS)
    r = hist["residual"]
    assert res < 1e-9
    assert np.all(r[1:] <= 2.0 * r[:-1] * (1 + 1e-12))


@pytest.mark.parametrize("name,D,N,dt,guess,max_it", [
    ("cartpole", 4, 2000, 0.01, "zeros", 45), ("matfac", 8, 1000, 0.05, "zeros", 25)])
def test_safeguards_converge_where_the_earlier_default_does_not(name, D, N, dt, guess, max_it):
    sol, res, it, _, truth = _solve(name, D, N, dt, guess, **SAFEGUARDS)
    assert np.all(np.isfinite(sol)) and res < 1e-9 and it <= max_it
    assert np.max(np.abs(sol - truth)) < 1e-6


@pytest.mark.parametrize("name,D,N,dt,guess,plain_iters", [
    ("cubic", 1, 1000, 0.05, "zeros", 7), ("cubic", 1, 1000, 0.05, "wrong_basin", 41)])
def test_safeguards_do_not_change_the_old_cases(name, D, N, dt, guess, plain_iters):
    _, res, it, _, _ = _solve(name, D, N, dt, guess, **SAFEGUARDS)
    assert res < 1e-9 and it <= plain_iters + 1


# Over EARLIER_DEFAULTS these four options give the paper's configuration, LaserConfig()'s defaults.
MEM = dict(accept_ref="best", step_cap=2.0, floor_switch_off=False, propose_first="lle")


def test_persistent_floor_converges_on_lorenz_long_horizon():
    """At N = 1e4 the horizon exponent of the iterate turns negative while the initial stretch is still
    expansive; without the persistent floor the retry ladder restarts from eps every iteration and stalls."""
    sol, res, it, hist, truth = _solve("lorenz", 3, 10000, 0.01, "x0_tiled", **MEM)
    assert np.all(np.isfinite(sol)) and res < 1e-9 and it <= 35
    assert np.max(np.abs(sol - truth)) < 1e-6


@pytest.mark.parametrize("name,D,N,dt,guess,plain_iters", [
    ("cubic", 1, 1000, 0.05, "zeros", 7), ("cubic", 1, 1000, 0.05, "wrong_basin", 41)])
def test_persistent_floor_reaches_zero_on_the_old_cases(name, D, N, dt, guess, plain_iters):
    _, res, it, hist, _ = _solve(name, D, N, dt, guess, **MEM)
    assert res < 1e-9 and it <= plain_iters + 1
    # with the damping proposed first, the persistent floor decays by nu_down per accepted iteration (1.03, 0.26,
    # 0.064, ...), so gamma is negligible rather than exactly zero at the end; the iteration count is unchanged
    assert hist["gamma"][-1] < 1e-2
