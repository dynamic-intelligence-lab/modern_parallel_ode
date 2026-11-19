"""LASER's iteration order, checked against the previous one: the Jacobians and
the exponent of the current iterate are formed at the top of the iteration, so a solve of K accepted iterations runs
K Jacobian passes and K exponents instead of K + 1 of each, and no number changes.

The references under tests/data/laser_reference.npz were generated with the previous order (the trial's terms_fn
formed the new iterate's Jacobians and the end of the body its exponent) by this module's generate(), on the CPU in
fp64; that version of parode/laser.py is not in this repository. The file holds 12 cases: the nine of CASES and the
vmapped chunk, which the tests compare, and two that no test reads (cubic_cold_lle_first_try,
expansive_linear_window).

  JAX_PLATFORMS=cpu python tests/test_laser_reference.py --generate OUT.npz

writes the ten compared cases from the code as it is. Every output of run_damped_newton (the iterate, the residual,
the iterations, the retries and the whole history, history_extra included) must be bit for bit the same. The cases
cover the exact and the carry exponent, plain first, the Jacobi-only loop, a trust radius, a caller without
residual_fn (the Jacobians stay carried there) and a vmapped chunk; the cubic cold guess runs the retry path with
lam_known, the Lorenz guess rejects and retries many times.
"""
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from parode.explicit_parallel import affine_terms, linear_terms, par_init, par_scan, step_functions  # noqa: E402
from parode.laser import LaserConfig, run_damped_newton  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from ode_problems import earlier_laser, euler_terms  # noqa: E402

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "laser_reference.npz")
TOL = 1e-9
MAX_ITER = 60
LASER = LaserConfig()                    # the paper's configuration (the defaults)
HISTORY = dict(return_history=True, history_extra=True)


def cubic(t, x, u=None):
    return x - x ** 3


def contractive(t, x, u=None):
    return -x + 0.3 * jnp.tanh(x) + jnp.sin(t)


A_EXP = jnp.array([[0.5, 1.0], [0.0, 0.3]])


def expansive_linear(t, x, u=None):
    return A_EXP @ x


def lorenz(t, x, u=None):
    return jnp.array([10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - 8.0 / 3.0 * x[2]])


ODES = dict(cubic=cubic, contractive=contractive, expansive_linear=expansive_linear, lorenz=lorenz)

# name -> (ode, x0, N, dt, guess rule, cfg, with residual_fn). N - 1 is a multiple of min_block for history_extra.
CASES = {
    "cubic_cold_lle": ("cubic", [1.5], 401, 0.05, "zeros", LASER, True),
    "cubic_cold_lle_no_residual_fn": ("cubic", [1.5], 401, 0.05, "zeros", LASER, False),
    "cubic_cold_carry": ("cubic", [1.5], 401, 0.05, "zeros", replace(LASER, exponent="carry"), True),
    "cubic_cold_plain": ("cubic", [1.5], 401, 0.05, "zeros", replace(LASER, propose_first="plain"), True),
    "cubic_cold_jacobi_only": ("cubic", [1.5], 401, 0.05, "zeros", earlier_laser(propose_first="plain", n_retry=0),
                               True),
    "contractive_lle": ("contractive", [0.7, -0.4], 101, 0.1, "zeros", earlier_laser(propose_first="lle"), True),
    "expansive_linear_lle": ("expansive_linear", [1.0, 1.0], 201, 0.05, "zeros", LASER, True),
    "lorenz_lle": ("lorenz", [1.0, 1.0, 1.0], 201, 0.005, "x0", LASER, True),
    "lorenz_trust": ("lorenz", [1.0, 1.0, 1.0], 201, 0.005, "x0", replace(LASER, trust_radius=10.0), True),
}
VMAP_X0S = [[1.5], [1.2], [-1.4], [0.8]]          # the vmapped chunk: cubic, LASER, N 201, dt 0.05, zero guess


def _guess(rule, x0, N):
    return jnp.zeros((N, x0.shape[0])) if rule == "zeros" else jnp.tile(x0[None], (N, 1))


def _run(name):
    """(X, residual, iterations, retries, history) of a case, as numpy."""
    ode, x0, N, dt, rule, cfg, with_residual = CASES[name]
    x0 = jnp.array(x0)
    terms_fn, residual_fn, scan_fn = euler_terms(ODES[ode], x0, dt, N)
    cfg = replace(cfg, **HISTORY)
    out = jax.jit(lambda g: run_damped_newton(g, terms_fn, scan_fn, dt, cfg, TOL, MAX_ITER,
                                              residual_fn=residual_fn if with_residual else None))(_guess(rule, x0, N))
    return _np(out)


def _run_vmap():
    N, dt = 201, 0.05
    cfg = replace(LASER, **HISTORY)

    def solve(x0):
        terms_fn, residual_fn, scan_fn = euler_terms(cubic, x0, dt, N)
        return run_damped_newton(jnp.zeros((N, 1)), terms_fn, scan_fn, dt, cfg, TOL, MAX_ITER, residual_fn=residual_fn)

    return _np(jax.jit(jax.vmap(solve))(jnp.array(VMAP_X0S)))


def _np(out):
    X, res, it, nr, hist = out
    flat = {"X": np.asarray(X), "res": np.asarray(res), "it": np.asarray(it), "nr": np.asarray(nr)}
    flat.update({f"hist_{k}": np.asarray(v) for k, v in hist.items()})
    return flat


def _all_runs():
    runs = {name: _run(name) for name in CASES}
    runs["cubic_vmap_lle"] = _run_vmap()
    return runs


def generate(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {}
    for name, flat in _all_runs().items():
        for k, v in flat.items():
            arrays[f"{name}/{k}"] = v
        h = flat["hist_n_trials"]
        print(f"{name}: residual {float(np.max(flat['res'])):.2e}, iterations {flat['it'].tolist()}, "
              f"retries {flat['nr'].tolist()}, largest n_trials {float(np.max(h)):g}")
    np.savez(path, **arrays)
    print("wrote", path, len(arrays), "arrays")


@pytest.fixture(scope="module")
def reference():
    if not os.path.exists(REFERENCE):
        pytest.skip("no tests/data/laser_reference.npz, which ships with the repository")
    return dict(np.load(REFERENCE))


@pytest.mark.parametrize("name", list(CASES) + ["cubic_vmap_lle"])
def test_bit_for_bit_against_the_previous_order(name, reference):
    flat = _run_vmap() if name == "cubic_vmap_lle" else _run(name)
    keys = sorted(k.split("/", 1)[1] for k in reference if k.startswith(name + "/"))
    assert keys and set(keys) == set(flat), (keys, sorted(flat))
    for k in keys:
        np.testing.assert_array_equal(flat[k], reference[f"{name}/{k}"], err_msg=f"{name}/{k}")


def test_the_references_exercise_the_paths(reference):
    """The cubic cold guess retries (lam_known), Lorenz retries many times, the Jacobi-only loop rejects."""
    assert reference["cubic_cold_lle/nr"] >= 1 and np.max(reference["cubic_cold_lle/hist_n_trials"]) >= 2
    assert reference["lorenz_lle/nr"] >= 3
    assert np.max(reference["cubic_vmap_lle/nr"]) >= 1
    assert np.max(reference["cubic_cold_jacobi_only/hist_n_trials"]) == 1 and reference["cubic_cold_plain/nr"] >= 1


def _count_jacobian_sites(cfg, N=101, dt=0.05):
    """The Jacobian call sites of one jitted solve, from the jaxpr text: the Jacobian function is a nested jit named
    jacobian_pass, printed once per call site. Returns (sites, sites before the while loop)."""
    x0 = jnp.array([1.5])
    ode_fn = prepare_ode_function(cubic)
    step = step_functions["euler"]
    times = dt * jnp.arange(N + 1)

    def jacobian_pass(X):
        return linear_terms(X, ode_fn, dt, step, times, None)

    jac = jax.jit(jacobian_pass)

    def residual_fn(X):
        return affine_terms(X, x0, ode_fn, dt, step, times, None)

    def terms_fn(X):
        return jac(X), residual_fn(X)

    def scan_fn(F, c):
        _, cs = par_scan(par_init(F, c[1:], c[0]))
        return jnp.vstack((c[0], cs))

    text = str(jax.make_jaxpr(lambda g: run_damped_newton(g, terms_fn, scan_fn, dt, cfg, TOL, MAX_ITER,
                                                          residual_fn=residual_fn))(jnp.zeros((N, 1))))
    marker = "name=jacobian_pass"
    assert "while[" in text and marker in text, text[:2000]
    return text.count(marker), text[:text.find("while[")].count(marker)


def test_the_jacobian_is_formed_once_per_iteration_inside_the_loop():
    """LASER without history: one Jacobian call site, in the loop body (K passes for K iterations); with the
    history one more after the loop, for the exponent of the final iterate."""
    assert _count_jacobian_sites(LASER) == (1, 0)
    assert _count_jacobian_sites(replace(LASER, return_history=True)) == (2, 0)


if __name__ == "__main__":
    if "--generate" in sys.argv:
        if len(sys.argv) != 3 or sys.argv[1] != "--generate":
            sys.exit("usage: python tests/test_laser_reference.py --generate OUT.npz")
        generate(sys.argv[2])
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
