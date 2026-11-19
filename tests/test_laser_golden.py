"""The golden test of LASER's loop, and tests of the window exponents (parode.laser.window_exponents, the lambda_w of
history_extra).

tests/data/laser_golden.json holds, per key, the traces of one run of explicit_par_integrate: iterations, n_trials,
gamma and the residual history per iteration, the sum of the solution, and the final residual and largest |x| of
the solution. The test compares the first five; it reads neither the final residual nor the largest |x|. No script in
the repository writes the file.
"""
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ode_problems import earlier_laser, make_guess, make_system  # noqa: E402
from parode.explicit_parallel import explicit_par_integrate  # noqa: E402
from parode.laser import window_exponents  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "laser_golden.json")


def _solve(name, D, N, dt, guess_kind, max_iter=100, **cfg):
    ode, x0, _ = make_system(name, D, jax.random.PRNGKey(0))
    truth = seq_integrate(prepare_ode_function(ode), x0, dt, N, "euler")
    guess = make_guess(guess_kind, x0, truth, N, jax.random.PRNGKey(1))
    sol, res, it, h = explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=1e-9,
                                             laser=earlier_laser(return_history=True, **cfg))
    it = int(it)
    return np.asarray(sol), float(res), it, {k: np.asarray(v)[:it] for k, v in h.items()}, np.asarray(truth)


@pytest.mark.parametrize("key", list(json.load(open(GOLDEN)).keys()))
def test_golden_traces_unchanged(key):
    """The loop must reproduce the recorded golden traces. A key's options are set over EARLIER_DEFAULTS."""
    g = json.load(open(GOLDEN))[key]
    name, N, guess, kw = key.split("|"); kw = json.loads(kw)
    D = {"cubic": 1, "hopfield": 16, "lorenz": 3, "cubic_coupled": 8}[name]
    dt = 0.01 if name == "lorenz" else 0.05
    sol, res, it, hist, _ = _solve(name, D, int(N), dt, guess, **kw)
    assert it == g["iters"]
    assert hist["n_trials"].astype(int).tolist() == g["n_trials"]
    np.testing.assert_allclose(hist["gamma"], np.array(g["gamma"]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(hist["residual"], np.array(g["residuals"]), rtol=1e-8, atol=1e-14)
    assert abs(float(np.sum(sol)) - g["sol_sum"]) <= 1e-8 * max(1.0, abs(g["sol_sum"]))


def test_window_exponents_constant_rate():
    dt, n, mu = 0.05, 37, 1.7
    F = jnp.broadcast_to(jnp.exp(mu * dt) * jnp.eye(2), (n, 2, 2))
    for mb in (1, 8):
        lam_max, lam_local = window_exponents(F, dt, min_block=mb)
        assert abs(float(lam_max) - mu) < 1e-9
        np.testing.assert_allclose(np.asarray(lam_local), mu, atol=1e-9)


def test_window_exponents_piecewise_rate():
    """Expansive first half, contractive second half: the local envelope is positive on the first
    half and negative deep inside the second."""
    dt, n = 0.05, 128
    rates = jnp.where(jnp.arange(n) < n // 2, 2.0, -1.0)
    Fd = jnp.exp(rates * dt)[:, None] * jnp.ones((n, 3))                  # diagonal (n, D)
    F = jax.vmap(jnp.diag)(Fd)                                             # dense
    lam_max, lam_local = window_exponents(F, dt, min_block=8)
    ll = np.asarray(lam_local)
    assert np.all(ll[: n // 2] > 1.9)                                      # every window there is expansive
    assert np.all(ll[n // 2:] < -0.9)                                      # every window there is contractive (window rule)
