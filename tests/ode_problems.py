"""ODE problems, initial guesses, the explicit Euler terms of LASER's loop and one reference estimator that the tests
of the solver library borrow.

make_system and make_guess were part of the research-stage benchmark driver, which the release reduces to the
Lorenz system of the paper (experiments/lorenz/run_benchmark.py). The tests of parode still solve the other
systems as fixtures, so the definitions they use were moved here: hopfield, cubic, cubic_coupled,
matfac and cartpole, with the guesses zeros, x0_tiled, wrong_basin and the system-specific
ones. The Lorenz system is the driver's own. euler_terms builds the functions that parode.laser.run_damped_newton
takes, as explicit_par_integrate builds them, for the tests that call the loop directly. estimate_lle_from_vector, at
the end of the file, is a sequential estimate of the finite-time exponent that tests/test_laser.py compares with
parode.laser.lle_dense.

The configuration class of parode.laser defaults to the paper's LASER. Its earlier defaults, for which most tests of
the library were written, are EARLIER_DEFAULTS; those tests build their configuration with earlier_laser.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.lorenz import run_benchmark  # noqa: E402
from parode.explicit_parallel import affine_terms, linear_terms, par_init, par_scan, step_functions  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402

# propose the damping only after a rejection, measure growth against the current residual, no per-step cap, drop the
# floor after two contractive iterates
EARLIER_DEFAULTS = dict(propose_first="plain", accept_ref="prev", step_cap=None, floor_switch_off=True)


def earlier_laser(**kw):
    """The LASER configuration with EARLIER_DEFAULTS in place of the defaults; kw sets any field, these included."""
    return LaserConfig(**{**EARLIER_DEFAULTS, **kw})


EXTRA_GUESSES = {}   # filled by make_system for system-specific guesses (e.g. upright), each a fixed state


def make_system(name, D, key, n_patterns=4, beta=8.0):
    """Return (ode_fn, x0, label). All ODEs take (t, x, u)."""
    k1, k2 = jax.random.split(key)
    if name == "hopfield":
        Xi = jnp.sign(jax.random.normal(k1, (D, n_patterns))) / np.sqrt(D)      # unit-norm binary patterns
        x0 = Xi[:, 0] + 0.6 * jax.random.normal(k2, (D,)) / np.sqrt(D)          # noisy copy of pattern 0

        def ode(t, x, u=None):
            return -x + Xi @ jax.nn.softmax(beta * Xi.T @ x)
        return ode, x0, f"modern Hopfield dynamics, {n_patterns} patterns, beta {beta:g}"
    if name == "cubic":
        x0 = 1.5 * jnp.ones(D) * (1.0 + 0.2 * jax.random.normal(k1, (D,)))

        def ode(t, x, u=None):
            return x - x ** 3
        return ode, x0, "cubic double well x' = x - x^3"
    if name == "cubic_coupled":
        L = jnp.diag(jnp.full(D, 2.0)) - jnp.eye(D, k=1) - jnp.eye(D, k=-1)
        x0 = 1.5 * jnp.ones(D) * (1.0 + 0.2 * jax.random.normal(k1, (D,)))

        def ode(t, x, u=None):
            return x - x ** 3 - 0.1 * L @ x
        return ode, x0, "Laplacian-coupled cubic double well"
    if name == "lorenz":
        assert D == 3
        return run_benchmark.make_system()
    if name == "matfac":
        # Gradient flow x' = -grad V of V(x) = (1/4) ||x x^T - M||_F^2, M = Q diag(mu) Q^T, mu_1 = 1 simple.
        # The origin is a saddle of V (Hessian -M: every direction expands at rate mu_i), the minima
        # +-sqrt(mu_1) q_1 are strictly contractive (rates -2 mu_1 along q_1, -(mu_1 - mu_i) along q_i).
        # From |x0| = 0.05 the truth spends about log(1 / 0.05) time units near the saddle.
        # The spectrum is mu = linspace(1, 0.1, D).
        mu = jnp.linspace(1.0, 0.1, D)
        Q, _ = jnp.linalg.qr(jax.random.normal(k1, (D, D)))
        M = Q @ jnp.diag(mu) @ Q.T
        v = jax.random.normal(k2, (D,))
        v = v / jnp.linalg.norm(v)
        v = jnp.where(v @ Q[:, 0] < 0, -v, v)                                   # positive component along q_1
        x0 = 0.05 * v

        def ode(t, x, u=None):
            # -(x x^T - M) x written as -(x.x) x + M x. The two are algebraically identical, since
            # (x x^T) x = x (x^T x), but the first materializes a D by D outer product at every step,
            # which under the vmap over time is an N by D by D intermediate: 21 GB at D 512, N 10^4.
            # That dominated the measured cost of every method on this system and exhausted the
            # device even for quasi-DEER, whose own storage is N by D. This form allocates nothing
            # larger than the D by D matrix M, which is shared across steps.
            return -(x @ x) * x + M @ x
        mu_label = "in [1, 0.1]"
        return ode, x0, f"rank-one matrix factorization gradient flow, D {D}, mu {mu_label}"
    if name == "cartpole":
        # Cart-pole (underactuated.mit.edu convention: state
        # [x, theta, xdot, thetadot], theta = 0 hanging, theta = pi upright) with viscous damping on both
        # velocities so the hanging state attracts (1.0: with 0.5 the pole still swings 0.13 rad at T = 20 s).
        # No control input. Released 0.1 rad from upright. The cart position is a neutral direction
        # (nothing depends on it), so the exponent along the settled truth is 0 rather than negative.
        assert D == 4
        gravity, pole_length, cart_mass, pole_mass, damping = 9.81, 0.5, 10.0, 1.0, 1.0
        total_mass = cart_mass + pole_mass
        EXTRA_GUESSES["upright"] = jnp.array([0.0, jnp.pi, 0.0, 0.0])

        def ode(t, x, u=None):
            theta, xdot, thetadot = x[1], x[2], x[3]
            sth, cth = jnp.sin(theta), jnp.cos(theta)
            denom = cart_mass + pole_mass * sth ** 2
            xddot = pole_mass * sth * (pole_length * thetadot ** 2 + gravity * cth) / denom - damping * xdot
            thetaddot = (-pole_mass * pole_length * thetadot ** 2 * cth * sth - total_mass * gravity * sth) / (
                pole_length * denom) - damping * thetadot
            return jnp.array([xdot, thetadot, xddot, thetaddot])
        return ode, jnp.array([0.0, jnp.pi - 0.1, 0.0, 0.0]), "damped cart-pole, no control (theta 0 hanging, theta pi upright)"
    raise ValueError(name)


def make_guess(kind, x0, truth, N, key):
    D = x0.shape[0]
    if kind == "zeros":
        return jnp.zeros((N, D))
    if kind == "x0_tiled":
        return jnp.tile(x0[None], (N, 1))
    if kind == "wrong_basin":
        return -jnp.tile(jnp.sign(truth[-1])[None] * jnp.maximum(jnp.abs(truth[-1]), 0.5)[None], (N, 1))
    if kind in EXTRA_GUESSES:
        point = EXTRA_GUESSES[kind]
        return jnp.tile(point[None], (N, 1))
    raise ValueError(kind)


def euler_terms(ode, x0, dt, N):
    """(terms_fn, residual_fn, scan_fn) of the explicit Euler recurrence from x0 over N steps of dt, for
    parode.laser.run_damped_newton: terms_fn(X) = (step Jacobians F, c = -r(X)), residual_fn(X) = c, scan_fn(F, c) =
    the plain DEER scan, as explicit_par_integrate builds them."""
    ode_fn = prepare_ode_function(ode)
    step = step_functions["euler"]
    times = dt * jnp.arange(N + 1)

    def residual_fn(X):
        return affine_terms(X, x0, ode_fn, dt, step, times, None)

    def terms_fn(X):
        return linear_terms(X, ode_fn, dt, step, times, None), residual_fn(X)

    def scan_fn(F, c):
        _, cs = par_scan(par_init(F, c[1:], c[0]))
        return jnp.vstack((c[0], cs))

    return terms_fn, residual_fn, scan_fn


# Third-party origin: estimate_lle_from_vector is copied, with small changes, from src/lle.py of
# https://github.com/lindermanlab/predictability_enables_parallelization
#     Copyright (c) 2025 Linderman Lab
#     MIT License
# Licence text: THIRD_PARTY_NOTICES.md at the repository root, section
# "predictability_enables_parallelization".
def estimate_lle_from_vector(jacobians, v, dt):
    """
    Estimate the (max) LLE given a sequence of Jacobians and an initial vector v.
    """
    D = jacobians.shape[-1] if jacobians.ndim >= 2 else 1
    v = v / jnp.linalg.norm(v)  # normalise once

    def body_fn(carry, J):
        v, logsum = carry
        v = J[0, 0] * v if D == 1 else J @ v  # advance tangent vec
        n = jnp.linalg.norm(v)

        # safe update: if n==0 => add -inf, keep v unchanged
        logsum += lax.cond(
            n == 0.0,
            lambda _: -jnp.inf,  # branch taken when norm is zero
            lambda n_: jnp.log(n_),  # otherwise log(n)
            n,
        )
        v = lax.cond(
            n == 0.0,
            lambda _: v,  # leave v as-is (or set to zeros_like(v))
            lambda n_: v / n_,  # renormalise
            n,
        )
        return (v, logsum), logsum

    (v_final, lyap_sum), step_logs = lax.scan(body_fn, (v, 0.0), jacobians)
    
    # Return continuous-time LLE
    return lyap_sum / (jacobians.shape[0] * dt) # divide LLE by dt to get continuous Lyapunov exponent
