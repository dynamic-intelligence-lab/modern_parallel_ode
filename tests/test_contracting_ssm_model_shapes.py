"""Smoke tests for shapes and finiteness on tiny problems."""
import jax.numpy as jnp
import jax.random as jr

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.appendix.shared.contracting_ssm_cell import init_cell_params, cell_ode_fn
from experiments.appendix.contracting_ssm.stacked_model import init_model, encode, pool_and_readout
from experiments.appendix.contracting_ssm.cell_solvers import solve


D = 4
NINP = 3
T = 8
NCLASS = 2


def test_cell_ode_shape_and_finite():
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    out = cell_ode_fn(0.0, jnp.zeros((D,)), jnp.ones((NINP,)), p, ("exp", 1.5, 0.0), ("off", 0.0, 0.0))
    assert out.shape == (D,)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_init_model_layer_count():
    for L in [1, 2, 5]:
        m = init_model(jr.PRNGKey(0), ninp=NINP, D=D, num_layers=L, nclass=NCLASS, A_param="exp", gamma_param="off")
        assert len(m.layers) == L, f"L={L}: got {len(m.layers)}"


def test_encode_shape():
    m = init_model(jr.PRNGKey(0), ninp=NINP, D=D, num_layers=1, nclass=NCLASS, A_param="exp", gamma_param="off")
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP))
    h_enc = encode(x_seq, m)
    assert h_enc.shape == (T, D)


def test_solve_no_nan_all_solvers():
    p = init_cell_params(jr.PRNGKey(0), NINP, D, A_param="exp", gamma_param="off")
    x_seq = jr.normal(jr.PRNGKey(1), (T, NINP))
    for solver in ["seq", "jacobi", "quasi"]:
        h = solve(x_seq, p, (solver, 1.0, 1e-5, 200, 50, "exp", 1.5, 0.0, "off", 0.0, 0.0))
        assert h.shape == (T, D), f"{solver}: bad shape {h.shape}"
        assert bool(jnp.all(jnp.isfinite(h))), f"{solver}: NaN/Inf"


def test_pool_and_readout_shape():
    m = init_model(jr.PRNGKey(0), ninp=NINP, D=D, num_layers=1, nclass=NCLASS, A_param="exp", gamma_param="off")
    h_seq = jnp.zeros((T, D))
    assert pool_and_readout(h_seq, m).shape == (NCLASS,)
