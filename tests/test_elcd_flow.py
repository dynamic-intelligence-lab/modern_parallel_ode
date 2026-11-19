"""Tests of the JAX port of the ELCD flow (tests/reference_flow.py, tests/reference_systems.py), CPU, fp64, tiny sizes.

- psi inverts phi to 1e-12 inside the tail bound and to 1e-12 |x| outside it (the outside set, entries up to about
  25, has coupling inputs beyond +-10; the measured round trip is 5e-14 relative there);
- Q is the Jacobian of phi (central differences inside one cell and one bin), D psi(phi(x)) Q(x) = I;
- the data-space field satisfies the chain rule Q(x) x' = f(phi(x));
- the relu patterns and bin indices have the documented shapes, the pattern flips across a hand-placed relu boundary
  of the first conditioner, the bin index steps across a hand-placed knot and across the tail bound;
- `state_from_flow` round-trips through `flow_from_state`, which rejects a missing, an extra and a misshaped key.

The tests import jax.
"""
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tests import reference_flow as fl  # noqa: E402
from experiments.elcd import model  # noqa: E402
from tests import reference_systems as systems  # noqa: E402

DIMS = [2, 4]


def _flow(D, seed=0):
    return fl.cast_flow(fl.random_flow(seed, D), jnp.float64)


def _elcd(D, seed=0):
    return model.cast(model.init_synthetic(seed, D, rate=1.0), jnp.float64)


def _leaves_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


# ----------------------------------------------------------------------------- the maps

@pytest.mark.parametrize("D", DIMS)
def test_psi_inverts_phi_inside_and_outside_the_tails(D):
    flow = _flow(D)
    rng = np.random.default_rng(1)
    inside = jnp.asarray(rng.normal(size=(16, D)))
    outside = jnp.asarray(10.0 * rng.normal(size=(16, D)))
    bins = jax.jit(jax.vmap(fl.bin_index, in_axes=(None, 0)))
    assert np.all(np.asarray(bins(flow, inside)) >= 0)
    assert np.any(np.asarray(bins(flow, outside)) == -1)
    for X in (inside, outside):                       # 1e-12 absolute, relative to |x| where |x| exceeds one
        back = jax.jit(fl.decode)(flow, jax.jit(fl.encode)(flow, X))
        assert np.max(np.abs(np.asarray(back) - np.asarray(X))) < 1e-12 * max(1.0, float(np.max(np.abs(np.asarray(X)))))


@pytest.mark.parametrize("D", DIMS)
def test_Q_matches_central_differences(D):
    flow, h = _flow(D), 1e-6
    x = np.asarray(np.random.default_rng(2).normal(size=D))
    phi, pattern, bins = jax.jit(fl.phi), jax.jit(fl.conditioner_pattern), jax.jit(fl.bin_index)
    p0, b0 = np.asarray(pattern(flow, jnp.asarray(x))), np.asarray(bins(flow, jnp.asarray(x)))
    cols = []
    for i in range(D):
        e = np.eye(D)[i]
        for sgn in (1.0, -1.0):                       # the stencil stays inside one relu cell and one bin
            xs = jnp.asarray(x + sgn * h * e)
            assert np.array_equal(np.asarray(pattern(flow, xs)), p0) and np.array_equal(np.asarray(bins(flow, xs)), b0)
        cols.append((np.asarray(phi(flow, jnp.asarray(x + h * e))) - np.asarray(phi(flow, jnp.asarray(x - h * e)))) / (2 * h))
    Q = np.asarray(jax.jit(fl.Q)(flow, jnp.asarray(x)))
    assert np.max(np.abs(Q - np.stack(cols, axis=1))) < 1e-6
    assert np.max(np.abs(Q - np.eye(D))) > 1e-2               # the flow is not the identity


@pytest.mark.parametrize("D", DIMS)
def test_decode_jacobian_inverts_Q(D):
    flow = _flow(D)
    x = jnp.asarray(np.random.default_rng(3).normal(size=D))
    z = jax.jit(fl.phi)(flow, x)
    Q = np.asarray(jax.jit(fl.Q)(flow, x))
    Dpsi = np.asarray(jax.jit(fl.decode_jacobian)(flow, z))
    assert np.max(np.abs(Dpsi @ Q - np.eye(D))) < 1e-10
    assert np.max(np.abs(Dpsi - np.linalg.inv(Q))) < 1e-10


@pytest.mark.parametrize("D", DIMS)
def test_data_ode_satisfies_the_chain_rule(D):
    elcd, flow = _elcd(D), _flow(D)
    x = jnp.asarray(np.random.default_rng(4).normal(size=D))
    xdot = jax.jit(lambda p, v: systems.data_ode_builder(p)(0.0, v))((elcd, flow), x)
    lhs = np.asarray(jax.jit(fl.Q)(flow, x)) @ np.asarray(xdot)
    rhs = np.asarray(jax.jit(lambda p, v: model.field(p[0], fl.phi(p[1], v)))((elcd, flow), x))
    assert np.max(np.abs(lhs - rhs)) < 1e-12
    assert np.max(np.abs(rhs)) > 1e-3


# ----------------------------------------------------------------------------- read-outs

@pytest.mark.parametrize("D", DIMS)
def test_patterns_and_bins_have_the_documented_shapes(D):
    flow = _flow(D)
    x = jnp.asarray(np.random.default_rng(5).normal(size=D))
    pattern = np.asarray(jax.jit(fl.conditioner_pattern)(flow, x))
    bins = np.asarray(jax.jit(fl.bin_index)(flow, x))
    S, B, H = flow.flow_steps, fl.NUM_BLOCKS, fl.HIDDEN
    assert pattern.shape == (S * 2 * B * H,) and pattern.dtype == bool
    assert pattern.any() and not pattern.all()
    n_tr = sum(int(c.transform_features.shape[0]) for c in flow.couplings)
    assert n_tr == D and bins.shape == (D,) and bins.dtype == np.int32
    assert np.all(bins >= -1) and np.all(bins < fl.NUM_BINS)
    X = jnp.stack((x, x))
    assert int(jax.jit(fl.cell_crossings)(flow, X)) == 0 and int(jax.jit(fl.knot_crossings)(flow, X)) == 0
    assert float(jax.jit(fl.kappa_Q)(flow, X)) >= 1.0


def test_pattern_flips_across_a_relu_boundary():
    D = 4
    flow = _flow(D)
    lu0, cpl0 = flow.linears[0], flow.couplings[0]
    idf = np.asarray(cpl0.identity_features)
    w, b = np.asarray(cpl0.net.W_in[0]), float(cpl0.net.b_in[0])   # unit 0 of the first relu layer
    u_on = -b * w / (w @ w)
    y0, e = np.zeros(D), np.zeros(D)
    y0[idf], e[idf] = u_on, w / np.linalg.norm(w)
    inv = jax.jit(fl.linear_inverse)
    x_plus, x_minus = inv(lu0, jnp.asarray(y0 + 1e-3 * e)), inv(lu0, jnp.asarray(y0 - 1e-3 * e))
    pattern = jax.jit(fl.conditioner_pattern)
    assert bool(pattern(flow, x_plus)[0]) and not bool(pattern(flow, x_minus)[0])
    assert int(jax.jit(fl.cell_crossings)(flow, jnp.stack((x_minus, x_plus)))) >= 1
    assert int(jax.jit(fl.cell_crossings)(flow, jnp.stack((x_plus, x_plus)))) == 0


def test_bin_index_steps_across_a_knot_and_the_tail():
    D = 2
    flow = _flow(D)
    lu0, cpl0 = flow.linears[0], flow.couplings[0]
    idf, trf = int(cpl0.identity_features[0]), int(cpl0.transform_features[0])
    u = jnp.array([0.3])
    (cumw, _, _, _, _), _ = jax.jit(fl.coupling_spline)(cpl0, u, flow.tail_bound)
    knot = float(cumw[0, 5])
    assert -10.0 < knot < 10.0
    inv, bins = jax.jit(fl.linear_inverse), jax.jit(fl.bin_index)

    def x_at(v):
        return inv(lu0, jnp.zeros(D).at[idf].set(u[0]).at[trf].set(v))

    assert int(bins(flow, x_at(knot + 1e-6))[0]) == 5 and int(bins(flow, x_at(knot - 1e-6))[0]) == 4
    assert int(bins(flow, x_at(10.0 - 1e-3))[0]) == fl.NUM_BINS - 1 and int(bins(flow, x_at(10.0 + 1e-3))[0]) == -1
    assert int(bins(flow, x_at(-10.0 + 1e-3))[0]) == 0 and int(bins(flow, x_at(-10.0 - 1e-3))[0]) == -1
    assert int(jax.jit(fl.knot_crossings)(flow, jnp.stack((x_at(knot - 1e-6), x_at(knot + 1e-6))))) >= 1
    assert int(jax.jit(fl.knot_crossings)(flow, jnp.stack((x_at(0.5), x_at(0.5))))) == 0


# ----------------------------------------------------------------------------- the state_dict mapping

def test_state_round_trip_and_key_checks():
    flow = fl.random_flow(3, 4)
    state = fl.state_from_flow(flow)
    assert set(state) == set(fl.flow_keys(2))
    assert state["transform.transform._transforms.0._transforms.0._transforms.0._permutation"].dtype == np.int64
    assert state["transform.transform._transforms.0._transforms.1.transform_net.initial_layer.weight"].shape == (fl.HIDDEN, 2)
    assert state["transform.transform._transforms.0._transforms.1.transform_net.final_layer.weight"].shape == (2 * 29, fl.HIDDEN)
    back = fl.flow_from_state(state, 4, 2, fl.HIDDEN)
    assert _leaves_equal(back, flow)
    bare = {k[len(fl.KEY_PREFIX):]: v for k, v in state.items()}          # the CompositeTransform's own keys
    assert _leaves_equal(fl.flow_from_state(bare, 4, 2, fl.HIDDEN), flow)
    with_field = dict(state, **{"model.P_s.0.weight": np.zeros((16, 4))})   # ELCD field keys are ignored
    assert _leaves_equal(fl.flow_from_state(with_field, 4, 2, fl.HIDDEN), flow)
    missing = dict(state)
    del missing["transform.transform._transforms.1._transforms.1.transform_net.blocks.1.linear_layers.0.bias"]
    with pytest.raises(KeyError):
        fl.flow_from_state(missing, 4, 2, fl.HIDDEN)
    extra = dict(state, **{"transform.transform._transforms.3._transforms.0._permutation": np.arange(4)})
    with pytest.raises(KeyError):
        fl.flow_from_state(extra, 4, 2, fl.HIDDEN)
    misshaped = dict(state)
    misshaped["transform.transform._transforms.2._transforms.1.bias"] = np.zeros(3)
    with pytest.raises(ValueError):
        fl.flow_from_state(misshaped, 4, 2, fl.HIDDEN)
    with pytest.raises(KeyError):
        fl.flow_from_state(state, 4, 3, fl.HIDDEN)                          # one step more than the state has

