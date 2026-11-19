"""flow_gatherfree.py is the first port of the flow (tests/reference_flow.py) with dense 0/1 matrices in place
of the index gathers and scatters. The arithmetic is
the same operation by operation (eagerly the outputs are bitwise equal), but XLA fuses the multiply-adds of the jitted
programs differently, so jitted outputs may differ in the last bits: the float outputs must agree to REL (relative to
the largest entry; the inverse map, its Jacobian and the field to max(REL, the conditioning floor), since kappa_Q
amplifies last-bit differences), the relu patterns and bin indices exactly. Checked on random flows (D 2, 3, 4; the odd D exercises
the unequal mask halves) inside and outside the tails: phi, psi, Q, decode_jacobian, the relu pattern, the bin index
and the data-space field (tests/reference_systems.py on the reference flow). Also asserts that the two modules
differ only in the primitives. CPU, fp64.
"""
import os
import re
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tests import reference_flow as fl  # noqa: E402
from experiments.elcd import flow_gatherfree as fg  # noqa: E402
from experiments.elcd import model  # noqa: E402
from tests import reference_systems as systems  # noqa: E402

FUNCTIONS = ("phi", "psi", "Q", "decode_jacobian", "conditioner_pattern", "bin_index")
REL = 1e-13


def _close(a, b, name, rel_tol):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
        assert np.array_equal(a, b), name
        return 0.0
    rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
    assert rel < rel_tol, f"{name}: relative difference {rel:.2e} (tolerance {rel_tol:.1e})"
    return rel


def _check_all(flow, X, elcd, rel_tol=REL):
    """The forward maps agree to REL; the inverse map, its Jacobian and the field amplify last-bit differences by
    kappa_Q, so their tolerance is max(REL, the conditioning floor of the model) when the caller passes it."""
    worst = {}
    for name in FUNCTIONS:
        a = jax.jit(jax.vmap(getattr(fl, name), in_axes=(None, 0)))(flow, X)
        b = jax.jit(jax.vmap(getattr(fg, name), in_axes=(None, 0)))(flow, X)
        worst[name] = _close(a, b, name, REL if name in ("phi", "conditioner_pattern", "bin_index") else rel_tol)
    ode_a = jax.jit(lambda p, V: jax.vmap(lambda x: systems.data_ode_builder(p)(0.0, x))(V))
    ode_b = jax.jit(lambda p, V: jax.vmap(lambda x: jnp.linalg.solve(fg.Q(p[1], x), model.field(p[0], fg.phi(p[1], x))))(V))
    worst["data ode"] = _close(ode_a((elcd, flow), X), ode_b((elcd, flow), X), "data ode", rel_tol)
    print("max relative difference flow vs flow_gatherfree:", {k: f"{v:.1e}" for k, v in worst.items()})


@pytest.mark.parametrize("D", [2, 3, 4])
def test_random_flows_agree_within_rel(D):
    flow = fl.cast_flow(fl.random_flow(D, D), jnp.float64)
    elcd = model.cast(model.init_synthetic(0, D, rate=1.0), jnp.float64)
    rng = np.random.default_rng(D)
    X = jnp.asarray(np.concatenate((rng.normal(size=(8, D)), 10.0 * rng.normal(size=(8, D)))))
    _check_all(flow, X, elcd)


def test_modules_differ_only_in_the_primitives():
    def body(text):
        i = text.index("# ----------------------------------------------------------------------------- the coupling")
        return text[i:]
    a = open(os.path.join(ROOT, "tests", "reference_flow.py")).read()
    b = open(os.path.join(ROOT, "experiments", "elcd", "flow_gatherfree.py")).read()
    la, lb = body(a).splitlines(), body(b).splitlines()
    changed = [l for l in lb if l not in la]
    # the coupling section swaps x[idx] / .at[idx].set for the selection matrices; nothing else may differ
    allowed = re.compile(r"_split\(|_assemble\(E_id|E_id, E_tr|E_id @ u|def _assemble|return E_id|D = x.shape\[0\]|"
                         r"E_id, E_tr = _selection|return E_id, E_tr")
    assert all(allowed.search(l) for l in changed), [l for l in changed if not allowed.search(l)]
