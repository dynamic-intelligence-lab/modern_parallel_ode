# Third-party origin: make_data_ode re-implements the data-space field of ELCD_Transform.forward of
# models/elcd.py in
# https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
#     Copyright (c) 2024 Sean Jaffe
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section "ELCD".
"""The data-space field of an ELCD model on the reference flow (tests/reference_flow.py), the reference of
tests/test_elcd_flow_gatherfree.py and itself tested by tests/test_elcd_flow.py. The data-space field is
ELCD_Transform.forward (models/elcd.py of the ELCD repository):

    z = phi(x),  z' = A(z) (z - z_eq),  x' = D psi(z) z' = Q(x)^{-1} f(phi(x)),   Q = D phi,

written here as one linear solve per evaluation (`make_data_ode`); torch composes D psi layer by layer instead, the
two agree to rounding times cond(Q). The three functions are those of the first port's systems.py, whose reader of
PyTorch exports is not in this repository.
"""
from typing import Tuple

import jax.numpy as jnp

from experiments.elcd.model import ELCDParams, field
from tests.reference_flow import FlowParams, Q, phi


def make_data_ode(params: ELCDParams, flow: FlowParams):
    """ode(t, x, u) = Q(x)^{-1} f(phi(x)). Build it inside the jitted function so that both pytrees stay traced."""
    def ode(t, x, u=None):
        return jnp.linalg.solve(Q(flow, x), field(params, phi(flow, x)))
    return ode


def unpack(p) -> Tuple[ELCDParams, FlowParams]:
    return p[0], p[1]


def data_ode_builder(p):
    """(elcd_params, flow) -> ode, the data-space field of the pair."""
    return make_data_ode(*unpack(p))
