"""SSM cell + MLP readout for the neural controller (the paper's SSM controller).

The cell ODE is the contracting SSM cell of
`experiments/appendix/shared/contracting_ssm_cell.py`:

    h_dot = -gamma(u) * h + A(u) * tanh(h) + tanh(MLP_b(u))

where A(u) and gamma(u) are diagonal in h. With ``gamma_param="sigmoid"``
and ``gamma_min > 0`` the per-coordinate Jacobian is uniformly bounded
above by ``-gamma_min``, giving a contraction floor — that is the
stability property we want for the closed-loop policy.

The cell input ``u`` is fed time-varying from the closed-loop rollout:
``u(t) = [x_arm(t), q_ref(t), qdot_ref(t)]``, dim = 2 * D_arm = 8 for the
2-link arm. The augmented state is ``z = [x_arm (D_arm), h (D_ssm)]`` and
is integrated by ``implicit_deer.{exp,euler}_par_integrate_implicit`` —
the cell does NOT own the solver; it just contributes its block of the
ODE.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from experiments.appendix.shared.contracting_ssm_cell import (
    init_cell_params,
    cell_ode_fn,
)


@dataclass(frozen=True)
class SSMCellConfig:
    """Static config for the SSM cell + readout. Carried as a Python object
    (not a JAX pytree) so it can be a static argument to JIT. The defaults
    are the paper's SSM controller: ``SSMCellConfig()``.

    Attributes:
        D_ssm: hidden state dimension of the SSM cell.
        ninp: dimension of the cell input u (= 2 * D_arm = 8 here).
        n_out: number of control outputs (= 2 for the 2-link arm).
        readout_mult: width factor of MLP-readout hidden layer (4*D_ssm).
        mlp_hidden: width of the cell's internal A/gamma/forcing MLPs.
        A_static, gamma_static: parametrization tuples consumed by
            cell_ode_fn (see contracting_ssm_cell.py): the sigmoid
            parametrizations, A in (-1.5, 0) and gamma in (0.01, 1.5).
    """
    D_ssm: int = 16
    ninp: int = 8
    n_out: int = 2
    readout_mult: int = 4
    mlp_hidden: int | None = None  # defaults to D_ssm in init_cell_params
    A_static: Tuple[str, float, float] = ("sigmoid", 1.5, 0.0)
    gamma_static: Tuple[str, float, float] = ("sigmoid", 0.01, 1.5)


def init_ssm_policy(key, cfg: SSMCellConfig):
    """Initialize SSM cell + 2-layer MLP readout.

    Returns a dict with keys:
        cell      — CellParams NamedTuple from contracting_ssm_cell
        W_out1, b_out1, W_out2, b_out2  — readout MLP (D_ssm -> 4*D_ssm -> n_out)
    """
    k_cell, k_r1, k_r2 = jax.random.split(key, 3)
    cell = init_cell_params(
        key=k_cell,
        ninp=cfg.ninp,
        D=cfg.D_ssm,
        mlp_hidden=cfg.mlp_hidden,
        A_param=cfg.A_static[0],
        gamma_param=cfg.gamma_static[0],
    )
    M = cfg.readout_mult * cfg.D_ssm
    # He init on hidden, small final layer so u_nn starts ~0.
    W_out1 = jnp.sqrt(2.0 / cfg.D_ssm) * jax.random.normal(k_r1, (cfg.D_ssm, M))
    b_out1 = jnp.zeros(M)
    W_out2 = 0.01 * jax.random.normal(k_r2, (M, cfg.n_out))
    b_out2 = jnp.zeros(cfg.n_out)
    return {
        "cell": cell,
        "W_out1": W_out1, "b_out1": b_out1,
        "W_out2": W_out2, "b_out2": b_out2,
    }


def readout_mlp(params, h):
    """2-layer MLP: h -> tanh(h W1 + b1) -> z W2 + b2 ."""
    z = jnp.tanh(h @ params["W_out1"] + params["b_out1"])
    return z @ params["W_out2"] + params["b_out2"]


def count_ssm_policy_params(params) -> int:
    """Total scalar parameter count (cell + readout)."""
    cell_n = sum(int(jnp.size(v)) for v in params["cell"])
    readout_n = sum(
        int(jnp.size(params[k])) for k in ("W_out1", "b_out1", "W_out2", "b_out2")
    )
    return cell_n + readout_n


def make_ssm_ode_cl(arm_ode, M_fn, V_fn, q_ref_fn, qdot_ref_fn,
                    kp, kd, cfg: SSMCellConfig):
    """Closed-loop ODE factory: returns ``ode_cl(t, z) -> dz``.

    z = [x_arm (D_arm=4), h (D_ssm)].
    u_pd = gravity comp + M @ desired_qddot   (PD baseline torque)
    u_nn = readout_mlp(params, h)             (learned augmentation)
    cell input u_cell = [x_arm, q_ref, qdot_ref] (dim 2*D_arm = ninp).
    """
    n = cfg.n_out

    def factory(params):
        def ode_cl(t, z):
            x = z[: 2 * n]                  # arm state (q, qdot)
            h = z[2 * n :]                  # SSM hidden state
            q, qdot = x[:n], x[n:]
            q_ref = q_ref_fn(t)
            qdot_ref = qdot_ref_fn(t)
            g_comp = jax.grad(V_fn)(q)
            M = M_fn(q)
            desired_qddot = -kp * (q - q_ref) - kd * (qdot - qdot_ref)
            u_pd = g_comp + M @ desired_qddot
            u_nn = readout_mlp(params, h)
            dx = arm_ode(t, x, u_pd + u_nn)
            u_cell = jnp.concatenate([x, q_ref, qdot_ref])
            dh = cell_ode_fn(t, h, u_cell, params["cell"],
                             cfg.A_static, cfg.gamma_static)
            return jnp.concatenate([dx, dh])
        return ode_cl

    return factory
