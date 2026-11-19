"""Stacked nonlinear SSM: encoder -> L cells (pre-norm + ODE + FFN + residual) -> readout.

The ODE solve is solver-agnostic at this layer — `apply_layer_block` takes an
`ode_solve_fn` callable so the same model definition runs against any backend.
"""
from __future__ import annotations
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from experiments.appendix.shared.contracting_ssm_cell import CellParams, init_cell_params, _xavier


class LayerParams(NamedTuple):
    cell: CellParams
    ln_scale: jnp.ndarray   # (D,)
    ln_shift: jnp.ndarray   # (D,)
    ffn_W1: jnp.ndarray     # (ffn_dim, D)
    ffn_b1: jnp.ndarray
    ffn_W2: jnp.ndarray     # (D, ffn_dim)
    ffn_b2: jnp.ndarray


class StackedSSMParams(NamedTuple):
    enc_W1: jnp.ndarray     # (D, ninp)
    enc_b1: jnp.ndarray
    enc_W2: jnp.ndarray     # (D, D)
    enc_b2: jnp.ndarray
    layers: Tuple[LayerParams, ...]
    readout_W: jnp.ndarray  # (nclass, D)
    readout_b: jnp.ndarray  # (nclass,)


def init_model(
    key,
    ninp: int,
    D: int,
    num_layers: int,
    nclass: int,
    ffn_mult: int = 2,
    init_alpha_range: Tuple[float, float] = (0.1, 1.5),
    A_param: str = "sigmoid",
    gamma_param: str = "sigmoid",
) -> StackedSSMParams:
    """Initialize encoder + L stacked cells + readout."""
    keys = jr.split(key, 3 + num_layers)
    k_enc1, k_enc2, k_read, *k_layers = keys

    enc_W1 = _xavier(k_enc1, (D, ninp))
    enc_W2 = _xavier(k_enc2, (D, D))
    readout_W = _xavier(k_read, (nclass, D))

    layers = []
    ffn_dim = ffn_mult * D
    for k_layer in k_layers:
        k_cell, k_ffn1, k_ffn2 = jr.split(k_layer, 3)
        layers.append(LayerParams(
            cell=init_cell_params(
                k_cell, ninp=D, D=D,
                init_alpha_range=init_alpha_range,
                A_param=A_param,
                gamma_param=gamma_param,
            ),
            ln_scale=jnp.ones((D,)),
            ln_shift=jnp.zeros((D,)),
            ffn_W1=_xavier(k_ffn1, (ffn_dim, D)),
            ffn_b1=jnp.zeros((ffn_dim,)),
            ffn_W2=_xavier(k_ffn2, (D, ffn_dim)),
            ffn_b2=jnp.zeros((D,)),
        ))

    return StackedSSMParams(
        enc_W1=enc_W1, enc_b1=jnp.zeros((D,)),
        enc_W2=enc_W2, enc_b2=jnp.zeros((D,)),
        layers=tuple(layers),
        readout_W=readout_W,
        readout_b=jnp.zeros((nclass,)),
    )


def encode(x_seq: jnp.ndarray, model: StackedSSMParams) -> jnp.ndarray:
    """x_seq: (T, ninp) -> (T, D)."""
    pre1 = jnp.tanh(x_seq @ model.enc_W1.T + model.enc_b1)
    return pre1 @ model.enc_W2.T + model.enc_b2


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, shift: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return scale * (x - mu) / jnp.sqrt(var + eps) + shift


def apply_layer_block(
    h_seq: jnp.ndarray,
    layer: LayerParams,
    ode_solve_fn: Callable[[jnp.ndarray, CellParams], jnp.ndarray],
) -> jnp.ndarray:
    """One layer: pre-norm -> ODE cell solve -> FFN -> residual.

    h_seq: (T, D) input to the residual stream.
    ode_solve_fn(x_seq, cell_params) -> (T, D) hidden trajectory from the ODE.

    The ODE cell consumes the pre-normalized stream as its input signal `u`,
    so the cell sees a clean, scale-controlled input regardless of upstream
    growth.
    """
    z = jax.vmap(_layer_norm, in_axes=(0, None, None))(
        h_seq, layer.ln_scale, layer.ln_shift,
    )
    h_ode = ode_solve_fn(z, layer.cell)
    pre1 = jax.nn.gelu(h_ode @ layer.ffn_W1.T + layer.ffn_b1)
    ffn_out = pre1 @ layer.ffn_W2.T + layer.ffn_b2
    return h_seq + ffn_out


def pool_and_readout(h_seq: jnp.ndarray, model: StackedSSMParams) -> jnp.ndarray:
    """h_seq: (T, D) -> logits (nclass,): the mean over time, then a linear readout."""
    h_pool = h_seq.mean(axis=0)
    return model.readout_W @ h_pool + model.readout_b
