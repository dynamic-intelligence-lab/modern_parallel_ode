# Third-party origin: this file is a port to JAX of the latent ELCD field of models/elcd.py in
# https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
#     Copyright (c) 2024 Sean Jaffe
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section "ELCD".
"""The latent ELCD field of Jaffe et al. (NeurIPS 2024) in JAX, with synthetic instances.

    f(z) = A(z) (z - z*),    A(z) = -P_s(z)^T P_s(z) - alpha I + P_a(z) - P_a(z)^T,

P_s and P_a the D x D row-major reshapes of Linear(D, H), tanh, Linear(H, D^2) (H = 16, alpha = 0.01 in
`models/elcd.py` of the ELCD repository: `build_model` lines 92-101, `self.eps = .01` line 106, `get_A_matrix`
lines 126-134, `forward` line 159). Torch's Linear computes x W^T + b with W of shape (out, in), so the weights
are used here as W1 @ z + b1 with W1 of shape (H, D) and W2 @ h + b2 with W2 of shape (D^2, H); the reshape is
row-major on both sides. sym A = -P_s^T P_s - alpha I <= -alpha I for every weight, so every parameter set,
trained or random, is a valid ELCD field (the equilibrium contraction of ELCD, Jaffe et al. 2024). The
Jacobian is J = A + M with M e = (DA[e])(z - z*), M(z*) = 0; `jacobian` forms it by forward-mode AD.

Synthetic instances (`init_synthetic`): the torch initialization (xavier_uniform weights, bias 0.01), then W1 of
both MLPs scaled by `gain` (the Lipschitz constant of A scales with it while ||A|| stays bounded by the tanh) and
b2 of P_s shifted by `rate` vec(I) so that P_s is about rate I and the settling rate at z* is about
alpha + rate^2.

Everything here is pure and jit-safe; parameters are passed as arguments (an `ELCDParams` pytree), never closed
over, so one compiled program serves every instance of the same shape.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

ALPHA = 0.01          # ELCD's `eps` (models/elcd.py:106), the guaranteed contraction rate of sym A
HIDDEN = 16           # the notebooks' hidden_dim
BIAS_INIT = 0.01      # torch init_weights: bias filled with 0.01


class ELCDParams(NamedTuple):
    W1s: jnp.ndarray   # (H, D)
    b1s: jnp.ndarray   # (H,)
    W2s: jnp.ndarray   # (D*D, H)
    b2s: jnp.ndarray   # (D*D,)
    W1a: jnp.ndarray   # (H, D)
    b1a: jnp.ndarray   # (H,)
    W2a: jnp.ndarray   # (D*D, H)
    b2a: jnp.ndarray   # (D*D,)
    z_star: jnp.ndarray   # (D,)
    alpha: jnp.ndarray    # ()

    @property
    def D(self) -> int:
        return int(self.z_star.shape[0])


# ----------------------------------------------------------------------------- the field

def _mlp_matrix(z, W1, b1, W2, b2):
    """Linear, tanh, Linear, reshaped row-major to (D, D)."""
    D = z.shape[-1]
    h = jnp.tanh(W1 @ z + b1)
    return (W2 @ h + b2).reshape(D, D)


def P_matrices(params: ELCDParams, z: jnp.ndarray):
    """(P_s(z), P_a(z)), each (D, D)."""
    Ps = _mlp_matrix(z, params.W1s, params.b1s, params.W2s, params.b2s)
    Pa = _mlp_matrix(z, params.W1a, params.b1a, params.W2a, params.b2a)
    return Ps, Pa


def A_matrix(params: ELCDParams, z: jnp.ndarray) -> jnp.ndarray:
    """The extended linearization A(z) = -P_s^T P_s - alpha I + P_a - P_a^T, (D, D)."""
    Ps, Pa = P_matrices(params, z)
    eye = jnp.eye(z.shape[-1], dtype=z.dtype)
    return -Ps.T @ Ps - params.alpha * eye + Pa - Pa.T


def field(params: ELCDParams, z: jnp.ndarray) -> jnp.ndarray:
    """f(z) = A(z) (z - z*)."""
    return A_matrix(params, z) @ (z - params.z_star)


def jacobian(params: ELCDParams, z: jnp.ndarray) -> jnp.ndarray:
    """J(z) = df/dz = A(z) + M(z) by forward-mode AD."""
    return jax.jacfwd(lambda y: field(params, y))(z)


def curvature_matrix(params: ELCDParams, z: jnp.ndarray) -> jnp.ndarray:
    """M(z) = J(z) - A(z): the i-th column is (dA/dz_i)(z) (z - z*); zero at z*."""
    return jacobian(params, z) - A_matrix(params, z)


def make_ode(params: ELCDParams):
    """The (t, x, u) callable parode's integrators expect (autonomous, no input). Build it inside the jitted
    function so that `params` stays a traced argument."""
    def ode(t, x, u=None):
        return field(params, x)
    return ode


# ----------------------------------------------------------------------------- instances

def _xavier_uniform(rng: np.random.Generator, out_features: int, in_features: int) -> np.ndarray:
    bound = np.sqrt(6.0 / (in_features + out_features))
    return rng.uniform(-bound, bound, size=(out_features, in_features))


def init_synthetic(seed: int, D: int, gain: float = 1.0, rate: float = 0.0, hidden: int = HIDDEN,
                   alpha: float = ALPHA, z_star: Optional[np.ndarray] = None) -> ELCDParams:
    """A synthetic ELCD instance in float64 numpy (cast with `cast`).

    seed: numpy seed of the torch-style initialization (xavier_uniform weights, bias BIAS_INIT).
    gain: multiplies W1 of both MLPs; the Lipschitz constant of A scales with it, ||A|| does not.
    rate: adds rate * vec(I) to b2 of P_s, so P_s is about rate I and sym A(z*) is about -(alpha + rate^2) I
          plus the rate-weighted cross terms; the settling rate at z* is read from eigvals(A(z*)), not assumed.
    """
    rng = np.random.default_rng(seed)
    D2 = D * D

    def mlp():
        W1 = _xavier_uniform(rng, hidden, D) * gain
        b1 = np.full(hidden, BIAS_INIT)
        W2 = _xavier_uniform(rng, D2, hidden)
        b2 = np.full(D2, BIAS_INIT)
        return W1, b1, W2, b2

    W1s, b1s, W2s, b2s = mlp()
    W1a, b1a, W2a, b2a = mlp()
    b2s = b2s + rate * np.eye(D).ravel()
    zs = np.zeros(D) if z_star is None else np.asarray(z_star, dtype=np.float64)
    return ELCDParams(W1s, b1s, W2s, b2s, W1a, b1a, W2a, b2a, zs, np.float64(alpha))


def cast(params: ELCDParams, dtype) -> ELCDParams:
    """Every leaf as a jnp array of `dtype`."""
    return ELCDParams(*[jnp.asarray(np.asarray(leaf), dtype=dtype) for leaf in params])
