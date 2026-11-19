"""Single nonlinear SSM cell: the cell of the contracting SSM of experiments/appendix/contracting_ssm and of the
recurrent robot controller of experiments/appendix/robot_control (neural/ssm_policy.py).

Continuous-time ODE driven by an input signal:

    h_dot = -gamma(u) * h + A(u) * tanh(h) + tanh(MLP_b(u))

where A(u) and gamma(u) are diagonal in h. A and gamma share an MLP backbone
``(g_W1, g_b1)`` (one hidden layer); their per-coordinate output heads
``g_W2/g_b2`` (for A) and ``gamma_W2/gamma_b2`` (for gamma) are independent.
``MLP_b`` is a separate MLP for the additive forcing.

Two parametrizations of A are supported via
``A_static = (A_param, alpha_max, alpha_min)``:

* ``A_param="exp"``: ``A_ii(u) = -exp(-g_i(u))``, unbounded below in (-inf, 0).
  Allows log-uniform timescale spread (HiPPO-style), but training can drift
  ``A_ii`` past Euler's stability bound. ``alpha_min`` is ignored for exp.
* ``A_param="sigmoid"``: with ``alpha_min=0`` (default) this is
  ``A_ii(u) = -alpha_max * sigmoid(g_i(u))``, capped in ``(-alpha_max, 0)``.
  With ``alpha_min > 0``:
  ``A_ii(u) = -(alpha_min + (alpha_max - alpha_min) * sigmoid(g_i(u)))``,
  capped in ``[-alpha_max, -alpha_min]``. This bounds the Euler stability
  factor when the state is unsaturated, but it does not guarantee uniform
  strict contraction for the ``A*tanh(h)`` cell: as ``|h|`` grows,
  ``sech²(h)`` can approach zero and the one-step propagator can approach 1.

Two parametrizations of gamma are supported via
``gamma_static = (gamma_param, gamma_min, gamma_max)``:

* ``gamma_param="off"``: no leak term, ``gamma(u) ≡ 0``: the A*tanh
  cell, with no gamma computation at all (the contracting-SSM runs without the
  gamma head). ``gamma_W2/gamma_b2`` are kept zero.
* ``gamma_param="sigmoid"``:
  ``gamma_ii(u) = gamma_min + (gamma_max - gamma_min) * sigmoid(...)``,
  capped in ``[gamma_min, gamma_max]``. With ``gamma_min > 0`` the diagonal
  Jacobian ``J = -gamma + A*sech²(h) ≤ -gamma_min`` is uniformly negative,
  giving a state-independent contraction floor of ``gamma_min``. Euler stability
  requires ``dt*(gamma_max + alpha_max) < 2``.

The defaults ``A_static = ("sigmoid", 1.5, 0.0)`` and ``gamma_static = ("sigmoid", 0.01, 0.4)``,
and ``A_param="sigmoid"``, ``gamma_param="sigmoid"`` of ``init_cell_params``, are the defaults of
experiments/appendix/contracting_ssm/run_train.py: the values of the paper's gamma_min sweep apart from gamma_min,
which it varies (the ablation runs pass gamma_param "off"). The robot controller passes its own
(experiments/appendix/robot_control/neural/ssm_policy.py).
"""
from __future__ import annotations
from typing import NamedTuple, Optional, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


class CellParams(NamedTuple):
    g_W1: jnp.ndarray   # (hidden, ninp)   <-- shared backbone for A and gamma
    g_b1: jnp.ndarray   # (hidden,)        <-- shared backbone for A and gamma
    g_W2: jnp.ndarray   # (D, hidden)      A-head
    g_b2: jnp.ndarray   # (D,)             A-head; HiPPO timescale spread lives here
    b_W1: jnp.ndarray
    b_b1: jnp.ndarray
    b_W2: jnp.ndarray
    b_b2: jnp.ndarray
    gamma_W2: jnp.ndarray  # (D, hidden)   gamma-head; shares g_W1/g_b1 backbone
    gamma_b2: jnp.ndarray  # (D,)          gamma-head bias


def _xavier(key, shape: Tuple[int, ...]) -> jnp.ndarray:
    fan_in = shape[-1]
    fan_out = shape[0]
    scale = jnp.sqrt(2.0 / (fan_in + fan_out))
    return jr.normal(key, shape) * scale


def _hippo_g_bias(key, D: int, alpha_range: Tuple[float, float]) -> jnp.ndarray:
    """Log-uniform spread of g bias so initial A_ii spans [-alpha_max, -alpha_min].

    A_ii = -exp(-g) = -alpha  =>  g = -log(alpha).
    With alpha drawn log-uniformly in [alpha_min, alpha_max], the per-neuron
    decay rates span the chosen range.
    """
    log_lo = jnp.log(alpha_range[0])
    log_hi = jnp.log(alpha_range[1])
    log_alpha = jr.uniform(key, (D,), minval=log_lo, maxval=log_hi)
    return -log_alpha


A_STATIC_DEFAULT: Tuple[str, float, float] = ("sigmoid", 1.5, 0.0)
GAMMA_STATIC_DEFAULT: Tuple[str, float, float] = ("sigmoid", 0.01, 0.4)


def init_cell_params(
    key,
    ninp: int,
    D: int,
    mlp_hidden: Optional[int] = None,
    init_alpha_range: Tuple[float, float] = (0.1, 1.5),
    A_param: str = "sigmoid",
    gamma_param: str = "sigmoid",
) -> CellParams:
    """Initialize a single ODE cell.

    Args:
        key: PRNG key.
        ninp: input dimension feeding A, gamma, and MLP_b.
        D: hidden state dimension.
        mlp_hidden: width of the hidden layer of g and MLP_b. Defaults to D.
        init_alpha_range: (alpha_min, alpha_max) used by the HiPPO init, which
            spreads the per-neuron initial timescales.
            For ``A_param="exp"``: alpha_target is drawn log-uniformly in this
            range and ``g_b2 = -log(alpha_target)``. Default (0.1, 1.5) stays
            inside Euler's stability region at dt=1 — for wider ranges, reduce
            dt.
            For ``A_param="sigmoid"``: range is unused; ``g_b2`` is drawn
            uniformly in [-2, 2] (gives A_ii ≈ ±88% of alpha_max coverage).
        A_param: "exp" (log-uniform timescales but unbounded) or "sigmoid"
            (the default; bounded in (-alpha_max, 0), Euler-stable). Note that
            alpha_max is set at solve time via ``A_static`` (see module
            docstring), so it doesn't affect the parameters here.
        gamma_param: "off" (gamma_W2/gamma_b2 = 0; no leak term) or "sigmoid"
            (the default; bounded leak rate, gives uniform contraction floor).
    """
    if mlp_hidden is None:
        mlp_hidden = D
    k_g, k_b, k_init, k_gamma_W2, k_gamma_b2 = jr.split(key, 5)

    if A_param == "exp":
        g_b2 = _hippo_g_bias(k_init, D, init_alpha_range)
    elif A_param == "sigmoid":
        # Sigmoid is bounded → log-uniform-in-A spread isn't available.
        # Use uniform-in-g spread, which gives a wide (if not log-uniform)
        # range of initial sigmoid(g) values.
        g_b2 = jr.uniform(k_init, (D,), minval=-2.0, maxval=2.0)
    else:
        raise ValueError(f"unknown A_param: {A_param!r}")

    # gamma head: shares (g_W1, g_b1) backbone with A. When "off", we keep
    # gamma_W2/gamma_b2 as zeros so gamma(u) ≡ 0 (param overhead is one (D,hidden)
    # matrix and one (D,) bias; compute overhead is short-circuited in cell_ode_fn).
    if gamma_param == "off":
        gamma_W2 = jnp.zeros((D, mlp_hidden))
        gamma_b2 = jnp.zeros((D,))
    elif gamma_param == "sigmoid":
        gamma_W2 = _xavier(k_gamma_W2, (D, mlp_hidden))
        # Uniform-in-g spread, mirroring the A sigmoid HiPPO init.
        gamma_b2 = jr.uniform(k_gamma_b2, (D,), minval=-2.0, maxval=2.0)
    else:
        raise ValueError(f"unknown gamma_param: {gamma_param!r}")

    k_g1, k_g2 = jr.split(k_g)
    k_b1, k_b2 = jr.split(k_b)

    return CellParams(
        g_W1=_xavier(k_g1, (mlp_hidden, ninp)),
        g_b1=jnp.zeros((mlp_hidden,)),
        g_W2=_xavier(k_g2, (D, mlp_hidden)),
        g_b2=g_b2,
        b_W1=_xavier(k_b1, (mlp_hidden, ninp)),
        b_b1=jnp.zeros((mlp_hidden,)),
        b_W2=_xavier(k_b2, (D, mlp_hidden)),
        b_b2=jnp.zeros((D,)),
        gamma_W2=gamma_W2,
        gamma_b2=gamma_b2,
    )


def _shared_pre(u: jnp.ndarray, params: CellParams) -> jnp.ndarray:
    """Hidden representation shared between A and gamma heads: tanh(g_W1·u + g_b1)."""
    return jnp.tanh(params.g_W1 @ u + params.g_b1)


def _A_diag(
    u: jnp.ndarray,
    params: CellParams,
    A_static: Tuple[str, float, float] = A_STATIC_DEFAULT,
) -> jnp.ndarray:
    """Diagonal entries A_ii(u). Branch on A_static[0]:

    - "exp"     : -exp(-g)            unbounded, log-uniform timescales
    - "sigmoid" : -(alpha_min + (alpha_max - alpha_min) * sigmoid(g))
                  capped in [-alpha_max, -alpha_min]; alpha_min=0 reduces
                  to the pure sigmoid form.
    """
    A_param, alpha_max, alpha_min = A_static
    pre1 = _shared_pre(u, params)
    g = params.g_W2 @ pre1 + params.g_b2
    if A_param == "exp":
        return -jnp.exp(-g)
    if A_param == "sigmoid":
        return -(alpha_min + (alpha_max - alpha_min) * jnn.sigmoid(g))
    raise ValueError(f"unknown A_param: {A_param!r}")


def _gamma_diag(
    u: jnp.ndarray,
    params: CellParams,
    gamma_static: Tuple[str, float, float] = GAMMA_STATIC_DEFAULT,
) -> jnp.ndarray:
    """Diagonal entries gamma_ii(u) ≥ 0. Branch on gamma_static[0]:

    - "off"     : returns zeros (no leak); short-circuits all gamma compute.
    - "sigmoid" : gamma_min + (gamma_max - gamma_min) * sigmoid(gamma_W2 @ pre + gamma_b2)
                  capped in [gamma_min, gamma_max].
    """
    gamma_param, gamma_min, gamma_max = gamma_static
    if gamma_param == "off":
        return jnp.zeros_like(params.gamma_b2)
    if gamma_param == "sigmoid":
        pre1 = _shared_pre(u, params)
        gpre = params.gamma_W2 @ pre1 + params.gamma_b2
        return gamma_min + (gamma_max - gamma_min) * jnn.sigmoid(gpre)
    raise ValueError(f"unknown gamma_param: {gamma_param!r}")


def _forcing(u: jnp.ndarray, params: CellParams) -> jnp.ndarray:
    """tanh(MLP_b(u)). Returns shape (D,)."""
    pre1 = jnp.tanh(params.b_W1 @ u + params.b_b1)
    pre2 = params.b_W2 @ pre1 + params.b_b2
    return jnp.tanh(pre2)


def cell_ode_fn(
    t: float,
    h: jnp.ndarray,
    u: jnp.ndarray,
    params: CellParams,
    A_static: Tuple[str, float, float] = A_STATIC_DEFAULT,
    gamma_static: Tuple[str, float, float] = GAMMA_STATIC_DEFAULT,
) -> jnp.ndarray:
    """Continuous-time RHS:
        h_dot = -gamma(u) * h + A(u) * tanh(h) + tanh(MLP_b(u))

    When ``gamma_static[0] == "off"`` the leak term is skipped entirely
    (no gamma computation): the A*tanh cell.
    """
    base = _A_diag(u, params, A_static) * jnp.tanh(h) + _forcing(u, params)
    if gamma_static[0] == "off":
        return base
    return base - _gamma_diag(u, params, gamma_static) * h
