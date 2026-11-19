# Third-party origin: this file is a port to JAX of the normalizing flow that
# utils.generate_mf_transform builds from the manifold_flow package of
# https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
#     Copyright (c) 2024 Sean Jaffe
#     MIT License
# That package is a modified copy of https://github.com/johannbrehmer/manifold-flow
#     Copyright (c) 2020 Johann Brehmer and Kyle Cranmer
#     MIT License
# which is based on https://github.com/bayesiains/nsf
#     Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios
#     MIT License
# Licence texts and list of changes: THIRD_PARTY_NOTICES.md at the repository root, sections "ELCD",
# "manifold-flow" and "Neural Spline Flows (nsf)".
"""The ELCD flow in JAX with gather-free primitives: dense 0/1 matrices instead of index gathers and scatters.

A direct port applies the permutation as x[perm], the inverse as x[argsort(perm)], selects the identity and transform
coordinates of a coupling by x[idx], reassembles them with .at[idx].set() and picks the spline bin with
take_along_axis. Every one of these is a gather or scatter with traced indices, and XLA's CPU backend compiles
batched gathers and scatters slowly and with a large footprint. Here the permutation is P @ x with
P[i, perm[i]] = 1 (inverse P^T y), the coordinate selection is E^T x with E[idx[j], j] = 1 (assembly
E_id u + E_tr v), and the bin is picked by a one-hot contraction over the K + 1 knots. All weights are 0 or 1 and
every value is finite, so each primitive returns exactly what the gather returns. tests/reference_flow.py is the
direct port with the gathers; tests/test_elcd_flow_gatherfree.py checks that the two modules agree.

The ELCD normalizing flow of Jaffe et al. (NeurIPS 2024) in JAX: a line-by-line port of the torch transform that
`generate_mf_transform` builds (utils.py of the ELCD repository), so that the paper's trained models run in parode's
parallel solvers with their exact data-space field. Every torch convention is reproduced to rounding; the source
lines are cited by file.

Layer sequence (`manifold_flow/vector_transforms.py`, `create_vector_transform` with flow_steps S, linear "lu", base
"rq-coupling"): for i in 0..S-1 a linear transform then a coupling, and a final linear transform. The wrapping
CompositeProjection with latent_dim = input_dim is the identity.

  linear    CompositeTransform([RandomPermutation, LULinear]) (`permutations.py`, `lu.py`): y = L U x[perm] + b. L is
            unit lower triangular with `lower_entries` at np.tril_indices(D, -1), U upper triangular with
            `upper_entries` at np.triu_indices(D, 1) and diagonal softplus(unconstrained_upper_diag) + 1e-3; torch's
            F.linear(x, U) then F.linear(., L, bias) is L (U x) + b for one vector. The inverse is two triangular
            solves and the inverse permutation argsort(perm).
  coupling  PiecewiseRationalQuadraticCouplingTransform (`coupling.py`) with the alternating mask
            (`various.create_alternating_binary_mask`, even for step 0: coordinates 0, 2, ... are transformed and
            1, 3, ... pass through; odd for step 1). The conditioner is ResidualNet(n_id -> n_tr (3 K - 1), hidden 30,
            2 blocks, relu, no dropout, no batch norm) (`nn/resnet.py`): t = W_in u + b_in, per block
            t <- t + W_1 relu(W_0 relu(t) + b_0) + b_1 (the activation precedes each linear), out = W_out t + b_out.
            The output is reshaped row-major to (n_tr, 3 K - 1) and split into K unnormalized widths, K heights and
            K - 1 interior derivatives (`_piecewise_cdf`); widths and heights are divided by sqrt(hidden_features)
            = sqrt(30) in place, the derivatives are not. The spline (`splines/rational_quadratic.py`, tails
            "linear", tail_bound B = 10) is the identity outside [-B, B]; inside, widths = 1e-3 + (1 - 1e-3 K)
            softmax over the bins, cumulative and scaled to [-B, B] with both end knots pinned, the same for the
            heights, derivatives 1e-3 + softplus with the two boundary values set to log(exp(1 - 1e-3) - 1) so that
            the slope at +-B is one, the bin from torch's `various.searchsorted` (last knot nudged by 1e-6, then
            sum(x >= knots) - 1), the rational-quadratic map of Durkan et al. forward and its closed-form inverse
            with the discriminant clamped at zero.

`FlowParams` is a NamedTuple pytree: a tuple of S + 1 `LUParams`, a tuple of S `CouplingParams` and the tail bound;
the index buffers are int32 leaves. Every function here takes the flow as a traced argument and is jit-safe. Read-outs
that the agreement test compares: `decode_jacobian` (D psi), `conditioner_pattern` (the relu signs of every hidden
unit of every conditioner) and `bin_index` (the spline bin of every transformed coordinate, -1 outside the tails).
`flow_from_state` maps every torch state_dict key (and refuses missing or leftover ones), `state_from_flow`
is its inverse, `random_flow` draws the torch initialization.
"""
from typing import Dict, Mapping, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

NUM_BINS = 10                 # generate_mf_transform: num_bins
TAIL_BOUND = 10.0             # generate_mf_transform: tail_bound
HIDDEN = 30                   # generate_mf_transform: hidden_features of the conditioner
NUM_BLOCKS = 2                # generate_mf_transform: num_transform_blocks
MIN_BIN_WIDTH = 1e-3          # splines.rational_quadratic defaults
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3
LU_EPS = 1e-3                 # LULinear eps
SEARCHSORTED_EPS = 1e-6       # various.searchsorted eps
SOFTPLUS_THRESHOLD = 20.0     # torch.nn.functional.softplus threshold
BOUNDARY_DERIVATIVE = float(np.log(np.exp(1.0 - MIN_DERIVATIVE) - 1.0))   # softplus^{-1}(1 - min_derivative)
KEY_PREFIX = "transform.transform."   # ELCD_Transform.transform (CompositeProjection) .transform (CompositeTransform)
FLOW_KEY_MARK = "_transforms."        # every flow key contains it; the ELCD field keys (model.P_s...) do not


class LUParams(NamedTuple):
    perm: jnp.ndarray                       # (D,) int32, RandomPermutation._permutation
    lower_entries: jnp.ndarray              # (D (D - 1) / 2,)
    upper_entries: jnp.ndarray              # (D (D - 1) / 2,)
    unconstrained_upper_diag: jnp.ndarray   # (D,)
    bias: jnp.ndarray                       # (D,)


class ResNetParams(NamedTuple):
    W_in: jnp.ndarray     # (H, n_id)   initial_layer.weight (torch (out, in))
    b_in: jnp.ndarray     # (H,)
    W0: jnp.ndarray       # (B, H, H)   blocks.k.linear_layers.0.weight
    b0: jnp.ndarray       # (B, H)
    W1: jnp.ndarray       # (B, H, H)   blocks.k.linear_layers.1.weight
    b1: jnp.ndarray       # (B, H)
    W_out: jnp.ndarray    # (n_tr (3 K - 1), H)   final_layer.weight
    b_out: jnp.ndarray    # (n_tr (3 K - 1),)


class CouplingParams(NamedTuple):
    identity_features: jnp.ndarray    # (n_id,) int32 buffer
    transform_features: jnp.ndarray   # (n_tr,) int32 buffer
    net: ResNetParams


class FlowParams(NamedTuple):
    linears: Tuple[LUParams, ...]           # S + 1
    couplings: Tuple[CouplingParams, ...]   # S
    tail_bound: jnp.ndarray                 # ()

    @property
    def D(self) -> int:
        return int(self.linears[0].perm.shape[0])

    @property
    def flow_steps(self) -> int:
        return len(self.couplings)

    @property
    def hidden(self) -> int:
        return int(self.couplings[0].net.b_in.shape[0])

    @property
    def num_bins(self) -> int:
        n_tr = int(self.couplings[0].transform_features.shape[0])
        return (int(self.couplings[0].net.b_out.shape[0]) // n_tr + 1) // 3


# ----------------------------------------------------------------------------- torch primitives

def _softplus(x):
    """torch.nn.functional.softplus: log(1 + e^x), the identity above the threshold."""
    return jnp.where(x > SOFTPLUS_THRESHOLD, x, jnp.log1p(jnp.exp(jnp.minimum(x, SOFTPLUS_THRESHOLD))))


def _searchsorted(knots, x):
    """various.searchsorted: knots (n, K + 1) with the last one nudged by eps, sum(x >= knots) - 1, in 0..K-1 for
    x in [knots_0, knots_K]."""
    knots = knots.at[..., -1].add(SEARCHSORTED_EPS)
    return (jnp.sum(x[..., None] >= knots, axis=-1) - 1).astype(jnp.int32)


def _pick(a, idx):
    """a[..., idx] by a one-hot contraction (exact: the weights are 0 and 1), no dynamic gather."""
    onehot = (jnp.arange(a.shape[-1], dtype=idx.dtype)[None, :] == idx[..., None]).astype(a.dtype)
    return jnp.sum(a * onehot, axis=-1)


def _perm_matrix(perm, D, dtype):
    """P with P[i, perm[i]] = 1, so that P @ x = x[perm] and P.T @ y = y[argsort(perm)]."""
    return (jnp.arange(D, dtype=perm.dtype)[None, :] == perm[:, None]).astype(dtype)


def _selection(D, idx, dtype):
    """E (D, n) with E[idx[j], j] = 1: E.T @ x = x[idx], E @ u scatters u to idx."""
    return (jnp.arange(D, dtype=idx.dtype)[:, None] == idx[None, :]).astype(dtype)


# ----------------------------------------------------------------------------- the linear transform

def _lu_factors(lu: LUParams):
    D = lu.bias.shape[0]
    lower = jnp.eye(D, dtype=lu.bias.dtype).at[np.tril_indices(D, -1)].set(lu.lower_entries)
    upper = jnp.zeros((D, D), lu.bias.dtype).at[np.triu_indices(D, 1)].set(lu.upper_entries)
    upper = upper + jnp.diag(_softplus(lu.unconstrained_upper_diag) + LU_EPS)
    return lower, upper


def linear_forward(lu: LUParams, x):
    """y = L U x[perm] + b (RandomPermutation then LULinear.forward_no_cache)."""
    lower, upper = _lu_factors(lu)
    P = _perm_matrix(lu.perm, x.shape[0], x.dtype)
    return lower @ (upper @ (P @ x)) + lu.bias


def linear_inverse(lu: LUParams, y):
    """x = (U^{-1} L^{-1} (y - b))[argsort(perm)] (LULinear.inverse_no_cache then Permutation.inverse)."""
    lower, upper = _lu_factors(lu)
    v = solve_triangular(lower, y - lu.bias, lower=True, unit_diagonal=True)
    P = _perm_matrix(lu.perm, y.shape[0], y.dtype)
    return P.T @ solve_triangular(upper, v, lower=False)


# ----------------------------------------------------------------------------- the coupling

def conditioner(net: ResNetParams, u):
    """ResidualNet.forward at u (n_id,): (raw (n_tr (3 K - 1),), relu signs (2 B H,) bool, in evaluation order)."""
    t = net.W_in @ u + net.b_in
    signs = []
    for k in range(net.W0.shape[0]):
        signs.append(t > 0)
        a = net.W0[k] @ jax.nn.relu(t) + net.b0[k]
        signs.append(a > 0)
        t = t + net.W1[k] @ jax.nn.relu(a) + net.b1[k]
    return net.W_out @ t + net.b_out, jnp.concatenate(signs)


def _knots(unnormalized, min_size, tail):
    """rational_quadratic_spline: softmax with the floor, cumulative, scaled to [-tail, tail], ends pinned.
    Returns (knots (n, K + 1), bin sizes (n, K))."""
    K = unnormalized.shape[-1]
    sizes = min_size + (1.0 - min_size * K) * jax.nn.softmax(unnormalized, axis=-1)
    cum = jnp.cumsum(sizes, axis=-1)
    cum = jnp.concatenate((jnp.zeros_like(cum[..., :1]), cum), axis=-1)
    cum = (tail - (-tail)) * cum + (-tail)
    cum = cum.at[..., 0].set(-tail).at[..., -1].set(tail)
    return cum, cum[..., 1:] - cum[..., :-1]


def coupling_spline(cpl: CouplingParams, u, tail):
    """The spline of the transformed coordinates given the identity coordinates u (n_id,): (cumw, w, cumh, h, d) with
    the knots (n_tr, K + 1), bin sizes (n_tr, K) and derivatives (n_tr, K + 1), plus the conditioner's relu pattern.
    `_piecewise_cdf` and the head of `unconstrained_rational_quadratic_spline`."""
    raw, pattern = conditioner(cpl.net, u)
    n_tr = cpl.transform_features.shape[0]
    raw = raw.reshape(n_tr, -1)
    K = (raw.shape[1] + 1) // 3
    scale = float(np.sqrt(cpl.net.b_in.shape[0]))      # sqrt(hidden_features), a Python float keeps the dtype
    uw, uh, ud = raw[:, :K] / scale, raw[:, K:2 * K] / scale, raw[:, 2 * K:]
    edge = jnp.full((n_tr, 1), BOUNDARY_DERIVATIVE, raw.dtype)
    d = MIN_DERIVATIVE + _softplus(jnp.concatenate((edge, ud, edge), axis=1))
    cumw, w = _knots(uw, MIN_BIN_WIDTH, tail)
    cumh, h = _knots(uh, MIN_BIN_HEIGHT, tail)
    return (cumw, w, cumh, h, d), pattern


def spline_forward(x, spline, idx):
    cumw, w, cumh, h, d = spline
    cw, bw, ch, bh = _pick(cumw, idx), _pick(w, idx), _pick(cumh, idx), _pick(h, idx)
    d0, d1 = _pick(d, idx), _pick(d[..., 1:], idx)
    delta = bh / bw
    theta = (x - cw) / bw
    t1 = theta * (1 - theta)
    numerator = bh * (delta * theta ** 2 + d0 * t1)
    denominator = delta + (d0 + d1 - 2 * delta) * t1
    return ch + numerator / denominator


def spline_inverse(y, spline, idx):
    cumw, w, cumh, h, d = spline
    cw, bw, ch, bh = _pick(cumw, idx), _pick(w, idx), _pick(cumh, idx), _pick(h, idx)
    d0, d1 = _pick(d, idx), _pick(d[..., 1:], idx)
    delta = bh / bw
    a = (y - ch) * (d0 + d1 - 2 * delta) + bh * (delta - d0)
    b = bh * d0 - (y - ch) * (d0 + d1 - 2 * delta)
    c = -delta * (y - ch)
    discriminant = jnp.maximum(b ** 2 - 4 * a * c, 0.0)
    root = (2 * c) / (-b - jnp.sqrt(discriminant))
    return root * bw + cw


def _split(x, cpl: CouplingParams):
    D = x.shape[0]
    E_id, E_tr = _selection(D, cpl.identity_features, x.dtype), _selection(D, cpl.transform_features, x.dtype)
    return E_id, E_tr, E_id.T @ x, E_tr.T @ x


def _assemble(E_id, E_tr, u, v):
    return E_id @ u + E_tr @ v


def coupling_forward(cpl: CouplingParams, x, tail):
    """CouplingTransform.forward_: (output (D,), relu pattern (2 B H,) bool, bin index (n_tr,) int32, -1 outside)."""
    E_id, E_tr, u, v = _split(x, cpl)
    spline, pattern = coupling_spline(cpl, u, tail)
    inside = (v >= -tail) & (v <= tail)
    vc = jnp.clip(v, -tail, tail)
    idx = _searchsorted(spline[0], vc)
    y = jnp.where(inside, spline_forward(vc, spline, idx), v)
    return _assemble(E_id, E_tr, u, y), pattern, jnp.where(inside, idx, -1)


def coupling_inverse(cpl: CouplingParams, y, tail):
    """CouplingTransform.inverse_ (the bin from the height knots)."""
    E_id, E_tr, u, v = _split(y, cpl)
    spline, _ = coupling_spline(cpl, u, tail)
    inside = (v >= -tail) & (v <= tail)
    vc = jnp.clip(v, -tail, tail)
    idx = _searchsorted(spline[2], vc)
    x = jnp.where(inside, spline_inverse(vc, spline, idx), v)
    return _assemble(E_id, E_tr, u, x)


# ----------------------------------------------------------------------------- the maps

def _encode_trace(flow: FlowParams, x):
    """(phi(x), relu patterns of every conditioner concatenated, bin indices of every coupling concatenated)."""
    y, patterns, bins = x, [], []
    for lu, cpl in zip(flow.linears[:-1], flow.couplings):
        y, pattern, idx = coupling_forward(cpl, linear_forward(lu, y), flow.tail_bound)
        patterns.append(pattern)
        bins.append(idx)
    return linear_forward(flow.linears[-1], y), jnp.concatenate(patterns), jnp.concatenate(bins)


def phi(flow: FlowParams, x):
    """The encode map x -> z, (D,) -> (D,) (ELCD_Transform.encode)."""
    return _encode_trace(flow, x)[0]


def psi(flow: FlowParams, y):
    """The exact inverse z -> x (ELCD_Transform.decode without the Jacobian)."""
    x = linear_inverse(flow.linears[-1], y)
    for lu, cpl in zip(flow.linears[-2::-1], flow.couplings[::-1]):
        x = linear_inverse(lu, coupling_inverse(cpl, x, flow.tail_bound))
    return x


def Q(flow: FlowParams, x):
    """D phi(x), (D, D), by forward-mode AD."""
    return jax.jacfwd(lambda v: phi(flow, v))(x)


def decode_jacobian(flow: FlowParams, z):
    """D psi(z), (D, D): torch's inv_jacobian of `decode`, equal to Q(psi(z))^{-1}."""
    return jax.jacfwd(lambda v: psi(flow, v))(z)


# ----------------------------------------------------------------------------- read-outs

def conditioner_pattern(flow: FlowParams, x):
    """The relu signs of every hidden unit of every conditioner at x, (S 2 B H,) bool: coupling by coupling, block by
    block, the block input's relu then the inner relu."""
    return _encode_trace(flow, x)[1]


def bin_index(flow: FlowParams, x):
    """The spline bin of every transformed coordinate of every coupling at x, (sum n_tr,) int32 in 0..K-1, -1
    outside the tails; coupling by coupling in the order of `transform_features`."""
    return _encode_trace(flow, x)[2]


# ----------------------------------------------------------------------------- torch state_dict

def _mask_features(D: int, step: int):
    """create_alternating_binary_mask(D, even=(step % 2 == 0)): (identity_features, transform_features)."""
    transform = np.arange(D)[step % 2::2]
    identity = np.arange(D)[1 - step % 2::2]
    return identity, transform


def _linear_keys(prefix: str):
    return [prefix + "0._permutation"] + [prefix + "1." + n for n in
                                          ("lower_entries", "upper_entries", "unconstrained_upper_diag", "bias")]


def _strip(key: str):
    i = key.find(FLOW_KEY_MARK)
    return key[i:] if i >= 0 else None


def flow_from_state(state: Mapping[str, np.ndarray], D: int, flow_steps: int, hidden: int = HIDDEN,
                    num_blocks: int = NUM_BLOCKS, num_bins: int = NUM_BINS, tail_bound: float = TAIL_BOUND) -> FlowParams:
    """FlowParams (float64 numpy leaves, int32 indices) from the tensors of the transform's state_dict. Any prefix
    before the first `_transforms.` is stripped, so the `transform.transform.` prefix of an ELCD_Transform export
    and the bare CompositeTransform keys both work; keys without `_transforms.` (the ELCD field) are ignored. Every
    flow key must be present with torch's shape and every flow key present must be used (KeyError otherwise)."""
    present = {_strip(k): np.asarray(v) for k, v in state.items() if _strip(k) is not None}
    used = set()

    def take(key, shape):
        if key not in present:
            raise KeyError(f"state_dict lacks {key}")
        used.add(key)
        value = present[key]
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{key}: shape {value.shape}, expected {shape}")
        return value

    def linear(prefix):
        keys = _linear_keys(prefix)
        n_tri = D * (D - 1) // 2
        perm = take(keys[0], (D,)).astype(np.int32)
        if sorted(perm.tolist()) != list(range(D)):
            raise ValueError(f"{keys[0]} is not a permutation of 0..{D - 1}")
        return LUParams(perm, take(keys[1], (n_tri,)).astype(np.float64), take(keys[2], (n_tri,)).astype(np.float64),
                        take(keys[3], (D,)).astype(np.float64), take(keys[4], (D,)).astype(np.float64))

    def coupling(prefix, step):
        identity, transform = _mask_features(D, step)
        idf = take(prefix + "identity_features", identity.shape).astype(np.int32)
        trf = take(prefix + "transform_features", transform.shape).astype(np.int32)
        if not (np.array_equal(idf, identity) and np.array_equal(trf, transform)):
            raise ValueError(f"{prefix}: the mask is not the alternating mask of step {step}")
        n_id, n_tr = len(idf), len(trf)
        net = prefix + "transform_net."

        def lin(name, out, inn):
            return take(f"{net}{name}.weight", (out, inn)).astype(np.float64), take(f"{net}{name}.bias", (out,)).astype(np.float64)

        W_in, b_in = lin("initial_layer", hidden, n_id)
        blocks0 = [lin(f"blocks.{k}.linear_layers.0", hidden, hidden) for k in range(num_blocks)]
        blocks1 = [lin(f"blocks.{k}.linear_layers.1", hidden, hidden) for k in range(num_blocks)]
        W_out, b_out = lin("final_layer", n_tr * (3 * num_bins - 1), hidden)
        resnet = ResNetParams(W_in, b_in, np.stack([w for w, _ in blocks0]), np.stack([b for _, b in blocks0]),
                              np.stack([w for w, _ in blocks1]), np.stack([b for _, b in blocks1]), W_out, b_out)
        return CouplingParams(idf, trf, resnet)

    linears = [linear(f"_transforms.{i}._transforms.0._transforms.") for i in range(flow_steps)]
    linears.append(linear(f"_transforms.{flow_steps}._transforms."))
    couplings = [coupling(f"_transforms.{i}._transforms.1.", i) for i in range(flow_steps)]
    leftover = sorted(set(present) - used)
    if leftover:
        raise KeyError(f"unmapped flow keys: {leftover}")
    return FlowParams(tuple(linears), tuple(couplings), np.float64(tail_bound))


def state_from_flow(flow: FlowParams, prefix: str = KEY_PREFIX) -> Dict[str, np.ndarray]:
    """The torch state_dict of the flow (numpy; index buffers int64), the inverse of `flow_from_state`."""
    out = {}

    def put_linear(p, lu):
        keys = _linear_keys(p)
        out[keys[0]] = np.asarray(lu.perm).astype(np.int64)
        for key, leaf in zip(keys[1:], (lu.lower_entries, lu.upper_entries, lu.unconstrained_upper_diag, lu.bias)):
            out[key] = np.asarray(leaf)

    S = flow.flow_steps
    for i, (lu, cpl) in enumerate(zip(flow.linears[:-1], flow.couplings)):
        put_linear(f"{prefix}_transforms.{i}._transforms.0._transforms.", lu)
        p = f"{prefix}_transforms.{i}._transforms.1."
        out[p + "identity_features"] = np.asarray(cpl.identity_features).astype(np.int64)
        out[p + "transform_features"] = np.asarray(cpl.transform_features).astype(np.int64)
        net = p + "transform_net."
        out[net + "initial_layer.weight"], out[net + "initial_layer.bias"] = np.asarray(cpl.net.W_in), np.asarray(cpl.net.b_in)
        for k in range(cpl.net.W0.shape[0]):
            out[f"{net}blocks.{k}.linear_layers.0.weight"] = np.asarray(cpl.net.W0[k])
            out[f"{net}blocks.{k}.linear_layers.0.bias"] = np.asarray(cpl.net.b0[k])
            out[f"{net}blocks.{k}.linear_layers.1.weight"] = np.asarray(cpl.net.W1[k])
            out[f"{net}blocks.{k}.linear_layers.1.bias"] = np.asarray(cpl.net.b1[k])
        out[net + "final_layer.weight"], out[net + "final_layer.bias"] = np.asarray(cpl.net.W_out), np.asarray(cpl.net.b_out)
    put_linear(f"{prefix}_transforms.{S}._transforms.", flow.linears[-1])
    return out


# ----------------------------------------------------------------------------- instances

def random_flow(seed: int, D: int, flow_steps: int = 2, hidden: int = HIDDEN, num_blocks: int = NUM_BLOCKS,
                num_bins: int = NUM_BINS, tail_bound: float = TAIL_BOUND) -> FlowParams:
    """A flow with the torch initialization in float64 numpy (cast with `cast_flow`): RandomPermutation buffers
    torch.randperm; LULinear's non-identity init, entries and unconstrained diagonal U(-1/sqrt(D), 1/sqrt(D)), plus
    a bias of the same law (torch zeros it; a trained model has one, so the tests exercise it); the conditioner's
    Linear layers with torch's default U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and biases, and the second
    linear of each residual block U(-1e-3, 1e-3) (ResidualBlock zero_initialization)."""
    if D < 2:
        raise ValueError(f"the coupling needs D >= 2, got {D}")
    rng = np.random.default_rng(seed)
    stdv = 1.0 / np.sqrt(D)
    n_tri = D * (D - 1) // 2

    def linear_params():
        return LUParams(rng.permutation(D).astype(np.int32), rng.uniform(-stdv, stdv, n_tri),
                        rng.uniform(-stdv, stdv, n_tri), rng.uniform(-stdv, stdv, D), rng.uniform(-stdv, stdv, D))

    def torch_linear(out, inn):
        bound = 1.0 / np.sqrt(inn)
        return rng.uniform(-bound, bound, (out, inn)), rng.uniform(-bound, bound, out)

    def coupling_params(step):
        identity, transform = _mask_features(D, step)
        W_in, b_in = torch_linear(hidden, len(identity))
        first = [torch_linear(hidden, hidden) for _ in range(num_blocks)]
        second = [(rng.uniform(-1e-3, 1e-3, (hidden, hidden)), rng.uniform(-1e-3, 1e-3, hidden)) for _ in range(num_blocks)]
        W_out, b_out = torch_linear(len(transform) * (3 * num_bins - 1), hidden)
        net = ResNetParams(W_in, b_in, np.stack([w for w, _ in first]), np.stack([b for _, b in first]),
                           np.stack([w for w, _ in second]), np.stack([b for _, b in second]), W_out, b_out)
        return CouplingParams(identity.astype(np.int32), transform.astype(np.int32), net)

    linears = tuple(linear_params() for _ in range(flow_steps + 1))
    couplings = tuple(coupling_params(i) for i in range(flow_steps))
    return FlowParams(linears, couplings, np.float64(tail_bound))


def cast_flow(flow: FlowParams, dtype) -> FlowParams:
    """Every floating leaf as a jnp array of `dtype`; the index buffers stay int32."""
    def leaf(a):
        a = np.asarray(a)
        if np.issubdtype(a.dtype, np.integer):
            return jnp.asarray(a, jnp.int32)
        return jnp.asarray(a, dtype)
    return jax.tree_util.tree_map(leaf, flow)
