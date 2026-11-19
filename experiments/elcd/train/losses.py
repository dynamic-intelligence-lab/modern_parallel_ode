"""The two losses of the paper and the rollouts behind the second.

Velocity loss: the model's data-space velocity F(x) = Q(x)^{-1} f(phi(x)) against the recorded velocity, mean
squared error over a batch of samples. The equilibrium z* = phi(x_eq) is recomputed from the current flow in every
loss evaluation, as the torch model does in its forward pass, so the flow's parameters receive its gradient.

The rollouts are solved by one of SOLVERS: "seq" (lax.scan with backpropagation) or the parallel-in-time solvers of
parode.implicit (plain DEER and LASER, `laser`) with the implicit-function-theorem adjoint; "scale-elk:<g>" and
"elk:<s2>" exist on the paper runs' path below. Parallel solves take an initial guess per rollout (L, D), excluding
z_0; the driver warm-starts it from the start's previous solution. A rollout whose solve did not converge (residual
above tol or non-finite) is dropped from the loss of that step.

The multi-start ("suffix") loss: the model starts at recorded states x_k of each training demonstration (all k in
0..N-2, or K drawn per step) and is rolled out L = N - 1 steps (h_max 0, which the driver passes; h_max caps L);
step j is compared with x_{k+j} while k + j <= N - 1 and masked beyond. The loss is the mean squared error over all
valid (start, step) pairs of the converged starts, each pair weighing the same. Every start is one rollout_one call
(rollout_batch over the flattened starts, optionally in chunks: the driver's auto_chunk sizes them for time and
memory).

The anchor, off by default so that the default path is the plain multi-start loss bit for bit: `anchor` a in [0, 1]
mixes in the loss of the rollouts from x_0 (the rows whose start is 0, each demonstration's first sample):
loss = (1 - a) suffix + a anchor, with the anchor term the plain pair-mean over those rows' valid pairs. The driver
ramps a and forces start 0 into every draw (the recipe of the model of panel A of the ELCD figure).

The paper runs, off by default so that the
default path is the loss above bit for bit: `suffix_loss(..., fallback=True)` solves every start with LASER's
stopping rule for all solvers (implicit stop "laser"; the solvers "scale-elk:<g>", constant scale-ELK, and "elk:<s2>",
ELK, exist only there) and re-solves the whole step with the sequential loop when any parallel solve failed, keeping the converged
parallel paths (solve_starts_fallback); no start is dropped unless the loop's own path is not finite.
"""
from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, vmap

from experiments.elcd import flow_gatherfree as FG
from experiments.elcd.model import ELCDParams, field, make_ode
from parode.implicit import attach_adjoint, implicit_forward, implicit_solve, seq_solve
from parode.laser import LaserConfig

SOLVERS = ("seq", "deer", "laser")
SCALE_ELK = "scale-elk"     # "scale-elk:<g>": constant scale-ELK at rate g >= 0 per unit time (stop "laser" only)
ELK = "elk"                 # "elk:<s2>": ELK with observation variance s2 > 0, lambda_LM = 1 / s2 (stop "laser" only)


class Theta(NamedTuple):
    """The trainable parameters: the latent field (its z_star leaf is overwritten by phi(x_eq)) and the flow."""
    elcd: ELCDParams
    flow: FG.FlowParams


def trainable_spec(theta: Theta):
    """A pytree of Theta's structure with True at the trained leaves: every floating leaf except the constants
    alpha and z_star of the field and tail_bound of the flow (if they were trained, the saved model would drop
    the drifted tail bound, and the reloaded model would differ from the trained one)."""
    import equinox as eqx
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, theta)
    return Theta(spec.elcd._replace(alpha=False, z_star=False), spec.flow._replace(tail_bound=False))


def partition(theta: Theta):
    """(params, static): the trained leaves and the fixed ones (the flow's integer index buffers and the constants
    alpha, z_star and tail_bound), as two pytrees of Theta's structure with None in the other's places
    (equinox.partition); `combine` reverses it."""
    import equinox as eqx
    return eqx.partition(theta, trainable_spec(theta))


def count_trainable(theta: Theta) -> int:
    import numpy as np
    params, _ = partition(theta)
    return int(sum(np.asarray(l).size for l in jax.tree_util.tree_leaves(params)))


def combine(params, static) -> Theta:
    import equinox as eqx
    return eqx.combine(params, static)


def parse_solver(name: str) -> Tuple[str, Optional[float]]:
    """(name, None) for a name of SOLVERS, (SCALE_ELK, g) for 'scale-elk:<g>' with g a finite rate >= 0, (ELK, s2) for
    'elk:<s2>' with s2 a finite variance > 0; ValueError otherwise."""
    if name in SOLVERS:
        return name, None
    head, sep, arg = str(name).partition(":")
    if head in (SCALE_ELK, ELK) and sep:
        try:
            v = float(arg)
        except ValueError:
            v = math.nan
        if math.isfinite(v) and (v >= 0.0 if head == SCALE_ELK else v > 0.0):
            return head, v
    raise ValueError(f"solver must be one of {SOLVERS}, '{SCALE_ELK}:<g>' with a rate g >= 0 or '{ELK}:<s2>' with a "
                     f"variance s2 > 0, got {name!r}")


def solver_kwargs(name: str, stop: str = "parode") -> Dict:
    """implicit_solve's keyword arguments for a solver name; stop "laser" (the paper runs) adds stop="laser", and
    scale-elk:<g> and elk:<s2> need it."""
    table = {"deer": {}, "laser": {"laser": LaserConfig()}}
    if name == "seq":
        raise ValueError("'seq' is the sequential rollout")
    base, value = parse_solver(name)
    if base in (SCALE_ELK, ELK):
        if stop != "laser":
            raise ValueError(f"{name!r} runs on the shared loop of stop 'laser' (the driver's --fallback seq)")
        return {"scale_rate" if base == SCALE_ELK else "elk_sigma2": value, "stop": "laser"}
    return dict(table[name]) if stop == "parode" else dict(table[name], stop=stop)


# ----------------------------------------------------------------------------- the model as trained

def with_equilibrium(theta: Theta, x_eq: jnp.ndarray) -> ELCDParams:
    """The latent field with z* = phi(x_eq) from the current flow."""
    return theta.elcd._replace(z_star=FG.phi(theta.flow, x_eq))


def data_velocity(theta: Theta, x: jnp.ndarray, x_eq: jnp.ndarray) -> jnp.ndarray:
    """F(x) = Q(x)^{-1} f(phi(x)) at one data point (D,)."""
    elcd = with_equilibrium(theta, x_eq)
    return jnp.linalg.solve(FG.Q(theta.flow, x), field(elcd, FG.phi(theta.flow, x)))


def velocity_loss(theta: Theta, xs: jnp.ndarray, vs: jnp.ndarray, x_eq: jnp.ndarray) -> jnp.ndarray:
    """Mean over samples and coordinates of |F(x_i) - v_i|^2 (torch's MSELoss on a batch)."""
    pred = vmap(data_velocity, in_axes=(None, 0, None))(theta, xs, x_eq)
    return jnp.mean((pred - vs) ** 2)


# ----------------------------------------------------------------------------- rollouts

def rollout_one(theta: Theta, x_eq: jnp.ndarray, x0: jnp.ndarray, guess: jnp.ndarray, dt: float, solver: str,
                tol: float, max_iter: int, finite_decode: bool = False):
    """One demonstration's rollout of H = guess.shape[0] steps: decoded path X (H+1, D) with X[0] = psi(phi(x0)),
    latent path Z (H+1, D), the solve's residual max|r| (0 for seq) and its accepted iterations (H for seq).

    finite_decode (the suffix loss; off by default): an unusable latent path (non-finite, or a
    parallel solve's residual above tol) is decoded from z_0 repeated and its X set to NaN afterwards. converged_mask
    then drops it as before, but no non-finite value enters psi, whose backward would otherwise turn the zero
    cotangent of a dropped path into NaN in the flow's gradient (0 * NaN) and void the whole step's gradient. The
    decode is also rematerialized (jax.checkpoint): its backward residuals, kept for every start, were the largest
    memory term of a suffix step (26 KB per solve-step at D 32, 5.4 KB at D 2, fp64), and the decode
    costs about 1 percent of a step."""
    elcd = with_equilibrium(theta, x_eq)
    z0 = FG.phi(theta.flow, x0)
    H = int(guess.shape[0])
    if solver == "seq":
        Z = seq_solve(make_ode, elcd, z0, dt, H, solver="rk4")
        res, it = jnp.zeros((), Z.dtype), jnp.asarray(H, jnp.int32)
    else:
        Z, res, it = implicit_solve(make_ode, elcd, z0, guess, dt, solver="rk4", tol=tol, max_iter=max_iter,
                                    unconverged="zero", **solver_kwargs(solver))
    if not finite_decode:
        return vmap(lambda z: FG.psi(theta.flow, z))(Z), Z, res, it
    ok = jnp.all(jnp.isfinite(Z)) & (True if solver == "seq" else res <= tol)
    return _finite_decode(theta, Z, z0, ok), Z, res, it


def _finite_decode(theta: Theta, Z: jnp.ndarray, z0: jnp.ndarray, ok) -> jnp.ndarray:
    """rollout_one's finite_decode: psi of Z where ok, else of z_0 repeated, then NaN there; rematerialized."""
    import equinox as eqx
    floats, fixed = eqx.partition(theta.flow, eqx.is_inexact_array)     # the index buffers stay closed over
    decode = jax.checkpoint(lambda fl, Zs: vmap(lambda z: FG.psi(eqx.combine(fl, fixed), z))(Zs))
    X = decode(floats, jnp.where(ok, Z, jnp.broadcast_to(z0, Z.shape)))
    return jnp.where(ok, X, jnp.nan)


def rollout_batch(theta: Theta, x_eq: jnp.ndarray, x0s: jnp.ndarray, guesses: jnp.ndarray, dt: float, solver: str,
                  tol: float, max_iter: int, finite_decode: bool = False):
    """rollout_one over a batch of demonstrations: X (B, H+1, D), Z (B, H+1, D), res (B,), it (B,)."""
    f = lambda x0, g: rollout_one(theta, x_eq, x0, g, dt, solver, tol, max_iter, finite_decode)
    return vmap(f)(x0s, guesses)


def converged_mask(X: jnp.ndarray, res: jnp.ndarray, tol: float, solver: str) -> jnp.ndarray:
    """(B,) 1.0 where the rollout is usable: finite, and (parallel solvers) residual at most tol."""
    finite = jnp.all(jnp.isfinite(X), axis=(1, 2))
    if solver == "seq":
        return finite.astype(X.dtype)
    return (finite & (res <= tol)).astype(X.dtype)


# ----------------------------------------------------------------------------- the suffix loss

def suffix_length(N: int, h_max: int = 0) -> int:
    """L, the steps every start is rolled out: min(h_max, N - 1), with h_max 0 meaning to the end (N - 1)."""
    if N < 2 or h_max < 0:
        raise ValueError(f"need N >= 2 and h_max >= 0, got N {N}, h_max {h_max}")
    return N - 1 if h_max == 0 else min(int(h_max), N - 1)


def suffix_index(starts: jnp.ndarray, N: int, L: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """(idx, valid), both (B, S, L): the demonstration sample k + j that step j = 1..L of the rollout from start k
    is compared with, clipped to N - 1, and whether k + j <= N - 1 (the pair exists)."""
    idx = starts[..., None] + jnp.arange(1, L + 1)
    return jnp.minimum(idx, N - 1), idx <= N - 1


def _gather_samples(arr: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """arr (B, N, D), idx (B, ...) -> (B, ..., D) with out[b, ...] = arr[b, idx[b, ...]]."""
    return vmap(lambda a, i: a[i])(arr, idx)


def demo_suffix_guesses(theta: Theta, x_eq: jnp.ndarray, pos: jnp.ndarray, starts: jnp.ndarray,
                        L: int) -> jnp.ndarray:
    """(B, S, L, D): each start's demonstration suffix in latent coordinates, phi(x_{k+1}), ..., phi(x_{k+L}), padded
    with z* = phi(x_eq) beyond the demonstration's end. A start's initial guess the first time it is drawn, and
    every guess of the sequential solver's runs (which take none)."""
    zs = vmap(vmap(lambda x: FG.phi(theta.flow, x)))(pos)
    z_star = FG.phi(theta.flow, x_eq)
    idx, valid = suffix_index(starts, pos.shape[1], L)
    return jnp.where(valid[..., None], _gather_samples(zs, idx), z_star)


def solve_starts(theta: Theta, x_eq: jnp.ndarray, x0s: jnp.ndarray, guesses: jnp.ndarray, dt: float, solver: str,
                 tol: float, max_iter: int, chunk: int = 0):
    """rollout_batch over n starts with finite_decode (an unusable start is dropped without voiding the gradient),
    x0s (n, D) and guesses (n, L, D). With 0 < chunk < n the starts are solved `chunk` at a time (lax.map over the
    chunks, the last one padded with copies of the last start and cut off after), so that the dense step Jacobians
    of one chunk, not of all starts, are in memory at once; the values are rollout_batch's.

    The sequential loop (solver "seq") backpropagates through every step, and lax.map would keep those activations
    for every chunk at once (61 GB on a task that no run list uses, D 8, N 7880, 256 starts per demonstration),
    so its chunks are rematerialized (jax.checkpoint): the backward pass recomputes
    one chunk's forward at a time, the chunks bound the memory, and a chunked step costs one more forward pass. The
    parallel solvers' implicit adjoint keeps no per-step activations and is not rematerialized."""
    n = int(x0s.shape[0])
    if chunk <= 0 or chunk >= n:
        return rollout_batch(theta, x_eq, x0s, guesses, dt, solver, tol, max_iter, finite_decode=True)

    def one_chunk(th, x0c, gc):
        return rollout_batch(th, x_eq, x0c, gc, dt, solver, tol, max_iter, finite_decode=True)

    if solver == "seq":
        one_chunk = jax.checkpoint(one_chunk)
    return _chunked(lambda x0c, gc: one_chunk(theta, x0c, gc), (x0s, guesses), chunk)


def _chunked(fn, arrays: Tuple[jnp.ndarray, ...], chunk: int):
    """fn over the leading axis (n) of arrays, `chunk` rows at a time: lax.map over the chunks, the last one padded
    with copies of the last row and cut off after; one call on all rows when chunk <= 0 or chunk >= n. fn takes one
    chunk of each array and returns a tuple of arrays with the chunk as their leading axis."""
    n = int(arrays[0].shape[0])
    if chunk <= 0 or chunk >= n:
        return fn(*arrays)
    n_chunks = -(-n // chunk)
    pad = n_chunks * chunk - n

    def padded(a):
        a = jnp.concatenate([a, jnp.broadcast_to(a[-1:], (pad,) + a.shape[1:])]) if pad else a
        return a.reshape((n_chunks, chunk) + a.shape[1:])

    out = lax.map(lambda c: fn(*c), tuple(padded(a) for a in arrays))
    return tuple(o.reshape((n_chunks * chunk,) + o.shape[2:])[:n] for o in out)


def solve_starts_fallback(theta: Theta, x_eq: jnp.ndarray, x0s: jnp.ndarray, guesses: jnp.ndarray, dt: float,
                          solver: str, tol: float, max_iter: int, chunk: int = 0):
    """The paper runs' solve of n starts: every parallel solve stops by LASER's rule (implicit stop
    "laser"), and when any of them failed the sequential loop re-solves the step. x0s (n, D), guesses (n, L, D).

    1. Encode once, with gradient: the field with z* = phi(x_eq), and z_0 = phi(x_0) of every start.
    2. Phase A, without gradient (implicit_forward): the parallel solves in the driver's chunks.
    3. One predicate per step, outside every vmap: ok = finite path and own residual <= tol per solve; bad = any
       solve not ok.
    4. One lax.cond: when bad, the loop (seq_solve, forward only) re-solves every start in one vmap (it is latency-
       bound, so all starts cost what one chunk costs); the final path is the parallel one where ok, else the loop's.
    5. Phase B, the gradient, in the same chunks: attach_adjoint gives each final path implicit_solve's IFT adjoint
       (at a loop path, backpropagation through the loop up to rounding), valid = a finite final path; then the
       decode (_finite_decode).

    No while loop and no cond is differentiated. Returns X, Z (n, L+1, D) the decoded and latent final paths; res,
    it, ok (n,) the parallel solves' own residuals, iterations and convergence; fell_back (scalar bool, the loop
    ran); valid (n,) the starts the loss uses; retries (n,) the parallel solves' rejected trials (LASER's hidden
    scans; 0 for the solvers without retries and for the loop's re-solve, which has none)."""
    L = int(guesses.shape[1])
    kw = dict(solver="rk4", tol=tol, max_iter=max_iter, **solver_kwargs(solver, stop="laser"))
    elcd = with_equilibrium(theta, x_eq)
    z0s = vmap(lambda x: FG.phi(theta.flow, x))(x0s)
    sg_elcd, sg_z0s = lax.stop_gradient(elcd), lax.stop_gradient(z0s)
    solve = vmap(lambda z0, g: implicit_forward(make_ode, sg_elcd, z0, g, dt, retries=True, **kw))
    Zp, res, it, retries = _chunked(solve, (sg_z0s, guesses), chunk)
    ok = jnp.all(jnp.isfinite(Zp), axis=(1, 2)) & (res <= tol)
    bad = ~jnp.all(ok)
    loop = lambda: vmap(lambda z0: seq_solve(make_ode, sg_elcd, z0, dt, L, solver="rk4"))(sg_z0s)
    Zs = lax.cond(bad, loop, lambda: Zp)
    Zf = jnp.where(ok[:, None, None], Zp, Zs)
    valid = jnp.all(jnp.isfinite(Zf), axis=(1, 2))

    def attach_decode(z0, Zrow, v):
        Z = attach_adjoint(make_ode, elcd, z0, Zrow, dt, valid=v, solver="rk4")
        return _finite_decode(theta, Z, z0, v), Z

    X, Z = _chunked(vmap(attach_decode), (z0s, Zf, valid), chunk)
    return X, Z, res, it, ok, bad, valid, retries


def suffix_loss(theta: Theta, x_eq: jnp.ndarray, pos: jnp.ndarray, starts: jnp.ndarray, guesses: jnp.ndarray,
                dt: float, solver: str, tol: float, max_iter: int, h_max: int,
                chunk: int = 0, anchor=0.0, fallback: bool = False):
    """The suffix loss: the mean squared error over every valid (start, step) pair of the converged starts.

    pos (B, N, D): the training demonstrations; starts (B, S): start indices in 0..N-2; guesses (B, S, L, D) with
    L = suffix_length(N, h_max): each start's latent guess, excluding z_0. Start k of demonstration b is rolled out
    from x_k (rollout_one: z_0 = phi(x_k), L latent RK4 steps at dt, decoded) and step j is compared with x_{k+j}
    for j = 1..L while k + j <= N - 1. Loss = sum of |x_j - x_{k+j}|^2 over those pairs of the converged starts /
    (their number * D); zero when no start converged (the driver then skips the step). An unconverged start (non-
    finite, or a parallel solve's residual above tol) is dropped.

    Returns (loss, aux) with aux = dict(X, Z (B, S, L+1, D): the decoded and latent paths from each start; res, it,
    mask (B, S): residual, accepted iterations and the converged mask; valid (B, S, L): the pairs that exist;
    n_pairs: the pairs counted; n_ok: the converged starts).

    L = 1 (h_max 1) is one RK4 step per start, which every solver computes by the sequential path (parode's parallel
    solvers need at least two steps; one step has no fixed point to iterate on): the same value and gradient,
    residual 0 and 1 iteration.

    anchor a (a Python float, or a traced scalar for a ramp): loss = (1 - a) suffix + a times the pair-mean
    over the valid pairs of the converged rows whose start is 0 (aux: loss_suffix, loss_anchor; n_anchor the anchor's
    pairs); the Python float 0.0 is the plain path with nothing added to the graph.

    fallback (the paper runs; a parallel solver at L >= 2): solve_starts_fallback, so that mask is the starts with a
    finite final path (a failed parallel solve is re-solved by the loop, not dropped), Z the final paths, and res,
    it the parallel solves' own; aux gains ok_par (B, S) (the parallel solves that converged), seq_fallback (the
    loop ran this step), latent_max (max|z| over the used final paths) and retries (B, S) (the parallel solves'
    rejected trials, int32; 0 without LASER)."""
    B, N, D = (int(s) for s in pos.shape)
    S = int(starts.shape[1])
    L = suffix_length(N, h_max)
    if tuple(starts.shape) != (B, S) or tuple(guesses.shape) != (B, S, L, D):
        raise ValueError(f"need starts (B, S) = ({B}, S) and guesses (B, S, L, D) = ({B}, {S}, {L}, {D}), "
                         f"got {tuple(starts.shape)} and {tuple(guesses.shape)}")
    solver = "seq" if L < 2 else solver
    x0s = _gather_samples(pos, starts).reshape(B * S, D)
    extra = {}
    if fallback and solver != "seq":
        X, Z, res, it, ok, fell_back, usable, retries = solve_starts_fallback(
            theta, x_eq, x0s, guesses.reshape(B * S, L, D), dt, solver, tol, max_iter, chunk)
        mask = usable.astype(X.dtype).reshape(B, S)
        extra = dict(ok_par=ok.reshape(B, S), seq_fallback=fell_back,
                     latent_max=jnp.max(jnp.where(usable[:, None, None], jnp.abs(Z), 0.0)),
                     retries=retries.reshape(B, S))
    else:
        X, Z, res, it = solve_starts(theta, x_eq, x0s, guesses.reshape(B * S, L, D), dt, solver, tol, max_iter,
                                     chunk)
        mask = converged_mask(X, res, tol, solver).reshape(B, S)
    X, Z = X.reshape(B, S, L + 1, D), Z.reshape(B, S, L + 1, D)
    idx, valid = suffix_index(starts, N, L)
    target = _gather_samples(pos, idx)                                  # (B, S, L, D): x_{k+j}
    use = mask[..., None] * valid.astype(X.dtype)                       # (B, S, L): the pairs counted
    Xs = jnp.where(use[..., None] > 0, X[:, :, 1:], target)             # a dropped pair contributes zero error
    n_pairs = jnp.sum(use)
    sq = jnp.sum((Xs - target) ** 2, axis=-1)                           # (B, S, L): zero at the pairs not counted
    loss = jnp.sum(use * sq) / (jnp.maximum(n_pairs, 1.0) * D)
    aux = dict(X=X, Z=Z, res=res.reshape(B, S), it=it.reshape(B, S), mask=mask, valid=valid, n_pairs=n_pairs,
               n_ok=jnp.sum(mask), **extra)
    if isinstance(anchor, float) and anchor == 0.0:
        return loss, aux
    row0 = use * (starts == 0)[..., None].astype(X.dtype)                # the pairs of the rollouts from x_0
    n_anchor = jnp.sum(row0)
    loss_anchor = jnp.sum(row0 * sq) / (jnp.maximum(n_anchor, 1.0) * D)
    a = jnp.asarray(anchor, X.dtype)
    aux.update(loss_suffix=loss, loss_anchor=loss_anchor, n_anchor=n_anchor)
    return (1.0 - a) * loss + a * loss_anchor, aux
