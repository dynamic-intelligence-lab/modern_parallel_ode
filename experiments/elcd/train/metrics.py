# Third-party origin: chamfer and truncate re-implement utils.dtwd_np and the stopping rule of
# utils.forward_step_until_converge in
# https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
#     Copyright (c) 2024 Sean Jaffe
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section "ELCD".
"""Evaluation of a trained ELCD model against demonstrations.

Two rollouts per demonstration: the native one (latent RK4 at the demonstration's step for as many steps as the
demonstration has, decoded) and the paper's protocol (data-space RK4 at dt 0.01, up to 1000 steps, cut at the
first step below 1e-5 in 2-norm). Metrics: the paper's Chamfer distance (its DTWD, `chamfer`) on
both rollouts; the matched-time RMSE of the native rollout; the settling-time ratio, the first time at which the
native rollout's speed drops below SPEED_TOL normalized units per second (the paper's 1e-5 per step at dt 0.01),
divided by the demonstration's duration, from a rollout of SETTLE_FACTOR times the demonstration's length; the
arrival-time ratio, the first time the rollout comes within ARRIVAL_RADIUS of the equilibrium divided by the
demonstration's own such time (the settling criterion measures the asymptotic contraction rate near the goal, which
a matched-time loss barely constrains; the arrival time measures when the path gets there); the end point error
|X_N - x_eq|; and the Chamfer of rollouts from perturbed starts (the paper's out-of-sample argument). The paper's
rollout runs 1000 steps or PAPER_COVER times the demonstration's duration, whichever is longer (`paper_steps`).

Phase-space systems (Demos.pos_dims set, data.lift_phase): every distance (both Chamfers, the RMSE, the arrival
and the end error, the perturbed Chamfer) uses only the first pos_dims coordinates of the rollouts, the
demonstrations and x_eq, while the rollouts, the perturbation of the starts and the two stopping rules (the
settling speed and the paper's until-converge cut) use the full state: a lifted rollout that follows a
demonstration through a momentary rest (the junctions of a chain) has zero position speed there without having
stopped. With pos_dims None nothing is sliced and every number is what it was before the option existed.

The Chamfer distances go through `chamfer_blocked`, which evaluates chamfer's expression over blocks
of trajectory rows: chamfer builds the whole (n, N, D) difference tensor, 51 GB for a rollout against
a demonstration at N 20000 and D 16 (npend:8-N20000, a task that no run list uses), which killed the final
evaluation of such a run (host OOM at 24 GB). The blocked version gives the same number bit for bit (every distance
by the same expression, the minima exact, the means over the same vectors).

The multi-start error (`multistart_error`) is the quantity the suffix loss trains,
measured on every demonstration, held-out ones included. The model starts at a recorded state x_k, rolls out to the
demonstration's last sample, and is compared with x_{k+1}, ..., x_{N-1} at matched times. The starts are every k
when the demonstration has at most MULTISTART_ALL samples (the naming contract: N <= 1000), else MULTISTART_MAX evenly
spaced ones. The error is the RMSE over all (start, time) pairs on the position coordinates, each pair weighing the
same. `evaluate` adds it to every row (multistart_rmse, multistart_starts) and `summarize` adds its train and
held-out mean and median; every other value is computed by the same code as before, so it does not change. The
rollouts from MULTISTART_PATHS of the starts (evenly spaced over the first 80 percent of the demonstration) are kept
for the notebook's figure of rollouts from later starts.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from experiments.elcd import flow_gatherfree as FG
from experiments.elcd import solvers
from experiments.elcd.model import field, make_ode
from experiments.elcd.train.data import Demos
from experiments.elcd.train.losses import Theta, with_equilibrium

PAPER_DT, PAPER_STEPS, PAPER_TOL = 0.01, 1000, 1e-5
PAPER_COVER = 2.0                          # the paper's rollout runs at least twice the demonstration's duration
DIVERGED = 10.0                            # a Chamfer above this (normalized units) counts as a diverged rollout
SPEED_TOL = PAPER_TOL / PAPER_DT           # 1e-3 normalized units per second
SETTLE_FACTOR = 3                          # the settling rollout covers three demonstration durations
ARRIVAL_RADIUS = 0.05                      # normalized units: the arrival criterion
PERTURB_SIGMA, PERTURB_COUNT = 0.1, 5
CHAMFER_BLOCK_BYTES = 2 ** 27              # largest (rows, N, D) float64 difference block of chamfer_blocked (128 MB)
MULTISTART_ALL = 1000                      # every recorded state is a start when a demonstration has at most this
                                           # many samples (N <= 1000, the naming contract)
MULTISTART_MAX = 128                       # otherwise this many evenly spaced starts (the first sample included)
MULTISTART_PATHS, MULTISTART_PATH_SPAN = 4, 0.8   # rollouts kept: 4 starts evenly spaced over the first 80 percent
MULTISTART_BLOCK = 2 ** 21                 # starts x samples x coordinates of one rollout block (bounds the memory)
# Measured on a 4-core CPU node (multistart_error alone): 7 demonstrations at N 985, D 2, all 984 starts
# 34 s, 1.3 GB peak; N 7880, D 32, 128 starts 759 s, 1.4 GB (2 GB at twice the block, 707 s); 10 demonstrations at
# N 20000, D 16, 128 starts 1181 s, 1.4 GB. The time is compute (RK4 and the decode at large D), not memory.
MULTISTART_KEYS = ("multistart_rmse",)     # the per-demonstration multi-start values that summarize averages


def chamfer(traj: np.ndarray, reference: np.ndarray) -> float:
    """The paper's distance (utils.dtwd_np of the ELCD repository): mean over reference points of the distance to
    the nearest trajectory point plus mean over trajectory points of the distance to the nearest reference point.
    traj (n, D), reference (N, D)."""
    d = np.linalg.norm(traj[:, None, :] - reference[None, :, :], axis=-1)
    return float(np.mean(np.min(d, axis=0)) + np.mean(np.min(d, axis=1)))


def truncate(traj: np.ndarray, tol: float = 1e-5, max_steps: Optional[int] = None) -> np.ndarray:
    """The notebooks' until-converge rule on a trajectory (n, D) including its start: keep the points up to and
    including the first step whose 2-norm is below tol."""
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    below = np.nonzero(steps < tol)[0]
    n = int(below[0]) + 1 if below.size else traj.shape[0] - 1
    if max_steps is not None:
        n = min(n, max_steps)
    return traj[: n + 1]


def chamfer_blocked(traj: np.ndarray, reference: np.ndarray, max_bytes: int = CHAMFER_BLOCK_BYTES) -> float:
    """chamfer(traj, reference), computed over blocks of trajectory rows so that no difference tensor
    exceeds max_bytes (float64). Bit-identical to chamfer: each distance is the same norm of the same
    difference, a block's row minima and the running column minimum are exact, and the two means see the same
    vectors. Arrays that fit in one block go straight to chamfer."""
    traj, reference = np.asarray(traj), np.asarray(reference)
    n, N, D = traj.shape[0], reference.shape[0], reference.shape[1]
    rows = max(1, int(max_bytes // max(1, N * D * 8)))
    if rows >= n:
        return chamfer(traj, reference)
    row_min, col_min = [], None
    for s in range(0, n, rows):
        d = np.linalg.norm(traj[s: s + rows, None, :] - reference[None, :, :], axis=-1)
        row_min.append(np.min(d, axis=1))
        m = np.min(d, axis=0)
        col_min = m if col_min is None else np.minimum(col_min, m)
    return float(np.mean(col_min) + np.mean(np.concatenate(row_min)))


def _native_rollout_fn(dt: float, n_steps: int):
    def run(theta: Theta, x_eq, x0):
        elcd = with_equilibrium(theta, x_eq)
        Z = solvers.seq_scan(make_ode(elcd), FG.phi(theta.flow, x0), dt, n_steps, "rk4")
        return vmap(lambda z: FG.psi(theta.flow, z))(Z)
    return jax.jit(vmap(run, in_axes=(None, None, 0)))


def _velocity_fn():
    """The model's data-space velocity at every sample of every demonstration: (n, N, D)."""
    from experiments.elcd.train.losses import data_velocity
    return jax.jit(vmap(vmap(data_velocity, in_axes=(None, 0, None)), in_axes=(None, 0, None)))


def paper_steps(duration: float) -> int:
    """The paper's rollout length: its 1000 steps at dt 0.01 (10 s), or PAPER_COVER times the demonstration's
    duration when that is longer (the chains last 8 to 38 s, which 10 s would truncate)."""
    return max(PAPER_STEPS, int(np.ceil(PAPER_COVER * duration / PAPER_DT)))


def _paper_rollout_fn(n_steps: int = PAPER_STEPS):
    def run(theta: Theta, x_eq, x0):
        elcd = with_equilibrium(theta, x_eq)

        def ode(t, x, u=None):
            return jnp.linalg.solve(FG.Q(theta.flow, x), field(elcd, FG.phi(theta.flow, x)))
        return solvers.seq_scan(ode, x0, PAPER_DT, n_steps, "rk4")
    return jax.jit(vmap(run, in_axes=(None, None, 0)))


def settling_time(X: np.ndarray, dt: float, speed_tol: float = SPEED_TOL) -> float:
    """The time of the first step whose speed is below speed_tol; inf when there is none."""
    speed = np.linalg.norm(np.diff(X, axis=0), axis=1) / dt
    below = np.nonzero(speed < speed_tol)[0]
    return float(below[0] * dt) if below.size else float("inf")


def arrival_time(X: np.ndarray, x_eq: np.ndarray, dt: float, radius: float = ARRIVAL_RADIUS) -> float:
    """The time of the first sample within `radius` of the equilibrium; inf when there is none."""
    near = np.nonzero(np.linalg.norm(X - x_eq, axis=1) < radius)[0]
    return float(near[0] * dt) if near.size else float("inf")


def positions_of(demos: Demos) -> Callable[[np.ndarray], np.ndarray]:
    """The map from states (..., D) to the positions the metrics compare: the first demos.pos_dims coordinates, or
    the identity (the array itself, nothing sliced) when pos_dims is None."""
    if demos.pos_dims is None:
        return lambda a: a
    p = int(demos.pos_dims)
    if not 0 < p <= demos.D:
        raise ValueError(f"{demos.name}: pos_dims {p} outside 1..{demos.D}")
    return lambda a: a[..., :p]


def multistart_indices(N: int, max_starts: int = MULTISTART_MAX, all_max: int = MULTISTART_ALL) -> np.ndarray:
    """The start indices k of the multi-start error for a demonstration of N samples. Every k in 0..N-2 (the last
    sample has nothing after it) when the demonstration has at most all_max samples (the naming contract: every
    recorded state when N <= 1000), or when its N - 1 starts number at most max_starts; otherwise max_starts indices
    evenly spaced from 0 to N - 2, rounded to the nearest sample (both ends included)."""
    if N < 2 or max_starts < 1:
        raise ValueError(f"need N >= 2 and max_starts >= 1, got N {N}, max_starts {max_starts}")
    n = N - 1
    if N <= all_max or n <= max_starts:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n - 1, max_starts)).astype(np.int64))


def multistart_path_starts(ks: np.ndarray, N: int, n_paths: int = MULTISTART_PATHS,
                           span: float = MULTISTART_PATH_SPAN) -> np.ndarray:
    """The starts whose rollouts multistart_error keeps for the figure of rollouts from later starts: for each target
    span * (N - 1) * j / n_paths (j = 1..n_paths; evenly spaced over the first `span` of the demonstration, x_0's
    rollout being the native one), the member of ks nearest to it, so a kept rollout is one the metric computed.
    Sorted, duplicates removed; empty for n_paths 0."""
    ks = np.asarray(ks, np.int64)
    if n_paths <= 0 or ks.size == 0:
        return np.zeros(0, np.int64)
    targets = span * (N - 1) * np.arange(1, n_paths + 1) / n_paths
    return np.unique(ks[np.abs(ks[None, :] - targets[:, None]).argmin(axis=1)])


def multistart_error(theta: Theta, demos: Demos, max_starts: int = MULTISTART_MAX, all_max: int = MULTISTART_ALL,
                     block: int = MULTISTART_BLOCK, n_paths: int = MULTISTART_PATHS) -> Dict[str, np.ndarray]:
    """The multi-start error of every demonstration, training and held-out.

    From each start x_k (k in multistart_indices(N, max_starts, all_max)) the model rolls out to the demonstration's
    last sample, and the rollout's step j is compared with x_{k+j} for j = 1..N-1-k, on the position coordinates
    (positions_of). The rollout is the native one (`_native_rollout_fn`: latent RK4 at the demonstration's dt,
    decoded), vmapped over the starts. Every start runs the same N - 1 steps, so one compiled program serves all of
    them, and the steps past the demonstration's end are masked out. The starts go through in blocks of at most
    `block` starts x samples x coordinates, which bounds the memory of the decode. The last block is padded with
    copies of its last start, so it has the same shape (no second compilation); the copies are dropped.

    Returns numpy arrays (float64 errors):
      k (S,)               the start indices
      rmse (n,)            per demonstration, the RMSE over all valid (start, time) pairs and position coordinates,
                           each pair weighing the same; nan when a valid pair is not finite
      sse_start (n, S)     the squared error of each start, summed over its pairs and the position coordinates
      pairs_start (S,)     each start's pair count, N - 1 - k
      sse_lag (n, N - 1)   the squared error at lag j = 1..N-1 (the j-th step after the start), summed over starts
      pairs_lag (N - 1,)   the number of starts that have a pair at lag j
      pos_dims ()          the number of coordinates compared
      paths_k (S',)        the starts whose rollouts are kept (multistart_path_starts, S' <= n_paths)
      paths (n, S', N, p)  those rollouts on the position coordinates, indexed by the step j after the start (row 0 is
                           the decoded x_k, row j the time of sample k + j), NaN past the demonstration's end (j > N - 1 - k)
    A horizon cap H reads off the lag sums: sqrt(sum_{j <= H} sse_lag / (pos_dims * sum_{j <= H} pairs_lag)). The
    kept rollouts change none of the other arrays."""
    N, n_demos = demos.N, int(demos.pos.shape[0])
    ks = multistart_indices(N, max_starts, all_max)
    S = int(ks.size)
    P = positions_of(demos)
    p = int(P(np.zeros(demos.D)).shape[-1])
    C = int(max(1, min(S, block // max(1, N * demos.D))))           # starts per block
    x_eq = jnp.asarray(demos.x_eq, theta.elcd.z_star.dtype)
    run = _native_rollout_fn(demos.dt, N - 1)                         # one jit for every block and demonstration
    lag = np.arange(N)
    pairs_start = (N - 1 - ks).astype(np.int64)
    pairs_lag = np.searchsorted(ks, N - 1 - lag[1:], side="right").astype(np.int64)
    sse_start, sse_lag = np.zeros((n_demos, S)), np.zeros((n_demos, N - 1))
    finite = np.ones(n_demos, bool)
    paths_k = multistart_path_starts(ks, N, n_paths)
    paths = np.full((n_demos, paths_k.size, N, p), np.nan)
    for i in range(n_demos):
        ref = P(np.asarray(demos.pos[i], np.float64))                   # (N, p)
        for a in range(0, S, C):
            kb = ks[a: a + C]
            m = int(kb.size)
            kp = np.concatenate([kb, np.repeat(kb[-1:], C - m)])
            X = np.asarray(run(theta, x_eq, jnp.asarray(demos.pos[i, kp], x_eq.dtype)), np.float64)[:m]   # (m, N, D)
            t = kb[:, None] + lag[None, :]                              # the demonstration's sample at each step
            valid = (lag[None, :] >= 1) & (t <= N - 1)
            with np.errstate(invalid="ignore", over="ignore"):
                diff = np.where(valid[..., None], P(X) - ref[np.minimum(t, N - 1)], 0.0)
                sq = np.sum(diff ** 2, axis=2)                          # (m, N): zero at the masked steps
            finite[i] &= bool(np.all(np.isfinite(diff)))
            sse_start[i, a: a + m] = np.sum(sq, axis=1)
            sse_lag[i] += np.sum(sq[:, 1:], axis=0)
            for q, k in enumerate(paths_k):                             # the kept rollouts that fall in this block
                hit = np.flatnonzero(kb == k)
                if hit.size:
                    paths[i, q, : N - k] = P(X[hit[0], : N - k])
    with np.errstate(invalid="ignore"):
        rmse = np.sqrt(np.sum(sse_start, axis=1) / (np.sum(pairs_start) * p))
    rmse = np.where(finite, rmse, np.nan)
    return dict(k=ks, rmse=rmse, sse_start=sse_start, pairs_start=pairs_start, sse_lag=sse_lag, pairs_lag=pairs_lag,
                pos_dims=np.int64(p), paths_k=paths_k, paths=paths)


def evaluate(theta: Theta, demos: Demos, perturb_seed: int = 0, keep_rollouts: bool = False,
             multistart: bool = True, max_starts: int = MULTISTART_MAX) -> Dict[str, object]:
    """Per demonstration: the metrics above, with train_demo True for the training demonstrations. Runs the
    rollouts on the current default device (call inside a Slurm job). keep_rollouts adds the native (n, N, D),
    settling (n, 3N, D), paper (n, 1001, D) and perturbed (n, PERTURB_COUNT, N, D) rollouts to the result, and the
    model's velocity at every demonstration sample, vel_pred (n, N, D), for the velocity-error mechanism figure.
    With demos.pos_dims set, the distances use the position coordinates only (module docstring); the kept rollouts
    are full states.

    multistart: after the metrics above, which it does not touch, add multistart_rmse and
    multistart_starts (the start count) to every row from multistart_error(theta, demos, max_starts), and that
    function's arrays to the result as `multistart`. False gives the rows without these two keys."""
    x_eq = jnp.asarray(demos.x_eq, theta.elcd.z_star.dtype)
    x0s = jnp.asarray(demos.pos[:, 0], x_eq.dtype)
    N, dt = demos.N, demos.dt
    native = np.asarray(_native_rollout_fn(dt, N - 1)(theta, x_eq, x0s))                      # (n, N, D)
    settle = np.asarray(_native_rollout_fn(dt, SETTLE_FACTOR * (N - 1))(theta, x_eq, x0s))    # (n, 3N, D)
    n_paper = paper_steps(demos.duration)
    paper = np.asarray(_paper_rollout_fn(n_paper)(theta, x_eq, x0s))                          # (n, n_paper + 1, D)
    rng = np.random.default_rng(perturb_seed)
    P = positions_of(demos)
    x_eq_pos = P(demos.x_eq)
    rows, perts = [], []
    for i in range(demos.pos.shape[0]):
        ref = P(demos.pos[i])
        starts = demos.pos[i, 0][None] + PERTURB_SIGMA * rng.standard_normal((PERTURB_COUNT, demos.D))
        pert = np.asarray(_native_rollout_fn(dt, N - 1)(theta, x_eq, jnp.asarray(starts, x_eq.dtype)))
        perts.append(pert)
        finite = bool(np.all(np.isfinite(native[i])))
        t_settle = settling_time(settle[i], dt) if np.all(np.isfinite(settle[i])) else float("inf")
        t_arrive = arrival_time(P(settle[i]), x_eq_pos, dt) if np.all(np.isfinite(settle[i])) else float("inf")
        t_arrive_demo = arrival_time(ref, x_eq_pos, dt)
        rows.append(dict(
            demo=i, train_demo=i < demos.n_train, finite=finite,
            chamfer_native=chamfer_blocked(P(native[i]), ref) if finite else float("nan"),
            chamfer_paper=chamfer_blocked(P(truncate(paper[i], PAPER_TOL, n_paper)), ref) if np.all(np.isfinite(paper[i])) else float("nan"),
            nstop_paper=int(truncate(paper[i], PAPER_TOL, n_paper).shape[0] - 1) if np.all(np.isfinite(paper[i])) else -1,
            rmse=float(np.sqrt(np.mean((P(native[i]) - ref) ** 2))) if finite else float("nan"),
            settling_time=t_settle, settling_ratio=t_settle / demos.duration,
            arrival_time=t_arrive, arrival_time_demo=t_arrive_demo,
            arrival_ratio=t_arrive / t_arrive_demo if np.isfinite(t_arrive_demo) and t_arrive_demo > 0 else float("inf"),
            end_error=float(np.linalg.norm(P(native[i, -1]) - x_eq_pos)) if finite else float("nan"),
            chamfer_perturbed=float(np.mean([chamfer_blocked(P(p), ref) for p in pert])) if np.all(np.isfinite(pert)) else float("nan")))
    out = dict(rows=rows, dt=dt, N=N, n_paper=n_paper)
    if multistart:
        ms = multistart_error(theta, demos, max_starts)
        for i, r in enumerate(rows):
            r.update(multistart_rmse=float(ms["rmse"][i]), multistart_starts=int(ms["k"].size))
        out["multistart"] = ms
    if keep_rollouts:
        vel_pred = np.asarray(_velocity_fn()(theta, jnp.asarray(demos.pos, x_eq.dtype), x_eq))
        out.update(native=native, settle=settle, paper=paper, perturbed=np.stack(perts), vel_pred=vel_pred)
    return out


def _splits(rows: List[Dict]):
    return (("train", [r for r in rows if r["train_demo"]]), ("heldout", [r for r in rows if not r["train_demo"]]))


def _mean_median(out: Dict[str, float], tag: str, key: str, sel: List[Dict]):
    """out[<tag>_<key>_mean]: the mean, inf when a value is not finite; out[<tag>_<key>_median]: the median."""
    vals = np.asarray([r[key] for r in sel], float)
    out[f"{tag}_{key}_mean"] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else float("inf")
    out[f"{tag}_{key}_median"] = float(np.median(vals))


def summarize_multistart(rows: List[Dict]) -> Dict[str, float]:
    """train_/heldout_multistart_rmse_mean and _median (the convention of summarize); empty when a row lacks the
    multi-start values (rows written by evaluate(multistart=False))."""
    out: Dict[str, float] = {}
    if not rows or not all(key in r for r in rows for key in MULTISTART_KEYS):
        return out
    for tag, sel in _splits(rows):
        for key in MULTISTART_KEYS:
            if sel:
                _mean_median(out, tag, key, sel)
    return out


def summarize(rows: List[Dict]) -> Dict[str, float]:
    """Means and medians over training and held-out demonstrations of the metrics that matter, the count of
    diverged paper-protocol rollouts (Chamfer non-finite or above DIVERGED), the finite count and the count. The
    multi-start keys (summarize_multistart) follow when the rows have them, so rows without them summarize exactly
    as before, keys and order alike."""
    out = {}
    for tag, sel in _splits(rows):
        if not sel:
            continue
        for key in ("chamfer_native", "chamfer_paper", "rmse", "settling_ratio", "arrival_ratio", "end_error", "chamfer_perturbed"):
            _mean_median(out, tag, key, sel)
        cp = np.asarray([r["chamfer_paper"] for r in sel], float)
        out[f"{tag}_chamfer_paper_diverged"] = int(np.sum(~np.isfinite(cp) | (cp > DIVERGED)))
        out[f"{tag}_finite"] = int(sum(r["finite"] for r in sel))
        out[f"{tag}_n"] = len(sel)
    out.update(summarize_multistart(rows))
    return out
