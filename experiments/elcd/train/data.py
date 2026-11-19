# Third-party origin: load_lasa_raw, normalize and load_pendulum re-implement the loaders
# lasa_to_torch, NormalisedTrajData and pendulum_to_torch of data.py in
# https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
#     Copyright (c) 2024 Sean Jaffe
#     MIT License
# Licence text and list of changes: THIRD_PARTY_NOTICES.md at the repository root, section "ELCD".
# The LASA handwriting dataset and the ELCD pendulum data are read at run time and are not part of
# this repository; their terms are in the section "Data that is downloaded, not shipped" of that file.
"""Demonstrations for ELCD training at their native sampling step.

`load_lasa(shape)` reproduces the paper's loader (data.py of the ELCD repository, `lasa_to_torch` with start 15 and
`NormalisedTrajData` with subtract_mean False): the 7 demonstrations of a LASA shape, samples 15 to 999 (985 per
demonstration), positions and velocities divided by the per-coordinate standard deviation of the 3 training
demonstrations (ddof 1, torch's default), the common end point as the equilibrium. The file's velocities are the
forward differences of each demonstration at its own sampling step (pos[k+1] - pos[k] = dt_i vel[k] to 1e-14),
and the demonstrations' steps differ from the shape's nominal step `dt` by up to 20 percent; the paper regresses
those velocities as they are and rolls out at dt 0.01. Here every demonstration is treated as sampled at the
shape's nominal step: `vel` holds the forward differences at that step, exactly consistent with a rollout on the
same grid, so that the velocity loss and the matched-time rollout loss target the same trajectories.

`load_chain(shapes)` concatenates shapes in time into one long demonstration per demonstration index: every shape
is resampled to a common step, rotated so that its start-to-end direction points along +x, placed to the right of
the previous shape, and joined by a bridge with a cosine speed profile (zero speed at both ends, like the shapes
themselves): a straight segment (bridge "line"). The 7 chain
demonstrations share the last shape's end point. Autonomous dynamics cannot reproduce a self-crossing path, so
`chain_crossings` counts the bridge crossings per demonstration and the loader picks, per shape, the vertical offset
with the fewest (or takes given `offsets`). Every shape box lies to the right of all earlier ones, so a bridge can
only cross the two shapes it joins and a chain's crossing count is the sum of its adjacent pairs' counts;
`search_chain_order` uses this to score all orderings of a set of shapes from the pairwise counts, breaks ties by
the normalization scale (the one closest to the given order's chain) and verifies the winner on the full chain.
`chain8z` is chain8's shapes in the ordering it found (CHAIN_Z, recorded below; the search never runs at load
time).

`lift_phase(d)` lifts any system to phase space: the state is (x, v / s) with the demonstration's
velocity v and a fixed per-coordinate scale s (the ddof-1 standard deviation of the training velocities over that
of the training positions, in 1/s), so both halves have comparable size and an autonomous field can follow a path
that crosses itself in position. Demos.pos_dims records how many leading coordinates are positions (metrics.py
scores those only). No paper run uses the lift; the tests of metrics.py do.

`python experiments/elcd/train/data.py` runs `search_chains`, the search behind CHAIN_Z.

`load_pendulum(n)` reads the paper's n-link pendulum demonstrations (10 demonstrations of 1000 samples at 0.1 s,
no normalization, the rest state as the equilibrium). pendulum.py ('npend:<links>-N<n>[-b<damping>]', our damped
n-link pendulum) lives beside this module and is reached through `load_system`.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

LASA_DIR = os.environ.get("ELCD_LASA_DIR", "data/lasa/DataSet")     # the DataSet folder of a clone of
# bitbucket.org/khansari/lasahandwritingdataset
PENDULUM_DIR = os.environ.get("ELCD_PENDULUM_DIR", "data/pendulum")
START = 15                     # the paper's start slice (lasa_to_torch(start=15))
N_TRAIN = 3                    # the paper's training demonstrations (num_demos)
CHAIN_DT = 0.005               # common step of a chain (the shapes' native steps are 0.003 to 0.007)
BRIDGE_STEPS = 100             # samples of a bridge, its end excluded
CHAIN_GAP = 0.15               # gap between consecutive shape boxes, as a fraction of the box width
CHAIN_OFFSETS = (0.0, 1.0, -1.0, 2.0, -2.0)   # vertical offsets tried per shape, in shape-box heights, in this order
BRIDGES = ("line",)
SEARCH_GAPS = (0.15, 0.3, 0.5, 1.0)   # gaps tried by search_chain_order, smallest first, until 0 crossings
CHAINS = {"chain8": ("Sshape", "JShape", "Angle", "Snake", "Multi_Models_1", "Worm", "Saeghe", "CShape")}
# The zero-crossing chains, recorded from search_chain_order (python experiments/elcd/train/data.py, on a CPU):
# shapes in chain order, the vertical offset of shapes 2.. (in box heights), the gap, and the
# per-demonstration crossings of the full chain, counted directly on the rebuilt chain (equal to the pairwise
# prediction). At chain8's own gap 0.15, 1536 of the 40320 orderings of its 8 shapes have 0 crossings (chain8 has 6);
# the recorded one is, among them, the one normalized most like chain8 (training-demonstration standard deviations
# 124.3 and 17.7 against chain8's 126.0 and 17.9, ratio 1.014; 89 of the 1536 are within 1.1). The zero-crossing
# ordering closest to chain8's (its first four shapes first, offsets 1, -1, 0, 0, -2, 0, 0) is not used: its
# stacked offsets give a y standard deviation of 58.3 (ratio 3.26), every shape squashed 3.3 times in the normalized
# coordinates, which would confound the Chamfer comparison with chain8.
# No 6-shape fallback is needed. load_system never searches.
CHAIN_Z: Dict[str, Dict[str, object]] = {
    "chain8z": dict(shapes=("CShape", "Saeghe", "JShape", "Snake", "Angle", "Worm", "Sshape", "Multi_Models_1"),
                    offsets=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0), gap=0.15, crossings=(0, 0, 0, 0, 0, 0, 0)),
}


class Demos(NamedTuple):
    name: str
    pos: np.ndarray            # (n_demos, N, D) positions, normalized
    vel: np.ndarray            # (n_demos, N, D) velocities in normalized units per second
    dt: float                  # sampling step (s)
    x_eq: np.ndarray           # (D,) the common end point, normalized
    n_train: int               # the first n_train demonstrations train, the others are held out
    pos_std: np.ndarray        # (D,) the normalization (positions were divided by it)
    segments: Tuple[Tuple[int, int, str], ...]   # (first sample, last sample + 1, shape) per segment
    crossings: Tuple[int, ...]  # bridge crossings per demonstration (chains; zeros otherwise)
    pos_dims: Optional[int] = None         # the leading coordinates that are positions (phase-space systems);
                                           # None: all D coordinates

    @property
    def N(self) -> int:
        return int(self.pos.shape[1])

    @property
    def D(self) -> int:
        return int(self.pos.shape[2])

    @property
    def duration(self) -> float:
        return float((self.N - 1) * self.dt)

    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.pos[: self.n_train], self.vel[: self.n_train]

    def samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """The training samples (n_train N, D) of the velocity loss, as the paper's data loader yields them, with
        the forward-difference velocities on the nominal grid."""
        pos = self.pos[: self.n_train]
        vel = self.vel[: self.n_train]
        return pos.reshape(-1, self.D), vel.reshape(-1, self.D)


# ----------------------------------------------------------------------------- raw LASA

def _unwrap(v):
    return v.ravel()[0] if v.dtype == object else v


def load_lasa_raw(shape: str, lasa_dir: str = LASA_DIR) -> Tuple[np.ndarray, np.ndarray, float]:
    """(pos (7, 985, 2), vel (7, 985, 2), dt) in the file's units after the paper's start slice."""
    from scipy.io import loadmat
    path = os.path.join(lasa_dir, f"{shape}.mat")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: set ELCD_LASA_DIR to the LASA DataSet folder "
                                f"(experiments/elcd/download_data.sh fetches the dataset)")
    m = loadmat(path)
    dt = float(np.asarray(m["dt"]).ravel()[0])
    demos = m["demos"].ravel()
    pos = np.stack([np.asarray(_unwrap(d["pos"]), np.float64)[:, START:].T for d in demos])
    vel = np.stack([np.asarray(_unwrap(d["vel"]), np.float64)[:, START:].T for d in demos])
    if pos.shape[1:] != (1000 - START, 2):
        raise ValueError(f"{shape}: unexpected demonstration shape {pos.shape}")
    return pos, vel, dt


def forward_differences(pos: np.ndarray, dt: float) -> np.ndarray:
    """vel[k] = (pos[k+1] - pos[k]) / dt, the last sample at rest: the data's own convention on a uniform grid."""
    vel = np.zeros_like(pos)
    vel[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
    return vel


def normalize(name: str, pos: np.ndarray, vel: np.ndarray, dt: float, n_train: int = N_TRAIN,
              segments: Optional[Sequence[Tuple[int, int, str]]] = None,
              crossings: Optional[Sequence[int]] = None) -> Demos:
    """The paper's normalization: divide positions and velocities by the per-coordinate ddof-1 standard deviation
    of the training demonstrations (over demonstrations and time); the equilibrium is the training demonstrations'
    common end point."""
    std = pos[:n_train].reshape(-1, pos.shape[2]).std(axis=0, ddof=1)
    pos_n, vel_n = pos / std, vel / std
    ends = pos_n[:, -1]
    x_eq = ends[:n_train].mean(axis=0)
    if np.max(np.abs(ends - x_eq)) > 1e-3:
        raise ValueError(f"{name}: the demonstrations do not share an end point (spread {np.max(np.abs(ends - x_eq)):.3g})")
    seg = tuple(segments) if segments is not None else ((0, int(pos.shape[1]), name),)
    cr = tuple(int(c) for c in crossings) if crossings is not None else tuple(0 for _ in range(pos.shape[0]))
    return Demos(name, pos_n, vel_n, float(dt), x_eq, n_train, std, seg, cr)


def load_lasa(shape: str, lasa_dir: str = LASA_DIR) -> Demos:
    pos, _, dt = load_lasa_raw(shape, lasa_dir)
    return normalize(shape, pos, forward_differences(pos, dt), dt)


# ----------------------------------------------------------------------------- chains

def resample(pos: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    """Linear interpolation of (n_demos, N, D) positions from step dt_in to step dt_out over the same duration T; the
    last output sample is the demonstration's own last sample, so a shared end point stays shared exactly.

    The samples are at k dt_out for k dt_out < T, then the last sample at T, so the last
    interval is shorter than dt_out (0.16 to 0.66 of it for chain8's shapes). A forward difference on the dt_out grid
    divides it by the full step, so a shape that still moves at its end (JShape's last raw step is 2.3 times its
    median speed) loses that fraction of its velocity one step before its end."""
    n, N, D = pos.shape
    t_in = np.arange(N) * dt_in
    t_out = np.arange(0.0, t_in[-1] - 1e-12, dt_out)
    out = np.empty((n, len(t_out) + 1, D))
    for i in range(n):
        for d in range(D):
            out[i, :-1, d] = np.interp(t_out, t_in, pos[i, :, d])
        out[i, -1] = pos[i, -1]
    return out


def _rotate_to_x(pos: np.ndarray) -> np.ndarray:
    """Rotate every demonstration (same rotation for all) so that the mean start-to-end direction is +x."""
    v = pos[:, -1].mean(axis=0) - pos[:, 0].mean(axis=0)
    ang = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return pos @ R.T


def _cosine_profile(steps: int) -> np.ndarray:
    """The bridge parameter at its steps - 1 interior samples: s = (1 - cos(pi tau)) / 2, zero speed at both ends."""
    tau = np.arange(1, steps) / steps
    return 0.5 * (1.0 - np.cos(np.pi * tau))


def line_bridge(a: np.ndarray, b: np.ndarray, steps: int) -> np.ndarray:
    """(steps - 1, D) samples strictly between a and b on the straight segment with a cosine speed profile: the
    speed is zero at a and at b, so the bridge joins two resting shapes without a velocity jump."""
    s = _cosine_profile(steps)
    return a[None] + s[:, None] * (b - a)[None]


def _segments_cross(p: np.ndarray, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Proper intersections of segment (p, q) with the segments (r[i], s[i]), boolean per i."""
    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p[None]), orient(r, s, q[None])
    return (o1 * o2 < 0) & (o3 * o4 < 0)


def chain_crossings(pos: np.ndarray, segments: Sequence[Tuple[int, int, str]]) -> List[int]:
    """Per demonstration, how many times its bridges cross the rest of its own path (proper crossings only)."""
    out = []
    for i in range(pos.shape[0]):
        path = pos[i]
        n = 0
        for k in range(1, len(segments)):
            b0, b1 = segments[k - 1][1] - 1, segments[k][0]          # the bridge runs from sample b0 to sample b1
            r, s = path[:-1], path[1:]
            for j in range(b0, b1):
                hit = _segments_cross(path[j], path[j + 1], r, s)
                hit[max(0, j - 1): j + 2] = False                    # neighbours share an end point
                n += int(hit.sum())
        out.append(n)
    return out


def chain_blocks(shapes: Sequence[str], dt: float = CHAIN_DT, lasa_dir: str = LASA_DIR) -> List[np.ndarray]:
    """Every shape's demonstrations resampled to step dt (resample) and rotated so that the
    start-to-end direction is +x, in the file's units: a list of (n_demos, M_k, 2) blocks, before placement."""
    blocks = []
    for shape in shapes:
        pos, _, dt_in = load_lasa_raw(shape, lasa_dir)
        blocks.append(_rotate_to_x(resample(pos, dt_in, dt)))
    return blocks


def assemble_chain(blocks: Sequence[np.ndarray], names: Sequence[str], bridge_steps: int = BRIDGE_STEPS,
                   gap: float = CHAIN_GAP, bridge: str = "line", offsets: Optional[Sequence[float]] = None,
                   count_crossings: bool = True
                   ) -> Tuple[np.ndarray, List[Tuple[int, int, str]], List[int], List[float]]:
    """Place the blocks left to right and join them by bridges: (pos (n_demos, N, 2) unnormalized, segments,
    per-demonstration crossings of the whole chain, the vertical offset chosen for blocks 2..).

    Block 1 ends at the origin; block k+1 starts gap times its own width to the right of everything placed, at the
    height of block k's end plus offset times its own height. offsets None: per block, the first offset of
    CHAIN_OFFSETS with the fewest crossings of the chain so far (a greedy search); given: those
    offsets. bridge "line" (line_bridge). count_crossings False
    (given offsets only) skips the crossing count, which dominates the cost of a placement, and returns an empty
    crossings list (search_chain_order's scale comparison)."""
    if len(blocks) < 2:
        raise ValueError("a chain needs at least two shapes")
    if bridge not in BRIDGES:
        raise ValueError(f"bridge must be one of {BRIDGES}, got {bridge!r}")
    if offsets is not None and len(offsets) != len(blocks) - 1:
        raise ValueError(f"{len(blocks)} shapes need {len(blocks) - 1} offsets, got {len(offsets)}")
    if offsets is None and not count_crossings:
        raise ValueError("the greedy offset search needs the crossing count")
    n = blocks[0].shape[0]
    placed = [blocks[0] - blocks[0][:, -1].mean(axis=0)[None, None]]     # shape 1 ends at the origin
    segments = [(0, placed[0].shape[1], names[0])]
    parts = [placed[0]]
    chosen: List[float] = []
    cr: List[int] = []
    for k in range(1, len(blocks)):
        prev = np.concatenate(parts, axis=1)
        block = blocks[k]
        width = block[..., 0].max() - block[..., 0].min()
        height = block[..., 1].max() - block[..., 1].min()
        x_shift = prev[..., 0].max() + gap * width - block[..., 0].min()
        y_base = prev[:, -1, 1].mean() - block[:, 0, 1].mean()             # align the bridge with the +x axis
        best = None
        for offset in (CHAIN_OFFSETS if offsets is None else (float(offsets[k - 1]),)):
            cand = block + np.array([x_shift, y_base + offset * height])[None, None]
            seg = segments + [(prev.shape[1] + bridge_steps - 1, prev.shape[1] + bridge_steps - 1 + cand.shape[1], names[k])]
            if bridge == "line":
                links = [line_bridge(prev[i, -1], cand[i, 0], bridge_steps) for i in range(n)]
            demos = np.stack([np.concatenate([prev[i], links[i], cand[i]]) for i in range(n)])
            cr = chain_crossings(demos, seg) if count_crossings else []
            if best is None or sum(cr) < best[0]:
                best = (sum(cr), cand, demos, seg, cr, offset)
            if best[0] == 0:
                break
        _, cand, demos, seg, cr, offset = best
        chosen.append(float(offset))
        parts = [demos]
        segments = list(seg)
    return parts[0], segments, list(cr), chosen


def load_chain(shapes: Sequence[str], name: Optional[str] = None, dt: float = CHAIN_DT,
               bridge_steps: int = BRIDGE_STEPS, gap: float = CHAIN_GAP, lasa_dir: str = LASA_DIR,
               bridge: str = "line", offsets: Optional[Sequence[float]] = None) -> Demos:
    """One long demonstration per demonstration index: the shapes in time order (resampled by chain_blocks),
    joined by bridges (see assemble_chain for bridge and offsets), normalized like a LASA shape."""
    if len(shapes) < 2:
        raise ValueError("a chain needs at least two shapes")
    pos, segments, cr, _ = assemble_chain(chain_blocks(shapes, dt, lasa_dir), list(shapes), bridge_steps,
                                          gap, bridge, offsets)
    return normalize(name or "chain_" + "_".join(shapes), pos, forward_differences(pos, dt), dt, segments=segments,
                     crossings=cr)


def _train_scale(pos: np.ndarray, n_train: int = N_TRAIN) -> np.ndarray:
    """(D,) the per-coordinate ddof-1 standard deviation of the training demonstrations: what normalize divides by."""
    return pos[:n_train].reshape(-1, pos.shape[2]).std(axis=0, ddof=1)


def _inversions(perm: Sequence[int]) -> int:
    """Kendall distance of a sequence of distinct indices to their sorted order (pairs out of order)."""
    return sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j])


def chain_pair_costs(blocks: Sequence[np.ndarray], names: Sequence[str], gap: float,
                     bridge_steps: int = BRIDGE_STEPS, bridge: str = "line") -> Tuple[np.ndarray, np.ndarray]:
    """(cost (K, K) int, offset (K, K)): the crossings summed over demonstrations of the 2-shape chain a -> b at
    its best offset, and that offset (the diagonal unused). A bridge crosses only the two shapes it joins, so the
    crossings of a chain are the sum of these over its adjacent pairs."""
    K = len(blocks)
    cost, offset = np.zeros((K, K), dtype=np.int64), np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            if a != b:
                _, _, cr, off = assemble_chain([blocks[a], blocks[b]], [names[a], names[b]], bridge_steps, gap, bridge)
                cost[a, b], offset[a, b] = sum(cr), off[0]
    return cost, offset


def search_chain_order(shapes: Sequence[str], length: Optional[int] = None, gaps: Sequence[float] = SEARCH_GAPS,
                       bridge_steps: int = BRIDGE_STEPS, bridge: str = "line", dt: float = CHAIN_DT,
                       lasa_dir: str = LASA_DIR) -> Dict[str, object]:
    """The ordering of `length` of the shapes (all by default) with the fewest bridge crossings, found exhaustively.

    For each gap in turn (smallest first, stopping at the first with a zero-crossing ordering): the pairwise costs
    of chain_pair_costs, then every ordered selection scored by the sum over its adjacent pairs (8! = 40320
    orderings of 8 shapes). Ties go to the most samples (subsets only), then to the ordering whose normalization
    scale is closest to that of the given order's chain (load_chain's default build at CHAIN_GAP): the largest
    |log| ratio of the per-coordinate standard deviations of the training demonstrations, which load_chain divides by.
    Vertical offsets that stack shapes up and down inflate the y scale and shrink every shape in the normalized
    coordinates that the model, the loss and the metrics see, so a reordered chain with another scale is not
    comparable with the given one. Then to the ordering closest to the given one (fewest pairs out of order), then
    to the lexicographically first. The winner is rebuilt as a full chain with its offsets (assemble_chain) and its
    crossings counted directly; `verified` says whether they equal the predicted sum. Returns shapes, offsets, gap,
    crossings (per demonstration), total, predicted, verified, the scale ratio (1 = the given chain's scale) and both
    scales, minimum-count orderings, and the per-gap minima."""
    blocks = chain_blocks(shapes, dt, lasa_dir)
    K = len(shapes)
    length = K if length is None else int(length)
    sizes = [b.shape[1] for b in blocks]
    tried = {}
    best = None
    for gap in gaps:
        cost, off = chain_pair_costs(blocks, shapes, gap, bridge_steps, bridge)
        scored = []
        for perm in itertools.permutations(range(K), length):
            c = int(sum(cost[perm[i], perm[i + 1]] for i in range(length - 1)))
            scored.append((c, -sum(sizes[p] for p in perm), _inversions(perm), perm))
        scored.sort()
        tried[float(gap)] = scored[0][0]
        n_min = sum(1 for s in scored if s[0] == scored[0][0])
        top = [s for s in scored if s[:2] == scored[0][:2]]          # the fewest crossings, then the most samples
        if best is None or scored[0][0] < best[0]:
            best = (scored[0][0], top, gap, off, n_min)
        if scored[0][0] == 0:
            break
    c, top, gap, off, n_min = best
    ref = _train_scale(assemble_chain(blocks, list(shapes), bridge_steps, CHAIN_GAP, bridge)[0])

    def offsets_of(perm: Sequence[int]) -> List[float]:
        return [float(off[perm[i], perm[i + 1]]) for i in range(length - 1)]

    def scale_of(perm: Sequence[int]) -> np.ndarray:
        return _train_scale(assemble_chain([blocks[p] for p in perm], [shapes[p] for p in perm], bridge_steps, gap,
                                           bridge, offsets_of(perm), count_crossings=False)[0])

    ranked = sorted((float(np.max(np.abs(np.log(scale_of(s[3]) / ref)))), s[2], s[3]) for s in top)
    mismatch, _, perm = ranked[0]
    order = [shapes[p] for p in perm]
    offsets = offsets_of(perm)
    pos, _, cr, _ = assemble_chain([blocks[p] for p in perm], order, bridge_steps, gap, bridge, offsets)
    return dict(shapes=tuple(order), offsets=tuple(offsets), gap=float(gap), crossings=tuple(int(x) for x in cr),
                total=int(sum(cr)), predicted=int(c), verified=int(sum(cr)) == int(c),
                scale_ratio=float(np.exp(mismatch)), scale=_train_scale(pos).tolist(), scale_given=ref.tolist(),
                n_minimal=int(n_min), minimum_by_gap=tried, bridge=bridge, bridge_steps=int(bridge_steps))


def search_chains() -> None:
    """The exhaustive ordering search for the 8 shapes of chain8; prints the CHAIN_Z entry to record above."""
    t0 = time.time()
    res8 = search_chain_order(CHAINS["chain8"])
    print(f"chain8z search ({time.time() - t0:.0f} s): {json.dumps(res8)}", flush=True)
    found = {"chain8z": res8}
    print("\n# to record in data.CHAIN_Z:")
    for name, r in found.items():
        print(f"    {name!r}: dict(shapes={r['shapes']!r}, offsets={r['offsets']!r}, gap={r['gap']!r}, "
              f"crossings={r['crossings']!r}),")
    # the given order, for comparison
    for name in ("chain8",):
        d = load_chain(CHAINS[name], name=name)
        print(f"{name}: crossings {list(d.crossings)} (total {sum(d.crossings)})")


def load_chain_z(spec: str) -> Demos:
    """A recorded zero-crossing chain of CHAIN_Z (no search at load time)."""
    if spec not in CHAIN_Z:
        raise ValueError(f"{spec!r} is not recorded in data.CHAIN_Z (python experiments/elcd/train/data.py runs the search)")
    cfg = CHAIN_Z[spec]
    return load_chain(cfg["shapes"], name=spec, gap=float(cfg["gap"]), offsets=cfg["offsets"])


# ----------------------------------------------------------------------------- phase space

def phase_scale(d: Demos) -> np.ndarray:
    """(D,) the lift's velocity scale s in 1/s: per coordinate, the ddof-1 standard deviation of the training
    velocities over that of the training positions (over demonstrations and time), so that v / s has the spread
    of the positions."""
    pos, vel = d.pos[: d.n_train].reshape(-1, d.D), d.vel[: d.n_train].reshape(-1, d.D)
    return vel.std(axis=0, ddof=1) / pos.std(axis=0, ddof=1)


def lift_phase(d: Demos) -> Demos:
    """The phase-space lift: state (x, v / s) of dimension 2 D with s = phase_scale(d); velocities the forward
    differences of the lifted state on the same grid (so the first half of the lifted velocity is v itself when v is
    the forward-difference velocity, as for LASA shapes and chains); x_eq = (x_eq, 0); pos_std = (pos_std, pos_std s), so that the lifted
    coordinates times pos_std are the file's positions and velocities; pos_dims = the original D (or its own
    pos_dims when d is lifted already); segments and the position-space crossings kept."""
    s = phase_scale(d)
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"{d.name}: the velocity scale {s} is not positive (a coordinate without motion?)")
    state = np.concatenate([d.pos, d.vel / s], axis=-1)
    return Demos("phase:" + d.name, state, forward_differences(state, d.dt), d.dt,
                 np.concatenate([d.x_eq, np.zeros(d.D)]), d.n_train, np.concatenate([d.pos_std, d.pos_std * s]),
                 d.segments, d.crossings, d.D if d.pos_dims is None else int(d.pos_dims))


# ----------------------------------------------------------------------------- pendulum

def load_pendulum(n_links: int, pendulum_dir: str = PENDULUM_DIR, dt: float = 0.1) -> Demos:
    """The paper's pendulum data, unnormalized: the loader's `pendulum_to_torch` (npz keys my_X_phy, my_X_dot_phy
    of shape (T, n_demos, D)); the equilibrium is the demonstrations' common last sample (the rest state)."""
    path = os.path.join(pendulum_dir, f"n-{n_links}.npz")
    with np.load(path) as data:
        pos = np.transpose(np.asarray(data["my_X_phy"], np.float64), (1, 0, 2))      # (n_demos, T, D)
        vel = np.transpose(np.asarray(data["my_X_dot_phy"], np.float64), (1, 0, 2))
    ends = pos[:, -1]
    x_eq = ends[:N_TRAIN].mean(axis=0)
    D = pos.shape[2]
    return Demos(f"pend{n_links}", pos, vel, float(dt), x_eq, N_TRAIN, np.ones(D), ((0, int(pos.shape[1]), f"pend{n_links}"),),
                 tuple(0 for _ in range(pos.shape[0])))


# ----------------------------------------------------------------------------- registry

def load_system(spec: str) -> Demos:
    """'lasa:<shape>'; 'chain8z' (the CHAIN_Z entries); 'pend:<n>' (the paper's pendulum);
    'npend:<links>-N<n>[-b<damping>]' (our pendulum, pendulum.py)."""
    if spec.startswith("lasa:"):
        return load_lasa(spec[5:])
    if spec in CHAIN_Z:
        return load_chain_z(spec)
    if spec.startswith("pend:"):
        return load_pendulum(int(spec[5:]))
    if spec.startswith("npend:"):
        from experiments.elcd.train import pendulum
        return pendulum.load_npend(spec[6:])
    raise ValueError(f"unknown system spec {spec!r}")


def describe(d: Demos) -> Dict[str, object]:
    return dict(name=d.name, n_demos=int(d.pos.shape[0]), n_train=d.n_train, N=d.N, D=d.D, dt=d.dt,
                duration_s=d.duration, pos_std=d.pos_std.tolist(), x_eq=d.x_eq.tolist(),
                segments=[list(s) for s in d.segments], crossings=list(d.crossings),
                speed_max=float(np.linalg.norm(d.vel, axis=-1).max()),
                pos_dims=None if d.pos_dims is None else int(d.pos_dims))


if __name__ == "__main__":
    search_chains()
