#!/usr/bin/env python3
"""Export the real data behind panel A of the ELCD figure (make_schematic_C.py): chain8z, the model of panel A, one
rollout from a mid-chain start, and the LASER iterates from a stale warm start.

    python experiments/elcd/schematic/export_chain8z.py [--alpha 2] [--start 4951] [--out <file>]

Writes an npz with (data-space coordinates, normalized units, dt = 0.005 s):
  X_star (N+1, 2)        the model's trajectory from the start: latent RK4 (solvers.seq_scan), decoded.  "x*(t)"
  guess_src (N+1, 2)     the stale warm start: the same start rolled out under the earlier checkpoint (--ckpt)
  iterates (K+1, N+1, 2) x^(0) = X_star + alpha (guess_src - X_star) encoded, then the k-th ACCEPTED LASER iterate
                         (LaserConfig(), tol 1e-9), decoded. iterates[0] is the guess.
  err (K+1,)             max_t |x^(k)(t) - x*(t)| (data space); res (K,) residual max|r| after each iteration;
  exact_prefix (K+1,)    first sample index where |z^(k) - z*| > 1e-6 (DEER/LASER: the first k steps are exact)
  grid_x, grid_y (G,), field_u, field_v (G, G)   the data-space field Q(x)^{-1} f(phi(x)) on a grid over the bbox
  x_eq (2,), start, N, dt, K, alpha, seq_steps_per_iter (from the L40S fp64 benchmark fit), model, ckpt
The training demonstrations from the same sample on (demo 0 provides the start) are not written: the LASA dataset
may not be redistributed, and make_schematic_C.py rebuilds them from the dataset (training_demos).
CPU, fp64, ~30 s. Run from the repository root (imports experiments.elcd). The two models it reads (MODEL, CKPT:
an earlier fp64 run) are not in this repository, so the script documents how schematic_data.npz was made. It writes
results/elcd/check/schematic_data.npz; --out experiments/elcd/schematic/schematic_data.npz rewrites the shipped file.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from parode.laser import LaserConfig, run_damped_newton  # noqa: E402
from experiments.elcd import flow_gatherfree as FG  # noqa: E402
from experiments.elcd import solvers  # noqa: E402
from experiments.elcd.model import field, make_ode  # noqa: E402
from experiments.elcd.record_names import PAPER_LASER  # noqa: E402
from experiments.elcd.train import data as D  # noqa: E402
from experiments.elcd.train.losses import with_equilibrium  # noqa: E402
from experiments.elcd.train.train_elcd import load_model  # noqa: E402

ROOT = "results/elcd/panel_a_models/"   # a link to the research folder of the two models (not in the repository)
MODEL = ROOT + f"chain8z_suffix-k64-anc0.25-strat-st0_scratch_{PAPER_LASER}_fp64_cos30k_s4.npz"   # best demo tracking (mean 0.12)
CKPT = ROOT + f"chain8z_suffix-k64-anc0.25-strat-st0_scratch_{PAPER_LASER}_fp64_cos30k_s4_at10000.npz"  # 10k of 30k steps: the stale warm start
START = 4951           # first sample of the Worm segment: Worm -> Sshape -> Multi_Models_1 (3 shapes, N = 2737)
TOL, MAX_ITER = 1e-9, 60
GRID = 64
# one LASER iteration against one sequential RK4 step on the paper's L40S fp64 benchmark (a run of the research
# project, not in the repository; latent coordinates, Sshape, N = 1000 and 10000): iteration ~ A + B N, sequential
# step ~ C.
BENCH_A, BENCH_B, BENCH_C = 5.8e-4, 3.5e-7, 4.65e-5


def seq_steps_per_iter(N: int) -> float:
    return (BENCH_A + BENCH_B * N) / BENCH_C


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=2.0, help="amplification of the stale warm start's deviation from x*")
    p.add_argument("--start", type=int, default=START)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out", default=os.path.join(REPO, "results", "elcd", "check", "schematic_data.npz"))
    args = p.parse_args(argv)
    t0 = time.time()
    demos = D.load_system("chain8z")
    theta = load_model(args.model)
    x_eq = jnp.asarray(demos.x_eq)
    elcd = with_equilibrium(theta, x_eq)
    flow = theta.flow
    ode = make_ode(elcd)
    N, dt = demos.N - 1 - args.start, float(demos.dt)
    x0 = jnp.asarray(demos.pos[0, args.start])
    z0 = FG.phi(flow, x0)
    dec = jax.jit(jax.vmap(lambda z: FG.psi(flow, z)))
    enc = jax.jit(jax.vmap(lambda x: FG.phi(flow, x)))
    Z_star = np.asarray(solvers.seq_scan(ode, z0, dt, N, "rk4"))
    X_star = np.asarray(dec(jnp.asarray(Z_star)))
    # stale warm start: the earlier checkpoint's rollout from the same start, in data space
    th2 = load_model(args.ckpt)
    e2 = with_equilibrium(th2, x_eq)
    Z2 = solvers.seq_scan(make_ode(e2), FG.phi(th2.flow, x0), dt, N, "rk4")
    X_src = np.asarray(jax.vmap(lambda z: FG.psi(th2.flow, z))(Z2))
    dev = np.linalg.norm(X_src - X_star, axis=1)
    print(f"start {args.start} N {N}: x* vs demo 0 mean {np.linalg.norm(X_star - demos.pos[0, args.start:], axis=1).mean():.3f}; "
          f"stale guess deviation mean {dev.mean():.3f} max {dev.max():.3f}  ({time.time() - t0:.0f}s)")

    lin = solvers.exact_step_linearizer(ode, dt, "rk4")
    terms_fn, residual_fn, scan_fn = solvers.make_triple(ode, z0, dt, N, "rk4", lin)
    cfg = LaserConfig()

    @jax.jit
    def run_k(G, m):
        X, res, k, nrej, _ = run_damped_newton(G, terms_fn, scan_fn, dt, cfg, TOL, m, residual_fn=residual_fn)
        return X, res, k, nrej

    def iterate_from(alpha):
        Xg = X_star + alpha * (X_src - X_star)
        G = enc(jnp.asarray(Xg))[1:]
        Zs, res, rej = [np.concatenate([np.asarray(z0)[None], np.asarray(G)])], [], []
        for k in range(1, MAX_ITER + 1):
            X, r, kk, nrej = run_k(G, jnp.int32(k))
            Zs.append(np.concatenate([np.asarray(z0)[None], np.asarray(X)]))
            res.append(float(r)); rej.append(int(nrej))
            if int(kk) < k or r < TOL:
                break
        Zs = np.stack(Zs)
        Xs = np.stack([np.asarray(dec(jnp.asarray(z))) for z in Zs])
        err = np.array([np.max(np.linalg.norm(Xs[k] - X_star, axis=1)) for k in range(len(Xs))])
        bad = np.linalg.norm(Zs - Z_star[None], axis=2) > 1e-6
        exact = np.array([int(np.argmax(b)) if b.any() else N + 1 for b in bad])
        return Xs, Zs, np.array(res), err, exact, rej[-1] if rej else 0

    Xs, Zs, res, err, exact, nrej = iterate_from(args.alpha)
    K = len(Xs) - 1
    print(f"alpha {args.alpha:g}: K={K} rejected={nrej}; max err per k: " + " ".join(f"{v:.2e}" for v in err) + f"; exact prefix {exact.tolist()}")

    # the data-space field on a grid over the trajectory's bounding box (plus margin)
    lo, hi = X_star.min(axis=0), X_star.max(axis=0)
    m = 0.18 * (hi - lo)
    gx, gy = np.linspace(lo[0] - m[0], hi[0] + m[0], GRID), np.linspace(lo[1] - m[1], hi[1] + m[1], GRID)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    pts = jnp.asarray(np.column_stack([GX.ravel(), GY.ravel()]))
    data_field = jax.jit(jax.vmap(lambda x: jnp.linalg.solve(FG.Q(flow, x), field(elcd, FG.phi(flow, x)))))
    UV = np.asarray(data_field(pts)).reshape(GRID, GRID, 2)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, X_star=X_star, guess_src=X_src, iterates=Xs, err=err, res=res,
             exact_prefix=exact, grid_x=gx, grid_y=gy, field_u=UV[..., 0], field_v=UV[..., 1], x_eq=demos.x_eq, start=args.start,
             N=N, dt=dt, K=K, alpha=args.alpha, seq_steps_per_iter=seq_steps_per_iter(N), model=args.model, ckpt=args.ckpt,
             segments=np.array([(a - args.start, b - args.start) for a, b, _ in demos.segments if b > args.start]),
             segment_names=np.array([s for a, b, s in demos.segments if b > args.start]))
    print(f"wrote {args.out}: N={N} K={K} seq_steps_per_iter={seq_steps_per_iter(N):.1f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
