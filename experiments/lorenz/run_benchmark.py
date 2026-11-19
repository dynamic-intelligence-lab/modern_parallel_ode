"""Benchmark driver of the bistable Lorenz experiment: DEER, scale-ELK, ELK and LASER on one problem.

For one configuration (N, dt, dtype, rho) of the Lorenz system, runs the methods below on the explicit Euler
DEER solver, records iterations, rejected trials, wall-clock (median of `--repeats` runs after one
compile run), residual, error against the sequential solution, and for LASER the per-iteration
lambda_h / gamma / step-norm history. One npz per run under --outdir (default results/lorenz/check/benchmark),
one JSON per configuration, plus a summary CSV to which every run appends one row per method.

Methods
  deer        plain explicit_par_integrate (stops on the update norm, parode default)
  scale-elk:K constant scale-ELK with k = K (J -> (1-K) J), no acceptance test, stopping on the residual
  elk:S       ELK (Gonzalez et al. 2024) with a fixed sigma^2 = S, parode.elk_baseline.elk_integrate, stopping
              on the residual
  laser       LASER: explicit_par_integrate with the default LaserConfig() (propose_first="lle",
              accept_ref="best", step_cap=2.0, floor_switch_off=False). The damping is proposed before the first
              trial, growth is measured against the best accepted residual with a per-step cap of 2, and the
              hysteresis floor persists

The Lorenz system and rho
  The system is Lorenz from x0 = (1, 1, 1), solved from the zero guess. --rho sets rho of the system; sigma 10
  and beta 8/3 stay fixed, and the default rho 10 is the bistable system of the paper figure. The tag gains
  _rho<value> only when rho != 10. When --rho is given (even at its default value) the rows carry four more
  columns: rho, lorenz_ic (always "paper", the name of x0 = (1, 1, 1)), x0 and truth_end (the two state vectors
  as space-separated floats). The paper's commands pass --rho 10 on every line, and the summaries that ship with
  the repository have these columns; summary_path sends rows with other columns to summary_lorenz.csv. The
  columns system, D, guess and seed and the file names carry the constants lorenz, 3, zeros and 0, as in the
  shipped records.

Run:  JAX_PLATFORMS=cpu python experiments/lorenz/run_benchmark.py --rho 10
"""
import argparse
import csv
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from parode.explicit_parallel import (  # noqa: E402
    affine_terms, euler_step, explicit_par_integrate, linear_terms, par_init, par_scan,
)
from parode.laser import LaserConfig  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402


# ----------------------------------------------------------------------------- systems

LORENZ_SIGMA, LORENZ_BETA = 10.0, 8.0 / 3.0
LORENZ_RHO = 10.0                                  # default rho: the bistable system of the paper figure


def lorenz_hopf_rho(sigma=LORENZ_SIGMA, beta=LORENZ_BETA):
    """rho at which C+- lose stability: sigma (sigma + beta + 3) / (sigma - beta - 1), 24.74 at sigma 10, beta 8/3."""
    return sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)


def lorenz_focus(rho, s, beta=LORENZ_BETA):
    """Equilibrium C = (s sqrt(beta (rho - 1)), s sqrt(beta (rho - 1)), rho - 1) with s = +1 or -1 (needs rho > 1).

    The components are computed from rho and beta in closed form, so the vector field vanishes at C up to the
    round-off of sqrt and of rho - 1."""
    if not rho > 1.0:
        raise ValueError(f"the foci C+- exist only for rho > 1, got rho {rho}")
    if s not in (1, -1):
        raise ValueError(f"s must be +1 or -1, got {s}")
    a = float(np.sqrt(beta * (rho - 1.0)))
    return jnp.array([s * a, s * a, rho - 1.0])


def make_system(rho=LORENZ_RHO):
    """Return (ode_fn, x0, label) of the Lorenz system with the given rho (that of --rho), from x0 = (1, 1, 1). The
    ODE takes (t, x, u)."""
    # Lorenz with sigma 10, beta 8/3 and rho from --rho. For 1 < rho < 24.74 (lorenz_hopf_rho) the origin is a
    # saddle with unstable rate (-(sigma + 1) + sqrt((sigma + 1)^2 + 4 sigma (rho - 1))) / 2 (5.5 at rho 10)
    # and the fixed points C+- = (+-sqrt(beta (rho - 1)), +-sqrt(beta (rho - 1)), rho - 1) are stable; at the
    # default rho 10 every trajectory off the stable manifold of the origin converges to one of them.
    from experiments.lorenz.lorenz_field import lorenz_dynamics
    sigma, beta_l = LORENZ_SIGMA, LORENZ_BETA
    rho = float(rho)
    x0 = jnp.array([1.0, 1.0, 1.0])

    def ode(t, x, u=None):
        return lorenz_dynamics(t, x, u, sigma=sigma, rho=rho, beta=beta_l)
    if rho == LORENZ_RHO:
        label = "Lorenz sigma 10, rho 10, beta 8/3 (pre-chaotic: origin saddle, C+- attract)"
    else:
        regime = ("origin saddle, C+- stable" if 1.0 < rho < lorenz_hopf_rho(sigma, beta_l)
                  else ("no foci" if rho <= 1.0 else "C+- not stable"))
        label = f"Lorenz sigma 10, rho {rho:.15g}, beta 8/3 ({regime})"
    return ode, x0, label


def sequential_truth(ode, x0, dt, N):
    """Sequential explicit Euler solution (N + 1, D)."""
    return seq_integrate(ode, x0, dt, N, "euler")


# ----------------------------------------------------------------------------- baselines

def scale_elk_step(X, x0, ode_fn, dt, times, k):
    """One scale-ELK iteration with the constant k (J -> (1-k) J; k = 0 is a DEER step): the new iterate and its
    residual max|r|. constant_scale_elk loops over it; run_bistable_figure.scaled_newton_history records it."""
    c = affine_terms(X, x0, ode_fn, dt, euler_step, times, None)
    F = (1.0 - k) * linear_terms(X, ode_fn, dt, euler_step, times, None)
    _, cs = par_scan(par_init(F, c[1:], c[0]))
    X_new = X + jnp.vstack((c[0], cs))
    c_new = affine_terms(X_new, x0, ode_fn, dt, euler_step, times, None)
    return X_new, jnp.max(jnp.abs(c_new))


def constant_scale_elk(ode, x0, guess, dt, k, tol, max_iter):
    """Scale-ELK with a fixed k (J -> (1-k) J), stopping on the residual. No acceptance test."""
    ode_fn = prepare_ode_function(ode)
    N = guess.shape[0]
    times = dt * jnp.arange(N + 1)

    def body(val):
        X, _, it = val
        X_new, res = scale_elk_step(X, x0, ode_fn, dt, times, k)
        return X_new, res, it + 1

    def cond(val):
        _, res, it = val
        return (res > tol) & (it < max_iter)

    X, res, it = jax.lax.while_loop(cond, body, (guess, jnp.array(jnp.inf, guess.dtype), 0))
    return jnp.vstack((x0, X)), res, it


# ----------------------------------------------------------------------------- driver

def run_method(method, ode, dt, tol, max_iter):
    """Return (timed_fn, diag_fn): jitted solve(x0, guess) for timing (no history, so the exponent
    is only computed where the algorithm needs it) and an optional history-returning twin for
    the rejection count and per-iteration traces (run once, untimed)."""
    if method == "deer":
        f = lambda x0, guess: explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=tol)  # noqa: E731
        return jax.jit(f), None
    if method == "laser":
        # LASER with the defaults: the damping is proposed before the first trial, growth is measured against the
        # best accepted residual (total excursion at most growth_cap) with a per-step cap of 2, and the hysteresis
        # floor persists (LM-style memory of the last working damping).
        cfg = LaserConfig()
        cfg_h = LaserConfig(return_history=True)
        f = lambda x0, guess: explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=tol, laser=cfg)  # noqa: E731
        g = lambda x0, guess: explicit_par_integrate(ode, x0, guess, dt, max_iter, "euler", tol=tol, laser=cfg_h)  # noqa: E731
        return jax.jit(f), jax.jit(g)
    if method.startswith("scale-elk:"):
        # Constant scale-ELK with the dense Jacobian in the scan.
        k = float(method.split(":")[1])
        f = lambda x0, guess: constant_scale_elk(ode, x0, guess, dt, k, tol, max_iter)  # noqa: E731
        return jax.jit(f), None
    if method.startswith("elk:"):
        # ELK (Gonzalez et al. 2024) with a fixed sigma^2 = 1/lambda_LM and the dense filter.
        from parode.elk_baseline import elk_integrate
        sigmasq = float(method.split(":")[1])
        f = lambda x0, guess: elk_integrate(ode, x0, guess, dt, sigmasq, tol, max_iter)  # noqa: E731
        return jax.jit(f), None
    raise ValueError(method)


def timed(fn, args, repeats):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))                # compile + first run
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times)), compile_s


def _vec_str(v):
    """A state vector as space-separated shortest round-trip floats, for one CSV cell."""
    return " ".join(repr(float(a)) for a in np.asarray(v).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--rho", type=float, default=None,
                    help="rho, default 10 (bistable), with sigma 10 and beta 8/3 fixed; C+- lose stability at "
                         "24.74. Given at all, it adds the Lorenz columns to the rows")
    ap.add_argument("--methods", default="deer,scale-elk:0.05,scale-elk:0.053,elk:10000,elk:30000,laser")
    ap.add_argument("--tol", type=float, default=None, help="default 1e-9 fp64 / 1e-5 fp32")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "lorenz", "check", "benchmark"))
    args = ap.parse_args()
    methods = args.methods.split(",")
    lorenz_options = args.rho is not None                           # given on the command line, at any value
    rho = LORENZ_RHO if args.rho is None else args.rho

    jax.config.update("jax_enable_x64", args.dtype == "fp64")
    tol = args.tol if args.tol is not None else (1e-9 if args.dtype == "fp64" else 1e-5)
    ode, x0, label = make_system(rho=rho)
    x0 = x0.astype(jnp.float64 if args.dtype == "fp64" else jnp.float32)
    truth = sequential_truth(ode, x0, args.dt, args.N)
    guess = jnp.zeros((args.N, 3), x0.dtype)          # the zero guess: the constant trajectory at the saddle
    # The largest |x| of the sequential solution, with and without x0, recorded in every row.
    truth_max = float(np.max(np.abs(np.asarray(truth))))
    truth_tail_max = float(np.max(np.abs(np.asarray(truth)[1:])))
    lorenz_cols = (dict(rho=rho, lorenz_ic="paper", x0=_vec_str(x0), truth_end=_vec_str(truth[-1]))
                   if lorenz_options else {})

    os.makedirs(args.outdir, exist_ok=True)
    # system lorenz, D 3, guess zeros and seed 0 in the tag and the rows: the values of the shipped records
    tag = f"lorenz_D3_N{args.N}_dt{args.dt}_{args.dtype}_zeros_s0"
    tag += "" if rho == LORENZ_RHO else f"_rho{rho:.15g}"          # rho 10: the file names of the shipped records
    dev = jax.devices()[0].device_kind
    print(f"{label}: D=3 N={args.N} dt={args.dt} {args.dtype} guess=zeros tol={tol:.0e} device={dev}")
    print(f"{'method':20s} {'iters':>5s} {'rej':>4s} {'wall ms':>9s} {'ms/it':>7s} {'compile s':>9s} {'residual':>9s} {'err vs seq':>10s} {'max|X|':>8s}  status")

    rows = []
    for method in methods:
        fn, diag_fn = run_method(method, ode, args.dt, tol, args.max_iter)
        try:
            out, wall, compile_s = timed(fn, (x0, guess), args.repeats)
            hist = jax.block_until_ready(diag_fn(x0, guess))[3] if diag_fn is not None else None
        except Exception as exc:                          # noqa: BLE001
            print(f"{method:20s} failed: {exc}")
            continue
        sol, res, it = out[0], float(out[1]), int(out[2])
        sol_np = np.asarray(sol)
        finite = bool(np.all(np.isfinite(sol_np))) and np.isfinite(res)
        err = float(np.max(np.abs(sol_np - np.asarray(truth)))) if finite else float("nan")
        rej = int(np.sum(np.asarray(hist["n_trials"])[:it] - 1)) if hist is not None else 0
        maxx = float(np.max(np.abs(sol_np))) if finite else float("nan")
        converged = finite and res < tol
        status = "ok" if converged else ("NaN" if not finite else "max_iter")
        print(f"{method:20s} {it:5d} {rej:4d} {wall * 1e3:9.1f} {wall * 1e3 / max(it, 1):7.2f} {compile_s:9.1f} {res:9.1e} {err:10.1e} {maxx:8.1e}  {status}")
        row = dict(system="lorenz", D=3, N=args.N, dt=args.dt, dtype=args.dtype, guess="zeros",
                   seed=0, method=method, iters=it, rejected=rej, wall_s=wall, compile_s=compile_s,
                   residual=res, err=err, max_abs=maxx, status=status, device=dev,
                   truth_max=truth_max, truth_tail_max=truth_tail_max, x0_max=float(np.max(np.abs(np.asarray(x0)))))
        row.update(lorenz_cols)
        rows.append(row)
        np.savez(os.path.join(args.outdir, f"{tag}_{method.replace(':', '')}.npz"),
                 **{k: np.asarray(v) for k, v in row.items()},
                 **({f"hist_{k}": np.asarray(v)[:it] for k, v in hist.items()} if hist is not None else {}))

    if not rows:                                          # every method failed (e.g. out of GPU memory)
        print("no method completed; nothing written")
        return
    with open(os.path.join(args.outdir, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    summary = summary_path(args.outdir, list(rows[0].keys()))
    write_header = not os.path.exists(summary) or os.path.getsize(summary) == 0
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def summary_path(outdir, fieldnames):
    """summary.csv, unless it exists with other columns, in which case summary_lorenz.csv.

    Appending rows with other columns (a run without --rho, say) under an existing header would misalign the
    file, so such rows go to summary_lorenz.csv; a clash there too is an error."""
    for name in ("summary.csv", "summary_lorenz.csv"):
        path = os.path.join(outdir, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return path
        with open(path, newline="") as f:
            if next(csv.reader(f), None) == fieldnames:
                return path
        print(f"{path} has other columns; trying the next summary file")
    raise RuntimeError(f"no summary file in {outdir} matches the columns {fieldnames}")


if __name__ == "__main__":
    main()
