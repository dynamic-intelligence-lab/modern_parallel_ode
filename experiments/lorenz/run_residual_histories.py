"""Residual after each iteration on the bistable Lorenz problem, with the best constants of the tuning sweep.

DEER, scale-ELK (k 0.053), ELK (sigma^2 3e4) and LASER, zero guess, tolerance 1e-9, at most 2,000 iterations. The
solvers are those of panel C of the bistable Lorenz figure (run_bistable_figure.py), unchanged.

  python experiments/lorenz/run_residual_histories.py OUT.json [N ...]     # default N: 10000 100000

ELK at N = 1e5 needs 1,838 iterations: run the default in a GPU job (slurm/residual_histories.slurm). The file
is rewritten after every N.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from experiments.lorenz import run_bistable_figure as R       # noqa: E402

DT, K, SIGMASQ, CAP = 0.01, 0.053, 3e4, 2000

import jax                                                  # noqa: E402

print("devices", jax.devices(), flush=True)


def done(N, name, v):
    """One line per finished solve, so that a stalled run shows where it stopped."""
    print(f"  N {N} {name}: {len(v) - 1} iterations, residual {float(v[-1]):.3e}", flush=True)
    return v


out = {}
for N in [int(a) for a in sys.argv[2:]] or [10000, 100000]:
    ode, ode_fn, x0, _, guess, times, _ = R.build(N, DT)
    res_laser, hist = R.laser_history(ode, x0, guess, DT)[:2]
    h = dict(laser=done(N, "laser", res_laser),
             deer=done(N, "deer", R.scaled_newton_history(ode_fn, x0, guess, DT, times, 0.0, max_iter=CAP)),
             scale_elk=done(N, "scale_elk", R.scaled_newton_history(ode_fn, x0, guess, DT, times, K, max_iter=CAP)),
             elk=done(N, "elk", R.elk_history(ode_fn, x0, guess, DT, times, SIGMASQ, max_iter=CAP)))
    out[str(N)] = {m: [float(x) for x in v] for m, v in h.items()}
    out[str(N)]["laser_rejected"] = int(hist["n_trials"].sum()) - len(res_laser) + 1
    print(N, "LASER rejected", out[str(N)]["laser_rejected"], flush=True)
    json.dump(dict(dt=DT, k=K, sigmasq=SIGMASQ, tol=R.TOL, max_iter=CAP, histories=out), open(sys.argv[1], "w"))
