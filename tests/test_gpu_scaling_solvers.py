"""The four solvers of the GPU-scaling experiment reach the sequential solution at small N, in the iteration counts
of the paper's GPU runs.

Each runner of experiments/appendix/gpu_scaling/solvers.py solves the contractive RNN of the paper (D = 64, rho = 0.6,
fp64, seed 0) at N = 1000 on the CPU, with the settings of run_single_gpu.py (cap of 2000 iterations; 120 for the Jacobi
scan). The final iterate must agree with the sequential rollout, and the iteration count must equal the one stored
for the same (solver, N, seed) in every shipped single-GPU result file (six GPU types).
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.appendix.gpu_scaling import solvers  # noqa: E402

N, D, RHO, SEED = 1000, 64, 0.6, 0
GPUS = ["a40", "a6000", "l40s", "rtx_pro_blackwell", "h100", "b200"]
# (runner, cap, bound on max_n ||x_n - x*_n||_2). Jacobi and quasi-DEER stop on the mixed tolerance (1e-7 absolute,
# 1e-4 relative); the Jacobi scan runs 120 iterations and Full-DEER converges quadratically.
CASES = {
    "jacobi": (solvers.run_jacobi, 2000, 1e-5),
    "jacobi_scan": (solvers.run_jacobi_scan, 120, 1e-12),
    "quasi_deer": (solvers.run_quasi_deer, 2000, 1e-5),
    "full_deer": (solvers.run_full_deer, 2000, 1e-12),
}


def stored_iters(algorithm):
    out = {}
    for g in GPUS:
        d = np.load(os.path.join(ROOT, "results", "gpu_scaling", f"gpu_scaling_{g}.npz"), allow_pickle=True)
        sel = (d["algorithm"] == algorithm) & (d["T"] == N) & (d["seed"] == SEED)
        assert sel.sum() == 1, (g, algorithm)
        out[g] = int(d["iters_to_tol"][sel][0])
    return out


@pytest.mark.parametrize("algorithm", list(CASES))
def test_solver_matches_sequential_and_gpu_count(algorithm):
    runner, cap, bound = CASES[algorithm]
    r = runner(T=N, D=D, rho=RHO, seed=SEED, n_repeats=1, max_iter=cap)
    assert not r.oom
    assert r.error_vs_truth < bound, r
    assert stored_iters(algorithm) == {g: r.iters_to_tol for g in GPUS}, r.iters_to_tol
