"""Correctness test: multi-GPU Jacobi must match single-GPU Jacobi.

Runs a small fixed-size sweep on both implementations and checks that the
error history and the final state agree to fp64 round-off.

On the CPU the module gives JAX 4 host devices (XLA_FLAGS below); on a GPU
machine it uses the GPUs. A case that needs more devices than there are is
skipped with its reason.

    JAX_PLATFORMS=cpu python -m pytest -q tests/test_gpu_scaling_multi_gpu.py
"""

from __future__ import annotations

import os
import sys

# Force JAX to see multiple "devices" for shard_map testing on a single-device box.
# Must be set before JAX initialises its backends; appended so that other XLA flags are kept.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

import jax

jax.config.update("jax_enable_x64", True)

import jax.random as jr
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.appendix.gpu_scaling.contractive_rnn import (  # noqa: E402
    gen_drivers,
    init_contractive_rnn,
    make_step_fn,
)
from experiments.appendix.gpu_scaling.jacobi_scan import (  # noqa: E402
    jacobi_scan,
    sequential_rollout,
)
from experiments.appendix.gpu_scaling.jacobi_multi_gpu import (  # noqa: E402
    make_jacobi_scan_multigpu,
    make_mesh,
)
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


def run_comparison(T: int, D: int, rho: float, max_iters: int,
                   n_devices: int, seed: int = 0) -> dict:
    """Run both implementations on identical inputs, return comparison metrics."""
    assert T % n_devices == 0, f"T={T} must be divisible by n_devices={n_devices}"

    key = jr.PRNGKey(seed)
    k_w, k_x0, k_drv, k_guess = jr.split(key, 4)

    W = init_contractive_rnn(D, rho, k_w)
    x0 = jr.normal(k_x0, (D,))
    drivers = gen_drivers(T, D, k_drv, scale=1.0)
    initial_guess = jr.normal(k_guess, (T, D))

    f = make_step_fn(W)
    x_true = sequential_rollout(f, x0, drivers)

    # ---- Single-device (reference) ----
    final_sg, err_sg = jacobi_scan(f, x0, initial_guess, drivers, x_true, max_iters)
    final_sg = jax.block_until_ready(final_sg)

    # ---- Multi-device ----
    mesh = make_mesh(n_devices)
    sharding_t = NamedSharding(mesh, P("t", None))
    sharding_rep = NamedSharding(mesh, P())

    initial_guess_sh = jax.device_put(initial_guess, sharding_t)
    drivers_sh = jax.device_put(drivers, sharding_t)
    x_true_sh = jax.device_put(x_true, sharding_t)
    x0_rep = jax.device_put(x0, sharding_rep)

    run_mg = make_jacobi_scan_multigpu(f, mesh, n_devices, max_iters)
    final_mg, err_mg = run_mg(x0_rep, initial_guess_sh, drivers_sh, x_true_sh)
    final_mg = jax.block_until_ready(final_mg)

    # Gather for comparison.
    final_mg_np = np.asarray(final_mg)
    final_sg_np = np.asarray(final_sg)
    err_mg_np = np.asarray(err_mg)
    err_sg_np = np.asarray(err_sg)

    return {
        "state_max_abs_diff": float(np.max(np.abs(final_sg_np - final_mg_np))),
        "err_history_max_abs_diff": float(np.max(np.abs(err_sg_np - err_mg_np))),
        "err_sg_final": float(err_sg_np[-1, 0]),
        "err_mg_final": float(err_mg_np[-1, 0]),
        "resid_sg_final": float(err_sg_np[-1, 1]),
        "resid_mg_final": float(err_mg_np[-1, 1]),
        "final_sg_norm": float(np.linalg.norm(final_sg_np)),
        "final_mg_norm": float(np.linalg.norm(final_mg_np)),
    }


CONFIGS = [
    dict(T=64,   D=16, rho=0.6, max_iters=30, n_devices=2),
    dict(T=128,  D=32, rho=0.6, max_iters=40, n_devices=4),
    dict(T=512,  D=64, rho=0.6, max_iters=50, n_devices=4),
    dict(T=1024, D=64, rho=0.6, max_iters=60, n_devices=4),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"T{c['T']}-D{c['D']}-n{c['n_devices']}")
def test_multi_gpu_matches_single_gpu(cfg):
    if jax.device_count() < cfg["n_devices"]:
        pytest.skip(f"needs {cfg['n_devices']} devices, has {jax.device_count()}")
    m = run_comparison(**cfg)
    assert m["state_max_abs_diff"] < 1e-10 and m["err_history_max_abs_diff"] < 1e-10, m

