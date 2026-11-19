"""Multi-GPU Jacobi T-sweep.

Runs the fixed-length Jacobi scan with the T axis sharded across
``--n-devices`` GPUs on a single node, for every T of ``--Ts`` and every seed
of ``--seeds``, and writes one row per (T, seed) with a ``seed`` column to
``<outdir>/gpu_scaling_mg<suffix>.npz`` (default ``--outdir``: ``results/gpu_scaling/check``,
so the shipped files of ``results/gpu_scaling`` stay as they are).  Each T is rounded up
to the nearest multiple of ``n_devices``.

Sequential rollout is *not* timed inside this runner.  For the speedups, use
the single-GPU ``seq_rollout_ms`` from ``gpu_scaling_<gpu>.npz`` at the same T
and seed as the baseline (sequential rollout is inherently single-device).
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
import time
from typing import List, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from experiments.appendix.gpu_scaling.jacobi_multi_gpu import (  # noqa: E402
    JacobiResultMG,
    run_single_multigpu,
)


def _print_banner() -> None:
    print("=" * 72)
    print(f"Host:         {socket.gethostname()}")
    print(f"Python:       {platform.python_version()}")
    print(f"JAX devices:  {jax.devices()}")
    print(f"Default:      {jax.default_backend()}")
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
        print(f"nvidia-smi:\n  " + smi.replace("\n", "\n  "))
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("=" * 72)
    print()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _round_up_Ts(Ts: Sequence[int], n_devices: int) -> List[int]:
    out: List[int] = []
    for T in Ts:
        T_adj = ((T + n_devices - 1) // n_devices) * n_devices
        if not out or out[-1] != T_adj:
            out.append(T_adj)
    return out


# Schema for serialised multi-seed multi-GPU sweeps.
_MG_SCALAR_KEYS = (
    "T", "D", "rho", "tol", "max_iters", "n_devices", "seed",
    "iters_to_conv", "wall_clock_ms", "compile_time_ms",
    "initial_err", "initial_resid", "final_err", "predicted_k",
    "wall_min_ms",
)


def _result_to_record(r: JacobiResultMG, seed: int) -> dict:
    rec = {k: getattr(r, k) for k in _MG_SCALAR_KEYS if k != "seed"}
    rec["seed"] = seed
    rec["err_history"] = r.err_history
    return rec


def _records_to_dict(records: List[dict]) -> dict:
    out = {k: np.array([r[k] for r in records]) for k in _MG_SCALAR_KEYS}
    out["err_history"] = np.stack([r["err_history"] for r in records])
    return out


def run_T_seed_sweep(
    n_devices: int,
    Ts: Sequence[int],
    seeds: Sequence[int],
    D: int = 64,
    rho: float = 0.6,
    tol: float = 1e-7,
    max_iters: int = 120,
) -> List[dict]:
    print("=" * 72)
    print(f"MULTI-GPU T-SWEEP  (n_devices={n_devices}, D={D}, rho={rho}, "
          f"tol={tol}, mode=fixed scan)")
    print(f"Ts={list(Ts)}  seeds={list(seeds)}")
    print("=" * 72)

    records: List[dict] = []
    t_start = time.time()
    for seed in seeds:
        print(f"\n-- seed={seed} --")
        for T in Ts:
            r = run_single_multigpu(
                T=T, D=D, rho=rho, tol=tol, max_iters=max_iters,
                n_devices=n_devices, seed=seed,
            )
            records.append(_result_to_record(r, seed))
            print(
                f"  T={T:>8d}  seed={seed}  n_devices={n_devices}  "
                f"iters={r.iters_to_conv:>3d}  pred_k={r.predicted_k:>3d}  "
                f"init_err={r.initial_err:.3e}  init_resid={r.initial_resid:.3e}  "
                f"final={r.final_err:.2e}  wall={r.wall_clock_ms:>8.2f}ms  "
                f"min={r.wall_min_ms:>8.2f}ms  "
                f"compile={r.compile_time_ms:>8.1f}ms",
                flush=True,
            )
    print(f"\n  (sweep took {time.time() - t_start:.1f}s, "
          f"{len(records)} new rows)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-devices", type=int, required=True,
        help="Number of GPUs to shard across.",
    )
    parser.add_argument(
        "--Ts", required=True,
        help="Comma-separated T values.",
    )
    parser.add_argument(
        "--seeds", default="0",
        help="Comma-separated seeds (default '0').",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix appended to output file names (e.g. '_b200_8x').",
    )
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results", "gpu_scaling", "check"),
                        help="Folder of the output file (the shipped files are in results/gpu_scaling).")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer.")

    _print_banner()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"gpu_scaling_mg{args.output_suffix}.npz")

    print(f"Saving to: {path}")
    print(f"n_devices = {args.n_devices}  seeds={seeds}")

    Ts = _round_up_Ts(_parse_int_list(args.Ts), args.n_devices)
    print(f"Ts (rounded up to multiple of {args.n_devices}) = {Ts}")

    t_total = time.time()
    new_records = run_T_seed_sweep(
        n_devices=args.n_devices,
        Ts=Ts,
        seeds=seeds,
    )

    np.savez_compressed(path, **_records_to_dict(new_records))
    print(f"\nTOTAL: {time.time() - t_total:.1f}s — wrote {path} "
          f"({len(new_records)} rows)")


if __name__ == "__main__":
    main()
