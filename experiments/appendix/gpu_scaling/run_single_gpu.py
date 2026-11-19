"""Run the Jacobi / Quasi-DEER / Full-DEER comparison sweep on a single GPU.

Saves ``<outdir>/gpu_scaling<suffix>.npz`` with the columns of ``SolverResult``; the default
``--outdir`` is ``results/gpu_scaling/check``, so the shipped files of ``results/gpu_scaling``
stay as they are.  OOM points mark the end of that algorithm's T-sweep on this GPU.

Multi-seed mode
---------------
Pass ``--seeds 0,1,2,3,4`` to run each (algo, T) at multiple seeds.  By
default seed 0 runs the full T-sweep and seeds 1+ run only the T values
listed in ``--multi-seed-Ts`` (default empty = same T-sweep for every seed).
With ``--append`` the runner loads any existing output file and skips
``(algorithm, T, seed)`` triples already present.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable, List, Sequence, Set, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from experiments.appendix.gpu_scaling.solvers import (  # noqa: E402
    DEER_RTOL_FP64,
    DEER_TOL_FP64,
    DEFAULT_MAX_ITER,
    SolverResult,
    results_to_dict,
    run_full_deer,
    run_jacobi,
    run_jacobi_scan,
    run_quasi_deer,
    summarise,
)


def _banner() -> None:
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
        print(f"nvidia-smi:   {smi}")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("=" * 72)
    print()


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_existing(path: str) -> Tuple[List[SolverResult], Set[Tuple[str, int, int]]]:
    """Load existing results from ``path`` for append mode.

    Returns the materialised SolverResult list (so we can re-emit them in the
    same npz) and a set of ``(algorithm, T, seed)`` keys already present.
    """
    if not os.path.isfile(path):
        return [], set()
    arr = np.load(path, allow_pickle=True)
    keys = ["algorithm", "T", "D", "rho", "tol", "rtol", "iters_to_tol",
            "wall_ms", "compile_ms", "seq_rollout_ms", "final_resid_blk",
            "final_update_linf", "error_vs_truth", "peak_mem_mb", "oom",
            "seed"]
    cols = {k: arr[k] for k in keys if k in arr.files}
    n = len(cols["T"])
    out: List[SolverResult] = []
    seen: Set[Tuple[str, int, int]] = set()
    for i in range(n):
        r = SolverResult(
            algorithm=str(cols["algorithm"][i]),
            T=int(cols["T"][i]),
            D=int(cols["D"][i]),
            rho=float(cols["rho"][i]),
            tol=float(cols["tol"][i]),
            rtol=float(cols["rtol"][i]),
            iters_to_tol=int(cols["iters_to_tol"][i]),
            wall_ms=float(cols["wall_ms"][i]),
            compile_ms=float(cols["compile_ms"][i]),
            seq_rollout_ms=float(cols["seq_rollout_ms"][i]),
            final_resid_blk=float(cols["final_resid_blk"][i]),
            final_update_linf=float(cols["final_update_linf"][i]),
            error_vs_truth=float(cols["error_vs_truth"][i]),
            peak_mem_mb=float(cols["peak_mem_mb"][i]),
            oom=bool(cols["oom"][i]),
            seed=int(cols["seed"][i]),
        )
        out.append(r)
        seen.add((r.algorithm, r.T, r.seed))
    return out, seen


def _run_sweep(
    runner: Callable[..., SolverResult],
    algorithm: str,
    Ts: Iterable[int],
    seeds: Sequence[int],
    leader_seed: int,
    multi_seed_Ts: Sequence[int],
    skip_keys: Set[Tuple[str, int, int]],
    D: int, rho: float,
    tol: float, rtol: float,
    n_repeats: int, max_iter: int,
) -> List[SolverResult]:
    """Run one algorithm over (seed, T) pairs; stop a seed-sweep on OOM.

    For each seed in ``seeds``: run T values from ``Ts`` if seed == leader_seed
    or if multi_seed_Ts is empty; otherwise run only T in ``multi_seed_Ts``.
    """
    print(f"\n--- {algorithm} ---")
    results: List[SolverResult] = []
    msTs = set(multi_seed_Ts)
    for seed in seeds:
        Ts_for_seed = list(Ts) if (seed == leader_seed or not msTs) \
            else [T for T in Ts if T in msTs]
        if not Ts_for_seed:
            continue
        print(f"  -- seed={seed} --  Ts={Ts_for_seed}")
        for T in Ts_for_seed:
            key = (algorithm, T, seed)
            if key in skip_keys:
                print(f"  T={T:>8d}  seed={seed}  SKIP (already in output)")
                continue
            t0 = time.time()
            r = runner(
                T=T, D=D, rho=rho, tol=tol, rtol=rtol,
                seed=seed, n_repeats=n_repeats, max_iter=max_iter,
            )
            results.append(r)
            speedup = r.seq_rollout_ms / r.wall_ms if not r.oom and r.wall_ms else float("nan")
            tag = "OOM" if r.oom else f"iters={r.iters_to_tol:>4d}"
            print(
                f"  T={T:>8d}  seed={seed}  {tag}  wall={r.wall_ms:>9.2f}ms  "
                f"seq={r.seq_rollout_ms:>9.2f}ms  speedup={speedup:>6.2f}x  "
                f"resid={r.final_resid_blk:>9.2e}  err={r.error_vs_truth:>9.2e}  "
                f"mem={r.peak_mem_mb:>8.1f}MB  (elapsed {time.time() - t0:.1f}s)",
                flush=True,
            )
            if r.oom:
                print(f"  (OOM at T={T} seed={seed}; stopping {algorithm} sweep for this seed)")
                break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for output npz (e.g. '_b200').")
    parser.add_argument("--max-T", type=int, default=10_000_000)
    parser.add_argument("--algorithms", default="jacobi,quasi_deer,full_deer",
                        help="Comma-separated list to run.")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated seeds (default '0').")
    parser.add_argument("--multi-seed-Ts", default="",
                        help="Comma-separated T values to run at non-leader "
                             "seeds. Empty = full T-sweep for every seed.")
    parser.add_argument("--append", action="store_true",
                        help="Load existing output, dedupe (algo,T,seed), and "
                             "append new rows.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help="Cap on the iterations of jacobi, quasi_deer and full_deer; "
                             "jacobi_scan runs min(cap, 120) sweeps.")
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results", "gpu_scaling", "check"),
                        help="Folder of the output file (the shipped files are in results/gpu_scaling).")
    args = parser.parse_args()
    chosen = {a.strip() for a in args.algorithms.split(",") if a.strip()}
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer.")
    leader_seed = seeds[0]
    multi_seed_Ts = _parse_int_list(args.multi_seed_Ts) if args.multi_seed_Ts else []

    _banner()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"gpu_scaling{args.output_suffix}.npz")
    print(f"Saving to: {path}")
    print(f"Algorithms: {sorted(chosen)}")
    print(f"D={args.D}  rho={args.rho}  tol={DEER_TOL_FP64}  rtol={DEER_RTOL_FP64}")
    print(f"max_T={args.max_T}  max_iter={args.max_iter}")
    print(f"seeds={seeds}  leader_seed={leader_seed}  multi_seed_Ts={multi_seed_Ts}")
    print(f"append={args.append}")

    candidates = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000,
                  3_000_000, 10_000_000]
    Ts = [T for T in candidates if T <= args.max_T]

    existing: List[SolverResult] = []
    skip_keys: Set[Tuple[str, int, int]] = set()
    if args.append:
        existing, skip_keys = _load_existing(path)
        print(f"Loaded {len(existing)} existing rows; will skip "
              f"{len(skip_keys)} (algo,T,seed) triples already present.")

    new_results: List[SolverResult] = []
    t_total = time.time()

    if "jacobi" in chosen:
        new_results += _run_sweep(
            run_jacobi, "jacobi", Ts, seeds, leader_seed, multi_seed_Ts,
            skip_keys, args.D, args.rho,
            DEER_TOL_FP64, DEER_RTOL_FP64,
            args.n_repeats, args.max_iter,
        )
    if "jacobi_scan" in chosen:
        # max_iter for scan is capped at 120 to match multi-GPU sweep schema.
        new_results += _run_sweep(
            run_jacobi_scan, "jacobi_scan", Ts, seeds, leader_seed, multi_seed_Ts,
            skip_keys, args.D, args.rho,
            DEER_TOL_FP64, DEER_RTOL_FP64,
            args.n_repeats, min(args.max_iter, 120),
        )
    if "quasi_deer" in chosen:
        new_results += _run_sweep(
            run_quasi_deer, "quasi_deer", Ts, seeds, leader_seed, multi_seed_Ts,
            skip_keys, args.D, args.rho,
            DEER_TOL_FP64, DEER_RTOL_FP64,
            args.n_repeats, args.max_iter,
        )
    if "full_deer" in chosen:
        new_results += _run_sweep(
            run_full_deer, "full_deer", Ts, seeds, leader_seed, multi_seed_Ts,
            skip_keys, args.D, args.rho,
            DEER_TOL_FP64, DEER_RTOL_FP64,
            args.n_repeats, args.max_iter,
        )

    print(f"\nTOTAL: {time.time() - t_total:.1f}s "
          f"({len(new_results)} new rows)")
    print()
    print(summarise(new_results))

    all_results = existing + new_results
    np.savez_compressed(path, **results_to_dict(all_results))
    print(f"\nWrote {path}  ({len(all_results)} total rows)")


if __name__ == "__main__":
    main()
