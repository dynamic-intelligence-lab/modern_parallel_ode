"""Shared harness comparing Jacobi, Quasi-DEER, and Full-DEER on the
contractive RNN: four runners, each returning one ``SolverResult`` row.

The Jacobi while loop (``run_jacobi``), quasi-DEER and Full-DEER terminate
when DEER's native mixed absolute/relative tolerance criterion is met:

    max_{t, i} (|yt_next - yt|_{t,i} - rtol * |yt|_{t,i}) < tol

with ``tol = 1e-7`` and ``rtol = 1e-4`` in fp64.  Jacobi's update equals the
negative residual, so this criterion applied to Jacobi is a relative-tolerance
version of the per-element residual.

The Jacobi scan (``run_jacobi_scan``) always runs ``max_iter = 120`` sweeps
and counts the first sweep whose error to the sequential solution,
max_t ||s_t - x*_t||_2, is below ``tol``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.appendix.gpu_scaling.contractive_rnn import (
    gen_drivers,
    init_contractive_rnn,
    make_step_fn,
)
from experiments.appendix.gpu_scaling.jacobi_scan import jacobi_scan, sequential_rollout
from parode.deer_map import deer_map_solve, deer_map_solve_full


# ---------------------------------------------------------------------------
# Shared DEER-style tolerance constants (the values parode/deer_map.py uses in fp64)
# ---------------------------------------------------------------------------

DEER_TOL_FP64 = 1e-7
DEER_RTOL_FP64 = 1e-4
DEFAULT_MAX_ITER = 2000   # the cap of the paper's runs (run_single_gpu.py --max-iter); counts stay below 110


# ---------------------------------------------------------------------------
# Jacobi with DEER-style stopping criterion
# ---------------------------------------------------------------------------


def jacobi_while_deer_criterion(
    f: Callable,
    x0: jnp.ndarray,
    initial_guess: jnp.ndarray,
    drivers: jnp.ndarray,
    tol: float = DEER_TOL_FP64,
    rtol: float = DEER_RTOL_FP64,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run Jacobi iteration until the DEER stopping criterion is met.

    Termination (the formula of ``parode.deer_map.deer_map_solve``):

        err = max_{t,i} (|yt_next - yt| - rtol * |yt|) < tol

    For Jacobi ``yt_next = f(y_prev)`` and ``yt_next - yt = -residual``, so
    this equals the per-element residual's relative-tolerance criterion.

    Returns
    -------
    final_states : (T, D)
    n_iters : int (device scalar)
    """

    def step(states: jnp.ndarray) -> jnp.ndarray:
        prev = jnp.concatenate([x0[None], states[:-1]], axis=0)
        return jax.vmap(f)(prev, drivers)

    def body(state):
        _err, yt, iiter = state
        yt_next = step(yt)
        new_err = jnp.max(jnp.abs(yt_next - yt) - rtol * jnp.abs(yt))
        return new_err, yt_next, iiter + 1

    def cond(state):
        err, _, iiter = state
        return jnp.logical_and(err > tol, iiter < max_iter)

    err0 = jnp.asarray(1e10, dtype=initial_guess.dtype)
    iiter0 = jnp.asarray(0, dtype=jnp.int32)

    final_err, final_states, n_iters = jax.lax.while_loop(
        cond, body, (err0, initial_guess, iiter0)
    )
    return final_states, n_iters


# ---------------------------------------------------------------------------
# Common audit metrics
# ---------------------------------------------------------------------------


def compute_residual_blk_l2(
    f: Callable, x0: jnp.ndarray, states: jnp.ndarray, drivers: jnp.ndarray
) -> jnp.ndarray:
    """max_t ||s_t - f_t(s_{t-1})||_2."""
    prev = jnp.concatenate([x0[None], states[:-1]], axis=0)
    new_states = jax.vmap(f)(prev, drivers)
    return jnp.max(jnp.linalg.norm(states - new_states, axis=-1))


def compute_final_update_linf(
    f: Callable,
    x0: jnp.ndarray,
    states: jnp.ndarray,
    drivers: jnp.ndarray,
    rtol: float = DEER_RTOL_FP64,
) -> jnp.ndarray:
    """DEER's err metric at the final iterate (for audit)."""
    prev = jnp.concatenate([x0[None], states[:-1]], axis=0)
    new_states = jax.vmap(f)(prev, drivers)
    return jnp.max(jnp.abs(new_states - states) - rtol * jnp.abs(states))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverResult:
    algorithm: str
    T: int
    D: int
    rho: float
    tol: float
    rtol: float
    iters_to_tol: int
    wall_ms: float           # median of n_repeats
    compile_ms: float
    seq_rollout_ms: float    # sequential rollout on same device
    final_resid_blk: float   # max_t ||yt - f(prev)||_2
    final_update_linf: float # DEER err metric at final iterate
    error_vs_truth: float    # max_t ||yt - x*||_2 (against sequential ground truth)
    peak_mem_mb: float
    oom: bool
    seed: int


def _memory_stats_mb() -> float:
    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats()
        return float(stats.get("peak_bytes_in_use", 0)) / 1e6
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Sequential rollout (shared denominator for speedup)
# ---------------------------------------------------------------------------


def time_sequential_rollout(
    f: Callable,
    x0: jnp.ndarray,
    drivers: jnp.ndarray,
    n_repeats: int = 3,
) -> float:
    @jax.jit
    def seq(x0_in: jnp.ndarray, drv: jnp.ndarray) -> jnp.ndarray:
        return sequential_rollout(f, x0_in, drv)

    _ = jax.block_until_ready(seq(x0, drivers))  # compile
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = jax.block_until_ready(seq(x0, drivers))
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Individual solver runners
# ---------------------------------------------------------------------------


def _setup_system(
    T: int, D: int, rho: float, seed: int
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build (f, x0, drivers, initial_guess, x_true) for a run."""
    key = jr.PRNGKey(seed)
    k_w, k_x0, k_drv, k_guess = jr.split(key, 4)
    W = init_contractive_rnn(D, rho, k_w)
    x0 = jr.normal(k_x0, (D,))
    drivers = gen_drivers(T, D, k_drv, scale=1.0)
    initial_guess = jr.normal(k_guess, (T, D))
    f = make_step_fn(W)
    x_true = sequential_rollout(f, x0, drivers)
    x_true = jax.block_until_ready(x_true)
    return f, x0, drivers, initial_guess, x_true


def _is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return ("resource_exhausted" in s
            or "out of memory" in s
            or "memory exhausted" in s
            or "cudaerrorout" in s)


def run_jacobi(
    T: int, D: int, rho: float,
    tol: float = DEER_TOL_FP64, rtol: float = DEER_RTOL_FP64,
    seed: int = 0, n_repeats: int = 3, max_iter: int = DEFAULT_MAX_ITER,
) -> SolverResult:
    f, x0, drivers, initial_guess, x_true = _setup_system(T, D, rho, seed)
    seq_ms = time_sequential_rollout(f, x0, drivers, n_repeats=n_repeats)

    @jax.jit
    def run(x0_in: jnp.ndarray, guess_in: jnp.ndarray, drv_in: jnp.ndarray):
        return jacobi_while_deer_criterion(
            f, x0_in, guess_in, drv_in, tol=tol, rtol=rtol, max_iter=max_iter
        )

    try:
        t0 = time.perf_counter()
        final, iters = run(x0, initial_guess, drivers)
        _ = jax.block_until_ready((final, iters))
        compile_ms = (time.perf_counter() - t0) * 1000.0

        times: List[float] = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            final, iters = run(x0, initial_guess, drivers)
            _ = jax.block_until_ready((final, iters))
            times.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = float(np.median(times))

        final_resid = float(compute_residual_blk_l2(f, x0, final, drivers))
        final_update = float(compute_final_update_linf(f, x0, final, drivers, rtol))
        err_vs_truth = float(jnp.max(jnp.linalg.norm(final - x_true, axis=-1)))
        peak_mb = _memory_stats_mb()
        oom = False
    except Exception as e:
        if not _is_oom(e):
            raise
        wall_ms = compile_ms = final_resid = final_update = err_vs_truth = float("nan")
        iters = -1
        peak_mb = float("nan")
        oom = True

    return SolverResult(
        algorithm="jacobi", T=T, D=D, rho=rho, tol=tol, rtol=rtol,
        iters_to_tol=int(iters), wall_ms=wall_ms, compile_ms=compile_ms,
        seq_rollout_ms=seq_ms, final_resid_blk=final_resid,
        final_update_linf=final_update, error_vs_truth=err_vs_truth,
        peak_mem_mb=peak_mb, oom=oom, seed=seed,
    )


def run_jacobi_scan(
    T: int, D: int, rho: float,
    tol: float = DEER_TOL_FP64, rtol: float = DEER_RTOL_FP64,
    seed: int = 0, n_repeats: int = 3, max_iter: int = 120,
) -> SolverResult:
    """Scan-based Jacobi using ``lax.scan(length=max_iter)``; run_single_gpu.py passes min(--max-iter, 120).

    ``wall_ms`` is the full ``max_iter``-iteration scan wall (NOT rescaled).
    ``speedups.py`` rescales by ``iters_to_tol / max_iter`` to obtain the
    equivalent early-stop wall — apples-to-apples with the multi-GPU scan
    sweep (``run_multi_gpu.py``).  We use scan instead of
    ``lax.while_loop`` because ``while_loop`` inside ``shard_map`` cannot
    pipeline collectives; fixing the primitive across single-GPU and
    multi-GPU keeps one methodology for both.  ``iters_to_tol`` is the first
    iteration whose error to the sequential solution, max_t ||s_t - x*_t||_2,
    is below ``tol``.
    """
    f, x0, drivers, initial_guess, x_true = _setup_system(T, D, rho, seed)
    seq_ms = time_sequential_rollout(f, x0, drivers, n_repeats=n_repeats)

    @jax.jit
    def run(x0_in: jnp.ndarray, guess_in: jnp.ndarray,
            drv_in: jnp.ndarray, x_true_in: jnp.ndarray):
        return jacobi_scan(f, x0_in, guess_in, drv_in, x_true_in, max_iter)

    try:
        t0 = time.perf_counter()
        _, err_history = run(x0, initial_guess, drivers, x_true)
        _ = jax.block_until_ready(err_history)
        compile_ms = (time.perf_counter() - t0) * 1000.0

        times: List[float] = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _, err_history = run(x0, initial_guess, drivers, x_true)
            _ = jax.block_until_ready(err_history)
            times.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = float(np.median(times))

        err_np = np.asarray(err_history)  # (max_iter, 2): [err_vs_truth, resid]
        below = err_np[:, 0] < tol
        iters = int(np.argmax(below)) + 1 if below.any() else -1
        final_resid = float(err_np[-1, 1])
        final_update = final_resid  # scan reports update size as resid column
        err_vs_truth = float(err_np[-1, 0])
        peak_mb = _memory_stats_mb()
        oom = False
    except Exception as e:
        if not _is_oom(e):
            raise
        wall_ms = compile_ms = final_resid = final_update = err_vs_truth = float("nan")
        iters = -1
        peak_mb = float("nan")
        oom = True

    return SolverResult(
        algorithm="jacobi_scan", T=T, D=D, rho=rho, tol=tol, rtol=rtol,
        iters_to_tol=int(iters), wall_ms=wall_ms, compile_ms=compile_ms,
        seq_rollout_ms=seq_ms, final_resid_blk=final_resid,
        final_update_linf=final_update, error_vs_truth=err_vs_truth,
        peak_mem_mb=peak_mb, oom=oom, seed=seed,
    )


def _make_deer_func(W: jnp.ndarray) -> Callable:
    """Adapter: turn f(x, c) -> W tanh(x) + c into (y, driver, params) -> ."""
    def func(y: jnp.ndarray, driver: jnp.ndarray, params: Any) -> jnp.ndarray:
        return W @ jnp.tanh(y) + driver
    return func


def _run_deer_variant(
    algorithm: str,
    solve_fn: Callable,  # deer_map_solve or deer_map_solve_full
    T: int, D: int, rho: float,
    tol: float, rtol: float,
    seed: int, n_repeats: int, max_iter: int,
    uses_key: bool,
) -> SolverResult:
    """quasi-DEER or Full-DEER from the zero guess. ``tol`` and ``rtol`` are stored in the row, and ``rtol`` enters
    the audit metric final_update_linf; deer_map_solve and deer_map_solve_full do not take them and use the same
    values (1e-7 and 1e-4) for float64 inputs."""
    # Build system.
    key = jr.PRNGKey(seed)
    k_w, k_x0, k_drv, k_probe = jr.split(key, 4)
    W = init_contractive_rnn(D, rho, k_w)
    x0 = jr.normal(k_x0, (D,))
    drivers = gen_drivers(T, D, k_drv, scale=1.0)
    f_jacobi = make_step_fn(W)  # for audit metrics
    x_true = jax.block_until_ready(sequential_rollout(f_jacobi, x0, drivers))

    seq_ms = time_sequential_rollout(f_jacobi, x0, drivers, n_repeats=n_repeats)

    func_deer = _make_deer_func(W)
    params = {"key": k_probe} if uses_key else {}
    yinit_guess = jnp.zeros((T, D), dtype=x0.dtype)

    # Small closure so JIT captures (T, D) shapes but function identity too.
    def runner(x0_in: jnp.ndarray, drv_in: jnp.ndarray, guess_in: jnp.ndarray):
        return solve_fn(
            func_deer,
            x0_in,
            drv_in,
            params,
            yinit_guess=guess_in,
            max_iter=max_iter,
        )

    run_jit = jax.jit(runner)

    try:
        t0 = time.perf_counter()
        yt, n_iters = run_jit(x0, drivers, yinit_guess)
        _ = jax.block_until_ready((yt, n_iters))
        compile_ms = (time.perf_counter() - t0) * 1000.0

        times: List[float] = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            yt, n_iters = run_jit(x0, drivers, yinit_guess)
            _ = jax.block_until_ready((yt, n_iters))
            times.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = float(np.median(times))

        final_resid = float(compute_residual_blk_l2(f_jacobi, x0, yt, drivers))
        final_update = float(compute_final_update_linf(f_jacobi, x0, yt, drivers, rtol))
        err_vs_truth = float(jnp.max(jnp.linalg.norm(yt - x_true, axis=-1)))
        peak_mb = _memory_stats_mb()
        oom = False
    except Exception as e:
        if not _is_oom(e):
            raise
        wall_ms = compile_ms = final_resid = final_update = err_vs_truth = float("nan")
        n_iters = -1
        peak_mb = float("nan")
        oom = True

    return SolverResult(
        algorithm=algorithm, T=T, D=D, rho=rho, tol=tol, rtol=rtol,
        iters_to_tol=int(n_iters), wall_ms=wall_ms, compile_ms=compile_ms,
        seq_rollout_ms=seq_ms, final_resid_blk=final_resid,
        final_update_linf=final_update, error_vs_truth=err_vs_truth,
        peak_mem_mb=peak_mb, oom=oom, seed=seed,
    )


def run_quasi_deer(
    T: int, D: int, rho: float,
    tol: float = DEER_TOL_FP64, rtol: float = DEER_RTOL_FP64,
    seed: int = 0, n_repeats: int = 3, max_iter: int = DEFAULT_MAX_ITER,
) -> SolverResult:
    return _run_deer_variant(
        "quasi_deer", deer_map_solve,
        T, D, rho, tol, rtol, seed, n_repeats, max_iter,
        uses_key=True,
    )


def run_full_deer(
    T: int, D: int, rho: float,
    tol: float = DEER_TOL_FP64, rtol: float = DEER_RTOL_FP64,
    seed: int = 0, n_repeats: int = 3, max_iter: int = DEFAULT_MAX_ITER,
) -> SolverResult:
    return _run_deer_variant(
        "full_deer", deer_map_solve_full,
        T, D, rho, tol, rtol, seed, n_repeats, max_iter,
        uses_key=False,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def results_to_dict(results: List[SolverResult]) -> Dict[str, np.ndarray]:
    keys = [
        "algorithm", "T", "D", "rho", "tol", "rtol",
        "iters_to_tol", "wall_ms", "compile_ms", "seq_rollout_ms",
        "final_resid_blk", "final_update_linf", "error_vs_truth",
        "peak_mem_mb", "oom", "seed",
    ]
    return {k: np.array([getattr(r, k) for r in results]) for k in keys}


def summarise(results: List[SolverResult]) -> str:
    lines = []
    hdr = (f"{'algo':>12s}  {'T':>8s}  {'iters':>5s}  {'wall_ms':>9s}  "
           f"{'seq_ms':>9s}  {'speedup':>8s}  {'resid':>10s}  "
           f"{'err_truth':>10s}  {'oom':>4s}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        speedup = r.seq_rollout_ms / r.wall_ms if r.wall_ms and not r.oom else float("nan")
        lines.append(
            f"{r.algorithm:>12s}  {r.T:>8d}  {r.iters_to_tol:>5d}  "
            f"{r.wall_ms:>9.2f}  {r.seq_rollout_ms:>9.2f}  {speedup:>8.2f}  "
            f"{r.final_resid_blk:>10.2e}  {r.error_vs_truth:>10.2e}  "
            f"{'YES' if r.oom else '':>4s}"
        )
    return "\n".join(lines)
