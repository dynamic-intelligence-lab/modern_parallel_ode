"""Multi-GPU Jacobi iteration via ``shard_map``.

The T axis (N in the paper) is sharded across n devices along a 1D mesh ``'t'``.  Each sweep:

1. Every device applies ``f`` locally to its slice of (states, drivers).
2. A single nearest-neighbour ``ppermute`` exchanges the last local state of
   each device to the next device, so that the "prev = shift(states)" needed
   by Jacobi can be reconstructed locally.  Device 0 overrides the received
   value with the fixed initial condition x0.
3. Errors and residuals are reduced globally with ``pmax``.

Communication per sweep: one (D,)-vector exchange and two scalar all-reductions
(the error to the sequential solution and the residual).
Compute per sweep per device: O(T/n * D^2), same constant factor as
single-GPU Jacobi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.appendix.gpu_scaling.contractive_rnn import (
    gen_drivers,
    init_contractive_rnn,
    make_step_fn,
)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def make_mesh(n_devices: int) -> Mesh:
    """Build a 1D mesh ``'t'`` over the first ``n_devices`` JAX devices.

    Raises if fewer devices are available than requested.
    """
    available = jax.device_count()
    if n_devices > available:
        raise RuntimeError(
            f"Requested {n_devices} devices but only {available} are available."
        )
    devices = np.array(jax.devices()[:n_devices])
    return Mesh(devices, axis_names=("t",))


# ---------------------------------------------------------------------------
# Sharded Jacobi scan
# ---------------------------------------------------------------------------


def _sharded_sequential_rollout(
    f: Callable,
    x0: jnp.ndarray,
    drivers: jnp.ndarray,
    mesh: Mesh,
) -> jnp.ndarray:
    """Compute ground truth (T, D) with ``drivers`` sharded along t.

    Sequential by nature — we just run ``lax.scan`` on the full array on one
    device (gathered), then return a sharded copy.  T up to 1e7 for fp64
    at D=64 is 5 GB: fits on one device during setup.  For genuinely
    T-sharded sequential rollout we would need a ring-scan which is a
    log-depth algorithm; not needed here since ground-truth is not the
    bottleneck.
    """
    # Gather drivers, run on the first device, re-shard.
    drivers_full = jax.device_put(drivers, jax.devices()[0])

    def body(carry: jnp.ndarray, c: jnp.ndarray):
        x_next = f(carry, c)
        return x_next, x_next

    _, states = jax.lax.scan(body, x0, drivers_full)
    # Re-shard along t.
    return jax.device_put(
        states, NamedSharding(mesh, P("t", None))
    )


def make_jacobi_scan_multigpu(
    f: Callable,
    mesh: Mesh,
    n_devices: int,
    max_iters: int,
):
    """Return a JIT-compiled multi-GPU Jacobi scan (fixed iterations).

    ``f``, ``mesh``, ``n_devices``, and ``max_iters`` are captured as static
    constants so that the returned ``run(x0, guess, drivers, x_true)`` has
    only traced arguments (JAX jit-safe).
    """
    perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]

    def _body(states: jnp.ndarray, drivers_sh: jnp.ndarray,
              x_true_sh: jnp.ndarray, x0_rep: jnp.ndarray):
        device_id = jax.lax.axis_index("t")
        last_received = jax.lax.ppermute(
            states[-1:], axis_name="t", perm=perm_right
        )
        prepend = jnp.where(device_id == 0, x0_rep, last_received[0])
        prev = jnp.concatenate([prepend[None], states[:-1]], axis=0)

        new_states = jax.vmap(f)(prev, drivers_sh)
        local_err = jnp.max(jnp.linalg.norm(new_states - x_true_sh, axis=-1))
        local_res = jnp.max(jnp.linalg.norm(states - new_states, axis=-1))
        global_err = jax.lax.pmax(local_err, "t")
        global_res = jax.lax.pmax(local_res, "t")
        return new_states, jnp.stack([global_err, global_res])

    def _shard_body(x0_rep, initial_guess_sh, drivers_sh, x_true_sh):
        def scan_on_shard(states_init, drivers_local, x_true_local, x0_val):
            def step(states, _):
                return _body(states, drivers_local, x_true_local, x0_val)
            return jax.lax.scan(step, states_init, None, length=max_iters)

        return shard_map(
            scan_on_shard,
            mesh=mesh,
            in_specs=(P("t", None), P("t", None), P("t", None), P()),
            out_specs=(P("t", None), P()),
            check_rep=False,
        )(initial_guess_sh, drivers_sh, x_true_sh, x0_rep)

    return jax.jit(_shard_body)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JacobiResultMG:
    T: int
    D: int
    rho: float
    tol: float
    max_iters: int
    n_devices: int
    iters_to_conv: int
    wall_clock_ms: float
    compile_time_ms: float
    initial_err: float
    initial_resid: float
    final_err: float
    err_history: np.ndarray
    predicted_k: int
    wall_min_ms: float = float("nan")   # fastest of the timed repeats


def _predict_k(initial_err: float, tol: float, rho: float) -> int:
    if initial_err <= tol:
        return 0
    return int(np.ceil(np.log(initial_err / tol) / np.log(1.0 / rho)))


def run_single_multigpu(
    T: int,
    D: int,
    rho: float,
    tol: float,
    max_iters: int,
    n_devices: int,
    seed: int = 0,
    driver_scale: float = 1.0,
    n_repeats: int = 3,
) -> JacobiResultMG:
    """Run one multi-GPU Jacobi experiment and return timing + metrics.

    T must be divisible by n_devices; we enforce this by rounding T up.

    Runs ``lax.scan`` with a fixed ``max_iters``: ``wall_clock_ms`` is the time
    for ``max_iters`` Jacobi sweeps (the median of ``n_repeats`` timed runs
    after one compilation run), ``err_history`` holds the error to the
    sequential solution and the residual after every sweep, and
    ``iters_to_conv`` is the first sweep whose error is below ``tol``.
    """
    if T % n_devices != 0:
        T_adj = ((T + n_devices - 1) // n_devices) * n_devices
        T = T_adj

    mesh = make_mesh(n_devices)

    key = jr.PRNGKey(seed)
    k_w, k_x0, k_drv, k_guess = jr.split(key, 4)

    W = init_contractive_rnn(D, rho, k_w)
    x0 = jr.normal(k_x0, (D,))
    drivers = gen_drivers(T, D, k_drv, scale=driver_scale)
    initial_guess = jr.normal(k_guess, (T, D))

    f = make_step_fn(W)

    # Ground truth (single-device sequential rollout, then shard).  Also used
    # for ``initial_err`` and for the per-iter err vs truth.
    x_true = _sharded_sequential_rollout(f, x0, drivers, mesh)

    # Shard inputs along t.
    sharding_t = NamedSharding(mesh, P("t", None))
    initial_guess_sh = jax.device_put(initial_guess, sharding_t)
    drivers_sh = jax.device_put(drivers, sharding_t)
    x0_rep = jax.device_put(x0, NamedSharding(mesh, P()))

    diff0 = jnp.asarray(initial_guess) - jnp.asarray(x_true)
    initial_err = float(jnp.max(jnp.linalg.norm(diff0, axis=-1)))

    run_mg = make_jacobi_scan_multigpu(f, mesh, n_devices, max_iters)

    t0 = time.perf_counter()
    final_states, err_history = run_mg(x0_rep, initial_guess_sh, drivers_sh, x_true)
    err_history = jax.block_until_ready(err_history)
    compile_time_ms = (time.perf_counter() - t0) * 1000.0

    wall_times: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _, err_history = run_mg(x0_rep, initial_guess_sh, drivers_sh, x_true)
        err_history = jax.block_until_ready(err_history)
        wall_times.append((time.perf_counter() - t0) * 1000.0)
    # Median of the repeats, the estimator of the single-GPU runner
    # (solvers.run_jacobi_scan). Until 2026-11-12 this was the minimum.
    wall_clock_ms = float(np.median(wall_times))
    wall_min_ms = float(min(wall_times))

    err_np = np.asarray(err_history)  # (max_iters, 2): [err, resid]

    initial_resid = float(err_np[0, 1])

    below = err_np[:, 0] < tol
    iters_to_conv = int(np.argmax(below)) + 1 if below.any() else -1

    return JacobiResultMG(
        T=T,
        D=D,
        rho=rho,
        tol=tol,
        max_iters=max_iters,
        n_devices=n_devices,
        iters_to_conv=iters_to_conv,
        wall_clock_ms=wall_clock_ms,
        compile_time_ms=compile_time_ms,
        initial_err=initial_err,
        initial_resid=initial_resid,
        final_err=float(err_np[-1, 0]),
        err_history=err_np,
        predicted_k=_predict_k(initial_err, tol, rho),
        wall_min_ms=wall_min_ms,
    )

