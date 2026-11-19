"""
The driver of the neural-controller table of the paper (tab:controllers): the three controllers mlp_large, mlp_small
and ssm_d16_contract (the paper's MLP large, MLP small and SSM) on the two-link arm, at the paper's gains kp = 3,
kd = 0.6 of the PD term.

For each (seed, controller, kp, solver) row, where solver "exp" is DEER-exp and "euler" is Euler-DEER:
  1. Train for 1200 optimizer steps (--epochs; one batch of 4 initial conditions per step) at N = 1000, every step
     solving the closed loop in parallel (implicit_deer.py: Newton's method with the implicit-function adjoint,
     at most deer_K = 50 Newton iterations).
  2. Time the training step at N = 1000 against the same step through the sequential integrator of the same
     discretization.

Output: the JSON file --out, one row per (seed, controller, kp, solver).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

# Disable XLA CUDA-Graph capture — JAX 0.6.2 fails on this workload otherwise.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + \
    " --xla_gpu_enable_command_buffer="

import json
import time
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

from experiments.appendix.robot_control.neural.arms import planar_arm, ARM_2LINK  # noqa: E402
from experiments.appendix.robot_control.neural.controller_utils import (  # noqa: E402
    init_mlp, apply_mlp,
    make_lissajous_reference,
)
from experiments.appendix.robot_control.neural.ssm_policy import (  # noqa: E402
    SSMCellConfig, init_ssm_policy, count_ssm_policy_params,
    make_ssm_ode_cl, readout_mlp,
)
from experiments.appendix.robot_control.neural.implicit_deer import (  # noqa: E402
    exp_par_integrate_implicit,
    euler_par_integrate_implicit,
)
from experiments.appendix.robot_control.neural.sequential_solvers import seq_integrate_dispatch  # noqa: E402

PAR_SOLVERS = {
    "exp": exp_par_integrate_implicit,
    "euler": euler_par_integrate_implicit,
}


# ─── Architectures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MLPSpec:
    name: str
    sizes: Tuple[int, ...]   # e.g. (8, 128, 128, 2)


@dataclass(frozen=True)
class SSMSpec:
    name: str
    cfg: SSMCellConfig


def _build_mlp_factory(kp, kd, q_ref_fn, qdot_ref_fn, arm_ode, M_fn, V_fn, n: int):
    """Closed-loop ODE factory for MLP controller. State z = x_arm."""
    def factory(params):
        def ode_cl(t, x):
            q, qdot = x[:n], x[n:]
            q_ref = q_ref_fn(t)
            qdot_ref = qdot_ref_fn(t)
            g_comp = jax.grad(V_fn)(q)
            M = M_fn(q)
            desired_qddot = -kp * (q - q_ref) - kd * (qdot - qdot_ref)
            u_pd = g_comp + M @ desired_qddot
            nn_input = jnp.concatenate([x, q_ref, qdot_ref])
            u_nn = apply_mlp(params, nn_input)
            return arm_ode(t, x, u_pd + u_nn)
        return ode_cl
    return factory


def _control_loss_mlp(params, traj, q_ref_fn, qdot_ref_fn, dt, alpha_ctrl, N):
    times = jnp.arange(N + 1) * dt

    def nn_at(t_and_x):
        t, x = t_and_x
        return apply_mlp(params, jnp.concatenate(
            [x, q_ref_fn(t), qdot_ref_fn(t)]
        ))

    u_nn_all = jax.vmap(nn_at)((times[:-1], traj[:-1]))
    return alpha_ctrl * jnp.sum(u_nn_all ** 2) * dt


def _control_loss_ssm(params, traj, dt, alpha_ctrl, n):
    u_nn_all = jax.vmap(lambda h: readout_mlp(params, h))(traj[:-1, 2 * n:])
    return alpha_ctrl * jnp.sum(u_nn_all ** 2) * dt


# ─── train + bench (per row) ────────────────────────────────────────


def train_and_bench_row(
    arch_spec,
    par_solver: str,
    kp: float,
    kd: float = 0.6,
    omega1: float = 2.0,
    amp1: float = 0.5,
    amp2: float = 0.4,
    n_epochs: int = 1200,
    lr: float = 1e-3,
    T: float = 10.0,
    dt: float = 0.01,
    deer_K: int = 50,
    tol: float = 1e-5,
    alpha_ctrl: float = 1e-3,
    batch_size: int = 4,
    seed: int = 0,
    n_warmup: int = 3,
    n_runs: int = 20,
):
    """Train one (controller, par_solver, kp) row for n_epochs optimizer steps and time its training step.

    deer_K is the cap of Newton iterations of the parallel solve and tol its stopping tolerance (largest entry of
    the Newton update)."""
    arm_ode, _, M_fn, V_fn, _ = planar_arm(**ARM_2LINK)
    n = 2
    N = int(T / dt)
    par_fn = PAR_SOLVERS[par_solver]

    q_ref_fn, qdot_ref_fn, q_hang = make_lissajous_reference(
        omega1=omega1, amp1=amp1, amp2=amp2,
    )

    is_mlp = isinstance(arch_spec, MLPSpec)
    if is_mlp:
        D_aug = 2 * n
        factory = _build_mlp_factory(
            kp, kd, q_ref_fn, qdot_ref_fn, arm_ode, M_fn, V_fn, n,
        )
        params = init_mlp(jax.random.PRNGKey(seed), list(arch_spec.sizes))
        n_params = sum(w.size + b.size for w, b in params)
        z0_eval = jnp.concatenate([q_hang, jnp.zeros(n)])
        ctrl_loss_fn = lambda p, traj: _control_loss_mlp(
            p, traj, q_ref_fn, qdot_ref_fn, dt, alpha_ctrl, N,
        )
    else:
        cfg = arch_spec.cfg
        H = cfg.D_ssm
        D_aug = 2 * n + H
        factory = make_ssm_ode_cl(
            arm_ode, M_fn, V_fn, q_ref_fn, qdot_ref_fn, kp, kd, cfg,
        )
        params = init_ssm_policy(jax.random.PRNGKey(seed), cfg)
        n_params = count_ssm_policy_params(params)
        z0_eval = jnp.concatenate([q_hang, jnp.zeros(n), jnp.zeros(H)])
        ctrl_loss_fn = lambda p, traj: _control_loss_ssm(
            p, traj, dt, alpha_ctrl, n,
        )

    key_ic = jax.random.PRNGKey(42)
    q0_perturb = 0.05 * jax.random.normal(key_ic, (batch_size, n))
    q0_batch = q_hang[None, :] + q0_perturb
    if is_mlp:
        z0_batch = jnp.concatenate([q0_batch, jnp.zeros((batch_size, n))], axis=1)
    else:
        H = arch_spec.cfg.D_ssm
        z0_batch = jnp.concatenate(
            [q0_batch, jnp.zeros((batch_size, n)), jnp.zeros((batch_size, H))],
            axis=1,
        )

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(params)
    guess = jnp.zeros((N, D_aug))

    arch_name = arch_spec.name
    print(f"\n--- {arch_name} | kp={kp} kd={kd} | par={par_solver} | "
          f"params={n_params}, D_aug={D_aug} ---")
    print(f"    T={T}, N={N}, K={deer_K}, lr={lr}, epochs={n_epochs}")

    def par_rollout(p, z0):
        return par_fn(factory, p, z0, guess, dt, deer_K, 0.0, tol)[0]

    def seq_rollout(p, z0):
        return seq_integrate_dispatch(factory(p), z0, dt, N, par_solver, t0=0.0)

    def batch_loss(p, rollout):
        """Tracking loss plus control penalty of rollout(p, z0), averaged over the batch."""
        def single(z0):
            traj = rollout(p, z0)
            times = jnp.arange(N + 1) * dt
            q_refs = jax.vmap(q_ref_fn)(times)
            tracking = jnp.sum((traj[:, :n] - q_refs) ** 2) * dt
            return tracking + ctrl_loss_fn(p, traj)
        return jnp.mean(jax.vmap(single)(z0_batch))

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda p: batch_loss(p, par_rollout))(params)
        updates, new_opt = optimizer.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), new_opt, loss

    @jax.jit
    def eval_seq_rk4(params):
        from parode.sequential import seq_integrate
        return seq_integrate(factory(params), z0_eval, dt, N, method="rk4")

    @jax.jit
    def eval_rmse(traj):
        times = jnp.arange(N + 1) * dt
        q_refs = jax.vmap(q_ref_fn)(times)
        return jnp.sqrt(jnp.mean((traj[:, :n] - q_refs) ** 2))

    print("    JIT compile...", end=" ", flush=True)
    t_jit = time.perf_counter()
    _, _, loss0 = train_step(params, opt_state)
    jax.block_until_ready(loss0)
    jit_time = time.perf_counter() - t_jit
    print(f"{jit_time:.1f}s, loss0={float(loss0):.4f}")

    if jnp.isnan(loss0):
        return {"status": "NaN_init", "arch": arch_name, "kp": kp,
                "par_solver": par_solver}

    # The initial parameters and optimizer state again (the compiling call above did not keep its update)
    if is_mlp:
        params = init_mlp(jax.random.PRNGKey(seed), list(arch_spec.sizes))
    else:
        params = init_ssm_policy(jax.random.PRNGKey(seed), arch_spec.cfg)
    opt_state = optimizer.init(params)

    # PD-only baseline
    pd_params = jax.tree.map(jnp.zeros_like, params)
    pd_traj = eval_seq_rk4(pd_params)
    pd_rmse = float(eval_rmse(pd_traj))
    print(f"    PD-only RMSE: {pd_rmse:.4f}")

    losses = []
    eval_every = 100
    eval_epochs = sorted(set(list(range(0, n_epochs, eval_every)) + [n_epochs - 1]))

    for epoch in range(n_epochs):
        t0 = time.perf_counter()
        params, opt_state, loss = train_step(params, opt_state)
        jax.block_until_ready(loss)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        losses.append(float(loss))
        if jnp.isnan(loss):
            print(f"    NaN at ep {epoch} — stopping")
            break

        if epoch in eval_epochs or epoch == n_epochs - 1:
            traj = eval_seq_rk4(params)
            rmse = float(eval_rmse(traj))
            print(f"    ep {epoch:>4d}: loss={float(loss):.4f}, "
                  f"RMSE={rmse:.4f} ({pd_rmse/max(rmse,1e-10):.1f}×), "
                  f"{elapsed_ms:.0f}ms")

    final_traj = eval_seq_rk4(params)
    final_rmse = float(eval_rmse(final_traj))
    final_ratio = pd_rmse / max(final_rmse, 1e-10)

    # ─── Time the training step at N=1000: train_step (parallel) against step_seq (sequential) ───
    print(f"    Bench training-step:", end=" ", flush=True)

    @jax.jit
    def step_seq(p, s):
        loss, grads = jax.value_and_grad(lambda q: batch_loss(q, seq_rollout))(p)
        updates, new_opt = optimizer.update(grads, s, params=p)
        return optax.apply_updates(p, updates), new_opt, loss

    s_par = optimizer.init(params)
    s_seq = optimizer.init(params)
    for _ in range(n_warmup):
        _, s_par, lp = train_step(params, s_par); jax.block_until_ready(lp)
        _, s_seq, ls = step_seq(params, s_seq); jax.block_until_ready(ls)

    par_times, seq_times = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, _, lp = train_step(params, s_par); jax.block_until_ready(lp)
        par_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _, _, ls = step_seq(params, s_seq); jax.block_until_ready(ls)
        seq_times.append(time.perf_counter() - t0)

    @jax.jit
    def par_iters_at(p):
        _, _, it = par_fn(factory, p, z0_batch[0], guess, dt, deer_K, 0.0, tol)
        return it

    par_iters = int(par_iters_at(params))
    t_par = float(np.mean(par_times))
    t_seq = float(np.mean(seq_times))
    speedup = t_seq / t_par
    print(f"par={t_par*1e3:.1f}ms seq={t_seq*1e3:.1f}ms speedup={speedup:.2f}× "
          f"par_iters={par_iters}")

    return {
        "status": "converged",
        "arch": arch_name, "kp": kp, "kd": kd, "par_solver": par_solver,
        "n_params": int(n_params), "D_augmented": int(D_aug),
        "T": T, "dt": dt, "N": N, "deer_K": deer_K, "tol": tol,
        "n_epochs_completed": len(losses),
        "jit_time_s": round(jit_time, 2),
        "pd_only_rmse": pd_rmse, "final_rmse": final_rmse,
        "rmse_improvement": final_ratio,
        "losses": losses,
        "t_par_train_mean_s": t_par,
        "t_par_train_std_s": float(np.std(par_times)),
        "t_seq_train_mean_s": t_seq,
        "t_seq_train_std_s": float(np.std(seq_times)),
        "train_speedup": speedup,
        "par_solver_iters": par_iters,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train and time the rows of table tab:controllers (seed x controller x kp x solver).")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="Comma-separated seeds (default '0,1,2,3,4', the paper's).")
    parser.add_argument("--kps", default="3",
                        help="Comma-separated kp values to run (default '3', the paper's).")
    parser.add_argument("--out", default="results/robot_control/check/bench_table_rerun.json",
                        help="Output json file (default results/robot_control/check/bench_table_rerun.json).")
    parser.add_argument("--epochs", type=int, default=1200,
                        help="Optimizer steps of training, one batch of 4 initial conditions each (paper: 1200).")
    parser.add_argument("--timing-runs", type=int, default=20,
                        help="Timed training steps per solver after 3 warm-up steps (paper: 20).")
    parser.add_argument("--archs", default="mlp_large,mlp_small,ssm_d16_contract",
                        help="Comma-separated subset of the three controllers.")
    parser.add_argument("--solvers", default="exp,euler", help="Comma-separated subset of exp, euler.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    kps = [float(s) for s in args.kps.split(",") if s.strip()]

    archs = [
        MLPSpec(name="mlp_large", sizes=(8, 128, 128, 2)),
        MLPSpec(name="mlp_small", sizes=(8, 64, 32, 2)),
        SSMSpec(name="ssm_d16_contract", cfg=SSMCellConfig()),
    ]
    archs = [a for a in archs if a.name in args.archs.split(",")]
    pds = [(kp, 0.6) for kp in kps]
    par_solvers = tuple(args.solvers.split(","))

    print("=" * 80)
    print("TABLE tab:controllers: controller × kp × par_solver — training-step speedup at N=1000")
    print(f"seeds={seeds}  kps={kps}  out={args.out}")
    print("=" * 80)

    rows = []

    for seed in seeds:
        for arch in archs:
            for kp, kd in pds:
                for par_solver in par_solvers:
                    label = f"{arch.name}_kp{kp:g}_{par_solver}_s{seed}"
                    res = train_and_bench_row(
                        arch_spec=arch, par_solver=par_solver,
                        kp=kp, kd=kd, seed=seed, n_epochs=args.epochs,
                        n_runs=args.timing_runs,
                    )
                    res["label"] = label
                    res["seed"] = seed
                    rows.append(res)

    fname = args.out
    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    with open(fname, "w") as f:
        json.dump(rows, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved {fname}")

    # ─── Summary table ───
    print("\n" + "=" * 100)
    print("SUMMARY (kd=0.6, dt=0.01, T=10, N=1000, K=50)")
    print("=" * 100)
    print(f"{'arch':<18} {'kp':>3}  {'solver':>6}  {'params':>7}  "
          f"{'PD_RMSE':>8} {'final':>8} {'×PD':>5}  "
          f"{'par_ms':>8} {'seq_ms':>9} {'speedup':>8} {'iters':>5}")
    print("-" * 100)
    for r in rows:
        if r.get("status") != "converged":
            print(f"{r.get('arch','?'):<18}  {r.get('kp','?'):>3}  "
                  f"{r.get('par_solver','?'):>6}   {r.get('status')}")
            continue
        print(f"{r['arch']:<18} {int(r['kp']):>3}  {r['par_solver']:>6}  "
              f"{r['n_params']:>7d}  {r['pd_only_rmse']:>8.4f} "
              f"{r['final_rmse']:>8.4f} {r['rmse_improvement']:>4.1f}×  "
              f"{r['t_par_train_mean_s']*1e3:>8.2f} "
              f"{r['t_seq_train_mean_s']*1e3:>9.2f} "
              f"{r['train_speedup']:>7.2f}× "
              f"{r['par_solver_iters']:>5d}")


if __name__ == "__main__":
    main()
