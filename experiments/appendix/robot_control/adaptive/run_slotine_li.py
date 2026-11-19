"""Jacobi iteration of the Slotine-Li adaptive-control rollout (the paper's fig:slotine-li-fixedpoint-rollout),
checked against the sequential rollout.

For the recurrence

    s_{n+1} = clip_mass(s_n + dt * rhs(s_n, n)),   s = (q1, q2, dq1, dq2, m_hat),   n = 0 .. N-2,

where rhs is the closed loop of a two-link arm under the Slotine-Li adaptive controller with an online estimate of
the payload mass, this script computes in float64

  1. the sequential rollout (lax.scan), which is the exact solution of the recurrence, and
  2. the plain Jacobi iteration S <- Phi(S), where Phi applies one explicit-Euler step of the closed loop to every
     time index at once, from the reference path plus noise, for N sweeps (fewer with --sweeps), recording after
     every sweep the distance to the sequential rollout and saving the iterates after 0, 500 and 2000 sweeps, which
     the figure draws (figure_slotine_li.py).

The model block below (between the two marker comments) is copied verbatim from a notebook of the authors (not in
this repository), without its plotting helpers. The keys of the written file end in _ref_noise_r1: the guess
"reference plus noise" and relaxation 1 (plain Jacobi), the paper's run. Runs on a CPU or a GPU, from the repository
root:
    python experiments/appendix/robot_control/adaptive/run_slotine_li.py --out results/robot_control/check
Importing the module turns on float64 in JAX (jax_enable_x64, for the whole process) and defines the model, phi,
sequential, iterate and make_guess; it runs no rollout and no iteration.
"""
import argparse
import os
import time
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ===== model block, verbatim from the notebook (settings .. state_rhs, without fk_np and blue_color) =====
dt = 0.002
T = 18.0
t = jnp.arange(0.0, T, dt)
N = t.shape[0]

l1, l2 = 1.0, 0.85
lc1, lc2 = 0.5, 0.425
m1, m2 = 1.2, 0.9
I1, I2 = 0.08, 0.05
g = 9.81

Lambda = jnp.diag(jnp.array([5.0, 5.0]))
Kd = jnp.diag(jnp.array([18.0, 14.0]))
Gamma = 1.4


@jax.jit
def payload_mass(time):
    return (
        0.25
        + 0.55 * (0.5 + 0.5 * jnp.sin(0.55 * time + 0.8))
        + 0.15 * (time > 9.0)
    )


@jax.jit
def desired_xy(time):
    x = 0.75 + 0.22 * jnp.sin(1.4 * time) + 0.12 * jnp.sin(3.1 * time + 0.5)
    y = 0.55 + 0.24 * jnp.sin(2.2 * time + 1.2) + 0.08 * jnp.cos(4.0 * time)
    return jnp.array([x, y])


@jax.jit
def inverse_kinematics_xy(xy):
    x, y = xy
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    c2 = jnp.clip(c2, -0.98, 0.98)
    s2 = jnp.sqrt(1.0 - c2 * c2)

    q2 = jnp.arctan2(s2, c2)
    q1 = jnp.arctan2(y, x) - jnp.arctan2(l2 * s2, l1 + l2 * c2)
    return jnp.array([q1, q2])


xy_d = jax.vmap(desired_xy)(t)
qd = jax.vmap(inverse_kinematics_xy)(xy_d)

qd_dot = jnp.gradient(qd, dt, axis=0)
qd_ddot = jnp.gradient(qd_dot, dt, axis=0)


@jax.jit
def dynamics_matrices(q, qdot, mp):
    q1, q2 = q
    dq1, dq2 = qdot

    m2e = m2 + mp

    a = I1 + I2 + m1 * lc1**2 + m2e * (l1**2 + lc2**2)
    b = m2e * l1 * lc2
    d = I2 + m2e * lc2**2

    M = jnp.array([
        [a + 2.0 * b * jnp.cos(q2), d + b * jnp.cos(q2)],
        [d + b * jnp.cos(q2), d],
    ])

    C = jnp.array([
        [-b * jnp.sin(q2) * dq2, -b * jnp.sin(q2) * (dq1 + dq2)],
        [b * jnp.sin(q2) * dq1, 0.0],
    ])

    G = jnp.array([
        (m1 * lc1 + m2e * l1) * g * jnp.cos(q1)
        + m2e * lc2 * g * jnp.cos(q1 + q2),
        m2e * lc2 * g * jnp.cos(q1 + q2),
    ])

    return M, C, G


@jax.jit
def payload_regressor(q, qdot, v, a_ref):
    M0, C0, G0 = dynamics_matrices(q, qdot, 0.0)
    M1, C1, G1 = dynamics_matrices(q, qdot, 1.0)

    tau0 = M0 @ a_ref + C0 @ v + G0
    tau1 = M1 @ a_ref + C1 @ v + G1

    return tau1 - tau0


@jax.jit
def state_rhs(s, qd_k, qd_dot_k, qd_ddot_k, time_k):
    q = s[:2]
    qdot = s[2:4]
    mp_hat = s[4]

    mp_true = payload_mass(time_k)

    q_tilde = q - qd_k
    qdot_tilde = qdot - qd_dot_k

    qdot_r = qd_dot_k - Lambda @ q_tilde
    qddot_r = qd_ddot_k - Lambda @ qdot_tilde

    s_slide = qdot - qdot_r

    M0, C0, G0 = dynamics_matrices(q, qdot, 0.0)

    Yp = payload_regressor(q, qdot, qdot_r, qddot_r)

    tau = (
        M0 @ qddot_r
        + C0 @ qdot_r
        + G0
        + Yp * mp_hat
        - Kd @ s_slide
    )

    M, C, Gvec = dynamics_matrices(q, qdot, mp_true)

    qddot = jnp.linalg.solve(M, tau - C @ qdot - Gvec)

    mp_hat_dot = -Gamma * (Yp @ s_slide)

    return jnp.array([
        qdot[0],
        qdot[1],
        qddot[0],
        qddot[1],
        mp_hat_dot,
    ])

# ===== end of the verbatim block =====


def fk_tip(q):
    """End-effector position, the quantity the paper's figure draws."""
    return jnp.array([l1 * jnp.cos(q[0]) + l2 * jnp.cos(q[0] + q[1]),
                      l1 * jnp.sin(q[0]) + l2 * jnp.sin(q[0] + q[1])])


def phi(S, s0):
    """One explicit-Euler step applied to every time index (the notebook's fixed_point_update without relaxation)."""
    rhs_all = jax.vmap(state_rhs)(S[:-1], qd[:-1], qd_dot[:-1], qd_ddot[:-1], t[:-1])
    nxt = S[:-1] + dt * rhs_all
    nxt = nxt.at[:, 4].set(jnp.clip(nxt[:, 4], 0.0, 2.0))
    return jnp.concatenate([s0[None, :], nxt], axis=0)


@jax.jit
def sequential(s0):
    """The same recurrence, one step after the other."""
    def body(s, inp):
        qd_k, qd_dot_k, qd_ddot_k, t_k = inp
        s_next = s + dt * state_rhs(s, qd_k, qd_dot_k, qd_ddot_k, t_k)
        s_next = s_next.at[4].set(jnp.clip(s_next[4], 0.0, 2.0))
        return s_next, s_next
    _, tail = jax.lax.scan(body, s0, (qd[:-1], qd_dot[:-1], qd_ddot[:-1], t[:-1]))
    return jnp.concatenate([s0[None, :], tail], axis=0)


HIST_COLS = ("err_state", "err_joint", "err_tip", "err_mass", "update")


@partial(jax.jit, static_argnames=("num_iters",))
def iterate(S0, s0, S_seq, ckpt_iters, num_iters):
    """The plain Jacobi iteration S <- Phi(S) for num_iters + 1 sweeps; returns the last iterate, the iterates after
    ckpt_iters sweeps and hist.

    hist[i] = (errors of the iterate after i sweeps against the sequential rollout, size of sweep i + 1):
      err_state  max_n ||S_n - S*_n||_2          all five states
      err_joint  max_n ||q_n - q*_n||_2          joint angles, rad
      err_tip    max_n ||tip_n - tip*_n||_2      end-effector position, m
      err_mass   max_n |m_hat_n - m_hat*_n|      payload-mass estimate, kg
      update     max_n ||S^{i+1}_n - S^i_n||_2   what the notebook prints as "final residual"
    """
    tip_seq = jax.vmap(fk_tip)(S_seq[:, :2])

    def metrics(S):
        d = S - S_seq
        tip = jax.vmap(fk_tip)(S[:, :2])
        return jnp.stack([
            jnp.max(jnp.linalg.norm(d, axis=1)),
            jnp.max(jnp.linalg.norm(d[:, :2], axis=1)),
            jnp.max(jnp.linalg.norm(tip - tip_seq, axis=1)),
            jnp.max(jnp.abs(d[:, 4])),
        ])

    def body(carry, i):
        S, saved = carry
        saved = jnp.where((ckpt_iters == i)[:, None, None], S[None], saved)
        m = metrics(S)
        S_new = phi(S, s0)
        upd = jnp.max(jnp.linalg.norm(S_new - S, axis=1))
        return (S_new, saved), jnp.concatenate([m, upd[None]])

    saved0 = jnp.zeros((ckpt_iters.shape[0],) + S0.shape, S0.dtype)
    (S_fin, saved), hist = jax.lax.scan(body, (S0, saved0), jnp.arange(num_iters + 1))
    return S_fin, saved, hist


def make_guess(s0):
    """The figure cell of the notebook: reference + N(0, 0.35^2) noise, mass estimate 0.7 + noise."""
    noise = 0.35 * jax.random.normal(jax.random.PRNGKey(0), (N, 5))
    S0 = jnp.zeros((N, 5))
    S0 = S0.at[:, 0:2].set(qd + noise[:, 0:2])
    S0 = S0.at[:, 2:4].set(qd_dot + noise[:, 2:4])
    S0 = S0.at[:, 4].set(jnp.clip(0.7 + noise[:, 4], 0.0, 2.0))
    return S0.at[0].set(s0)


INITIAL_STATE = jnp.array([qd[0, 0] + 0.35, qd[0, 1] - 0.25, 0.0, 0.0, 0.1])     # the notebook's initial state
CKPT = (0, 500, 2000)                  # the sweeps the paper's figure draws


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/robot_control/check")
    p.add_argument("--sweeps", type=int, default=0, help="number of sweeps; 0 (default): N, as in the paper")
    args = p.parse_args()
    print("JAX devices:", jax.devices())
    print(f"N = {N}, dt = {dt}, dtype = {t.dtype}")
    s0 = INITIAL_STATE

    t0 = time.perf_counter()
    S_seq = jax.block_until_ready(sequential(s0))
    print(f"sequential rollout: {time.perf_counter() - t0:.2f} s (with compile), "
          f"finite: {bool(jnp.all(jnp.isfinite(S_seq)))}")
    fp_res = float(jnp.max(jnp.abs(phi(S_seq, s0) - S_seq)))
    print(f"fixed-point check of the sequential rollout, max |Phi(S*) - S*| = {fp_res:.3e}")
    assert fp_res < 1e-9, "the sequential rollout is not a fixed point of Phi"
    tip_dist = np.linalg.norm(np.asarray(jax.vmap(fk_tip)(S_seq[:, :2])) - np.asarray(xy_d), axis=1)
    print(f"sequential rollout against the reference path: max tip distance {np.max(tip_dist):.3e} m, "
          f"after t = 2 s {np.max(tip_dist[int(2.0 / dt):]):.3e} m")

    ckpt_iters = jnp.array(CKPT)
    out = dict(t=np.asarray(t), xy_d=np.asarray(xy_d), mass_true=np.asarray(jax.vmap(payload_mass)(t)),
               ckpt_iters=np.asarray(CKPT), hist_cols=np.array(HIST_COLS), s0=np.asarray(s0), N=N, dt=dt, x64=1)

    S0 = make_guess(s0)
    t0 = time.perf_counter()
    _, saved, hist = iterate(S0, s0, S_seq, ckpt_iters, args.sweeps or N)
    hist = np.asarray(jax.block_until_ready(hist))
    sec = time.perf_counter() - t0
    out["hist_ref_noise_r1"] = hist
    out["ckpt_ref_noise_r1"] = np.asarray(saved)

    def first_below(col, thr):
        ok = np.nan_to_num(hist[:, col], nan=np.inf) < thr
        return int(np.argmax(ok)) if ok.any() else -1

    print(f"\nJacobi iteration from the reference path plus noise, {len(hist) - 1} sweeps in {sec:.1f} s (with "
          "compile); distance to the sequential rollout after k sweeps:")
    for k in (100, 200, 500, 1000, 2000, N):
        if k < len(hist):
            print(f"  k = {k:5d}: end effector {hist[k, 2]:.2e} m, mass estimate {hist[k, 3]:.2e} kg")
    print(f"first sweep with the end-effector distance below 1e-3 m: {first_below(2, 1e-3)}, below 1e-6 m: "
          f"{first_below(2, 1e-6)}; with the state distance below 1e-6: {first_below(0, 1e-6)} "
          f"= {first_below(0, 1e-6) / N:.3f} N", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "fixed_point_check_fp64.npz")
    np.savez_compressed(path, **out)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
