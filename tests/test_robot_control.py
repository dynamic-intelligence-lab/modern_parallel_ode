"""Tests of the robot-control experiment (CPU, about a minute).

1. The parallel solvers of the table (DEER-exp and Euler-DEER with the implicit-function adjoint,
   experiments/appendix/robot_control/neural/implicit_deer.py) against the sequential integrator of the same
   discretization: the trajectory, and the gradient of the tracking loss against autodiff through the sequential
   integrator, for the MLP-small and the SSM controller at their initial parameters, kp = 3, N = 100.
2. The table of the paper (tab:controllers) recomputed from the shipped result files equals its printed cells.
3. The Jacobi iteration of the Slotine-Li rollout (run_slotine_li.py), 2000 sweeps, against the shipped result file
   of the figure (fig:slotine-li-fixedpoint-rollout).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import experiments.appendix.robot_control.neural.run_controller_table as rct  # noqa: E402  (sets jax_enable_x64)
from experiments.appendix.robot_control.adaptive import run_slotine_li as sl  # noqa: E402
from experiments.appendix.robot_control.neural.sequential_solvers import seq_integrate_dispatch  # noqa: E402
from experiments.appendix.robot_control.neural.ssm_policy import (  # noqa: E402
    SSMCellConfig, init_ssm_policy, make_ssm_ode_cl)
from experiments.appendix.robot_control.neural.table_controllers import table  # noqa: E402
from experiments.appendix.robot_control.neural.arms import ARM_2LINK, planar_arm  # noqa: E402
from experiments.appendix.robot_control.neural.controller_utils import init_mlp, make_lissajous_reference  # noqa: E402

DT, N, KP, KD = 0.01, 100, 3.0, 0.6
MAX_ITER, TOL = 50, 1e-10
TRAJ_ATOL = 1e-12        # measured on x86-64 Linux, JAX 0.6.2: at most 9.4e-15
GRAD_RTOL = 1e-12        # measured: at most 5.0e-15
SLOTINE_ATOL = 1e-11     # measured against the shipped file (written on an NVIDIA L40S): at most 6.1e-13


def _problem(arch):
    """Closed loop, initial parameters (seed 0) and first initial condition of the paper's training batch."""
    arm_ode, _, M_fn, V_fn, _ = planar_arm(**ARM_2LINK)
    q_ref_fn, qdot_ref_fn, q_hang = make_lissajous_reference(omega1=2.0, amp1=0.5, amp2=0.4)
    q0 = q_hang + 0.05 * jax.random.normal(jax.random.PRNGKey(42), (4, 2))[0]
    if arch == "mlp_small":
        factory = rct._build_mlp_factory(KP, KD, q_ref_fn, qdot_ref_fn, arm_ode, M_fn, V_fn, 2)
        params = init_mlp(jax.random.PRNGKey(0), [8, 64, 32, 2])
        z0 = jnp.concatenate([q0, jnp.zeros(2)])
    else:
        cfg = SSMCellConfig()          # the paper's SSM controller: D_ssm 16, sigmoid A and gamma
        factory = make_ssm_ode_cl(arm_ode, M_fn, V_fn, q_ref_fn, qdot_ref_fn, KP, KD, cfg)
        params = init_ssm_policy(jax.random.PRNGKey(0), cfg)
        z0 = jnp.concatenate([q0, jnp.zeros(2), jnp.zeros(16)])
    q_refs = jax.vmap(q_ref_fn)(jnp.arange(N + 1) * DT)
    return factory, params, z0, lambda traj: jnp.sum((traj[:, :2] - q_refs) ** 2) * DT


@pytest.mark.parametrize("arch", ["mlp_small", "ssm"])
@pytest.mark.parametrize("solver", ["exp", "euler"])
def test_parallel_solver_equals_sequential(arch, solver):
    factory, params, z0, loss = _problem(arch)
    guess = jnp.zeros((N, z0.size))

    def par(p):
        return rct.PAR_SOLVERS[solver](factory, p, z0, guess, DT, MAX_ITER, 0.0, TOL)

    def seq(p):
        return seq_integrate_dispatch(factory(p), z0, DT, N, solver)

    traj_par, update, iters = jax.jit(par)(params)
    traj_seq = jax.jit(seq)(params)
    assert int(iters) < MAX_ITER and float(update) <= TOL
    assert float(jnp.max(jnp.abs(traj_par - traj_seq))) < TRAJ_ATOL
    g_par = jax.jit(jax.grad(lambda p: loss(par(p)[0])))(params)
    g_seq = jax.jit(jax.grad(lambda p: loss(seq(p))))(params)
    flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree.leaves(g)])
    rel = jnp.linalg.norm(flat(g_par) - flat(g_seq)) / jnp.linalg.norm(flat(g_seq))
    assert float(rel) < GRAD_RTOL


def test_table_from_shipped_results():
    cells, notes = table(os.path.join(ROOT, "results", "robot_control"))
    assert cells == [
        ["PD only", "0.584", "--", "--"],
        ["MLP large (17.9k params)", "0.096 +- 0.001", "66.6x +- 20.6", "13.6x +- 1.8"],
        ["MLP small (2.7k params)", "0.117 +- 0.001", "73.5x +- 20.6", "19.8x +- 7.3"],
        ["SSM (2.3k params)", "0.108 +- 0.001", "50.4x +- 0.8", "28.1x +- 2.5"],
    ]
    assert notes == [
        "MLP large: converged seeds [0, 1, 2, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: "
        "DEER-exp [9, 10, 22, 24], Euler-DEER [23, 24, 25, 32]",
        "MLP small: converged seeds [0, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: "
        "DEER-exp [9, 17], Euler-DEER [9, 21]",
        "SSM: converged seeds [0, 1, 2, 3, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: "
        "DEER-exp [7, 7, 7, 7, 7], Euler-DEER [6, 6, 7, 7, 7]",
    ]


def test_slotine_li_iterates_equal_the_shipped_file():
    """The iterates after 0, 500 and 2000 sweeps (the figure's panels) and the distances of every sweep to the
    sequential rollout equal those of results/robot_control/fixed_point_check_fp64.npz; the iterate is within 1e-6 of
    the sequential rollout from sweep 1827 = 0.203 N on (the 0.2N of the paper's text)."""
    ref = np.load(os.path.join(ROOT, "results", "robot_control", "fixed_point_check_fp64.npz"))
    s0 = sl.INITIAL_STATE
    S_seq = sl.sequential(s0)
    _, saved, hist = sl.iterate(sl.make_guess(s0), s0, S_seq, jnp.array(sl.CKPT), 2000)
    assert tuple(ref["ckpt_iters"]) == sl.CKPT and int(ref["N"]) == sl.N
    np.testing.assert_allclose(np.asarray(saved), ref["ckpt_ref_noise_r1"], rtol=0, atol=SLOTINE_ATOL)
    np.testing.assert_allclose(np.asarray(hist), ref["hist_ref_noise_r1"][:2001], rtol=0, atol=SLOTINE_ATOL)
    below = np.asarray(hist)[:, 0] < 1e-6
    assert int(np.argmax(below)) == 1827
