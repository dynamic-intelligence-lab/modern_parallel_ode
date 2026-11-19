"""Parallel-in-time ODE solvers: the Newton (DEER) iteration with its stabilized variants (LASER, ELK).

Two families of solvers, with different conventions.

ODE solvers, for x' = ode(t, x[, u]) on a grid of step dt, stopping on a tolerance tol:
  explicit_parallel   DEER with an explicit Euler or RK4 step (explicit_par_integrate), and LASER through its
                      argument laser=LaserConfig()
  laser               LASER's damped Newton loop (run_damped_newton), its configuration and the exponents it uses
  quasi_parallel      quasi-DEER with the exact diagonal of the Euler step Jacobians (quasi_par_integrate)
  exp_parallel        the building blocks of DEER with an exponential step; the loop that drives them is
                      experiments/appendix/robot_control/neural/implicit_deer.py
  elk_baseline        ELK, the DEER step damped by a Kalman filter, dense and diagonal (elk_integrate)
  implicit            a forward solve by plain DEER, guarded DEER, LASER, scale-ELK or ELK with an implicit-function
                      adjoint, for training through the solve
  sequential          the sequential Euler or RK4 rollout (seq_integrate) with the steps of sequential_integrators

Discrete-map solvers, for y[n+1] = func(y[n], driver[n], params) (jacobi: f(y[n], driver[n])):
  deer_map            quasi-DEER with a Hutchinson estimate of the diagonal (deer_map_solve) and full DEER
                      (deer_map_solve_full); they stop on a relative change of the iterate, with tolerances set by
                      the dtype
  jacobi              the Jacobi iteration (jacobi_alg), which stops on a merit function given as an argument; the
                      Jacobi solver of the contracting-SSM experiment (the GPU-scaling experiment has its own, in
                      experiments/appendix/gpu_scaling/)

ode_utils holds the wrapper of the ODE's signature and the input lookup that the ODE solvers share.

The names a new problem needs are importable from the package: explicit_par_integrate, LaserConfig, implicit_solve
and seq_integrate (quickstart.py, examples/own_ode.py). Importing any module of the package imports JAX: set the
JAX_* environment variables before it.
"""
from parode.explicit_parallel import explicit_par_integrate
from parode.implicit import implicit_solve
from parode.laser import LaserConfig
from parode.sequential import seq_integrate

__all__ = ["LaserConfig", "explicit_par_integrate", "implicit_solve", "seq_integrate"]
