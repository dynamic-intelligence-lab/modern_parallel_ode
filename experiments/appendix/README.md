# Experiments of the paper's appendix

The paper's main experiments are in [`../lorenz/`](../lorenz/) and [`../elcd/`](../elcd/). Its appendix has four more:

- [`gpu_scaling/`](gpu_scaling/): on a contractive RNN, the plain Jacobi iteration is the fastest parallel solver
  on six GPU types, and on long rollouts it scales to several GPUs.
- [`contracting_ssm/`](contracting_ssm/): a contracting state-space model is trained in parallel on sequences of
  17,984 steps. Without the term that makes it contracting, the parallel solvers mostly fail to converge.
- [`robot_control/`](robot_control/): on a two-link arm, neural controllers are trained through a parallel solve,
  and an adaptive controller is simulated by Jacobi sweeps.
- [`euler_step_size/`](euler_step_size/): with a fixed number of Jacobi sweeps, forward Euler has a best step size.

`shared/` holds the state-space cell that `contracting_ssm/` and `robot_control/` both use.
