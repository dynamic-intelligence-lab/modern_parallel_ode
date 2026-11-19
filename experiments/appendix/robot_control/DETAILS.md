# Robot control experiment: details

The short version is in [README.md](README.md). This file has every file, result file and command.

The appendix "Parallel-in-Time Robotic Control and Simulation" (label `subsec:robotic_control`, also
`app:robotic_control`) studies a planar two-link arm. Its details are in the appendix sections labelled
`app:neural_robot_controller` and `app:adaptive_robot`.

**Neural controller training (table `tab:controllers`).** The arm has point masses of 1 kg at the ends of two
links of 1 m, gravity 9.81 and joint damping 0.1 (`planar_arm(**ARM_2LINK)`). It tracks a Lissajous reference of
its joint angles around the hanging configuration q = (-pi/2, 0). The torque is a gravity-compensated PD term
scaled by the mass matrix, with kp = 3 and kd = 0.6, plus a learned residual torque from one of three controllers:

| paper | `--archs` name | network | parameters |
|---|---|---|---|
| MLP large | `mlp_large` | MLP 8-128-128-2 of the arm state and the reference | 17,922 |
| MLP small | `mlp_small` | MLP 8-64-32-2 | 2,722 |
| SSM | `ssm_d16_contract` | a hidden state of 16 dimensions that follows the contracting SSM cell, driven by the arm state and the reference, and an MLP readout 16-64-2 | 2,322 |

Each controller is trained with AdamW for 1200 steps, each on a batch of 4 initial conditions near the hanging
configuration (learning rate 1e-3 with a cosine schedule, weight decay 1e-4, gradient clipping at norm 10). The
loss is the integral of the squared joint-angle error plus 1e-3 times the integral of the squared learned torque,
both as sums over the time grid, averaged over the batch; the horizon is T = 10 s with N = 1000 steps of
dt = 0.01. Every training step solves the closed loop in parallel, from the zero guess, with DEER-exp (Newton's
method on the exponential-integrator discretization, `--solvers exp`) or Euler-DEER (on the explicit-Euler
discretization, `--solvers euler`). The Newton loop stops when the largest entry of the Newton update is at most
1e-5, or after 50 iterations, and the gradient comes from the implicit-function adjoint at the converged
trajectory. After training the driver times the training step (the
mean of 20 steps after 3 warm-up steps) against the same step through the sequential integrator of the same
discretization, differentiated by autodiff through `lax.scan`; the speedup is the ratio of the two means. The
final RMSE is that of the joint angles of one sequential RK4 rollout of the trained controller from the hanging
configuration at rest; the PD-only RMSE is the same rollout without the learned torque.

**Adaptive controller (figure `fig:slotine-li-fixedpoint-rollout`).** A second two-link arm (links of 1 m and
0.85 m with their centres of mass at half length, masses 1.2 kg and 0.9 kg, moments of inertia 0.08 and 0.05
kg m^2, gravity 9.81, no joint friction) carries a payload whose mass changes in time (a sinusoid between 0.25 kg
and 0.80 kg, plus 0.15 kg from t = 9 s on) and enters the mass matrix, the Coriolis and the gravity terms. Its end
effector follows a path in the plane, turned into joint angles by inverse kinematics, under a Slotine-Li adaptive
controller (Lambda = diag(5, 5), K_D = diag(18, 14), adaptation gain 1.4) whose estimate of the payload mass
follows the gradient rule and is clipped to [0, 2] kg; the reference velocities and accelerations are finite
differences of the joint path. The state (q1, q2, dq1, dq2, mass estimate) starts 0.35 and -0.25 rad off the
reference, at rest, with a mass estimate of 0.1 kg. The closed loop over T = 18 s is discretized by explicit
Euler with dt = 0.002, N = 9000 points. All N states are found at once by the Jacobi iteration S <- Phi(S), where
Phi applies one Euler step to every point, from the reference path plus Gaussian noise of standard deviation 0.35
(mass estimate 0.7 kg plus noise), in float64; after every sweep the iterate is compared with the sequential
rollout, the fixed point. The figure draws the iterates after 0, 500 and 2000 sweeps.

All commands are run from the repository root. With a CUDA build of JAX on a machine without a usable GPU, a JAX
process first logs `Jax plugin configuration error` with a traceback and then runs on the CPU; the message can be
ignored.

## Files

| file | what it does |
|---|---|
| `neural/run_controller_table.py` | The driver of the table. For each seed, controller and solver it trains, computes the final and the PD-only RMSE, times the training step and writes one row to the JSON file `--out`. `--seeds`, `--epochs` (optimizer steps), `--timing-runs`, `--archs` and `--solvers` shorten a run; their defaults are the paper's. The solvers are named `exp` (DEER-exp) and `euler` (Euler-DEER), the SSM controller `ssm_d16_contract`; `deer_K` in a row is the cap of 50 Newton iterations. |
| `neural/implicit_deer.py` | DEER-exp and Euler-DEER with the implicit-function adjoint (`exp_par_integrate_implicit`, `euler_par_integrate_implicit`), on the Newton steps of `parode`; they return the trajectory, the largest entry of the last Newton update and the number of iterations. It adapts `deer_implicit` of micro_deer (`THIRD_PARTY_NOTICES.md`, section 13). |
| `neural/sequential_solvers.py` | The sequential exponential integrator, and the sequential counterpart of each parallel solver: the baselines of the speedups. |
| `neural/ssm_policy.py` | The recurrent controller: the SSM cell with its readout and the closed loop; `SSMCellConfig()` is the paper's controller. |
| `neural/table_controllers.py` | Prints the table from the two result files, with the converged seeds and the Newton iterations of every converged run at the end of training. |
| `neural/slurm/run_controller_table.slurm` | Job template of the training runs. |
| `adaptive/run_slotine_li.py` | The Jacobi iteration of the adaptive-control rollout against the sequential rollout. Writes `fixed_point_check_fp64.npz` into `--out`: the time grid, the reference path, the true payload mass, the iterates after 0, 500 and 2000 sweeps and, for every sweep, the largest distances of the iterate to the sequential rollout (state, joint angles, end effector, mass estimate) and the size of the sweep's update. `--sweeps` shortens the run. The model block is copied from a notebook of the authors, which is not in this repository. Importing the module turns on float64 in JAX and runs no rollout. |
| `adaptive/figure_slotine_li.py` | Draws `slotine_li_jacobi_3panel.pdf` from that file and prints the first sweep from which the iterate is within 1e-6 of the sequential rollout. The plotting code is copied from the same notebook; its panel titles say "Optimization step" for a Jacobi sweep. |
| `neural/arms.py` | `planar_arm`, the arm's equations of motion from its Lagrangian, and `ARM_2LINK`, the arm of the table. The arm of the Slotine-Li rollout is defined in `adaptive/run_slotine_li.py`. |
| `neural/controller_utils.py` | `init_mlp`, `apply_mlp` and `make_lissajous_reference`. |
| `../shared/contracting_ssm_cell.py` | The contracting nonlinear SSM cell, copied from the research code of the contracting-SSM experiment without its unconstrained A and its uniform initial spread, which no experiment of the paper uses. The controller uses its sigmoid parametrizations of A and gamma. |
| `../../../tests/test_robot_control.py` | The tests of section 3. |

From `parode/`, the experiment runs `exp_parallel.py` (`_compute_Ab`, `_augmented_expm`, `_exp_step_and_phi`,
`linear_and_affine_terms`, `newton_step`), `explicit_parallel.py`
(`euler_step`, `affine_terms`, `linear_terms`, `combine_fc`, `par_scan`, `par_init`, `newton_step`),
`sequential.py` (`seq_integrate`), `sequential_integrators.py` (`euler`, `rk4`) and `ode_utils.py`;
`laser.py` is imported along the way (by `explicit_parallel.py`).

## Data

The experiment needs no external data. The result files behind the table and the figure ship with the
repository (3 files, 2.2 MB):

| path under `results/robot_control/` | written by | device | used for |
|---|---|---|---|
| `bench_table.json` | the driver, seed 0, on 2026-05-03 | NVIDIA B200 | seed 0 of every cell of the table |
| `bench_table_seeds1-4_kp3.json` | the driver, `--seeds 1,2,3,4 --kps 3`, on 2026-08-31 | NVIDIA B200 | seeds 1 to 4 |
| `fixed_point_check_fp64.npz` | the research form of `run_slotine_li.py`, on 2026-11-12, reduced to the paper's run | NVIDIA L40S | the Slotine-Li figure, the sweep count of the text |

The device is that of the job logs, which do not ship; the files record neither the device nor the code or
package versions they were produced with. A row holds the settings (`kp`, `kd`, `T`, `dt`, `N`, `deer_K`, `tol`),
`n_params`, `losses` (one per training step), `pd_only_rmse`, `final_rmse`, the mean and standard deviation of
the timed training steps, `train_speedup` and `par_solver_iters`, the Newton iterations of the parallel solve at
the trained parameters for the first initial condition of the batch.

- `bench_table.json` was written by an earlier form of the driver that had no options and ran kp = 1 and kp = 3.
  Its six rows at kp = 1, which the paper does not use, were removed for the release: the file is the original
  (389,786 bytes, sha256 `674dd1e5...`) without them, and the remaining rows are byte for byte as written. Its
  rows have no `seed` field; `table_controllers.py` reads them as seed 0.
- The rows of both files also hold `checkpoints`, diagnostics recorded every 100 training steps by the driver of
  that time (loss, tracking RMSE, Newton iterations, spectral radius, size of the learned torque). The driver no
  longer computes them and nothing reads them.
- `fixed_point_check_fp64.npz` holds the arrays of the paper's run cut from the 17 MB output of the research form
  of `run_slotine_li.py`, which made six runs (two relaxations, three guesses) and saved eight checkpoints each:
  the arrays are bit for bit those of that output. `run_slotine_li.py` writes the same arrays. The keys of the
  iterates and of the distances end in `_ref_noise_r1`: the guess (the reference plus noise) and the relaxation
  (1, plain Jacobi) of the paper's run among the six.
- `figs/robot_control/slotine_li_jacobi_3panel.pdf` is the paper's figure file, byte for byte, as the research
  form of `figure_slotine_li.py` drew it from the full output.
- A run whose loss became NaN during training stops and still reads `status` "converged", with `final_rmse` NaN,
  `n_epochs_completed` the steps it ran and `par_solver_iters` 50. A run whose loss is NaN at the initial
  parameters reads `status` "NaN_init" and has no other result. `table_controllers.py` counts a seed as converged
  when its `final_rmse` is finite.

## Paper artifacts

Levels: **A** recomputed for this release on a CPU in minutes and compared exactly or to a stated tolerance; **B**
re-tabulated from the result files that ship with the repository; **C** needs a GPU, the command is given and was
exercised at small scale only; **D** cannot be reproduced from this repository. Nothing in this experiment is
level D.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| Table `tab:controllers`: the PD-only row and the final RMSE column | `table_controllers.py` (section 1) | the two result files | CPU, under 1 s | B; the PD-only RMSE and the parameter counts are A |
| The same table: the two speedup columns | `table_controllers.py` (section 1) | the two result files | the timings: one NVIDIA B200 | B as a reading of the shipped rows; C as a measurement |
| The training runs behind the table | `neural/slurm/run_controller_table.slurm` (section 2) | none | one NVIDIA B200: 35 min for seed 0 (12 rows, kp = 1 and 3), 68 min for seeds 1 to 4 | C; the first two training steps of the six seed-0 rows and the failure at initialization of MLP small with Euler-DEER, seed 1, are A |
| Figure `fig:slotine-li-fixedpoint-rollout`, `slotine_li_jacobi_3panel.pdf` | `figure_slotine_li.py` (section 4) | `fixed_point_check_fp64.npz` | CPU, 2 s | B; the result file is A: `run_slotine_li.py` recomputes it in 40 s on a CPU |
| Numbers of the paragraph "Adaptive controller" and of the figure caption: N = 9000, the iterate coincides with the sequential rollout after about 0.2N sweeps, the panels at 0, 500 and 2000 sweeps | `run_slotine_li.py` and `figure_slotine_li.py` print the sweep (section 4) | `fixed_point_check_fp64.npz` | CPU | A |
| Numbers of the paragraph "Neural optimal controller" and of the table caption: the factor of 5 to 6 over the PD baseline (0.58 to about 0.10), 2.3k, 2.7k and 17.9k parameters, the gap of 0.012, 14-28x with Euler-DEER and 50-74x with DEER-exp, 6 to 32 Newton iterations, seven iterations of DEER-exp for the SSM, 5/5, 4/5 and 2/5 converged seeds, the settings of the caption | `table_controllers.py` (section 1), see below | the two result files | CPU, under 1 s | B |

How the numbers of the text follow from the output of `table_controllers.py`: the factors over the PD baseline
are 0.584 divided by the final RMSE of each controller (0.584/0.096 = 6.1, 0.584/0.117 = 5.0, 0.584/0.108 =
5.4); the gap of 0.012 is 0.108 - 0.096; the Euler-DEER column ranges from 13.6 to 28.1 and the DEER-exp column
from 50.4 to 73.5; the Newton iterations of the converged runs range from 6 to 32, and those of the SSM with
DEER-exp are 7 for every seed. The settings of the caption and of the text are recorded in every row that ran:
`tol` 1e-5, `deer_K` 50 (the cap of Newton iterations), `kp` 3, `kd` 0.6, `dt` 0.01, `T` 10, `N` 1000; and
`n_epochs_completed` is 1200 for every converged run.

## Commands

`make smoke` runs two training steps of MLP small (both solvers) and of the SSM with Euler-DEER (section 2) and
2000 sweeps of the Slotine-Li iteration (section 4); `make figures` prints the table (section 1) and redraws the
Slotine-Li figure from the shipped file (section 4).

### 1. The table

```sh
python experiments/appendix/robot_control/neural/table_controllers.py
```

prints

```
Controller                 & Final RMSE & Speedup (DEER-exp) & Speedup (Euler-DEER)
PD only                    & 0.584 & -- & --
MLP large (17.9k params)   & 0.096 +- 0.001 & 66.6x +- 20.6 & 13.6x +- 1.8
MLP small (2.7k params)    & 0.117 +- 0.001 & 73.5x +- 20.6 & 19.8x +- 7.3
SSM (2.3k params)          & 0.108 +- 0.001 & 50.4x +- 0.8 & 28.1x +- 2.5
MLP large: converged seeds [0, 1, 2, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: DEER-exp [9, 10, 22, 24], Euler-DEER [23, 24, 25, 32]
MLP small: converged seeds [0, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: DEER-exp [9, 17], Euler-DEER [9, 21]
SSM: converged seeds [0, 1, 2, 3, 4] of [0, 1, 2, 3, 4]; Newton iterations at the end of training: DEER-exp [7, 7, 7, 7, 7], Euler-DEER [6, 6, 7, 7, 7]
```

Each cell is the mean and the sample standard deviation (ddof = 1) over the converged seeds of 0 to 4 at kp = 3;
the final RMSE is that of the controller trained with DEER-exp. The same seeds converge with both solvers, which
the script asserts. A path as argument reads the two files from another directory.

### 2. The training runs

The paper's rows come from two jobs of the driver on one NVIDIA B200. Fill in `YOUR_ACCOUNT` and
`YOUR_GPU_PARTITION` in the template first; its two arguments are the seeds and the output file.

```sh
mkdir -p logs
E=NONE,HOME=$HOME,USER=$USER,PATH=$PATH
S=experiments/appendix/robot_control/neural/slurm/run_controller_table.slurm
sbatch --export=$E $S 0 results/robot_control/check/bench_table.json
sbatch --export=$E $S 1,2,3,4 results/robot_control/check/bench_table_seeds1-4_kp3.json
python experiments/appendix/robot_control/neural/table_controllers.py results/robot_control/check     # after both jobs
```

Without arguments the template runs seeds 0 to 4 into `results/robot_control/check/bench_table_rerun.json`, and so
does the driver without arguments: a rerun does not overwrite the shipped files.

Two training steps of the six seed-0 rows on a CPU:

```sh
export JAX_PLATFORMS=cpu
python experiments/appendix/robot_control/neural/run_controller_table.py --seeds 0 --epochs 2 --timing-runs 1 \
    --out results/robot_control/check/seed0.json
python experiments/appendix/robot_control/neural/run_controller_table.py --seeds 1 --epochs 2 --timing-runs 1 \
    --archs mlp_small --solvers euler --out results/robot_control/check/nan_init.json
```

The first command took 6 min on four CPU cores, with a peak of 8.1 GB of memory; the two lines of `make smoke`,
MLP small with both solvers and the SSM with Euler-DEER, took 75 s and 26 s with peaks of 0.9 and 0.7 GB. The
`losses` of each row are its first two training losses; they equal the first two `losses` of the seed-0 rows of
`bench_table.json` to 2.5x10^-15 relative, and `pd_only_rmse` and `n_params` agree as well. With `--epochs`
other than 1200 the cosine schedule differs from the second update on, so later losses, the final RMSE and
`par_solver_iters` are not comparable with the shipped rows. The speedups on
a CPU are below 1 (0.13 to 0.50 in this run) and say nothing about the GPU timings. The second command stops at
the initial parameters with `status` "NaN_init", as the shipped row of that seed does (10 s).

### 3. Tests

```sh
JAX_PLATFORMS=cpu python -m pytest -q tests/test_robot_control.py
```

runs in about a minute on four CPU cores:

- DEER-exp and Euler-DEER with the implicit-function adjoint against the sequential integrator of the same
  discretization, for MLP small and the SSM at their initial parameters, at N = 100 with a stopping tolerance of
  1e-10: the trajectories agree to 1e-12 and the gradients of the tracking loss agree with autodiff through the
  sequential integrator to 1e-12 relative (measured: at most 9.4x10^-15 and 5.0x10^-15);
- the table recomputed from the shipped files equals the printed cells, the converged seeds and the Newton
  iterations of section 1;
- 2000 sweeps of the Slotine-Li iteration against the shipped `fixed_point_check_fp64.npz`: the iterates after 0,
  500 and 2000 sweeps and the distances of every sweep agree to 1e-11 (measured: at most 6.1x10^-13; the file was
  written on an NVIDIA L40S), and the iterate is within 1e-6 of the sequential rollout from sweep 1827 on.

### 4. The Slotine-Li figure

```sh
python experiments/appendix/robot_control/adaptive/figure_slotine_li.py
python tools/compare_pdf.py
```

draws `figs/robot_control/slotine_li_jacobi_3panel.pdf` from the shipped file and compares the PDF with the
committed one by text and pixels (`git checkout -- figs/robot_control` restores the committed bytes). The first
command prints that the iterate is within 1e-6 of the sequential rollout from sweep 1827 = 0.203 N on. The
distance is the largest Euclidean distance over the 9000 points of the five-dimensional state.

The result file recomputed on a CPU, into `results/robot_control/check/`:

```sh
JAX_PLATFORMS=cpu python experiments/appendix/robot_control/adaptive/run_slotine_li.py --out results/robot_control/check
python experiments/appendix/robot_control/adaptive/figure_slotine_li.py results/robot_control/check results/robot_control/check
python tools/compare_pdf.py results/robot_control/check/slotine_li_jacobi_3panel.pdf \
    figs/robot_control/slotine_li_jacobi_3panel.pdf
```

The first command took 40 s on four CPU cores (0.3 GB) and prints, among other lines,

```
  k =   500: end effector 6.08e-02 m, mass estimate 4.43e-01 kg
  k =  2000: end effector 2.72e-08 m, mass estimate 4.86e-08 kg
  k =  9000: end effector 5.12e-16 m, mass estimate 6.66e-16 kg
first sweep with the end-effector distance below 1e-3 m: 950, below 1e-6 m: 1649; with the state distance below 1e-6: 1827 = 0.203 N
```

The distances are the largest over the 9000 points. `--sweeps 2000`, the line of `make smoke`, stops after the
third panel (12 s) and gives the same figure.
