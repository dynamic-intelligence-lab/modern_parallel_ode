# Contracting state-space model on EigenWorms: details

The short version is in [README.md](README.md). This file has every file, result file and command.

A sequence classifier whose layers are a continuous-time diagonal state-space cell,

    dh/dt = -gamma(u) * h + A(u) * tanh(h) + tanh(MLP_b(u)),

with A(u) in [-alpha_max, -alpha_min] and gamma(u) in [gamma_min, gamma_max] (sigmoid heads; alpha_max is the
second value of `--init_alpha_range`, whose first value the sigmoid heads do not use), discretized by
explicit Euler with dt = 1. In every training step each layer's recurrence over the whole sequence is solved by
one of three solvers: the sequential loop, Jacobi iterations or quasi-DEER (Newton's method with the diagonal of
the step Jacobian and a parallel scan). The experiment trains the model on EigenWorms (5 classes, 6 channels,
17,984 time steps) with each solver and compares accuracy and the wall time per epoch, sweeping the leak floor
gamma_min, and repeats the training without the gamma head (the cell without the leak term).

In the paper this is the appendix subsection "Parallel Training of Contractive State-Space Models" (label
`subsec:contracting_ssm`) with figure `fig:ssm_exp` (`figs/fig_ssm.pdf` in the paper's source; here
`figs/contracting_ssm/ssm_ablation_bars.pdf`, its panel B only) and table `tab:eigenworms_gamma_min`, and the
appendix "Architecture and Training Details" (label `app:contracting_ssm_setup`) with table
`tab:contracting_ssm_hparams`. The cell module `../shared/contracting_ssm_cell.py` is also the cell of the
recurrent controller of the robot-control experiment.

All commands are run from the repository root. The training runs of the paper took 29.8 GPU hours; their records
(Slurm logs and wandb run folders) are not in the repository. What ships is the table of their results,
`results/contracting_ssm/runs.csv`, from which the table and panel B of the figure are drawn again. For this
release the training code was run on a CPU at small scale and for the first two epochs at full length.

## Solver names

| paper | `--solver` | what runs in the forward pass |
|---|---|---|
| sequential | `seq` | `parode.sequential.seq_integrate`: the Euler recurrence in a `lax.scan` |
| Jacobi | `jacobi` | `parode.jacobi.jacobi_alg`: every state is replaced by the Euler step of the previous state of the current iterate, until the largest residual max_n \|h_n - h_{n-1} - dt f(h_{n-1}, u_n)\| of the iterate is below `--tol` or `--max_iter_jacobi` sweeps were made |
| quasi-DEER | `quasi` | `parode.quasi_parallel.quasi_par_integrate`: Newton steps on the Euler recurrence in which each step Jacobian is replaced by its diagonal, computed exactly with one forward-mode product per state coordinate, and solved by a parallel scan; it stops after the first step that starts from an iterate whose largest residual is below `--tol`, or after `--max_iter_newton` steps |

Both parallel solvers start from the zero trajectory. The backward pass is the same for the three (`solve` in
`cell_solvers.py`, a `jax.custom_vjp`): the adjoint of the Euler recurrence along the trajectory that the solver
returned, solved by a parallel scan over the diagonal one-step propagators, followed by the gradient of every Euler
step with respect to the parameters and the inputs.

## What a run does

`run_train.py` parses the options and calls `training.train`:

- `eigenworms_data.load_eigenworms` reads the three splits (train 181, validation 39, test 39 sequences of shape
  17,984 x 6), z-scores every channel with the mean and standard deviation of the training split, and keeps every
  `--decimate_stride`-th time step (1 in the paper's runs; 64 gives 281 steps for the small-scale commands below).
- `stacked_model.init_model` builds the model: an encoder (6 -> D -> D, tanh), `--num_layers` blocks of LayerNorm
  (eps 1e-5), the cell solve, a GELU feed-forward layer of width `--ffn_mult` x D and a residual connection, then
  the mean over time and a linear readout to 5 classes. The cell (`../shared/contracting_ssm_cell.py`) reads the
  normalized stream as its input u; its A and gamma heads share one tanh hidden layer of width D.
- Training: cross-entropy, AdamW (optax) with global gradient clipping, a warmup-cosine learning-rate schedule with
  min(`--warmup_steps`, steps / 4) warmup steps over `--epochs` epochs, batches of `--batch_size` sequences in a
  random order every epoch (the last incomplete batch is dropped: 22 steps per epoch with batches of 8).
- After every epoch a validation pass records the loss and accuracy and, per solver call (sequence x layer), the
  iteration count, the residual and the one-step Jacobian of the returned trajectory. Training stops early after
  `--patience` epochs without a higher validation accuracy. The test split is evaluated with the parameters of the
  epoch with the highest validation accuracy.
- Every epoch prints one line; the run ends with a `TEST (best val)` line and `FINAL:`. With `--wandb_project` (off
  by default) the same numbers are logged to wandb; the paper's runs logged them.

The numbers that the table and the figure use, as the code computes them:

| name in `runs.csv` | definition |
|---|---|
| `test_acc` | the mean over the test batches (8, 8, 8, 8 and 7 sequences) of the share of correctly classified sequences |
| `wall_time_per_epoch_s` | the wall time of the last epoch of the run: its training steps and its validation pass |
| `converged_frac` | in the last validation pass, the share of solver calls (sequence x layer) that stopped before the iteration cap, averaged over the validation batches. For `seq` it is 0, because the sequential solve counts one iteration and its cap is 1 |
| `iters_median` | the median iteration count of the solver calls of the last validation pass, averaged over the batches |
| `epochs_completed` | the number of epochs the run trained before it stopped |

## Files

| file | what it does |
|---|---|
| `run_train.py` | The command line of one training run; `--help` lists the options. The defaults are the values of the paper's runs, apart from gamma_min (default 0.01), which the sweep varies; the submit scripts (`slurm/submit_gamma_min_sweep.sh`, `slurm/submit_ablation.sh`) still pass every value. |
| `training.py` | `train`: the training loop, the validation and test passes, the printed lines and the optional wandb logging. |
| `cell_solvers.py` | `solve` (forward solve by `seq`, `jacobi` or `quasi`, and the custom_vjp backward pass) and `solve_diagnostics` (the same forward solve with its iteration count, residual and Jacobian statistics). |
| `stacked_model.py` | The encoder, the residual blocks and the readout. |
| `../shared/contracting_ssm_cell.py` | The cell: parameters, initialization, A(u), gamma(u), the forcing and the right-hand side. |
| `eigenworms_data.py` | Loads, normalizes and decimates the data; the batch iterator. |
| `loss.py` | Cross-entropy and accuracy. |
| `naming.py` | The wandb run name. |
| `download_data.sh` | Downloads the data and checks its SHA-256. |
| `export_runs.py` | Reads the run records (Slurm logs and local wandb run folders) and writes a runs table, by default `results/contracting_ssm/check/runs.csv`; from the paper's records it gives the shipped `runs.csv`. |
| `table_gamma_min.py` | Prints the rows of `tab:eigenworms_gamma_min` from `runs.csv`. |
| `figure_ablation.py` | Draws panel B of `fig:ssm_exp` from `runs.csv` into `figs/contracting_ssm/ssm_ablation_bars.pdf`. |
| `slurm/train.slurm` | Job template of one run (fill in `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION`). |
| `slurm/submit_gamma_min_sweep.sh` | Submits the 75 runs of the table: 3 solvers x gamma_min in {0.01, 0.02, 0.05, 0.1, 0.2} x seeds 0 to 4. `DRY_RUN=1` prints the `sbatch` lines. |
| `slurm/submit_ablation.sh` | Submits the 4 runs without the gamma head: quasi and jacobi x seeds 0 and 1, iteration caps 1,000 and 5,000. |

From `parode/`, the runs use `sequential.py` and `sequential_integrators.py` (`seq`), `jacobi.py` (`jacobi`),
`quasi_parallel.py` (`quasi`, and its `combine_fc` in the adjoint scan of all three), `explicit_parallel.py`
(`euler_step` and `affine_terms`, which `quasi_parallel.py` imports; `laser.py` is imported along the way) and
`ode_utils.py`.

The tests are `tests/test_contracting_ssm_*.py` (part of `make test`): agreement of the three solvers with the
sequential Euler solution and of the custom_vjp gradient with autodiff through a sequential `lax.scan`, among them
the paper's configuration in float64 (`test_custom_vjp_paper_configuration`); the bounds of A and gamma and the
contraction floor of the leak; shapes; and `test_contracting_ssm_table.py`, which recomputes the 65 cells of
`tab:eigenworms_gamma_min` and the 23 of 25 pairs of the text from `runs.csv`.

## Data

EigenWorms (Yemini et al., 2013) in the train, validation and test split of Morrill et al. (2021), as the DEER
repository (Lim et al., `https://github.com/machine-discovery/deer`, BSD 3-Clause licence) ships it: three pickles
of PyTorch tensors in `experiments/04_rnn_eigenworms/neuralrde_split`, at commit `409140f`.

```sh
bash experiments/appendix/contracting_ssm/download_data.sh
```

downloads them into `data/eigenworms` (107 MB, git-ignored) and checks their SHA-256, which are those of the files
the paper's runs read; a second call only checks. `EIGENWORMS_DIR` points the code to another folder. Reading the
pickles needs PyTorch; the CPU build is enough:

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

(`pip install -e ".[eigenworms]"` names the same version; from PyPI it is the CUDA build. torch's own
dependencies are not pinned in `requirements-lock.txt`.) The requirements that wandb saved for each of the 79 runs
list torch 2.8.0, JAX 0.6.2 and wandb 0.25.1.

## Result files

`results/contracting_ssm/runs.csv` (79 rows, 36 kB) was written by `export_runs.py` from the records of the
paper's runs. One row per run:

| rows | `group` | solvers | device |
|---|---|---|---|
| 75 | `gamma_min_sweep` | seq, quasi, jacobi; gamma_min 0.01, 0.02, 0.05, 0.1, 0.2; seeds 0 to 4 | NVIDIA B200 |
| 2 | `ablation` | quasi, seeds 0 and 1, without the gamma head | NVIDIA GeForce RTX 3090 |
| 2 | `ablation_b200` | jacobi, seeds 0 and 1, without the gamma head | NVIDIA B200 |

Columns: `group`, `solver`, `gamma_param`, `gamma_min` (empty without the gamma head), `seed`, `max_iter_newton`,
`max_iter_jacobi`, `gpu` (from the run's wandb metadata), the five numbers defined above, and `args`, the run's
command line without the wandb group. The labels of `group` come from the wandb group names of the records:
`v2_gamma_min_sweep_<date>_<user>` for the paper's sweep (the submit script of this repository writes
`gamma_min_sweep_<date>_<user>`; `export_runs.py` reads both), `ablation_stability_<date>_<user>` for the runs
without the gamma head. The jacobi runs without the gamma head were run twice with the same arguments, the second
time on an NVIDIA B200 under a group name set outside the submit script, `ablation_stability_b200_<date>_<user>`;
`export_runs.py` keeps these and leaves out the first ones, and the paper's panel B is drawn from them (the panel
drawn from `runs.csv` equals it). Runs of the records that the paper does not use (an ablation with an unconstrained
A, earlier sweeps) are left out.

`figs/contracting_ssm/ssm_ablation_bars.pdf` is the Matplotlib file of panel B as the paper's figure received it;
`figure_ablation.py` draws it again.

## Paper artifacts

Levels: **A** recomputed for this release on a CPU and compared exactly or to a stated tolerance; **B** redrawn or
re-tabulated from the result file that ships; **C** needs GPU hours or the run records, the command is given and
was exercised at small scale; **D** cannot be reproduced from this repository.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| `tab:eigenworms_gamma_min`, test accuracy (mean of the three solvers' means over 5 seeds; the +- is the standard deviation of the `seq` runs) | `table_gamma_min.py` | `runs.csv` | CPU, under 1 s | B |
| `tab:eigenworms_gamma_min`, wall time per epoch (mean +- standard deviation over 5 seeds) and speedups (`seq` time over parallel time, paired by seed, mean +- standard deviation) | `table_gamma_min.py` | `runs.csv` | the same | B as a reading of `runs.csv`; C to measure again (one NVIDIA B200 per run) |
| `tab:contracting_ssm_hparams` | none; the values are the command lines of the runs (`args` of `runs.csv`, the submit scripts) and the defaults of `run_train.py` and the code | `runs.csv` | - | B |
| `fig:ssm_exp` panel B, the Matplotlib file (`ssm_ablation_bars.pdf`) | `figure_ablation.py` | `runs.csv` | CPU, 3 s | B |
| `fig:ssm_exp` as printed (`figs/fig_ssm.pdf`): panel A, the architecture diagram, and panel B with its two axes placed one above the other | none | - | - | D |
| Numbers of the text: test accuracies of the three solvers equal in 23 of 25 (gamma_min, seed) pairs; the range of the row accuracies (83.1 to 85.4 %); the Jacobi speedup growing from 4.2x to 36.0x with gamma_min | `table_gamma_min.py`, `tests/test_contracting_ssm_table.py` | `runs.csv` | CPU, under 1 s | B |
| N = 17,984 time steps, 5 classes | `download_data.sh`, then `load_eigenworms` (the first line a run prints) | the data | CPU, seconds | A |
| rho <= 0.99 at (dt, alpha_max, gamma_max, gamma_min) = (1, 1.5, 0.4, 0.01) | arithmetic of the runs' values: max(1 - 0.01, \|1 - 1.9\|); the runs print the largest one-step propagator (`jac=` mean/p99/max) | - | - | A |
| The 79 training runs | `slurm/submit_gamma_min_sweep.sh`, `slurm/submit_ablation.sh`, then `export_runs.py` | the data | 29.8 GPU hours: the 75 sweep runs 23.0 on NVIDIA B200 (`seq` 19.4, longest run 1.0 h; `jacobi` 2.7; `quasi` 0.9), the 4 runs without the gamma head 6.8 (quasi 3.2 on NVIDIA GeForce RTX 3090, jacobi 3.7 on NVIDIA B200) | C; the first two epochs of three runs at full length are A on a CPU (section 3) |

The GPU hours are the wandb run times of the 79 records, which include start-up, compilation, every epoch and the
test pass.

## Commands

### 1. Data

```sh
bash experiments/appendix/contracting_ssm/download_data.sh
```

### 2. The table and panel B from the shipped results

```sh
python experiments/appendix/contracting_ssm/table_gamma_min.py
python experiments/appendix/contracting_ssm/figure_ablation.py
python tools/compare_pdf.py
```

The first command prints the five rows of the table as they are in the LaTeX source, and a note for the row whose
three solver mean accuracies differ by 0.5 points or more:

```
    0.01 & 84.8 & 5.2 & 51.2 & 5.9 & 0.56 & 0.04 & 12.12 & 1.02 & 91$\times$ & 8.7 & 4.2$\times$ & 0.4 \\
    0.02 & 85.4 & 6.8 & 45.8 & 1.5 & 0.52 & 0.03 & 7.83 & 0.38 & 89$\times$ & 6.0 & 5.9$\times$ & 0.3 \\
    0.05 & 85.0 & 6.6 & 49.5 & 3.6 & 0.46 & 0.02 & 4.06 & 0.07 & 108$\times$ & 9.2 & 12.2$\times$ & 1.0 \\
    0.10 & 84.8 & 5.5 & 50.2 & 6.0 & 0.42 & 0.02 & 2.25 & 0.06 & 121$\times$ & 17.2 & 22.3$\times$ & 3.1 \\
    0.20 & 83.1 & 8.2 & 45.0 & 1.0 & 0.39 & 0.01 & 1.25 & 0.02 & 117$\times$ & 4.2 & 36.0$\times$ & 1.0 \\
note: gamma_min 0.02: mean accuracies seq 85.71, quasi 85.71, jacobi 84.71
```

The second overwrites `figs/contracting_ssm/ssm_ablation_bars.pdf` and prints the bars: "stable" are the quasi
and jacobi runs of the sweep at gamma_min = 0.05 (5 seeds, caps 200 and 1,000), "unstable" the runs without the
gamma head (2 seeds, caps 1,000 and 5,000); the bars are the mean, the error bars the standard deviation over
seeds. The third compares the PDF with the committed one by text and pixels; `make figures` runs the second and
the third. `git checkout -- figs/contracting_ssm` restores the committed bytes.

### 3. Training on a CPU at small scale

```sh
export JAX_PLATFORMS=cpu
ABLATION="--gamma_param off --max_iter_newton 1000 --max_iter_jacobi 5000"
for s in seq quasi jacobi; do
    python experiments/appendix/contracting_ssm/run_train.py --solver $s --gamma_min 0.05 --decimate_stride 64 --epochs 2
done
for s in quasi jacobi; do
    python experiments/appendix/contracting_ssm/run_train.py --solver $s $ABLATION --decimate_stride 64 --epochs 2
done
```

The defaults of `run_train.py` are the values of the paper's runs, seed 0 included; the commands add the paper's
gamma_min = 0.05, or the options of the runs without the gamma head, and run on every 64th time step (T = 281) for
two epochs; each run takes 20 to 35 s on four CPU cores. With the gamma head the three solvers print the same losses
and accuracies to the printed digits (training loss 1.5735 and 1.3786, validation accuracy 0.539, test accuracy
0.386) and test losses that agree to 7e-7 (1.5392017 for `seq`, 1.5392018 for `quasi`, 1.5392024 for `jacobi`). In
the validation passes the quartiles and the largest iteration count of the solves are 5, 6, 6 and 7 for quasi-DEER
and 38 to 39, 45, 56 and 99 to 100 for Jacobi. Without the gamma head the training loss is 12.98 and 4.45 and the
largest one-step propagator is 1.000; at T = 281 both solvers still reach the tolerance (Jacobi in at most T
sweeps), so this shows the code path and not the failure to converge of the paper's runs at full length. The last
digits of the losses can differ on another machine.

The paper's runs at full length on a CPU: replace `--decimate_stride 64 --epochs 2` by `--patience 0`. The default
60 epochs keep the paper's learning-rate schedule, and `--patience 0` ends the run after the first epoch whose
validation accuracy is not higher than the best one; for seed 0 at gamma_min = 0.05 that is after two epochs. These
runs take about 4, 7 and 10 minutes for `seq`, `quasi` and `jacobi` on four CPU cores and up to 3.2 GB of memory.

### 4. The paper's runs on GPUs

Before submitting: `export_runs.py` below reads the wandb summary and metadata of each run from the local wandb run
folder and finds that folder through the run URL that wandb writes into the job's `.err` file. It therefore needs
runs made with wandb logging on and online: add `--wandb_project <project>` to `COMMON` in the two submit scripts
and `pip install wandb` (the paper's runs used wandb 0.25.1). Without wandb a run's numbers are only in its printed
lines.

```sh
mkdir -p logs
bash experiments/appendix/contracting_ssm/slurm/submit_gamma_min_sweep.sh     # the 75 runs of the table
bash experiments/appendix/contracting_ssm/slurm/submit_ablation.sh            # the 4 runs without the gamma head
```

Fill in `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION` in `slurm/train.slurm` first; the table's runs used one NVIDIA B200
each. `DRY_RUN=1` in front prints the `sbatch` lines; their arguments are those of the `args` column of
`runs.csv`, row for row, without `--init_g_mode hippo` and `--pool mean` (options with a single value that the
code no longer has). Each job writes `logs/contracting_ssm-<job id>.out` with an `args:` line and the printed
lines. To turn the records into a runs table:

```sh
python experiments/appendix/contracting_ssm/export_runs.py logs wandb      # writes results/contracting_ssm/check/runs.csv
python experiments/appendix/contracting_ssm/table_gamma_min.py results/contracting_ssm/check/runs.csv
python experiments/appendix/contracting_ssm/figure_ablation.py results/contracting_ssm/check/runs.csv
```

The figure script always writes `figs/contracting_ssm/ssm_ablation_bars.pdf` (`git checkout -- figs/contracting_ssm`
restores it).
