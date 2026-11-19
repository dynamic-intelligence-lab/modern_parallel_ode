# ELCD experiment: details

The short version is in [README.md](README.md). This file has every file, result file and command.

Extended Linearized Contracting Dynamics (ELCD; Jaffe et al., NeurIPS 2024) is a latent vector field
dz/dt = A(z)(z - z*) with A + A^T <= -2 alpha I, composed with a learned diffeomorphism to the data space. The
experiment trains ELCD models with a multi-start rollout loss: at every optimizer step, 64 rollouts per training
demonstration (192 in all) start at recorded states and are compared with the demonstration at matched times.
The rollouts are solved by a parallel-in-time Newton solver (DEER, scale-ELK, ELK or LASER, with the gradient
from an implicit-function adjoint) or by the sequential RK4 loop with backpropagation. The experiment compares
the training time and the failed solves of the solvers, and the rollout loss with the velocity loss of Jaffe et
al.

There are two tasks: `chain8z`, eight LASA handwriting shapes chained into 7 demonstrations of N = 7,689
samples each (D = 2), and `npend:8-N5000`, an eight-link pendulum (N = 5,000, D = 16). The appendix repeats the
comparison of the two losses on the original tasks of Jaffe et al. (`lasa:<Shape>` for the 30 LASA shapes and
`pend:8`).

In the paper this is the section "Stable Imitation Learning" (label `subsec:learned_contracting_model`) and the
appendix "Training ELCD on Long Rollouts" (label `app:elcd`).

The paper's runs took about 1,200 GPU hours. Their records (about 4.5 GB) are not in the repository, and the
runs were not repeated for this release. The repository has the code, the run lists, one table and one cache
derived from the records, and the figures as this code wrote them. For this release, the first optimizer steps
of paper runs were repeated on a CPU and compared with the records, and the figures and tables were regenerated
from the records.

All commands are run from the repository root. LASER is in `parode/laser.py`; the solvers take it as
`laser=LaserConfig()`, whose defaults are the paper's configuration.

## Names

| paper | code |
|---|---|
| DEER | `--solver deer` |
| scale-ELK with rate g | `--solver scale-elk:<g>` |
| ELK with constant sigma^2 (Levenberg-Marquardt damping 1/sigma^2) | `--solver elk:<sigma^2>` |
| LASER | `--solver laser`, which is `parode.laser.LaserConfig()` |
| sequential solver | `--solver seq`; in the records `loop_full` (all 10,000 steps) and `loop_time<n>` (n timed steps) |
| multi-start rollout loss | `--loss suffix --starts 64` |
| velocity loss | `--loss vel` |
| fallback step | a step whose log entry has `seq_fallback` set |
| training time | the sum of `wall` over the log entries of a record; evaluations are not in it |
| Chamfer distance, multi-start RMSE | the summary keys `*_chamfer_native_*` and `*_multistart_rmse_*` of an evaluation |
| chunk C | `--start-chunk` of the driver; `--par-chunk` and `--loop-chunk` of the cell runner |
| chain8z, pendulum, fp32, fp64 | `--system chain8z`, `--system npend:8-N5000`, `--dtype fp32`, `--dtype fp64` |

## Files

Used for the paper's runs, figures and numbers:

| file | what it does |
|---|---|
| `model.py` | The latent field of ELCD: A(z), the field and its Jacobian. A JAX port of the model of Jaffe et al. |
| `flow_gatherfree.py` | The diffeomorphism, a rational-quadratic spline coupling flow, ported to JAX without gather operations: `phi` (encode) and `psi` (decode). |
| `solvers.py` | `seq_scan`, the sequential RK4 rollout of every evaluation (`train/metrics.py`); its other functions serve `schematic/export_chain8z.py`. |
| `train/train_elcd.py` | The driver. One process trains one model on one task with one loss and one solver, and writes a JSON record and the parameters (`.npz`); `load_model` reads the parameters back (the figure builders, `sym_jacobian_positive.py` and the schematic export call it). `--help` lists every option and the docstring explains them; the options of the recipe of panel A's model (`--x0-anchor`, `--anchor-ramp`, `--start-draw`) are off by default. |
| `train/losses.py` | The velocity loss, the multi-start ("suffix") loss, the solver names, and the sequential re-solve of a step whose parallel solves failed (`solve_starts_fallback`). |
| `train/metrics.py` | The evaluation of a model on every demonstration: Chamfer distances (`chamfer` and `truncate` re-implement the distance and the stopping rule of Jaffe et al.), matched-time errors, the multi-start RMSE. |
| `train/data.py` | The tasks (`load_system`): LASA shapes, `chain8z` (eight shapes chained in the order recorded in `CHAIN_Z`), the pendulum data of Jaffe et al. `python experiments/elcd/train/data.py` repeats the search behind `CHAIN_Z`. |
| `train/pendulum.py` | The damped n-link pendulum: the simulation, `generate_npend` (the demonstrations of `npend:8-N5000`) and the fit of the model to the data of Jaffe et al. (`fit_paper`; `python experiments/elcd/train/pendulum.py` repeats the fits behind `PAPER_MODEL`). |
| `train/run_cell.py` | One cell of the paper runs (task, precision, seed): calls the driver once per solver in an order that rotates over the seeds, with calibration probes in between, writes `cell.json`, and skips runs whose record is complete. |
| `train/run_init_eval.py` | The evaluation of a cell's initial model (`init_eval.json`). |
| `train/select_chunk.py`, `train/select_scale_elk_rate.py` | Read the pilot and the tuning records and pick the chunk C and the rate g of scale-ELK (the settings table of the appendix). |
| `train/summarize_records.py` | One row per record of a records root, as CSV (`slurm/summarize_records.slurm`). On `appendix/` its rows give the table of training times on the original tasks (`tab:elcd_lasa_speed`). |
| `train/sym_jacobian_positive.py` | The share of the training samples at which the symmetric part of the latent Jacobian has a positive eigenvalue (appendix, "Contraction of ELCD"). |
| `configs/cells/*.txt` | The run lists: one argument line per run or per cell (see "How a paper run is started"). |
| `slurm/` | Job templates. `run_cell.slurm` (one cell per array task), `train_velocity_loss.slurm` (the velocity-loss lists), `train_array.slurm` (the long runs), `run_init_eval.slurm`, `summarize_records.slurm`, `table_training_time.slurm`, `figure_appendix.slurm`. Each header gives the `sbatch` line. Their job names (`-J`), which name the log files, differ from the file names. |
| `record_names.py` | LASER's name in the paper's records, mapped to `laser` for every reader of records (standard library only). |
| `figures/records.py` | Loaders and tables over a records root (NumPy and pandas). `ELCD_PAPER_ROOT` sets the root. |
| `figures/table_training_time.py` | `hours_table` (hours, speedup, fallback steps and final Chamfer distance per run); its command line writes `hours_table.csv` and prints these numbers per seed. Also the order and the labels of the methods and `save`, which the other builders use. |
| `figures/panel_data.py`, `figures/loss_comparison.py`, `figures/long_runs.py` | `load_parts`, the inputs of the ELCD figure (the hours table, the rollouts of the best seed of each loss, the long runs); the velocity records and the rollouts of trained models; the tables of the long runs. |
| `figures/figure_appendix.py` | The four appendix figures. |
| `figures/build_figure_elcd_notebook.py`, `figures/figure_elcd.ipynb` | The builder writes the notebook; the notebook draws every part of the ELCD figure and composes `figs/elcd/elcd_figure_traj_left.pdf`. Edit the builder, not the notebook. |
| `schematic/make_schematic_C.py`, `make_schematic.py`, `arm3d.py`, `make_nn_spaces.py` | Draw panel A of the ELCD figure as SVG and PDF from `schematic/schematic_data.npz` and the LASA dataset (need `rsvg-convert` and Ghostscript); `make_schematic.py` holds the projection and the colours the others share. They import each other by file name, so they run as scripts (the notebook runs them in their folder); the C of `make_schematic_C.py` is a variant letter of the drawing, not panel C. |
| `schematic/export_chain8z.py` | Wrote `schematic_data.npz`, from which panel A is drawn: one LASER solve of a rollout of a trained model. The script and the driver options of that model's recipe (`--x0-anchor`, `--anchor-ramp`, `--start-draw`) are kept to document how the file was made; rerunning the script needs that model, an earlier fp64 run, and its checkpoint at step 10,000 of 30,000, which are not in the repository. |
| `../../tools/run_figure_notebooks.slurm` | Builds and executes the figure notebooks; a bash script that also runs without Slurm. |
| `../../figs/colors.json` | The colors of the methods and the losses. |

From `parode/`, the paper's runs use `implicit.py` (the forward solve, the stopping rule and the adjoint),
`laser.py` (LASER), `explicit_parallel.py` (the parallel scan) and `elk_baseline.py` (ELK); `implicit.py` also
imports `sequential.py` and `ode_utils.py`, and `elk_baseline.py` imports `quasi_parallel.py`.

In the repository because a test imports them; the paper's results do not use them:

| file | why it is here |
|---|---|
| `../../tests/reference_flow.py`, `../../tests/reference_systems.py` | The first port of the flow, with index gathers, and the data-space field on it: the reference of `tests/test_elcd_flow_gatherfree.py`, which checks the values of `flow_gatherfree.py` against them. `tests/test_elcd_flow.py` tests the two files themselves. |

## Data

| task | files | source |
|---|---|---|
| `chain8z`, `lasa:<Shape>` | `data/lasa/DataSet/*.mat`, 30 files | The LASA handwriting dataset (Khansari-Zadeh and Billard, 2011), `https://bitbucket.org/khansari/lasahandwritingdataset`, commit `38304f7c0ac4708b0fd38331d94d02095ad0ccfd` |
| `pend:8` | `data/pendulum/n-8.npz` | The pendulum data of Jaffe et al., `https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics` (MIT licence), commit `87c171f428c93e199da3eb32a60e03bcaa940d93`, folder `data/pendulum`. The paper's runs do not read `n-2.npz` and `n-4.npz`; the fit `python experiments/elcd/train/pendulum.py` does. |
| `npend:8-N5000` | `data/npend/8-N5000.npz` | Generated by `pendulum.generate_npend('8-N5000')`: unit point masses on unit links, 7 initial conditions from a fixed seed, RK4, damping calibrated to b = 4.8914. NumPy only, no input file. |

```sh
bash experiments/elcd/download_data.sh
```

clones LASA at that commit (21 MB), downloads the three pendulum files (3.3 MB), simulates
`data/npend/8-N5000.npz` (9 MB, about 4 CPU minutes) and checks SHA-256 digests. A second call repeats nothing.
`data/` is git-ignored. The LASA dataset is free for non-commercial academic use and must not be modified or
distributed without the permission of its authors (`data/lasa/Readme.txt`), so the dataset is not in this
repository. The cache of the seed-trajectories figure and the data file of panel A hold model rollouts (each
starts at a recorded state) and constants of the task (the goal, the segment boundaries, the normalization), and
no demonstration; the scripts that draw from them read the demonstrations from the dataset.

The code reads `data/lasa/DataSet`, `data/pendulum` and `data/npend` relative to the working directory;
`ELCD_LASA_DIR`, `ELCD_PENDULUM_DIR` and `ELCD_NPEND_DIR` override them.

## Records and result files

A run writes `<name>.json` and `<name>.npz` (the parameters) at `--out`, without it under `results/elcd/check/train/`
with `<name>` the run name the driver builds from the options. The record holds `config` (every option of the driver,
and `init` "scratch" and `diag_every` 0, the fixed values of two options that were removed), `data` (the task: N, D,
dt, segments, normalization), `device`, `device_kind`, `jax`, `trainable` (the number of trained parameters),
`provenance` (host, Slurm ids, GPU state, git commit, hashes of the modules; at the start and at the end), `suffix`
(rollout length, solves per step, chunk, warm-start buffer), `solve` (cap, tolerance, fallback, the measured cost of
a sequential re-solve), `log` (one entry per optimizer step: loss, gradient norm, `wall`, converged solves,
iterations, failed solves by cause, `seq_fallback`, LASER's `retries`, warm-start statistics, learning rate; the
velocity loss logs one entry per epoch, and the long LASER run one per 100 steps, `--log-every 100`), `evals` (per
evaluation: the step, one row per demonstration, the summary, the seconds it took), `final` (the last summary),
`train_seconds` and `total_seconds`. With `--time-only` a run has no evaluation and no `.npz`.

The cell runner writes to `<recdir>/<system>/<dtype>/s<seed>/<item>.json`, with `<item>` one of `deer`,
`laser`, `scale-elk<g>`, `elk<s2>`, `loop_time<n>`, `loop_full`, plus `probe_*.json`, `cell*.json`,
`gpu_<item>.csv` and snapshots `<item>_at<step>.{json,npz}`. The paper's records name LASER `laser-lle`, its
earlier name, in their file names and fields, and so do the strings `model` and `ckpt` of the shipped
`schematic/schematic_data.npz`, which give the research paths of panel A's two models; every reader of records maps
it to `laser` in one place, `experiments/elcd/record_names.py`. The module names and paths in the `provenance` of
the paper's records are those of the research code, not of this repository. The run lists put the records under
`results/elcd/records/{main,velocity,long,appendix,tune,elk_time,pilot*}`; the rerun behind a number of the
paper (scale-ELK, pendulum, fp64, seed 2) is under `rerun_slow/` (see "How a paper run is started"). The figure
builders read the main records under `ELCD_PAPER_ROOT` (default `results/elcd/records/main`) and `velocity/`,
`long/` and `appendix/` beside it.

What ships:

| file | content | used for |
|---|---|---|
| `results/elcd/hours_table.csv` | 124 rows: task (`chain8z`, `npend-8-N5000`), precision, seed, solver (`seq`, `deer`, `scale-elk<g>`, `laser`), training hours, speedup over the sequential solver, fallback steps, final training Chamfer distance. Written by `table_training_time.py` from the main records, with the record of scale-ELK on the pendulum, fp64, seed 2 replaced by its rerun in `rerun_slow/`, as in the paper: line 99 has the rerun's 8.250 h (over `main/` alone the builder writes the first run's 11.860 h there); every other row is the same either way. | the three tables of training times, the numbers of the text that derive from them |
| `figs/elcd/appendix/seed_trajectories.pkl` | A pickle written by `figure_appendix.py`: for the ten seeds of each loss, the final training Chamfer distance and the rollouts of the trained model from five states of the first chained demonstration. The demonstration is not in the file; the builder reads it from the LASA dataset. The file sits beside the figures because `--cached` reads it from the `--out` folder. | the seed-trajectories figure |
| `figs/elcd/appendix/{mechanism_chain8z_fp32_seed0,mechanism_pendulum_fp32_seed0,seed_trajectories_chain8z_fp32,original_tasks_loss_fp32}.pdf` | The four appendix figures as the builder wrote them, under the file names of the paper's source (`figs/elcd_appendix/`). | reference copies |
| `figs/elcd/elcd_figure_traj_left.pdf` | The ELCD figure of the main text as the notebook wrote it. | reference copy |
| `schematic/schematic_data.npz` | The solve of panel A: the solution, the LASER iterates, the errors and the field on a grid. The training demonstrations are not in the file; `make_schematic_C.py` rebuilds them from the LASA dataset. | panel A |

## How a paper run is started

A run list has one argument line per array task. Three scripts read the lines:

| lines | script | Slurm template |
|---|---|---|
| with `--recdir` and `--solvers` | `train/run_cell.py`, which calls the driver once per item | `slurm/run_cell.slurm` |
| with `--loss` and `--out` (the velocity loss, the long runs) | `train/train_elcd.py` | `slurm/train_velocity_loss.slurm`, `slurm/train_array.slurm` |
| `init_eval.txt` | `train/run_init_eval.py` | `slurm/run_init_eval.slurm` |

Line 11 of `main_chain8z.txt` is

```
--system chain8z --dtype fp32 --seed 0 --solvers deer,scale-elk:0.01,scale-elk:0.03 --save-at 6000 --recdir results/elcd/records/main --full-loop --par-chunk 32
```

- `--system`, `--dtype`, `--seed`: the cell. The seed sets the initial weights and the draws of the starts.
- `--solvers`: the parallel solvers. Each is one driver process with the protocol of the paper:
  `--loss suffix --starts 64 --lr-schedule cosine --steps 10000 --eval-every 500 --fallback seq --max-iter 0
  --tol 1e-5` (`1e-7` in fp64). `--max-iter 0` caps a solve at its own length N - 1;
  `--fallback seq` re-solves a step with the sequential solver when one of its solves has not converged.
- `--par-chunk 32`: the driver's `--start-chunk`, the solves per vmapped chunk. `--loop-chunk` is the same for
  the sequential solver.
- The sequential solver is timed over 200 steps by default (`loop_time200.json`; `--loop-time 0`: not at all).
  `--full-loop` adds its run over all 10,000 steps, last (`loop_full.json`).
- `--save-at 6000`: an evaluation and the model at step 6,000, besides the evaluations every 500 steps.
- `--recdir`: the records root. `--cell-tag T` names the summary of a second job on the same cell `cellT.json`.
- `--time-only n` stops every parallel solver after n steps, without evaluation (pilots, tuning, ELK timing).
- Before the first item and after every item the runner times 20 steps of the sequential solver and of DEER on a
  fixed workload (chain8z, fp64, seed 0); `--no-probes` leaves this out.
- The runner refuses a repository with modified tracked files (`--allow-dirty`) and stops when a run did not
  use the GPU (`--allow-cpu`). A run whose record is complete is skipped, so a resubmitted task runs what is
  missing.

`python experiments/elcd/train/run_cell.py $(sed -n 11p experiments/elcd/configs/cells/main_chain8z.txt) --dry-run`
prints the driver arguments of every item of that cell without running anything.

Run lists and paper results (lines in parentheses):

| paper result | run lists | records |
|---|---|---|
| chain8z: panel C and the rollout-loss model of panel B of the ELCD figure, the two chain8z tables, the mechanism figure, the final Chamfer distances per solver | `main_chain8z.txt` (20: DEER, scale-ELK at g = 0.01 and 0.03, 200 timed steps of the sequential solver; fp64 seeds 0 to 9, then fp32; the full sequential run on seeds 0 to 2; panel C shows g = 0.03 only, `MAIN_DROP` of `table_training_time.py`), `main_chain8z_laser.txt` (20: LASER), `main_chain8z_loop_s3-9.txt` (14: the full sequential runs of seeds 3 to 9) | `main/chain8z` |
| pendulum: its table and its mechanism figure | `main_npend.txt` (6), `main_npend_laser.txt` (6), `main_npend_loop.txt` (1), `main_npend_loop_fp32.txt` (1), `main_npend_loop_s12.txt` (4) | `main/npend-8-N5000` |
| pendulum, fp64, seed 2: the training time of scale-ELK in its table (8.250 h) | no run list: the driver arguments of scale-ELK on line 3 of `main_npend.txt` (`run_cell.py ... --dry-run` prints them), with `--out` under `rerun_slow/` (1; the first run, in `main/`, took 11.860 h, 355 of its steps 10 to 84 s each, not counting the compiling first step) | `rerun_slow/npend-8-N5000` |
| velocity loss on chain8z: the table of the two losses, the velocity model of panel B, the seed-trajectories figure | `velocity_chain8z.txt` (32), `velocity_chain8z_cos_s3-9.txt` (7) | `velocity/chain8z` |
| initial models: first row of the table of the two losses | `init_eval.txt` (50) | `init_eval.json` in the cells of `main/` and `appendix/` |
| long runs: the curves of panel B, the paragraph "Long runs" | `long_chain8z.txt` (2: LASER for 10^6 steps, the velocity loss for 43,290 epochs) | `long/chain8z/fp32/s1` |
| original tasks: the figure of the two losses and the table of training times | `appendix_speed.txt` (24: DEER, LASER and 200 timed sequential steps on `lasa:{Sshape,JShape,Angle}` and `pend:8`), `appendix_loss.txt` (162: LASER on the other 27 shapes), `appendix_vel.txt` (186: the velocity loss on all 31 tasks), `appendix_pend_loop192.txt` (3: `pend:8` in fp64 again, the sequential solver with all 192 rollouts in one chunk; the lines also list `deer,laser`, whose items are those of `appendix_speed.txt` and are skipped when their records are complete) | `appendix/` |
| settings table, rate g | `tune.txt` (8), `tune_g01.txt` (4), `tune_g003.txt` (4), `tune_g01_npend.txt` (4), `tune_g003_npend.txt` (2), `tune_split_s11.txt` (3: line 6 of `tune.txt`, one job per rate), lines 2 to 5 of `next_c1_g001.txt`; seeds 10 and 11, 2,000 steps | `tune/` |
| settings table, chunk C | `pilot.txt` (12), `chunk.txt` (8), `chunk_c2.txt` (1), `chunk_npend.txt` (2), line 1 of `next_c1_g001.txt` | `pilot*/` |
| ELK's time per step | `elk_time.txt` (2; line 1 is the fp32 timing of the text) | `elk_time/` |

Hardware and time of the paper's runs, read from the records (`device_kind`, `jax`, and the sum of
`total_seconds`, which covers compilation, the optimizer steps and the evaluations):

| records | runs | GPU | hours |
|---|---|---|---|
| `main/` | 150: 98 with a parallel solver, 26 full and 26 timed sequential runs | NVIDIA B200 | 769, of which the 26 full sequential runs 294 |
| `rerun_slow/` | 1 | NVIDIA B200 | 9 |
| `velocity/` | 39 | NVIDIA B200 | 28 |
| `long/` | 2 | NVIDIA B200 | 47 |
| `appendix/` | 420 | the 72 timed records (`appendix_speed.txt`) on NVIDIA B200, the others on 11 GPU models | 219 |
| `tune/` | 33 | NVIDIA B200 | 53 |
| `elk_time/` | 6 | NVIDIA B200 | 36 |
| the calibration probes of `main/`, `appendix/` and `tune/` | 765 | NVIDIA B200 | 48 |

These records were written with JAX 0.6.2. The templates ask for one GPU, 4 CPU cores and 24 GB of host memory
per task; the header of `slurm/run_cell.slurm` gives the time limits of the main and the appendix lists
(4 days for `main_chain8z.txt`, 6 days for `main_npend.txt`). The training hours of single runs are in
`hours_table.csv`: on chain8z 0.24 to 0.51 h with a parallel solver and 9.6 to 10.7 h with the sequential solver
in fp32, 2.5 to 3.6 h and 9.2 to 10.0 h in fp64. Evaluations come on top and can take longer than the training:
the record of LASER on chain8z, fp64, seed 0 has 3.04 h of steps and 18.0 h in total.

## Paper artifacts

Levels: **A** recomputed for this release on a CPU in minutes and compared exactly or to the stated tolerance;
**B** redrawn or re-tabulated from the result files that ship with the repository; **C** needs GPU hours or the
run records, which are not in the repository: the command is given; the training runs were exercised at small
scale only, and the commands that read the records were run on the records for this release (the times in the
table are from those runs); **D** cannot be reproduced from this repository. The commands are in the next
section.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| ELCD figure of the main text, `elcd_figure_traj_left.pdf` (label `fig:elcd`) | the notebook `figures/figure_elcd.ipynb` (section 4) | records `main/` and `velocity/` (drawn: `chain8z/fp32`), `long/chain8z/fp32/s1`; LASA; `schematic/schematic_data.npz`; `figs/colors.json`; `rsvg-convert`, Ghostscript | CPU, 1 min 38 s and 0.8 GB; the template asks for 16 GB and 45 min for all figure notebooks | C |
| Panel A of that figure alone (the rollout and the iterate cards; the notebook adds the encoder and decoder drawing of `make_nn_spaces.py`, which reads no data), and its numbers (7 iterations; one iteration costs as much as 33 sequential steps; sample 232 of 2,737) | `schematic/make_schematic_C.py` (section 4), with the script's defaults: the whole panel with four cards; the notebook draws the desk and the cards apart, with three cards (`A_OVERRIDES` of `build_figure_elcd_notebook.py`) | `schematic/schematic_data.npz`, LASA | CPU, 6 s | B; recomputing the data file is D |
| Tables of training time and fallback steps per seed (labels `tab:elcd_chain8z_fp32`, `tab:elcd_chain8z_fp64`, `tab:elcd_pendulum`); speedups, medians and final Chamfer distances per solver in the text | a pandas pivot of the CSV (section 4); the median, minimum and maximum rows are taken over the ten seeds | `hours_table.csv` | CPU, seconds | B; the CSV itself is C (`table_training_time.py` over `main/` with the one record of `rerun_slow/` in place, 1 min) |
| Settings table (label `tab:elcd_settings`) | the tolerance is `TOL` in `run_cell.py` and C is `--par-chunk` in the main run lists; g is picked by `select_scale_elk_rate.py`, C by `select_chunk.py` | `run_cell.py`, run lists; records `tune/`, `pilot*/` | CPU, seconds | B for the values as run; the choice of C and g is C |
| Mechanism figure of chain8z, `mechanism_chain8z_fp32_seed0.pdf` (label `fig:elcd_mechanism`) | `figure_appendix mechanism` | records `main/chain8z/fp32/s0/{deer,laser}.json` | CPU, seconds | C |
| Mechanism figure of the pendulum, `mechanism_pendulum_fp32_seed0.pdf` (label `fig:elcd_pendulum_mechanism`) | `figure_appendix pendulum` | records `main/npend-8-N5000/fp32/s0/{deer,laser,loop_full}.json` | CPU, seconds | C |
| Seed-trajectories figure, `seed_trajectories_chain8z_fp32.pdf` (label `fig:elcd_seed_trajectories`) | `figure_appendix trajectories --cached` | `seed_trajectories.pkl`, LASA, `figs/colors.json` | CPU, 7 s | B; without `--cached` C (the 20 models of `main/` and `velocity/`, LASA; 2 min and 0.7 GB; the template asks for 16 GB and 20 min) |
| Figure of the two losses on the original tasks, `original_tasks_loss_fp32.pdf` (label `fig:elcd_lasa_loss`) | `figure_appendix original` | records `appendix/` (372 runs) | CPU, 30 s | C |
| Table of the two losses on chain8z (label `tab:elcd_velocity`) | no script writes it: medians over seeds 0 to 9 of `train_chamfer_native_mean`, `heldout_chamfer_native_mean` and `train_multistart_rmse_mean` in the last evaluation's summary, and of the summed `wall` of the log | records `main/chain8z/fp32/s*/laser.json`, `velocity/chain8z/fp32/s*/vel-*.json`, `init_eval.json`; `hours_table.csv` for the time of the rollout loss | CPU, seconds | C |
| Table of training times on the original tasks (label `tab:elcd_lasa_speed`) | `train/summarize_records.py` (section 4) writes one row per record: the table has the medians over seeds 0 to 2 of `step_hours` (DEER, LASER) and the maxima of `fallback_steps` (DEER, LASER); the sequential time is `compile_s` plus 9,999 times `median_s` of the 200-step run. `table_training_time` prints the same hours and fallback steps per seed | records `appendix/` | CPU, 35 s | C |
| The numbers of the paragraph "Long runs" (the notebook draws the curves of panel B from the same evaluations) | `long_runs` | records `long/chain8z/fp32/s1`, `main/`, `velocity/` | CPU, 1 s | C |
| Share of training samples with a positive eigenvalue (22 to 85 %, appendix "Contraction of ELCD") | `train/sym_jacobian_positive.py fp32 laser` | the ten models `results/elcd/records/main/chain8z/fp32/s*/laser.npz` (this path is fixed in the script); LASA | CPU, 14 s | C |
| Tuning (33 runs, 53 GPU hours) and ELK's time per step | `train/select_scale_elk_rate.py <tune root>`; the logs of `elk_time/` | records | CPU, seconds | C |
| Facts about the tasks and the model: N = 7,689 samples and 38.4 s on chain8z, the chaining order, N = 5,000 and D = 16 on the pendulum with b = 4.89, 9,608 and 32,428 parameters | printed by the driver at the start of a run; `download_data.sh` for the pendulum | LASA; none for the pendulum | CPU, seconds; 5 min for the pendulum | A |
| The runs themselves | `slurm/run_cell.slurm` and the other templates (section 3) | LASA, the pendulum files | the table of the previous section | C; the first steps of four runs are A |

## Commands

On a machine without a GPU, set `JAX_PLATFORMS=cpu`. The CUDA build of JAX then still logs one plugin error at
start-up and runs on the CPU. The commands of sections 2 and 4 write under `results/elcd/check/`, which is
git-ignored; the figure notebook of section 4 redraws `figs/elcd/elcd_figure_traj_left.pdf` in place (see there how
to restore it).

### 1. Data

```sh
bash experiments/elcd/download_data.sh
```

### 2. Small runs on a CPU

A run that needs no downloaded data: three steps with LASER on a generated two-link pendulum (N = 400, D = 4),
with the protocol of the paper and its 192 solves per step.

```sh
export ELCD_NPEND_DIR=results/elcd/check/npend
python -c "from experiments.elcd.train.pendulum import generate_npend; print(generate_npend('2-N400-b2'))"
JAX_PLATFORMS=cpu python experiments/elcd/train/train_elcd.py --system npend:2-N400-b2 --dtype fp64 --seed 0 \
    --loss suffix --starts 64 --lr-schedule cosine --steps 3 --solver laser --fallback seq \
    --max-iter 0 --tol 1e-7 --out results/elcd/check/a/laser.json
unset ELCD_NPEND_DIR
```

chain8z in fp64 with the driver arguments of line 1 of `main_chain8z_laser.txt` (LASER, seed 0), three steps, 4
starts per demonstration instead of 64 (12 solves per step), and the final evaluation only:

```sh
JAX_PLATFORMS=cpu python experiments/elcd/train/train_elcd.py --system chain8z --dtype fp64 --seed 0 \
    --loss suffix --starts 4 --lr-schedule cosine --steps 3 --solver laser --eval-every 0 \
    --fallback seq --max-iter 0 --tol 1e-7 --start-chunk 2 --out results/elcd/check/b/laser.json
```

The evaluation of chain8z took 3.4 to 11.6 minutes on a CPU (4 idle cores and 3 shared ones). `--time-only <n>` stops
a run after n steps without an evaluation.

The same cell through the cell runner, with the 192 solves per step of the paper, one step, no evaluation:

```sh
python experiments/elcd/train/run_cell.py $(sed -n 1p experiments/elcd/configs/cells/main_chain8z_laser.txt) \
    --recdir results/elcd/check/cell --no-probes --time-only 1 --allow-cpu
```

A complete small cell, three steps and the final evaluation, on the first line of `appendix_loss.txt`
(`lasa:BendedLine`, fp32, seed 0, LASER):

```sh
python experiments/elcd/train/run_cell.py $(sed -n 1p experiments/elcd/configs/cells/appendix_loss.txt) \
    --recdir results/elcd/check/cell --steps 3 --eval-every 0 --allow-cpu
```

One step of the sequential solver on the chain8z cell (in chunks of 16 rollouts to bound the memory; the paper's
runs took all 192 in one chunk) and the first epoch of the velocity loss (line 1 of `velocity_chain8z.txt`):

```sh
JAX_PLATFORMS=cpu python experiments/elcd/train/train_elcd.py --system chain8z --dtype fp64 --seed 0 \
    --loss suffix --starts 64 --lr-schedule cosine --steps 10000 --solver seq --tol 1e-7 \
    --start-chunk 16 --time-only 1 --out results/elcd/check/seq/loop_time1.json
JAX_PLATFORMS=cpu python experiments/elcd/train/train_elcd.py \
    $(sed -n 1p experiments/elcd/configs/cells/velocity_chain8z.txt) \
    --time-only 231 --out results/elcd/check/vel/vel-const100.json
```

The evaluation of the initial model of one cell of the original tasks (line 27 of `init_eval.txt`:
`lasa:Sshape`, fp32, seed 0):

```sh
JAX_PLATFORMS=cpu python experiments/elcd/train/run_init_eval.py \
    $(sed -n 27p experiments/elcd/configs/cells/init_eval.txt) --out results/elcd/check/init/init_eval.json
```

Step 0 of a record:

```sh
python -c "import json, sys; e = json.load(open(sys.argv[1]))['log'][0]; print({k: e.get(k) for k in ('loss', 'grad_norm', 'converged', 'iters_max', 'chunk_iters', 'retries')})" \
    results/elcd/check/cell/chain8z/fp64/s0/laser.json
```

### 3. The paper's runs on a GPU

```sh
mkdir -p logs
env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable --array=1-20 --time=4-00:00:00 \
    --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH,CELLS=experiments/elcd/configs/cells/main_chain8z.txt \
    experiments/elcd/slurm/run_cell.slurm
```

after filling in the account and the partition in the template. The array size is the number of lines of the
list. The headers of `train_velocity_loss.slurm` (`CELLS`, `BLOCK`), `train_array.slurm` (`JOBS`) and
`run_init_eval.slurm` give the lines for the other lists. Without Slurm, one cell is
`python experiments/elcd/train/run_cell.py $(sed -n <line>p <list>)` on a machine with one GPU. For this
release one job of `slurm/table_training_time.slurm` ran on a CPU node (4 minutes); no other template was submitted.

### 4. Figures and tables

From the files that ship and the LASA dataset:

```sh
python -c "
import pandas as pd
t = pd.read_csv('results/elcd/hours_table.csv')
for col in ('hours', 'fallback_steps'):
    print(t.pivot_table(index=['system', 'dtype', 'seed'], columns='solver', values=col).round(3).to_string())
    print(t.groupby(['system', 'dtype', 'solver'])[col].agg(['median', 'min', 'max']).round(3).to_string())
"
mkdir -p results/elcd/check/figs
cp figs/elcd/appendix/seed_trajectories.pkl results/elcd/check/figs/
python -m experiments.elcd.figures.figure_appendix trajectories --cached --out results/elcd/check/figs
python tools/compare_pdf.py results/elcd/check/figs/seed_trajectories_chain8z_fp32.pdf \
    figs/elcd/appendix/seed_trajectories_chain8z_fp32.pdf
python experiments/elcd/schematic/make_schematic_C.py                     # results/elcd/check/figs/panel_a.*
```

From the run records, once they are under `results/elcd/records/` (or `ELCD_PAPER_ROOT` points to the
folder of the main records, with `velocity/`, `long/` and `appendix/` beside it):

```sh
OUT=results/elcd/check/figs
REC=results/elcd/records
JAX_PLATFORMS=cpu python -m experiments.elcd.figures.table_training_time --out $OUT   # hours_table.csv
F=npend-8-N5000/fp64/s2/scale-elk0.03.json      # the shipped hours_table.csv: main/ with this record from rerun_slow/
M=results/elcd/check/main; mkdir -p $M; cp -rsf "$(cd $REC/main && pwd -P)"/. $M   # symbolic links
ln -sf "$(cd $REC/rerun_slow && pwd -P)/$F" $M/$F
JAX_PLATFORMS=cpu python -m experiments.elcd.figures.table_training_time --root $M --out $OUT/shipped
JAX_PLATFORMS=cpu python -m experiments.elcd.figures.figure_appendix mechanism pendulum trajectories original --out $OUT
JAX_PLATFORMS=cpu python -m experiments.elcd.figures.long_runs --out $OUT
JAX_PLATFORMS=cpu python experiments/elcd/train/sym_jacobian_positive.py fp32 laser
python experiments/elcd/train/select_scale_elk_rate.py $REC/tune --out $OUT/selection.json       # the rate g
python experiments/elcd/train/select_chunk.py $REC/pilot*                              # the chunk C
python experiments/elcd/train/summarize_records.py $REC/appendix --csv $OUT/summary.csv
ONLY=elcd bash tools/run_figure_notebooks.slurm                      # figs/elcd/
```

Over `main/` alone, line 99 of `hours_table.csv` (pendulum, fp64, seed 2, scale-ELK) has the first run's 11.860 h;
the second call of `table_training_time` reads a copy of `main/` with the rerun's record in its place and writes the
shipped file.

On the paper's records each of these commands took between 1 s and 2 min 38 s on 3 CPU cores and at most 0.9 GB;
the Slurm templates of the builders ask for 16 to 24 GB. Without `--out`, `select_scale_elk_rate.py` writes
`selection.json` into the records. The last line runs a Slurm template with `bash`: the template changes to
`$SLURM_SUBMIT_DIR`, so inside a Slurm allocation put `SLURM_SUBMIT_DIR=$PWD` in front of it. It rewrites the
notebook from its builder and writes the executed copy to `figs/notebooks/` and the parts of the figure to
`figs/elcd/`, the shipped `elcd_figure_traj_left.pdf` included; `run_cell.py` then refuses to run until that file
is restored (`git checkout -- figs/elcd/elcd_figure_traj_left.pdf`). The tests
`tests/test_elcd_{records,panel_data,loss_comparison,long_runs}.py` run the loaders and the tables on small synthetic
records.
