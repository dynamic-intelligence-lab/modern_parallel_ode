# GPU scaling of Jacobi and DEER on a contractive RNN: details

The short version is in [README.md](README.md). This file has every file, result file and command.

The non-autonomous contractive RNN x_{n+1} = W tanh(x_n) + c_n with D = 64 and rho = 0.6, in float64, is solved for
N steps (`T` in the code and in the result files) by four parallel-in-time solvers, and their wall-clock time on a
GPU is compared with that of the sequential rollout (a `lax.scan`) on the same GPU. W = rho U V^T, where U Sigma V^T
is the singular value decomposition of a matrix with standard normal entries, so every singular value of W equals
rho; the inputs c_n and the initial state x_0 are standard normal. W, c and x_0 are drawn from
`jax.random.PRNGKey(seed)` for each of the seeds 0 to 4.

In the paper this is the appendix section "GPU Acceleration of Reservoir Computing" (label `subsec:gpu_scaling`)
with its figure (label `fig:singlegpu-jacobi`, `figs/gpu_scaling/gpu_scaling.pdf`; the paper's source
includes this file as `figs/gpu_scaling.pdf`) and table (label
`tab:multigpu-jacobi`), and the appendix "Experimental Protocol" (label `app:gpu_scaling_setup`). The appendix on
Jacobi convergence cites its iteration counts.

All commands are run from the repository root. Two targets of the top-level `Makefile` cover part of section 1 and
2: `make figures` redraws the figure and compares it with the committed PDF, and `make smoke` runs both runners at
N <= 3000 (the multi-GPU one on four CPU devices).

## Solvers and what is timed

| name in the code | in the paper | what runs | stops when |
|---|---|---|---|
| `jacobi` | Jacobi (counts only) | `solvers.jacobi_while_deer_criterion`, a `lax.while_loop` of Jacobi sweeps from a standard normal guess | max over n, i of \|Delta s\| - 10^-4 \|s\| < 10^-7 (the DEER criterion), cap 2,000 |
| `jacobi_scan` | Jacobi (bars and table) | `jacobi_scan.jacobi_scan`, a `lax.scan` of exactly 120 sweeps from the same guess; each sweep also computes the error to the sequential solution and the residual | the count is the first sweep with max_n \|\|s_n - x*_n\|\|_2 < 10^-7 |
| `quasi_deer` | quasi-DEER | `parode.deer_map.deer_map_solve` from the zero guess: Newton with a diagonal Jacobian estimated by one Rademacher (Hutchinson) probe per time step; the probes are drawn once and reused in every iteration | the DEER criterion, cap 2,000 |
| `full_deer` | Full-DEER | `parode.deer_map.deer_map_solve_full` from the zero guess, with dense D x D Jacobians | the DEER criterion, cap 2,000 |
| `run_multi_gpu.py` | multi-GPU Jacobi | `jacobi_multi_gpu.make_jacobi_scan_multigpu`: the 120-sweep scan with the time axis sharded over n devices (`shard_map`); per sweep one `ppermute` of the last state to the next device and two `pmax` all-reductions (error and residual) | as `jacobi_scan` |

Every time is the median of 3 timed runs after one untimed run that compiles. A speedup is the sequential rollout
time divided by the solver time, per seed, then averaged over the seeds (mean and sample standard deviation):

- quasi-DEER and Full-DEER: sequential time of the same row / `wall_ms`;
- Jacobi on one GPU: sequential time of the same row / (`wall_ms` x iterations / 120), the scan time rescaled to
  the number of sweeps it needs (in `speedups.py`, the figure and the table this rescaled scan is called `jacobi`;
  the `jacobi` rows of the files, the while loop, give only iteration counts);
- Jacobi on n B200: sequential time of the single-B200 `jacobi_scan` row with the same N and seed /
  (`wall_clock_ms` x iterations / 120). The 1x row of the table is therefore the B200 Jacobi bar of the figure.

The sequential time is measured in the same job as the solver. The single-GPU files hold rows from several jobs
per GPU type (see Data), and their sequential times for the same N differ by up to 29 % between jobs (H100: 890.7
and 1149.4 ms at N = 10^5; B200 up to 25 %, L40S up to 16 %, the others up to 5 %).

## Files

| file | what it does |
|---|---|
| `contractive_rnn.py` | W, the inputs and the step map f(x, c) = W tanh(x) + c. |
| `jacobi_scan.py` | `sequential_rollout` (the baseline and the reference solution) and `jacobi_scan`. |
| `solvers.py` | The four single-GPU runners: build the system of a seed, time the sequential rollout and the solver, return a `SolverResult` row; an out-of-memory error gives a row with `oom = True`. |
| `run_single_gpu.py` | Single-GPU sweep: N from 10^3 to `--max-T` (1, 3, 10, 30, ... x 10^k; default 10^7, as in the paper's jobs), the solvers of `--algorithms`, the first seed of `--seeds` at every N and the other seeds at the N of `--multi-seed-Ts` (at every N if it is empty); a solver's sweep for a seed stops at its first out-of-memory N. Two defaults leave out runs of the paper: `--algorithms` is `jacobi,quasi_deer,full_deer`, without `jacobi_scan` (the Jacobi of the figure and the table), and `--seeds` is `0`; the paper's jobs passed the solvers of each pass and `--seeds=0,1,2,3,4` with `--multi-seed-Ts` (section 4). Writes `gpu_scaling<suffix>.npz` into `--outdir` (default `results/gpu_scaling/check`); `--append` keeps the rows of an existing file and skips the (solver, N, seed) already in it. |
| `jacobi_multi_gpu.py` | The sharded Jacobi scan and `run_single_multigpu`. |
| `run_multi_gpu.py` | Multi-GPU sweep over `--Ts` and `--seeds` on `--n-devices` devices; writes `gpu_scaling_mg<suffix>.npz` into `--outdir` (default `results/gpu_scaling/check`). |
| `speedups.py` | Loads `results/gpu_scaling/` (another folder if the environment variable `GPU_SCALING_RESULTS` names one), computes the per-seed speedups and their mean and standard deviation (imported by the two scripts below). |
| `figure_single_gpu.py` | Draws `figs/gpu_scaling/gpu_scaling.pdf`. A cell without a finite speedup is drawn as an out-of-memory mark, whether its rows ran out of memory or do not exist because the sweep stopped at a smaller N (Full-DEER at N = 10^7 on all GPUs but the B200). |
| `table_multi_gpu.py` | Prints the cells of the multi-GPU table with two decimals, then rounded once to one decimal. |
| `slurm/run_single_gpu.slurm`, `slurm/run_multi_gpu.slurm` | Job templates of the GPU runs (section 4). |

`speedups.py`, the figure script and the table script were taken from the notebook that drew the paper's figure,
which is not in the repository. `table_multi_gpu.py` prints its LaTeX tabular and writes no file; the paper's table
was typed from these numbers. From `parode/`, the experiment runs `deer_map.py` (`deer_map_solve`,
`deer_map_solve_full`, `quasi_diag_estimator`, `_par_solve`, `_par_solve_full`) and the scan operators `combine_fc`
of `quasi_parallel.py` and `explicit_parallel.py`; the other modules are imported along the way.

## Data

The experiment needs no external data. The result files behind the figure and the table ship with the repository
(`results/gpu_scaling/`, 9 files, 124,505 bytes). Each row of a single-GPU file is one (solver, N, seed); each row of
a multi-GPU file one (N, seed) with the error and residual history of the 120 sweeps.

| file | rows | device (nvidia-smi) | written |
|---|---|---|---|
| `gpu_scaling_a40.npz` | 70 | NVIDIA A40 | 2026-04 and 05 |
| `gpu_scaling_a6000.npz` | 70 | NVIDIA RTX A6000 | 2026-04 and 05 |
| `gpu_scaling_l40s.npz` | 70 | NVIDIA L40S | 2026-04 and 05 |
| `gpu_scaling_rtx_pro_blackwell.npz` | 75 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition | 2026-04 and 05; quasi-DEER at N = 10^7 for seeds 1 to 4 on 2026-11-12 |
| `gpu_scaling_h100.npz` | 75 | NVIDIA H100 80GB HBM3 | the same |
| `gpu_scaling_b200.npz` | 76 | NVIDIA B200 | the same |
| `gpu_scaling_mg_b200_{2,4,8}x.npz` | 15 each | 2, 4 and 8 NVIDIA B200 of one node | 2026-11-12 and 13 |

- Single-GPU files: seed 0 at every N from 10^3 to the first out-of-memory N (up to 10^7; the seed-0 Full-DEER rows
  of every file end with out-of-memory rows at two consecutive N, while the runner of this repository stops at the
  first); seeds 0 to 4 at N = 10^5 and 10^6 for Jacobi, quasi-DEER and Full-DEER; seeds 0 to 4 at 10^5, 10^6 and
  10^7 for the Jacobi scan; quasi-DEER at N = 10^7 for seeds 0 to 4 on the three GPUs where it fits. The seed-0 rows
  of the first pass were written on 2026-04-13 by an earlier version of `run_single_gpu.py` that is not in the
  repository, the seeds 1 to 4 of the first pass by another earlier version; this release's code gives the same
  iteration counts, but how those versions timed cannot be checked. The scan rows and the rows of
  2026-11-12 were written by the code of this repository.
- Multi-GPU files: written by `run_multi_gpu.py` as it is here, which times with the median of the 3 repeats and
  also stores the minimum (`wall_min_ms`). An earlier set of multi-GPU files (2026-05), timed with the minimum, gave
  the table of earlier versions of the paper; it is not shipped.

No result file records the code version or the package versions it was produced with. Some columns are diagnostics
that no figure, table or test uses: `compile_ms`, `peak_mem_mb`, `final_resid_blk`, `final_update_linf` and
`error_vs_truth` of the single-GPU files, `predicted_k`, `err_history`, `wall_min_ms`, `initial_err`,
`initial_resid` and `final_err` of the multi-GPU files.

## Paper artifacts

Levels: **A** recomputed for this release on a CPU and compared exactly; **B** redrawn or re-tabulated from the
result files that ship with the repository; **C** needs GPUs (the command is given and was exercised at small scale
on the CPU only); **D** cannot be reproduced from this repository.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| Figure `fig:singlegpu-jacobi` (`gpu_scaling.pdf`) | `figure_single_gpu.py` (section 1) | the six single-GPU files | CPU, 3 s | B; the timings behind it C |
| Table `tab:multigpu-jacobi`, 12 cells | `table_multi_gpu.py` (section 1) | `gpu_scaling_b200.npz`, the three multi-GPU files | CPU, 1 s | B; the timings C |
| Numbers of the paragraph before the table: 548x and 3,209x at N = 10^7 (factor 5.9), factor 2.5 at N = 10^6, multi-GPU slower at N = 10^5 | quotients of the table cells (section 1) | as the table | as the table | B |
| Iteration counts of `app:gpu_scaling_setup`: 6 (Full-DEER), 86 to 106 (quasi-DEER), 20 to 22 (Jacobi), 22 to 23 (Jacobi scan), essentially independent of N from 10^3 to 10^7 | `run_single_gpu.py` on the CPU (section 2) | none | CPU: 25 min for the single-GPU rows, 4 min per multi-GPU file | A for every stored row with N <= 10^5 (216 rows: 36 (solver, N, seed) on six GPU types) and for the 15 multi-GPU rows at N = 10^5; B above |
| Protocol of `app:gpu_scaling_setup` (W, inputs, guesses, tolerances, cap, scan of 120) | the code, `run_single_gpu.py` at small N | none | CPU, seconds | A |
| Out-of-memory marks of the figure and their thresholds in `app:gpu_scaling_setup` | read from the `oom` rows | the six single-GPU files | - | B; observing them needs the GPUs (C) |
| Speedups (every bar and cell), the GPU models | `slurm/run_single_gpu.slurm`, `slurm/run_multi_gpu.slurm` (section 4) | none | six GPU types, about 16.4 GPU hours single-GPU; 2, 4 and 8 B200 for 16 to 20 min each | C; the list of GPU models is D |
| Sharded Jacobi equals single-device Jacobi | `tests/test_gpu_scaling_multi_gpu.py` | none | CPU, 11 s | A |

## Commands

### 1. The figure and the table from the shipped files

```sh
python experiments/appendix/gpu_scaling/figure_single_gpu.py
python tools/compare_pdf.py
python experiments/appendix/gpu_scaling/table_multi_gpu.py
```

The first command overwrites `figs/gpu_scaling/gpu_scaling.pdf`; the second compares it with the
committed file by text and pixels (`git checkout -- figs/gpu_scaling` restores the committed bytes). The third
prints the table; its last block holds the cells as the paper prints them. The numbers of the paragraph before the
table are quotients of these cells (3209.06 / 547.64 = 5.86, 1457.67 / 591.04 = 2.47).

### 2. Iteration counts on the CPU

```sh
export JAX_PLATFORMS=cpu
OUT=results/gpu_scaling/check
R="python experiments/appendix/gpu_scaling/run_single_gpu.py --max-T=100000 --seeds=0,1,2,3,4 --multi-seed-Ts=100000 --n-repeats=1 --outdir=$OUT"
$R --algorithms=jacobi,jacobi_scan,quasi_deer --output-suffix=_cpu
$R --algorithms=full_deer --output-suffix=_cpu_full_deer
XLA_FLAGS=--xla_force_host_platform_device_count=8 python experiments/appendix/gpu_scaling/run_multi_gpu.py --n-devices=8 \
    --Ts=100000 --seeds=0,1,2,3,4 --outdir=$OUT --output-suffix=_cpu_8x
```

These are the runs of the GPU files at every N <= 10^5 (seed 0 from N = 10^3, seeds 1 to 4 at N = 10^5), on the CPU
and with one timed repeat; `--n-devices` 2 and 4 run the same way. The `iters` column printed for each row must
equal `iters_to_tol` (`iters_to_conv` for the multi-GPU files) of the same (solver, N, seed) in the shipped files.
Times on a CPU are not comparable with the GPU times. Full-DEER at N = 10^5 needs about 12 GB of memory.

### 3. Tests

```sh
JAX_PLATFORMS=cpu python -m pytest -q tests/test_gpu_scaling_multi_gpu.py tests/test_gpu_scaling_solvers.py tests/test_gpu_scaling_table.py
```

- `test_gpu_scaling_multi_gpu.py`: the sharded scan on 2 and 4 CPU devices equals the single-device scan (final
  state and the error and residual histories of 30 to 60 sweeps within 10^-10; they are identical in practice).
- `test_gpu_scaling_solvers.py`: the four solvers at N = 1000, seed 0, reach the sequential solution, in the
  iteration counts stored for all six GPU types (20, 22, 86 and 6).
- `test_gpu_scaling_table.py`: the 12 cells of the table, rounded once, and the quoted ranges of iteration counts,
  from the shipped files.

### 4. The GPU jobs

Fill in `YOUR_ACCOUNT` and `YOUR_GPU_PARTITION` in each template first, and add your site's option for the GPU model.

```sh
mkdir -p logs
E=NONE,HOME=$HOME,USER=$USER,PATH=$PATH
for g in a40 a6000 l40s rtx_pro_blackwell h100 b200; do
  sbatch --export=$E experiments/appendix/gpu_scaling/slurm/run_single_gpu.slurm $g     # one job per GPU type
done
for n in 2 4 8; do
  sbatch --export=$E --gres=gpu:$n experiments/appendix/gpu_scaling/slurm/run_multi_gpu.slurm $n
done
```

The single-GPU template runs the three passes the shipped files were written with (the paper ran pass 3 only on the
B200, H100 and RTX Pro 6000, where quasi-DEER fits at N = 10^7), each appending to
`results/gpu_scaling/check/gpu_scaling_<gpu>.npz`; the multi-GPU template writes
`results/gpu_scaling/check/gpu_scaling_mg_b200_<n>x.npz`. `results/gpu_scaling/check` is the default `--outdir` of
both runners, so the shipped files of `results/gpu_scaling` stay as they are. The jobs are named `gpu_scaling_1gpu`
and `gpu_scaling_mgpu` and write `logs/<job name>-<job id>.out`. The paper's jobs used
about 6.1, 7.9 and 2.4 GPU hours for the three single-GPU passes summed over the GPU types, and 16 to 20 minutes for
each multi-GPU job. The figure and the table from the new files:

```sh
GPU_SCALING_RESULTS=results/gpu_scaling/check python experiments/appendix/gpu_scaling/figure_single_gpu.py
GPU_SCALING_RESULTS=results/gpu_scaling/check python experiments/appendix/gpu_scaling/table_multi_gpu.py
```

The figure script writes `figs/gpu_scaling/gpu_scaling.pdf` in either case; `git checkout -- figs/gpu_scaling`
restores the committed file.

`speedups.py` reads only `gpu_scaling_<tag>.npz` for the six tags in its `GPU_ORDER` and
`gpu_scaling_mg_b200_{2,4,8}x.npz`: a single-GPU file with another tag is ignored without a message, and a GPU of
the list without a file is drawn with out-of-memory marks at every N. The multi-GPU template writes
`gpu_scaling_mg_b200_<n>x.npz` on any GPU, and the multi-GPU table takes its sequential times from
`gpu_scaling_b200.npz`. On other GPUs, pass one of the six tags to the single-GPU template, or add the new tag to
`GPU_ORDER` and `GPU_LABELS` in `speedups.py`.
