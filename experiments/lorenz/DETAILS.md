# Bistable Lorenz experiment: details

The short version is in [README.md](README.md). This file has every file, result file and command.

The Lorenz system with sigma = 10, rho = 10, beta = 8/3 is bistable: the origin is a saddle and the two foci C+
and C- are stable. The trajectory from x0 = (1, 1, 1) is
discretized by explicit Euler with dt = 0.01 and solved in float64 for N = 10^3, 10^4 and 10^5 steps by Newton
iterations with a parallel scan, started from the zero guess, which is the constant trajectory at the saddle.
The experiment records how many iterations and how much time DEER, scale-ELK, ELK and LASER need to bring the
residual max_n ||r_n||_inf below 10^-9, and the time of the sequential solver, the N Euler steps one after
another.

In the paper this is the section "Bistable Dynamics with an Unstable Initial Guess" (label
`subsec:beyond_global_stability`) and the appendix "Details of the Bistable Lorenz Experiment" (label
`app:lorenz`). `python quickstart.py` at the repository root runs DEER and LASER on the same problem in a few
seconds.

All commands are run from the repository root. LASER is in `parode/laser.py`; the solvers take it as
`laser=LaserConfig()`, whose defaults are the paper's configuration. With a CUDA build of JAX on a machine without a
usable GPU, a JAX process first logs `Jax plugin configuration error` with a traceback and then runs on the CPU; the
message can be ignored.

## Solver names

| paper | `--methods` name in `run_benchmark.py` | what runs |
|---|---|---|
| DEER | `deer` | the full Newton step, `parode.explicit_parallel.explicit_par_integrate` |
| scale-ELK with constant k | `scale-elk:<k>` | every step Jacobian multiplied by 1 - k (the ELCD driver's `scale-elk:<g>` takes the rate g instead: 1 - k = exp(-g dt)) |
| ELK with constant sigma^2 | `elk:<sigma^2>` | `parode.elk_baseline.elk_integrate` |
| LASER | `laser` | `LaserConfig()`, the defaults: `propose_first="lle", accept_ref="best", step_cap=2.0, floor_switch_off=False` |

The driver writes one row per run: `iters` (for LASER the accepted iterations), `rejected` (LASER's rejected
trial steps), `status` (`ok`, `NaN` for a non-finite iterate, `max_iter`), `wall_s` (the median of `--repeats`
solves after one compilation run), `residual` (max|r| of the returned trajectory; for `deer` the size of its last
update, on which plain DEER stops), `err` (largest difference to the sequential Euler solution) and
`device`. The paper's table prints `NaN` as "nf", in the iteration and in the time columns. Every other run in
the table ends `ok`: ELK with sigma^2 = 10^4 at N = 10^5 with a cap of 5,000 iterations, every other run with at
most 2,000.

## Files

Used for the paper's figures and numbers:

| file | what it does |
|---|---|
| `run_benchmark.py` | The benchmark driver. Builds the system, the sequential Euler solution and the guess, runs a list of methods, and writes one `.npz` per method, one `.json` per configuration and a row per method appended to `summary.csv` in `--outdir`, by default `results/lorenz/check/benchmark/`. Source of the iteration counts and times of the Lorenz table and of the tuning sweep. |
| `run_seq_timing.py` | Time of the sequential solver, the jitted `lax.scan` of the Euler step (`parode.sequential.seq_integrate`), at N = 10^3, 10^4 and 10^5, on the problem of `run_bistable_figure.build` and with the timing protocol of the driver (`timed`: one compilation run, then the median of 10 runs), into a CSV file. Asserts that the last state equals `truth_end` of LASER's rows in `table_timing/cap200/summary.csv` to 10^-12. Does not overwrite an existing file. Source of the sequential row of the Lorenz table. |
| `run_bistable_figure.py` | Computes every number of the bistable Lorenz figure (trajectories, exponents, conditioning of the merit functional, residual histories) into `lorenz_bistable.json` and `lorenz_bistable_cache.npz` in `--outdir`, by default `results/lorenz/check/bistable_figure/`; the shipped files, which the notebook reads, are in `results/lorenz/bistable_figure/`. `run_residual_histories.py` reuses its problem setup and its solver wrappers. |
| `build_figure_bistable_notebook.py` | Writes `figure_bistable.ipynb`. Edit the builder, not the notebook. |
| `figure_bistable.ipynb` | Draws `figs/lorenz/lorenz_bistable.{pdf,png}` from the JSON and the npz. Tracked without outputs. |
| `tuning_sweep.py` | Standard library only. The loader and the outcome rule for the sweep's `summary.csv` files, used by both appendix figure scripts. Run as `tuning_sweep.py [SWEEP_DIR]`, it lists the runs of `SWEEP_DIR/stage{A,B}` that stopped at the cap of 200 iterations with a bounded iterate and writes `SWEEP_DIR/stage{A,B}_cap2000/plan.txt`; `SWEEP_DIR` is by default `results/lorenz/check/tuning_sweep`, where the templates write. |
| `figure_tuning.py` | Draws `figs/lorenz/lorenz_baseline_tuning.{pdf,png}` from the stage A and B summaries. |
| `run_residual_histories.py` | Residual after every iteration for DEER, scale-ELK (k = 0.053), ELK (sigma^2 = 3x10^4) and LASER, by default at N = 10^4 and 10^5, into a JSON file. |
| `figure_residual.py` | Draws `figs/lorenz/lorenz_residual_long_horizons.{pdf,png}` from that JSON file. |
| `run_spectra.py` | Spectra at the fixed points and exponents of the solution and of the zero guess for rho in {5, 10, 15, 20, 23}, into `--out`, by default `results/lorenz/check/spectra.csv` (the shipped file is `results/lorenz/spectra.csv`). The paper quotes the rho = 10 rows. The script holds the power iteration and the dense exponent it uses. |
| `slurm/` | Job templates: `tuning_sweep_{A,B}.slurm` (the sweep with 200 iterations), `tuning_sweep_{A,B}_cap2000.slurm` (the reruns with 2,000), `table_timing.slurm` (the times of the table and the residual histories), `table_timing_cap5000.slurm` (ELK sigma^2 = 10^4 at N = 10^5 with 5,000 iterations, the one run of the table that needs more than 2,000), `table_timing_seq.slurm` (the sequential row of the table), `residual_histories.slurm` (the residual histories alone; the shipped histories come from it), `run_benchmark_common.sh` (the driver call they share). |
| `lorenz_field.py` | `lorenz_dynamics`, the vector field. |
| `../../tools/run_figure_notebooks.slurm` | Builds and executes the figure notebooks; a bash script that also runs without Slurm. |
| `../../figs/colors.json` | The method colors of all three figures. |

The driver has one system, `lorenz`, and the four methods above; its module docstring describes them. From
`parode/`, the Lorenz results run `explicit_parallel.py`, `laser.py`, `elk_baseline.py`, `sequential.py`,
`sequential_integrators.py` and `ode_utils.py`; `quasi_parallel.py` is imported along the way (by
`elk_baseline.py`).

## Data

The experiment needs no external data. The result files behind the figures and the table ship with the
repository (14 files, 347 kB):

| path under `results/lorenz/` | written by | device | used for |
|---|---|---|---|
| `bistable_figure/lorenz_bistable.json`, `bistable_figure/lorenz_bistable_cache.npz` | `run_bistable_figure.py` | CPU | the bistable Lorenz figure |
| `tuning_sweep/stage{A,B}/summary.csv` | the driver, `slurm/tuning_sweep_{A,B}.slurm` | NVIDIA L40S | iteration counts of the table, tuning figure |
| `tuning_sweep/stage{A,B}_cap2000/{plan.txt,summary.csv}` | `tuning_sweep.py`; the driver, `slurm/tuning_sweep_{A,B}_cap2000.slurm` | NVIDIA L40S | the same, for runs that needed more than 200 iterations |
| `table_timing/cap{200,2000}/summary.csv` | the driver, `slurm/table_timing.slurm` | NVIDIA B200 | times of the table |
| `table_timing/cap5000/summary.csv` | the driver, `slurm/table_timing_cap5000.slurm` | NVIDIA B200 | iteration count and time of ELK sigma^2 = 10^4 at N = 10^5 in the table |
| `table_timing/seq/summary.csv` | `run_seq_timing.py`, `slurm/table_timing_seq.slurm` | NVIDIA B200 (stated in the paper's source; the script records the JAX name of the device, so the file's `device` column reads `cuda:0`) | the sequential row of the table and the speedups over it |
| `table_timing/residual_histories.json` | `run_residual_histories.py`, `slurm/residual_histories.slurm` | NVIDIA B200 (the job's log, which is not in the repository; the file has no device field) | residual figure |
| `spectra.csv` | `run_spectra.py` | not recorded | exponents quoted in the text |

The folders are named after the runs. In `tuning_sweep/`, stage A is scale-ELK at the 16 values of k with DEER and
LASER, stage B is ELK at the 11 values of sigma^2, both with at most 200 iterations, and `stage{A,B}_cap2000` holds
the runs of each stage that stopped at 200 iterations with a bounded iterate, rerun with at most 2,000. In
`table_timing/`, `cap<n>` holds runs with at most n iterations and `seq` the sequential solver;
`residual_histories.json` is there because `slurm/table_timing.slurm` writes it as its last step. In the paper's
run that step did not finish within the job's time limit, and the shipped file comes from
`slurm/residual_histories.slurm`, which runs it alone.

The summaries record the device, `seq/summary.csv` only as `cuda:0`. No result file records the code version or
the package versions it was produced with. The runs wrote the methods scale-ELK and LASER under earlier names; the
`method` column of the shipped files and `stageA_cap2000/plan.txt` carry the driver's names, and ten rows of two
research-stage LASER variants, which no figure or table read, were deleted from `stageA/summary.csv`. The seven `summary.csv` files of the driver were written
with six further columns (`guard_C`, `n_patterns`, `beta`, `force`, `gain`, `spectrum`) that the driver filled
for other systems and that were empty in every row; they were removed from six of the files together with the
code that wrote them, and from `table_timing/cap5000/summary.csv` when that file was added. No other cell
was changed. `table_timing/seq/summary.csv`, written by `run_seq_timing.py` with its own seven columns,
ships as written.

## Paper artifacts

Levels: **A** recomputed for this release on a CPU in minutes and compared exactly (iteration counts, statuses)
or to the stated tolerance; **B** redrawn or re-tabulated from the result files that ship with the repository;
**C** needs a GPU, the command is given and was exercised at small scale only; **D** cannot be reproduced from
this repository. No artifact of this experiment is level D. CPU times are those measured for this release on a
loaded shared node, except the three runs marked as such, which ran in CPU jobs; a range is the two measurements
of the same run.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| Bistable Lorenz figure, `lorenz_bistable.pdf` (label `fig:lorenz_bistable`) | `run_bistable_figure.py`, then the notebook (section 1) | none; the notebook reads `bistable_figure/lorenz_bistable{.json,_cache.npz}` and `figs/colors.json` | CPU; 1 to 4 min for the numbers (the shipped JSON records 38 s; one run on the loaded node took 9 min), 22 to 38 s for the notebook | A |
| Table of iterations and times (label `tab:lorenz`), iteration counts | `run_benchmark.py` (section 2) | none | recorded on one NVIDIA L40S, ELK sigma^2 = 10^4 at N = 10^5 on one NVIDIA B200; on a CPU 26 to 35 s (N = 10^3), 60 to 72 s (N = 10^4), 34 to 38 s (N = 10^5 without ELK), 24 min for ELK sigma^2 = 10^4 and 20 min for ELK sigma^2 = 3x10^4 at N = 10^5 (each in a CPU job) | A |
| The same table, times | `slurm/table_timing.slurm` and `slurm/table_timing_cap5000.slurm` (section 5) | none | one NVIDIA B200; the recorded solves add up to 14 min | C; B as a reading of the shipped summaries |
| The same table, sequential row (the N Euler steps one after another) | `run_seq_timing.py` (section 2) and `slurm/table_timing_seq.slurm` (section 5) | `table_timing/cap200/summary.csv`, whose `truth_end` the script checks | one NVIDIA B200; the recorded solves add up to 16 s. On a CPU 2 s (in a CPU job), with the times of that CPU (section 2) | C; B as a reading of the shipped summary |
| Wall-clock statements: LASER the fastest solver at every horizon (`app:lorenz`, `subsec:beyond_global_stability` and the introduction), 1.3x, 4.1x and 4.0x faster than the fastest baseline at N = 10^3, 10^4 and 10^5 (`app:lorenz`), about 4x at N >= 10^4 (`subsec:beyond_global_stability`); 2.0x, 9.8x and 28x faster than the sequential solver (`app:lorenz`), over 9x at N >= 10^4 (`subsec:beyond_global_stability`) | quotients of the times of the table (section 2) | the timing summaries | as the times of the table | as the times of the table |
| Tuning figure, `lorenz_baseline_tuning.pdf` (label `fig:lorenz_tuning`) | `figure_tuning.py` (section 3) | `tuning_sweep/stage{A,B}{,_cap2000}/summary.csv`, `figs/colors.json` | CPU, 3 s. The sweep behind it: one NVIDIA L40S, the recorded solves add up to 30 min | B; the sweep is C, recomputed at N = 10^3 only |
| Statements of the tuning paragraph of `app:lorenz`: 16 values of k and 11 of sigma^2; scale-ELK non-finite for k <= 0.052 at N >= 10^4; best k = 0.05 at N = 10^3 and 0.053 at N >= 10^4; ELK best at sigma^2 = 3x10^4 at every horizon and non-finite at sigma^2 = 10^5 | `figure_tuning.py` prints the best constant of each horizon; the rest is read from the summaries | the stage A and B summaries | as the tuning figure | B; A at N = 10^3, where both grids were recomputed |
| Residual figure, `lorenz_residual_long_horizons.pdf` (label `fig:lorenz_residual`) | `figure_residual.py` (section 3) | `table_timing/residual_histories.json`, the stage A and B summaries, `figs/colors.json` | CPU, 3 s. The histories: one NVIDIA B200, about two minutes according to the template; on a CPU 30 to 40 s for N = 10^4 | B; the histories are A at N = 10^4 and C at N = 10^5 |
| Exponents in the text: lambda_h = +5.32 for the guess, -0.19 to -0.40 along the solution; DEER's first step at N = 10^3 (iterate 1.5x10^23, residual 1.3x10^44); k = 1 - exp(-lambda_h dt) = 0.0518 | `run_spectra.py`, `run_bistable_figure.py` (section 4) | none | CPU; 23 to 42 s and 1 to 4 min | A |

"The recorded solves add up to" is the sum of `compile_s` plus the timed repeats times `wall_s` over the rows of
the shipped summaries; process start-up and the sequential reference solution are not in it.

## Commands

Two targets of the top-level `Makefile` cover part of this section: `make smoke` runs the first driver command
of section 2 (N = 10^3), and `make figures` redraws the three Lorenz figures from the shipped result files (the
second command of section 1 and the commands of section 3) and compares them with the committed PDFs.

### 1. The bistable Lorenz figure

```sh
JAX_PLATFORMS=cpu python experiments/lorenz/run_bistable_figure.py
ONLY=lorenz bash tools/run_figure_notebooks.slurm
python tools/compare_pdf.py
```

The first command recomputes the numbers into `results/lorenz/check/bistable_figure/`; with
`--outdir results/lorenz/bistable_figure` it would overwrite the shipped numbers, which the notebook reads. The
second rebuilds the notebook from its builder, executes a copy into `figs/notebooks/` and so writes
`figs/lorenz/lorenz_bistable.{pdf,png}` from the shipped numbers; `git checkout -- figs/lorenz` restores the
committed PDF. The third compares every tracked PDF that changed with the committed one, by text and by pixels (the
bytes differ by the creation date); it needs `pdftotext` and `pdftoppm` (poppler-utils) and Pillow.

None of the three commands compares the recomputed numbers with the shipped ones. With `CHECK_MEASURE=1` in
front of the second command, the script runs the recomputation itself (into `CHECK_DIR`, by default
`results/lorenz/check/bistable_figure`, the folder of the first command) and prints the recomputed exponents,
condition numbers, iteration counts and statuses next to those of the shipped file. Inside a Slurm allocation the
second command needs `SLURM_SUBMIT_DIR=$PWD` in front (section 5).

Panel C draws ELK at sigma^2 = 10^4, the better of the two values (10^2 and 10^4) that
`run_bistable_figure.py` runs. The residual figure of section 3 uses the best constants of the tuning sweep
instead: k = 0.053 and sigma^2 = 3x10^4.

### 2. The table of iterations and times

```sh
ARGS="--dt 0.01 --dtype fp64 --tol 1e-9 --rho 10"
M=deer,scale-elk:0.05,scale-elk:0.053,elk:10000,elk:30000,laser
OUT=results/lorenz/check
export JAX_PLATFORMS=cpu
python experiments/lorenz/run_benchmark.py $ARGS --N 1000 --max-iter 200 --methods $M --outdir $OUT/cap200
python experiments/lorenz/run_benchmark.py $ARGS --N 10000 --max-iter 2000 --repeats 1 --methods $M \
    --outdir $OUT/cap2000
python experiments/lorenz/run_benchmark.py $ARGS --N 100000 --max-iter 2000 --repeats 1 \
    --methods deer,scale-elk:0.05,scale-elk:0.053,laser --outdir $OUT/cap2000
python experiments/lorenz/run_benchmark.py $ARGS --N 100000 --max-iter 5000 --repeats 1 --methods elk:10000 \
    --outdir $OUT/cap5000
python experiments/lorenz/run_benchmark.py $ARGS --N 100000 --max-iter 2000 --repeats 1 --methods elk:30000 \
    --outdir $OUT/cap2000_elk30000
```

These are the arguments of the job templates (`slurm/run_benchmark_common.sh`), on the CPU and with fewer
timed repeats. The columns `iters`, `rej` and `status` that the driver prints must read:

| method | N = 10^3 | N = 10^4 | N = 10^5 |
|---|---|---|---|
| `deer` | 2, NaN | 2, NaN | 1, NaN |
| `scale-elk:0.05` | 90, ok | 2, NaN | 2, NaN |
| `scale-elk:0.053` | 94, ok | 227, ok | 227, ok |
| `elk:10000` | 33, ok | 214, ok | 2063, ok (fourth command) |
| `elk:30000` | 27, ok | 190, ok | 1838, ok (fifth command) |
| `laser` | 22, 10 rejected, ok | 21, 10 rejected, ok | 21, 10 rejected, ok |

ELK at N = 10^5 is left out of the third command, which takes 34 to 38 s without it: one ELK iteration takes
0.31 to 0.34 s on the CPU used here, and the driver solves at least twice (one compilation run and `--repeats`
timed runs). The fourth command runs the one solve of the table that needs more than 2,000 iterations, ELK with
sigma^2 = 10^4 at N = 10^5, with the cap of 5,000 of `slurm/table_timing_cap5000.slurm`; it took 24 min in a CPU job
with 4 cores: a compilation run and one timed run of 2,063 iterations each, 0.34 s per iteration. The fifth
command runs ELK with sigma^2 = 3x10^4 at N = 10^5, which converges within 2,000 iterations, in a directory of its
own (in `$OUT/cap2000` it would overwrite the per-configuration `.json` of the third command); it took 20 min in a
CPU job with 4 cores: two solves of 1,838 iterations, 0.32 s per iteration.

The sequential row of the table:

```sh
OUT=results/lorenz/check
mkdir -p $OUT
JAX_PLATFORMS=cpu python experiments/lorenz/run_seq_timing.py $OUT/seq_summary.csv
```

The script times the N Euler steps one after another with the driver's protocol (one compilation run, then the
median of 10 runs) at N = 10^3, 10^4 and 10^5, and stops with an assertion unless the last state equals
`truth_end` of LASER's rows in `table_timing/cap200/summary.csv` to 10^-12. It does not overwrite an
existing file: a second run prints `exists, not rerun`. In a CPU job with 4 cores it took 2 s and printed
`|end - truth_end|` 6.2e-15, 1.2e-14 and 1.2e-14 (0.0 in the B200 record). Its times are those of the CPU, not
those of the table, and they depend on the node: in the runs made for this release 0.024 to 0.025, 0.148 to 0.164
and 1.37 to 2.40 ms, against 14.4, 123 and 1,280 ms on the B200.

The paper's table has no generating script. Its cells are those of the shipped summaries: the sequential row is
`wall_s` of `table_timing/seq/summary.csv`, the other rows follow the two rules below, with the two
exceptions stated there. This prints the rows of the driver's summaries:

```sh
python - <<'EOF'
import csv
for cap in ("cap200", "cap2000", "cap5000"):
    for r in csv.DictReader(open(f"results/lorenz/table_timing/{cap}/summary.csv")):
        print(cap, r["N"], r["method"], r["iters"], r["rejected"], r["status"], f"{1e3 * float(r['wall_s']):.3f} ms")
EOF
```

and this the sequential row and its quotient by LASER's time:

```sh
python - <<'EOF'
import csv
T = "results/lorenz/table_timing"
laser = {r["N"]: float(r["wall_s"]) for r in csv.DictReader(open(f"{T}/cap200/summary.csv"))
         if r["method"] == "laser"}
for r in csv.DictReader(open(f"{T}/seq/summary.csv")):
    print("seq", r["N"], f"{1e3 * float(r['wall_s']):.3f} ms", f"{float(r['wall_s']) / laser[r['N']]:.2f} x LASER")
EOF
```

- Iteration counts: the run with 200 iterations, or the run with 2,000 if that one stopped at `max_iter`, or the
  run with 5,000 (`cap5000`) for ELK sigma^2 = 10^4 at N = 10^5, which stops at `max_iter` with 2,000 as well.
  17 of the 18 counts were recorded on the L40S (`tuning_sweep/stage{A,B}` and `stage{A,B}_cap2000`),
  which has no run with 5,000 iterations; the 36 rows of the B200 summaries `cap200` and `cap2000` have the same
  `iters`, `rejected` and `status`. The count 2,063 is that of the B200 row in `cap5000`.
- Times: `wall_s` of `table_timing`, the median of ten solves, from `cap200` if the run converged within
  200 iterations, from `cap2000` if it converged within 2,000 and from `cap5000` otherwise (16,664.6 ms, printed
  as 16,700). Eleven of the thirteen printed times of the DEER, scale-ELK, ELK and LASER rows follow from this
  rule. The other two are the values of `cap2000` although the run converged within 200 iterations: scale-ELK
  k = 0.05 at N = 10^3 (9.3 ms printed; 9.247 ms in `cap200`, 9.255 ms in `cap2000`) and ELK sigma^2 = 3x10^4
  at N = 10^4 (607 ms printed; 606.3 ms and 607.0 ms). The two files differ by 0.1 % in these cells.
- Speedups: the text's 1.3x, 4.1x and 4.0x at N = 10^3, 10^4 and 10^5 are the printed time of the fastest
  baseline, scale-ELK (k = 0.05 at N = 10^3, k = 0.053 at N >= 10^4), divided by that of LASER: 9.3/7.3,
  51.3/12.6 and 179/45.2. With the unrounded times of the rule above they are 1.27, 4.08 and 3.97. LASER's time
  is the smallest of the converged solves at every horizon.
- Sequential row: `wall_s` of `seq/summary.csv`, 14.406, 123.097 and 1,280.112 ms, printed as 14.4, 123 and
  1,280. Divided by LASER's unrounded times of `cap200` (7.275, 12.556 and 45.207 ms) they give 1.98, 9.80 and
  28.3, the text's 2.0x, 9.8x and 28x and its "over 9x at N >= 10^4". LASER's time is below the sequential one at
  every horizon.

### 3. The two appendix figures

```sh
python experiments/lorenz/figure_tuning.py
python experiments/lorenz/figure_residual.py
python tools/compare_pdf.py
```

Both scripts read the shipped result files and overwrite the PDF and PNG in `figs/lorenz/`;
`git checkout -- figs/lorenz` restores the committed PDFs. `figure_residual.py` asserts that the
histories end at the iteration counts of the sweep.

The residual histories at N = 10^4 on a CPU, drawn into a directory of their own:

```sh
OUT=results/lorenz/check
mkdir -p $OUT
JAX_PLATFORMS=cpu python experiments/lorenz/run_residual_histories.py $OUT/residual_histories.json 10000
python experiments/lorenz/figure_residual.py $OUT/residual_histories.json $OUT/lorenz_residual_N10000
```

`figure_residual.py` draws horizons at which DEER is non-finite after its first step, N >= 10^4. It
stops with an assertion on a file for N = 10^3, where DEER takes two steps. With a single horizon in the file
the legend is wider than the figure and its first entry is cut off.

### 4. The exponents quoted in the text

```sh
JAX_PLATFORMS=cpu python experiments/lorenz/run_spectra.py
```

It writes `results/lorenz/check/spectra.csv`; `--out results/lorenz/spectra.csv` would overwrite the shipped file.
A rerun reproduces the rows of rho = 5 and 10 (the paper quotes rho = 10) to 2x10^-14 relative; the rows of rho = 15,
20 and 23 at N >= 10^4 depend on rounding and can differ by order one.

| number in the paper | where it is |
|---|---|
| lambda_h = +5.32 for the zero guess | `lambda_h_guess` = 5.3217 in `bistable_figure/lorenz_bistable.json` (N = 10^3); `lam_zero` in the rho = 10 rows of `spectra.csv` (5.3217 at every N) |
| -0.19 to -0.40 along the solution (the section quotes -0.19, the value at N = 10^3) | `lam_truth` in the rho = 10 rows of `spectra.csv`: -0.1877, -0.3811, -0.4024 at N = 10^3, 10^4, 10^5; `lambda_h_truth` in the JSON for N = 10^3 |
| DEER's first step at N = 10^3: the iterate reaches 1.5x10^23, its residual 1.3x10^44, the next step is non-finite | `runs.deer.first_step_max_abs_iterate` = 1.49x10^23, `runs.deer.first_step_residual` = 1.30x10^44 and `runs.deer.residual` = [1.08, 1.30x10^44, NaN] in the JSON |
| k = 1 - exp(-lambda_h dt) = 0.0518 | computed in `figure_tuning.py` from the constant `LAMBDA_GUESS = 5.3217`; `k_eig` in the rho = 10 rows of `spectra.csv` |

### 5. The GPU jobs

The iteration counts of the sweep and the times of the table come from these jobs. Fill in `YOUR_ACCOUNT` and
`YOUR_GPU_PARTITION` in each template first.

```sh
mkdir -p logs
E=NONE,HOME=$HOME,USER=$USER,PATH=$PATH
S=experiments/lorenz/slurm
sbatch --export=$E $S/tuning_sweep_A.slurm         # scale-ELK grid, DEER, LASER; 200 iterations
sbatch --export=$E $S/tuning_sweep_B.slurm         # ELK grid; 200 iterations
python experiments/lorenz/tuning_sweep.py          # after both: lists the runs that stopped at 200
sbatch --export=$E $S/tuning_sweep_A_cap2000.slurm # those runs with 2,000 iterations
sbatch --export=$E $S/tuning_sweep_B_cap2000.slurm
sbatch --export=$E $S/table_timing.slurm           # times of the table, residual histories
sbatch --export=$E $S/table_timing_cap5000.slurm   # ELK sigma^2 = 1e4 at N = 1e5 with 5,000 iterations
sbatch --export=$E $S/table_timing_seq.slurm       # the sequential row of the table
sbatch --export=$E $S/residual_histories.slurm     # the residual histories alone, as for the shipped file
```

The sweep runs 16 values of k and 11 values of sigma^2 at N = 1000, 3000, 10000, 30000 and 100000; the paper
reports N = 10^3, 10^4 and 10^5.

- The templates write into `results/lorenz/check/` (the variable `OUT` of each), apart from the shipped records
  in `results/lorenz/`, and `tuning_sweep.py` plans from and into `results/lorenz/check/tuning_sweep` by default.
  The driver appends to an existing `summary.csv`, so a second submission adds rows to the files of the first;
  the figure scripts keep the last row for each pair of N and method, and they read the shipped files.
  `run_seq_timing.py` does not overwrite an existing file: a second `table_timing_seq.slurm` prints
  `exists, not rerun` and stops.
- `table_timing.slurm` ends with the residual histories, and `residual_histories.slurm` runs that step alone into
  the same file; the shipped file comes from the second (section "Data").
- The templates change to `$SLURM_SUBMIT_DIR`. To run one with `bash` inside an existing Slurm allocation, set
  `SLURM_SUBMIT_DIR=$PWD` in front of the command; this also holds for `run_figure_notebooks.slurm`.
