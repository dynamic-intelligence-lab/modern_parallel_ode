# Reproducing the paper

```sh
make smoke      # quick start, examples/own_ode.py, the Lorenz benchmark at N = 1000, three ELCD training steps, the
                # GPU-scaling runners at N <= 3000, two training steps of the robot controllers MLP small and SSM,
                # 2000 Slotine-Li sweeps; 3.4 to 5 minutes on four CPU cores
make figures    # redraws the Lorenz figures, the first figure, the GPU-scaling figure, panel B of the
                # contracting-SSM figure, the Slotine-Li figure and the Euler figure, compares them with the committed
                # PDFs, prints the robot-control table; 1 to 2 minutes
```

`make figures` leaves the eight redrawn PDFs modified, and `experiments/elcd/train/run_cell.py` refuses to start while
tracked files are modified; `git checkout -- figs experiments/fig1/output` restores them.

The small-scale training runs of the contracting-SSM experiment need its downloaded data and are not part of
`make smoke`; its README gives them (three runs of about 30 s on four CPU cores).

Every figure and table of the paper has one of four levels:

- **A** recomputed on a CPU in minutes and compared exactly (iteration counts) or to a stated tolerance;
- **B** redrawn or re-tabulated from result files that ship with the repository;
- **C** needs GPU hours or the run records, which are not in the repository; the command is given. The training
  runs were exercised at small scale only; the commands that read the run records were run on the records;
- **D** cannot be reproduced from this repository.

| paper artifact | where | level |
|---|---|---|
| First figure (`main_idea.pdf` in the paper's source, label `fig:main_idea`) | `experiments/fig1/`; its PDF is `experiments/fig1/output/deer/figure1_horizontal_deer_text.pdf`, not under `figs/` | B from the shipped arrays. The arrays the figure draws recompute in about 7 minutes on a CPU (A); the full chaotic DEER array (230 iterations, about 2 CPU hours) was not recomputed |
| Bistable Lorenz figure (`lorenz_bistable.pdf`) | `experiments/lorenz/` | A |
| Lorenz table: iteration counts | `experiments/lorenz/` | A |
| Lorenz table: wall times | `experiments/lorenz/` | C (GPU timings on one NVIDIA B200, the sequential row included; the recorded solves add up to 14 min); B as a reading of the shipped summaries |
| Two Lorenz appendix figures | `experiments/lorenz/` | B |
| Euler step-size figure (`Euler_Error_Analysis_Plot_one_panel.pdf`) and the numbers about its example | `experiments/appendix/euler_step_size/` | A for the data and the numbers; the PDF is a redraw, the paper's was drawn outside this repository |
| ELCD figure (`elcd_figure_traj_left.pdf`) | `experiments/elcd/` | C: drawn from the run records; its panel A alone is B |
| ELCD tables of training time and fallback steps per seed | `experiments/elcd/` | B from `hours_table.csv` |
| The ~35x speedup in total training time (abstract, panel C of the ELCD figure) | `experiments/elcd/` | B: the median over the ten seeds of the `speedup` column of `results/elcd/hours_table.csv`, chain8z, fp32: 34.9 (DEER), 35.3 (scale-ELK, g = 0.03), 36.6 (LASER) |
| ELCD seed-trajectories figure | `experiments/elcd/` | B |
| ELCD settings table (`tab:elcd_settings`) | `experiments/elcd/` | B for the values as run; the choice of C and g is C |
| The other ELCD appendix figures and tables | `experiments/elcd/` | C: they need the run records |
| The ELCD training runs | `experiments/elcd/` | C: about 1,200 GPU hours, the timed runs on NVIDIA B200. The first optimizer steps of four runs are A |
| GPU-scaling figure and multi-GPU table (appendix) | `experiments/appendix/gpu_scaling/` | B from the shipped result files; the timings C: about 16.4 GPU hours over six GPU types, and 2, 4 and 8 NVIDIA B200 for 16 to 20 min each. Iteration counts A for N <= 10^5 |
| Contracting-SSM table of accuracy and speedup over gamma_min (`tab:eigenworms_gamma_min`) and hyperparameter table | `experiments/appendix/contracting_ssm/` | B from `runs.csv`; the wall times are C (NVIDIA B200) |
| Contracting-SSM figure (`fig_ssm.pdf`) | `experiments/appendix/contracting_ssm/` | Panel B's Matplotlib file B; the printed figure, a composite with the architecture diagram of panel A, D |
| The contracting-SSM training runs | `experiments/appendix/contracting_ssm/` | C: 29.8 GPU hours. The first two epochs of three runs at full length are A on a CPU |
| Robot-control table (`tab:controllers`) | `experiments/appendix/robot_control/` | B from the two shipped result files |
| Slotine-Li figure (`slotine_li_jacobi_3panel.pdf`) | `experiments/appendix/robot_control/` | B from the shipped arrays; they recompute in 40 s on a CPU (A) |
| The robot-control training runs and timings | `experiments/appendix/robot_control/` | C: the paper's two jobs took 35 and 68 min on one NVIDIA B200. The first two training steps of the six seed-0 runs are A |
| Theorems 1 and 2 | `lean/ModernODE-Lean-ode/` | checked by Lean (`./verify.sh`) |
| Tables of formulas: the updates (`tab:updates`), the sweeps of the parallel scan (`tab:upsweep`, `tab:downsweep`), Euler as Jacobi step maps (`tab:euler-jacobi`) | none | no code needed |

On other hardware the float64 iteration counts should be the same. In float32 a count can differ by one, and the
ELCD training solves can differ from the first step on. Wall times and speedups will differ.

The long experiments were not rerun for this release. The figures were regenerated with this code and compared
with the files used in the paper by text and pixels (`tools/compare_pdf.py`): no difference. For the
contracting-SSM figure the file compared is panel B. The Euler step-size figure is a redraw of a figure that was
drawn outside this repository; its marks equal the paper's.

## Environment

`requirements-lock.txt` holds the versions of the paper's environment for every package this installs; without
`-c requirements-lock.txt` the packages named in `pyproject.toml` still have those versions, but their own
dependencies resolve to whatever is current. `pip install -e .` alone installs the solver library with JAX and
NumPy, whose versions `pyproject.toml` pins exactly. In an environment with another JAX, `pip install --no-deps -e .`
adds `parode` and leaves the installed packages as they are; the tests and experiments were run with the pinned
versions only. The scripts under
`experiments/` and `tests/` put the repository root on `sys.path` themselves and are run from the repository root;
`examples/own_ode.py` imports the installed package.
The ELCD figure also needs `rsvg-convert` (librsvg) and Ghostscript; `tools/compare_pdf.py` needs `pdftotext` and
`pdftoppm` (poppler-utils). The contracting-SSM experiment reads its data with PyTorch
(`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`; the extra `.[eigenworms]` names the
same version).

Clone the repository with git: `tools/compare_pdf.py` without arguments (the last step of `make figures`) and the
check of `experiments/elcd/train/run_cell.py` for modified tracked files call git. From an archive without `.git`,
copy a PDF before redrawing it and give `compare_pdf.py` the redrawn file and the copy; give `run_cell.py` the option
`--allow-dirty`.

## Tests

```sh
make test
```

runs one pytest process per module (`jax_enable_x64` is a process-wide flag) on the CPU: 35 modules with 319 tests,
20 to 28 minutes on four cores, of which `tests/test_elcd_train.py` and `tests/test_elcd_suffix_metrics.py` take
11 to 15 minutes. A single module: `JAX_PLATFORMS=cpu python -m pytest -q tests/test_laser_reference.py`.

- `tests/test_laser_reference.py` compares LASER on ten cases with reference arrays computed by the solver version
  that produced the paper's DEER and scale-ELK runs, bit for bit.
- `tests/data/laser_golden.json` pins five runs of LASER's loop, among them the Lorenz run of the paper (22
  iterations); `tests/test_laser_golden.py` compares them.
- Some tests compare bit for bit with values stored under `tests/data/`. They were computed on x86-64 Linux with the
  pinned package versions; on another platform the last bits can differ.
- In a fresh clone 317 tests pass and two skip and say why: one needs the LASA dataset (next section), and one is a
  memory probe that runs on request.

## Data and results that are not in the repository

- The LASA handwriting dataset (Khansari-Zadeh and Billard, 2011) is free for non-commercial academic use and may
  not be redistributed. `bash experiments/elcd/download_data.sh` clones it from its authors' repository into
  `data/`, together with the pendulum files of Jaffe et al. No file of the dataset and no array of its
  demonstrations is in this repository; the scripts that draw demonstrations rebuild them from the download.
- The run records of the ELCD experiment (about 4.5 GB) are not in the repository.
- The EigenWorms data of the contracting-SSM experiment (107 MB) is not in the repository:
  `bash experiments/appendix/contracting_ssm/download_data.sh` downloads it from the DEER repository and checks its SHA-256.
  The records of that experiment's runs are not in the repository either; `results/contracting_ssm/runs.csv` holds
  their results.
