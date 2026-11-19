# Run from the repository root. PYTHON selects the interpreter (default: python on the PATH).
PYTHON ?= python
TESTS := $(sort $(wildcard tests/test_*.py))
SMOKE := results/smoke
NOTEBOOKS := SLURM_SUBMIT_DIR=$(CURDIR) PYTHON=$(PYTHON) bash tools/run_figure_notebooks.slurm

.PHONY: test smoke figures
.DEFAULT_GOAL := smoke

# One pytest process per module: jax_enable_x64 is a process-wide flag that most test modules set at import, and
# tests/test_implicit_fp32.py needs it off. On four CPU cores the whole suite takes 20 to 28 minutes; two modules
# (test_elcd_train.py, test_elcd_suffix_metrics.py) account for 11 to 15 minutes of it.
test:
	@rc=0; for t in $(TESTS); do echo "== $$t"; JAX_PLATFORMS=cpu $(PYTHON) -m pytest -q $$t || rc=1; done; exit $$rc

# Each experiment's code path at small scale, on the CPU, without downloaded data (3.4 to 5 min on four cores): the
# quick start, the template for an ODE of one's own (examples/own_ode.py, which imports the installed parode: the
# line puts this tree's on PYTHONPATH), the Lorenz benchmark of the paper's table at N = 1000 (iterations 2 NaN, 90,
# 94, 33, 27 and 22 with 10 rejected steps), three optimizer steps of the ELCD driver with LASER and the paper's
# protocol on a generated two-link pendulum, followed by its evaluation; the four solvers of the GPU-scaling
# experiment at N = 1000 and 3000 (iterations 20, 22, 86 and 6) and its multi-GPU Jacobi scan on four CPU devices (22
# iterations); two training steps of the robot controllers MLP small (DEER-exp and Euler-DEER) and SSM (Euler-DEER),
# and 2000 sweeps of the Slotine-Li Jacobi iteration. Outputs go to results/smoke (git-ignored).
smoke:
	JAX_PLATFORMS=cpu $(PYTHON) quickstart.py
	JAX_PLATFORMS=cpu PYTHONPATH=. $(PYTHON) examples/own_ode.py
	JAX_PLATFORMS=cpu $(PYTHON) experiments/lorenz/run_benchmark.py --dt 0.01 --dtype fp64 --tol 1e-9 --rho 10 \
	    --N 1000 --max-iter 200 --methods deer,scale-elk:0.05,scale-elk:0.053,elk:10000,elk:30000,laser \
	    --outdir $(SMOKE)/lorenz
	ELCD_NPEND_DIR=$(SMOKE)/npend $(PYTHON) -c "from experiments.elcd.train.pendulum import generate_npend; print(generate_npend('2-N400-b2'))"
	ELCD_NPEND_DIR=$(SMOKE)/npend JAX_PLATFORMS=cpu $(PYTHON) experiments/elcd/train/train_elcd.py \
	    --system npend:2-N400-b2 --dtype fp64 --seed 0 --loss suffix --starts 64 \
	    --lr-schedule cosine --steps 3 --solver laser --fallback seq --max-iter 0 --tol 1e-7 \
	    --out $(SMOKE)/elcd/laser.json
	JAX_PLATFORMS=cpu $(PYTHON) experiments/appendix/gpu_scaling/run_single_gpu.py --max-T=3000 --n-repeats=1 \
	    --algorithms=jacobi,jacobi_scan,quasi_deer,full_deer --outdir=$(SMOKE)/gpu_scaling --output-suffix=_cpu
	JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
	    $(PYTHON) experiments/appendix/gpu_scaling/run_multi_gpu.py --n-devices=4 --Ts=1000,3000 \
	    --outdir=$(SMOKE)/gpu_scaling --output-suffix=_cpu_4x
	JAX_PLATFORMS=cpu $(PYTHON) experiments/appendix/robot_control/neural/run_controller_table.py --seeds 0 --epochs 2 --timing-runs 1 \
	    --archs mlp_small --out $(SMOKE)/robot_control/mlp_small.json
	JAX_PLATFORMS=cpu $(PYTHON) experiments/appendix/robot_control/neural/run_controller_table.py --seeds 0 --epochs 2 --timing-runs 1 \
	    --archs ssm_d16_contract --solvers euler --out $(SMOKE)/robot_control/ssm_euler.json
	JAX_PLATFORMS=cpu $(PYTHON) experiments/appendix/robot_control/adaptive/run_slotine_li.py --sweeps 2000 --out $(SMOKE)/robot_control

# The figures that need neither downloaded data nor run records, from the result files that ship: the bistable
# Lorenz figure and the paper's first figure (their notebooks), the two Lorenz appendix figures, the GPU-scaling
# figure, panel B of the contracting-SSM figure (from results/contracting_ssm/runs.csv), the Slotine-Li figure of the
# robot-control experiment and the Euler step-size figure (recomputed, about 30 s); it also prints the robot-control
# table (tab:controllers) from its two result files. The figures overwrite the tracked PDFs, whose bytes then differ
# by the creation date; the last line compares text and pixels with the committed files.
# `git checkout -- figs experiments/fig1/output` restores the committed bytes. The ELCD figures need the LASA
# dataset and the run records: experiments/elcd/README.md.
figures:
	ONLY=lorenz $(NOTEBOOKS)
	ONLY=figure1 $(NOTEBOOKS)
	$(PYTHON) experiments/lorenz/figure_tuning.py
	$(PYTHON) experiments/lorenz/figure_residual.py
	$(PYTHON) experiments/appendix/gpu_scaling/figure_single_gpu.py
	$(PYTHON) experiments/appendix/contracting_ssm/figure_ablation.py
	$(PYTHON) experiments/appendix/robot_control/adaptive/figure_slotine_li.py
	$(PYTHON) experiments/appendix/robot_control/neural/table_controllers.py
	$(PYTHON) experiments/appendix/euler_step_size/figure_step_size.py
	$(PYTHON) tools/compare_pdf.py
