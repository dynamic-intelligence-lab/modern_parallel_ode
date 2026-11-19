#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_time
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA B200: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Appendix of the bistable Lorenz experiment, on one NVIDIA B200.
# 1. Wall-clock time of the rows of the Lorenz table of that appendix. Same settings as the tuning sweep
#    (run_benchmark_common.sh): x0 paper (1,1,1), zero guess, fp64, tol 1e-9; 200 iterations, and the same
#    methods again with 2,000 for the runs that need more. 10 timed repeats after the compile run (the sweep used 3).
# 2. Residual after each iteration at N = 1e4 and 1e5 with the best constants (run_residual_histories.py). In the
#    paper's run this step did not finish within the job's time limit; the shipped histories come from
#    residual_histories.slurm, which runs it alone.
# One job, 7 processes; in the paper's run the six driver processes of step 1 took 12 minutes. Writes into OUT,
# results/lorenz/check/table_timing, apart from the shipped records in results/lorenz/table_timing.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/table_timing.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
BENCH_REPEATS=10
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/table_timing
M="deer,scale-elk:0.05,scale-elk:0.053,elk:10000,elk:30000,laser"
for CAP in 200 2000; do
  for N in 1000 10000 100000; do
    run_cfg "$OUT/cap$CAP" "$CAP" "$M" "$N"
  done
done
echo "== residual histories $(date +%T)"
$PY experiments/lorenz/run_residual_histories.py "$OUT/residual_histories.json"
echo "== done $(date)"
