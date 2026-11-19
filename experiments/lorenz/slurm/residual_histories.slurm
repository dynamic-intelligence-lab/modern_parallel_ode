#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --mem=16G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_resid
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA B200: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Residual after each iteration on the bistable Lorenz problem at N = 1e4 and 1e5 (run_residual_histories.py):
# the last step of table_timing.slurm as a job of its own. About two minutes; the timeout bounds a
# stalled run. Writes OUT, results/lorenz/check/table_timing/residual_histories.json, apart from the shipped file.
# The shipped file comes from this job: in the paper's run, that step of table_timing.slurm did not finish within
# the job's time limit.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/residual_histories.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/table_timing/residual_histories.json
mkdir -p "$(dirname "$OUT")"
timeout 900 $PY experiments/lorenz/run_residual_histories.py "$OUT"
echo "== exit $? at $(date)"
