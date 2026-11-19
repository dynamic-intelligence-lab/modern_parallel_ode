#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --mem=16G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_seq_time
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA B200: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# The sequential row of the Lorenz table: the jitted lax.scan of the Euler step at N = 1e3, 1e4 and 1e5, with the
# table's problem and timing protocol (run_seq_timing.py). A few minutes. Writes OUT,
# results/lorenz/check/table_timing/seq/summary.csv, apart from the shipped record. The script does not overwrite
# an existing file: a second run prints "exists, not rerun" and stops.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/table_timing_seq.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PYTHON:-python}
echo "== job ${SLURM_JOB_ID:-none} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT=results/lorenz/check/table_timing/seq/summary.csv
mkdir -p "$(dirname "$OUT")"
$PY experiments/lorenz/run_seq_timing.py "$OUT"
echo "== exit $? at $(date)"
