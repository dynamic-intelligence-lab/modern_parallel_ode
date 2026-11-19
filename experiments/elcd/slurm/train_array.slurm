#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J elcd_train
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION; one GPU per task. The long chain8z runs of the paper ran on
# one NVIDIA B200 each (add your site's option for that GPU model, for example --constraint=<model>).
#
# One ELCD training run per array task: line SLURM_ARRAY_TASK_ID (1-based) of the job list JOBS holds the
# arguments of experiments/elcd/train/train_elcd.py (sbatch --export splits on commas, so the list lives in
# a file). Submit from the repository root with
#   mkdir -p logs
#   sbatch --parsable --array=1-<n>%<k> --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH,JOBS=<file> \
#          experiments/elcd/slurm/train_array.slurm
# (python is taken from the PATH; add PYTHON=<interpreter> or an ELCD_*_DIR data directory to --export to change the
# defaults).
# The long chain8z runs of the paper are the two lines of JOBS=experiments/elcd/configs/cells/long_chain8z.txt: --array=1 with
# --time=4-00:00:00 (LASER, suffix loss, 1,000,000 steps) and --array=2 with --time=2-00:00:00 (velocity loss,
# 43,290 epochs).
# Site setup (module load, environment activation) goes here if the job needs it.
: "${JOBS:?set JOBS to the job list file}"
cd "${SLURM_SUBMIT_DIR:-.}"
REPO=$PWD
PY=${PYTHON:-python}
mkdir -p logs
unset JAX_PLATFORMS
LINE=$SLURM_ARRAY_TASK_ID
ARGS=$(sed -n "${LINE}p" "$JOBS")
echo "== job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID line $LINE on $(hostname), $(date), repo $REPO"
echo "== args: $ARGS"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
$PY experiments/elcd/train/train_elcd.py $ARGS
rc=$?
echo "== exit $rc at $(date)"
exit $rc
