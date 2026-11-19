#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J contracting_ssm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. One training run of the contracting SSM on EigenWorms; all
# arguments are passed to run_train.py. submit_gamma_min_sweep.sh and submit_ablation.sh submit the paper's runs
# through this template. The runs of the gamma_min table used one NVIDIA B200: add your site's option for that GPU
# model (for example --constraint=<model>) to compare timings.
#
# Submit from the repository root (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch experiments/appendix/contracting_ssm/slurm/train.slurm --solver quasi ...
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
echo "args: $*"                          # export_runs.py reads the arguments of a run from this line
${PYTHON:-python} experiments/appendix/contracting_ssm/run_train.py "$@"
