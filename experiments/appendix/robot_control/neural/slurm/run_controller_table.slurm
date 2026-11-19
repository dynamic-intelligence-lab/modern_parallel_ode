#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J robot_controller_table
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA B200: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Neural-controller table (tab:controllers): train the three controllers (MLP large, MLP small, SSM) with DEER-exp
# and Euler-DEER for 1200 epochs at kp = 3, kd = 0.6, N = 1000, and time the training step against the sequential
# integrator of the same discretization. Two arguments: the seeds and the output file. The paper's numbers come from
# two jobs of the driver:
#   seeds 0        -> results/robot_control/bench_table.json              (35 min on one B200; that job ran an
#                     earlier form of the driver, which also trained at kp = 1)
#   seeds 1,2,3,4  -> results/robot_control/bench_table_seeds1-4_kp3.json (68 min on one B200)
# Without arguments the job runs seeds 0 to 4 into results/robot_control/check/bench_table_rerun.json, which leaves
# the shipped files alone.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH \
#       experiments/appendix/robot_control/neural/slurm/run_controller_table.slurm \
#       1,2,3,4 results/robot_control/check/bench_table_seeds1-4_kp3.json
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PYTHON:-python}
echo "== job ${SLURM_JOB_ID:-} on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1), $(date)"
$PY experiments/appendix/robot_control/neural/run_controller_table.py --seeds "${1:-0,1,2,3,4}" \
    --out "${2:-results/robot_control/check/bench_table_rerun.json}"
echo "== done $(date)"
