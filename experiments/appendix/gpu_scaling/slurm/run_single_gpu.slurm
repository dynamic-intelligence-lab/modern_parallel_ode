#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J gpu_scaling_1gpu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION, and add your site's option for the GPU model (for example
# --constraint=<model>). One job per GPU type; the first argument names the result file
# results/gpu_scaling/check/gpu_scaling_<tag>.npz (the runner's default --outdir; the shipped files are in
# results/gpu_scaling). The paper's tags: a40, a6000, l40s, rtx_pro_blackwell, h100, b200.
#
# Single-GPU runs of fig:singlegpu-jacobi, in the three passes the paper's files were written with (each pass appends
# to the file and skips the (solver, N, seed) rows already in it):
#   1. Jacobi (while loop), quasi-DEER and Full-DEER: seed 0 at every N from 1e3 to 1e7, seeds 1 to 4 at N = 1e5
#      and 1e6. A solver's sweep stops at its first out-of-memory N; 1e7 is the runner's default --max-T.
#   2. Jacobi with a fixed scan of 120 iterations (the figure's Jacobi bars): seeds 0 to 4 at N = 1e5, 1e6, 1e7.
#   3. quasi-DEER at N = 1e7 for seeds 1 to 4 (it runs out of memory there on the 48 GB GPUs).
# The paper's jobs for these passes used about 6.1, 7.9 and 2.4 GPU hours, summed over the GPU types; quasi-DEER at
# N = 1e7 alone takes 9 to 13 minutes per seed on a B200, H100 or RTX Pro 6000.
#
# Submit from the repository root:
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH \
#       experiments/appendix/gpu_scaling/slurm/run_single_gpu.slurm b200
# Site setup (module load, environment activation) goes here if the job needs it.
set -euo pipefail
GPU=${1:?gpu tag of results/gpu_scaling/check/gpu_scaling_<tag>.npz}
cd "${SLURM_SUBMIT_DIR:-.}"
export JAX_PLATFORMS=cuda                # fail instead of running on the CPU
PY=${PYTHON:-python}
R="$PY experiments/appendix/gpu_scaling/run_single_gpu.py --output-suffix=_$GPU --seeds=0,1,2,3,4 --append"
$R --algorithms=jacobi,quasi_deer,full_deer --multi-seed-Ts=100000,1000000
$R --algorithms=jacobi_scan --multi-seed-Ts=100000,1000000,10000000
$R --algorithms=quasi_deer --multi-seed-Ts=100000,1000000,10000000
echo "== done $(date)"
