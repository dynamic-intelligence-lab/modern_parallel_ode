#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --mem=512G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J gpu_scaling_mgpu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION; the paper used NVIDIA B200 GPUs of one node (add your site's
# option for the GPU model). The number of GPUs is the first argument and must match --gres:
#   mkdir -p logs
#   for n in 2 4 8; do
#     sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH --gres=gpu:$n \
#         experiments/appendix/gpu_scaling/slurm/run_multi_gpu.slurm $n
#   done
# tab:multigpu-jacobi: Jacobi with a fixed scan of 120 iterations, the time axis sharded over n GPUs, seeds 0 to 4 at
# N = 1e5, 1e6, 1e7; every time is the median of 3 timed runs after one compilation run. Writes (and overwrites)
# results/gpu_scaling/check/gpu_scaling_mg_b200_<n>x.npz (the runner's default --outdir; the shipped files are in
# results/gpu_scaling). The 1x row of the table and the sequential baseline of every row come from the single-GPU
# B200 file (run_single_gpu.slurm b200). The paper's jobs took 16 to 20 minutes each.
# Site setup (module load, environment activation) goes here if the job needs it.
set -euo pipefail
NDEV=${1:?number of GPUs: 2 | 4 | 8}
cd "${SLURM_SUBMIT_DIR:-.}"
export JAX_PLATFORMS=cuda                # fail instead of running on the CPU
PY=${PYTHON:-python}
$PY experiments/appendix/gpu_scaling/run_multi_gpu.py --n-devices=$NDEV --output-suffix=_b200_${NDEV}x \
    --Ts=100000,1000000,10000000 --seeds=0,1,2,3,4
echo "== done $(date)"
