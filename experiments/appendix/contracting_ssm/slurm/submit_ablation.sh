#!/bin/bash
# The runs without the gamma head on eigenworms ("unstable" in panel B of fig:ssm_exp) — negative results.
#
# --gamma_param off --A_param sigmoid: the A*tanh cell without the leak term. tanh saturates,
# sech^2(h) -> 0, the one-step propagator -> 1, and the parallel solvers stop converging.
#
# Solvers: quasi (Newton) and jacobi (fixed-point). Both expected to saturate
# their iter caps. training.py prints the per-layer iterations, residual and
# jac_max (and logs them to wandb when it is on).
#
# Sweep: 2 solvers x 2 seeds = 4 jobs.
#
# Held fixed (matches the gamma_min sweep for direct comparison):
#   L=4, D=32, decimate_stride=1, batch_size=8, lr=3e-4, weight_decay=1e-4,
#   epochs=60, patience=30, init_alpha_range 0.1,1.5, alpha_min=0.0,
#   dt=1.0, tol=1e-4.
# Solver iter budgets raised to expose stalling rather than mask it:
#   --max_iter_newton 1000  --max_iter_jacobi 5000
#
# Compute target: any GPU. The paper's quasi runs ran on an NVIDIA GeForce RTX 3090; its two jacobi runs were
# submitted again on an NVIDIA B200 under the group name ablation_stability_b200_<date>_<user>, and those are the
# ones the figure uses. 16h walltime.
#
# Usage:
#   bash experiments/appendix/contracting_ssm/slurm/submit_ablation.sh             # submit all 4 jobs
#   DRY_RUN=1 bash experiments/appendix/contracting_ssm/slurm/submit_ablation.sh   # print sbatch lines, do not submit
#   SOLVERS="quasi" SEEDS="0" bash experiments/appendix/contracting_ssm/slurm/submit_ablation.sh
#                                                    # smoke-test one cell

set -euo pipefail

GROUP="ablation_stability_$(date -u +%F)_$(whoami)"
TAG="ablation_stability"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/train.slurm"

SOLVERS_ARR=( ${SOLVERS:-quasi jacobi} )
SEEDS_ARR=( ${SEEDS:-0 1} )

# Architecture + optimizer (shared across all jobs).
# export_runs.py needs local wandb run folders: before submitting, add --wandb_project <project> and install wandb.
COMMON=( --num_layers 4 --hidden_dim 32 --decimate_stride 1
         --batch_size 8 --lr 3e-4 --weight_decay 1e-4
         --epochs 60 --patience 30
         --init_alpha_range 0.1,1.5
         --alpha_min 0.0
         --max_iter_newton 1000 --max_iter_jacobi 5000 --tol 1e-4
         --wandb_group "$GROUP" )

# Partition and account come from the template (train.slurm).
SBATCH_ARGS=( --time=16:00:00
              --comment="$TAG" )

n=0
for SOLVER in "${SOLVERS_ARR[@]}"; do
    for SEED in "${SEEDS_ARR[@]}"; do
        n=$((n+1))
        CMD=( sbatch "${SBATCH_ARGS[@]}" "$TEMPLATE"
              --solver "$SOLVER"
              --seed "$SEED"
              --gamma_param off --A_param sigmoid
              "${COMMON[@]}" )
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            echo "[dry-run] ${CMD[*]}"
        else
            "${CMD[@]}"
            echo "submitted: solver=$SOLVER seed=$SEED group=$GROUP"
        fi
    done
done

echo
echo "$n jobs ${DRY_RUN:+would be }submitted to group $GROUP (tag $TAG)"
echo "watch with: squeue -u $(whoami)"
