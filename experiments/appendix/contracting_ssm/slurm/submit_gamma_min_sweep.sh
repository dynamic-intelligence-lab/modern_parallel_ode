#!/bin/bash
# gamma_min sweep on eigenworms, all on B200.
#
# Goal: the paper's table tab:eigenworms_gamma_min — vary the contraction floor
# gamma_min and report mean +/- std test_acc, per-epoch wall time, and
# parallel-vs-sequential speedup across seeds, for {seq, quasi, jacobi}.
#
# Sweep: 3 solvers x 5 gamma_min x 5 seeds = 75 jobs.
#   gamma_min in {0.01, 0.02, 0.05, 0.1, 0.2}
#   seed in {0, 1, 2, 3, 4}
#   solver in {seq, quasi, jacobi}
#
# Held fixed:
#   L=4, D=32, decimate_stride=1, A_param=sigmoid, alpha_min=0.0,
#   init_alpha_range 0.1,1.5, gamma_param=sigmoid, gamma_max=0.4,
#   epochs=60, patience=30, batch_size=8, lr=3e-4, weight_decay=1e-4,
#   max_iter_newton=200, max_iter_jacobi=1000, tol=1e-4.
#
# Stability check: dt*(gamma_max + alpha_max) = 1*(0.4 + 1.5) = 1.9 < 2. OK.
# Caveat: at gamma_min=0.2 the leak range is [0.2, 0.4] — only 2x.
#
# Compute target: one NVIDIA B200 per job. 16h walltime so seq at decimate_stride=1 (T=17984) finishes 60
# epochs even at gamma_min=0.01.
#
# Usage:
#   S=experiments/appendix/contracting_ssm/slurm/submit_gamma_min_sweep.sh
#   bash $S                                                 # submit all 75 jobs
#   DRY_RUN=1 bash $S                                       # print sbatch lines, do not submit
#   SOLVERS="quasi" GAMMA_MINS="0.05" SEEDS="0" bash $S     # smoke-test one cell

set -euo pipefail

GROUP="gamma_min_sweep_$(date -u +%F)_$(whoami)"     # the paper's runs: v2_gamma_min_sweep_<date>_<user>
TAG="gamma_min_sweep"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/train.slurm"

# Allow narrowing the grid via env vars for smoke tests.
SOLVERS_ARR=( ${SOLVERS:-seq quasi jacobi} )
GAMMA_MINS_ARR=( ${GAMMA_MINS:-0.01 0.02 0.05 0.1 0.2} )
SEEDS_ARR=( ${SEEDS:-0 1 2 3 4} )

# export_runs.py needs local wandb run folders: before submitting, add --wandb_project <project> and install wandb.
COMMON=( --A_param sigmoid
         --num_layers 4 --hidden_dim 32 --decimate_stride 1
         --batch_size 8 --lr 3e-4 --weight_decay 1e-4
         --epochs 60 --patience 30
         --init_alpha_range 0.1,1.5
         --alpha_min 0.0
         --gamma_param sigmoid --gamma_max 0.4
         --max_iter_newton 200 --max_iter_jacobi 1000 --tol 1e-4
         --wandb_group "$GROUP" )

# Partition and account come from the template (train.slurm). The paper's runs used --constraint=<B200 feature>.
SBATCH_ARGS=( --time=16:00:00
              --comment="$TAG" )

n=0
for SOLVER in "${SOLVERS_ARR[@]}"; do
    for GAMMA_MIN in "${GAMMA_MINS_ARR[@]}"; do
        for SEED in "${SEEDS_ARR[@]}"; do
            n=$((n+1))
            CMD=( sbatch "${SBATCH_ARGS[@]}" "$TEMPLATE"
                  --solver "$SOLVER"
                  --gamma_min "$GAMMA_MIN"
                  --seed "$SEED"
                  "${COMMON[@]}" )
            if [[ "${DRY_RUN:-0}" == "1" ]]; then
                echo "[dry-run] ${CMD[*]}"
            else
                "${CMD[@]}"
                echo "submitted: solver=$SOLVER gamma_min=$GAMMA_MIN seed=$SEED group=$GROUP"
            fi
        done
    done
done

echo
echo "$n jobs ${DRY_RUN:+would be }submitted to group $GROUP (tag $TAG, B200)"
echo "watch with: squeue -u $(whoami)"
