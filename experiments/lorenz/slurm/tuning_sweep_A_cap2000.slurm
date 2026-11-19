#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_hA2k
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA L40S: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Lorenz tuning sweep, stage A reruns at max_iter 2000. Every scale-ELK run of stage A that
# stopped at max_iter 200 with a bounded iterate (cap, not div) is rerun unchanged except for --max-iter 2000,
# so "cannot converge" is separated from "slow".
# The runs come from OUT/plan.txt, OUT = results/lorenz/check/tuning_sweep/stageA_cap2000 (apart from the shipped
# records), one line "<N> <methods>" per configuration, written from the rows of tuning_sweep_A.slurm before
# submission by
#   python experiments/lorenz/tuning_sweep.py
# Budget: a scale-ELK run at N 1e5 took about 0.47 ms per iteration on the L40S, so 2000 iterations x 4 runs x
# 10 values is about 40 s per configuration.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/tuning_sweep_A_cap2000.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/tuning_sweep/stageA_cap2000
cat "$OUT/plan.txt"
mapfile -t LINES < "$OUT/plan.txt"      # read first, so no process in the loop can consume the plan from stdin
for line in "${LINES[@]}"; do
  read -r N M <<< "$line"
  [ -n "$N" ] && run_cfg "$OUT" 2000 "$M" "$N"
done
echo "-- rows:"; wc -l < "$OUT/summary.csv"
echo "== done $(date)"
