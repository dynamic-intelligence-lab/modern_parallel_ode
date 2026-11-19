#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_elk_front
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA B200: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# ELK with sigma^2 = 1e4 at N = 1e5, the one solve of the Lorenz table that does not reach the tolerance within
# 2,000 iterations: the timing protocol of table_timing.slurm (x0 paper (1,1,1), zero guess, fp64,
# tol 1e-9, 10 timed repeats after the compile run) with a cap of 5,000 iterations. About four minutes. Writes
# into OUT/cap5000, OUT = results/lorenz/check/table_timing, apart from the shipped records.
# The template that produced the paper's row also ran a diagnostic of where ELK's iterate agrees with the solution
# after every iteration (front of agreement); no number of the paper comes from it, and it is not part of the
# release. The job name, lorenz_elk_front, is the one the paper's run had.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/table_timing_cap5000.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
BENCH_REPEATS=10
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/table_timing
run_cfg "$OUT/cap5000" 5000 "elk:10000" 100000
echo "== done $(date)"
