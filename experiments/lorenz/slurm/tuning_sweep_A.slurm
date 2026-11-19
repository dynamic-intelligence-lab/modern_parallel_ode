#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_hA
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA L40S: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Lorenz tuning sweep. Stage A: constant scale-ELK scale-elk:k over the k grid, plus deer and LASER.
# Configurations: x0 paper (1,1,1), zero guess, N in {1000, 3000, 10000, 30000, 100000}, max_iter 200.
# One process per N (see run_benchmark_common.sh). Expected run time well under an hour: the
# slowest solve of this stage (scale-ELK at N 1e5, 200 iterations) took 0.09 s per run on the L40S.
# Writes into OUT, results/lorenz/check/tuning_sweep/stageA, apart from the shipped records.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/tuning_sweep_A.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/tuning_sweep/stageA
K="0.03 0.04 0.045 0.05 0.051 0.052 0.053 0.055 0.06 0.07 0.08 0.1 0.15 0.2 0.3 0.5"
M="$(for k in $K; do printf "scale-elk:%s," $k; done)deer,laser"
NS="${BENCH_NS:-1000 3000 10000 30000 100000}"
for N in $NS; do
  run_cfg "$OUT" 200 "$M" "$N"
done
echo "-- rows:"; wc -l < "$OUT/summary.csv"
echo "== done $(date)"
