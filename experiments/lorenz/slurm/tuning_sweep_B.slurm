#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH -J lorenz_hB
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION. The paper runs used one NVIDIA L40S: add your site's option
# for that GPU model (for example --constraint=<model>) to reproduce the timings.
#
# Lorenz tuning sweep. Stage B: ELK elk:s over the sigma^2 grid.
# Configurations: x0 paper (1,1,1), zero guess, N in {1000, 3000, 10000, 30000, 100000}, max_iter 200.
# One process per N (see run_benchmark_common.sh). Expected run time well under an hour: the
# slowest Lorenz configuration (elk at N 1e5, 200 iterations) takes 1.7 s per run.
# Writes into OUT, results/lorenz/check/tuning_sweep/stageB, apart from the shipped records.
#
# Submit from the repository root with (python is taken from the PATH; add PYTHON=<interpreter> to --export to use
# another one):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/lorenz/slurm/tuning_sweep_B.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
unset JAX_PLATFORMS                      # the submitting shell may have pinned JAX to the CPU
cd "${SLURM_SUBMIT_DIR:-.}"
source experiments/lorenz/slurm/run_benchmark_common.sh
header
OUT=results/lorenz/check/tuning_sweep/stageB
S2="1 3 10 30 100 300 1000 3000 10000 30000 100000"
M="$(for s in $S2; do printf "elk:%s," $s; done)"; M=${M%,}
NS="${BENCH_NS:-1000 3000 10000 30000 100000}"
for N in $NS; do
  run_cfg "$OUT" 200 "$M" "$N"
done
echo "-- rows:"; wc -l < "$OUT/summary.csv"
echo "== done $(date)"
