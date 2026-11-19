#!/bin/bash
# Shared configuration runner of the Lorenz tuning sweep, stages A and B and their
# max_iter 2000 reruns. Sourced by tuning_sweep_{A,B,A_cap2000,B_cap2000}.slurm,
# table_timing.slurm, table_timing_cap5000.slurm and residual_histories.slurm; it runs nothing
# by itself.
#
# The interpreter is `python` on the PATH; set PYTHON=<interpreter> to use another one.
#
# Fixed settings: Lorenz sigma 10, rho 10, beta 8/3 from x0 (1, 1, 1), zero guess, explicit Euler dt 0.01, fp64,
# tol 1e-9 (the driver's fp64 default, passed explicitly), 3 timed repeats after the compile run. --rho 10 is
# passed on every line so that every row carries the rho, lorenz_ic, x0 and truth_end columns, as the shipped
# summaries do; the file names are those of the default run.
#
# One python process per configuration with the whole method list, since at D 3 nothing runs out of
# memory. A process that fails or exceeds the limit loses that configuration only.
#
# Log format:
#   "== cfg <args>"                               before every configuration
#   "  [lost] methods=<list> <args> (exit <rc>)"  when the process exits non-zero (124 is the timeout)
#   "   cfg_end exit=<rc> seconds=<s>"            after every configuration
set -u
cd "${SLURM_SUBMIT_DIR:-.}"              # the repository root: submit (or run) from there
PY=${PYTHON:-python}
TMO="${BENCH_TIMEOUT:-3600}"             # seconds per configuration process
REPEATS="${BENCH_REPEATS:-3}"
COMMON="--dt 0.01 --dtype fp64 --tol 1e-9 --rho 10 --repeats $REPEATS"

# run_cfg <outdir> <max_iter> <comma method list> <N>
run_cfg () {
  local out=$1 maxit=$2 m=$3 n=$4
  mkdir -p "$out"
  local args="--N $n --max-iter $maxit"
  echo "== cfg $args outdir=$out $(date +%T)"
  echo "   methods=$m"
  local t0; t0=$(date +%s)
  timeout "$TMO" $PY experiments/lorenz/run_benchmark.py $COMMON $args --methods "$m" --outdir "$out"
  local rc=$?
  [ $rc -ne 0 ] && echo "  [lost] methods=$m $args (exit $rc)"
  echo "   cfg_end exit=$rc seconds=$(( $(date +%s) - t0 ))"
}

# header: job id, node, GPU and JAX devices
header () {
  echo "== job ${SLURM_JOB_ID:-none} on $(hostname), $(date)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  $PY -c "import jax; print('jax', jax.__version__, jax.devices())"
}
