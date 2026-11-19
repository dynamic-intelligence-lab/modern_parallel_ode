#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH -c 4 --mem=16G --time=02:00:00 --open-mode=append
#SBATCH -J elcd_init_eval
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Template: fill in YOUR_ACCOUNT and YOUR_CPU_PARTITION (a CPU job).
#
# The step-0 evaluation of the paper cells (experiments/elcd/train/run_init_eval.py): array task t runs line t
# of CELLS, the script's full argument list including --out, on the CPU (JAX_PLATFORMS=cpu; the initial model is
# built from the seed alone and the evaluation is the driver's metrics.evaluate). A line whose --out exists is skipped
# by the script. Submit from the repository root (python is taken from the PATH; add PYTHON=<interpreter> or an
# ELCD_*_DIR data directory to --export to change the defaults). The paper's list is init_eval.txt, 50 lines
# (--array=1-50): the 26 main cells and the 24 appendix_speed cells.
#   mkdir -p logs
#   env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable --array=1-<n> \
#       --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH,CELLS=experiments/elcd/configs/cells/<name>.txt experiments/elcd/slurm/run_init_eval.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
: "${CELLS:?set CELLS to the cell list}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$PWD
PY=${PYTHON:-python}
mkdir -p logs
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO"
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLS")
echo "== job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID) on $(hostname), $(date), repo $REPO ($(git -C "$REPO" rev-parse --short HEAD)), line $SLURM_ARRAY_TASK_ID: $ARGS"
[ -z "$ARGS" ] && { echo "empty line"; exit 0; }
$PY experiments/elcd/train/run_init_eval.py $ARGS
rc=$?
echo "== exit $rc at $(date)"
exit $rc
