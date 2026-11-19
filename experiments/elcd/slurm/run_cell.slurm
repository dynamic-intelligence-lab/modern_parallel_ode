#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH -c 4 --mem=24G --time=24:00:00 --open-mode=append
#SBATCH -J elcd_paper
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION; one GPU per task. The timed cells of the paper (the main lists
# and appendix_speed) ran on one NVIDIA B200 each: add your site's option for that GPU model (for example
# --constraint=<model>) to reproduce the times. The appendix_loss cells, which report no time, ran on mixed GPU models.
#
# One cell of the ELCD paper runs per array task:
# line SLURM_ARRAY_TASK_ID (1-based) of the list CELLS holds the arguments of
# experiments/elcd/train/run_cell.py, which runs the probes, the items in a Williams order, the loop's timing
# and, if asked, the loop's full run, all on this one GPU, and skips any run whose record is complete (so a requeued
# or resubmitted task reruns only what is missing). Submit from the repository root with a clean environment (inside
# an interactive allocation its SLURM_* variables would leak into the job; python is taken from the PATH, add
# PYTHON=<interpreter> or an ELCD_*_DIR data directory to --export to change the defaults):
#   mkdir -p logs
#   env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable --array=1-<n> --time=<T> \
#       --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH,CELLS=experiments/elcd/configs/cells/<name>.txt experiments/elcd/slurm/run_cell.slurm
# The paper's arrays ran with the --time limits listed below (a chain8z cell with the full loop took about 3 days, a
# pendulum cell in fp64 about 5 days). A task stopped at its limit can be resubmitted, but its interrupted item starts
# again from step 0 (the driver cannot resume), so the limit must exceed the longest single item. The runner exits 3
# when its first probe fails (a machine problem).
# Run lists of the paper, with n (the number of lines) and the time limit of the array:
#   main_chain8z.txt         20  4 days  deer and scale-elk; fp64 seeds 0-9 then fp32 seeds 0-9; --full-loop on seeds 0-2
#   main_chain8z_laser.txt   20  24 h    laser on the same cells
#   main_npend.txt            6  6 days  deer and scale-elk on npend:8-N5000; fp64 seeds 0-2 then fp32 seeds 0-2
#   main_npend_laser.txt      6  36 h    laser on the same cells
#   main_npend_loop.txt       1  7 days  the sequential loop's full run on npend:8-N5000, fp64 seed 0
#   appendix_speed.txt       24  12 h    deer, laser and the loop's timing on lasa:{Angle,JShape,Sshape} and pend:8
#   appendix_loss.txt       162  4 h     laser on the other 27 LASA shapes (10 h on the slower GPU models)
# The remaining lists (the other full loops, tuning, chunk sweeps, ELK timing) are submitted the same way.
# Site setup (module load, environment activation) goes here if the job needs it.
: "${CELLS:?set CELLS to the cell list}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$PWD
PY=${PYTHON:-python}
mkdir -p logs
unset JAX_PLATFORMS
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLS")
echo "== job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID) on $(hostname), $(date), repo $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
echo "== cell: $ARGS"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>/dev/null || true
$PY experiments/elcd/train/run_cell.py $ARGS
rc=$?
echo "== exit $rc at $(date)"
exit $rc
