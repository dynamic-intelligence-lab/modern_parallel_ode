#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH -c 1 --mem=8G --time=00:30:00
#SBATCH -J elcd_appx_post
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_CPU_PARTITION (a CPU job).
#
# After the appendix arrays (the appendix_speed and appendix_loss lists of run_cell.slurm and the appendix_vel
# list of train_velocity_loss.slurm, job ids <JA>, <JB>, <JC>):
# the long summary of every record under the appendix root (summarize_records.py, the rows behind tab:elcd_lasa_speed),
# printed to the log and written next to the records. The appendix root is appendix/ beside
# the main records (ELCD_PAPER_ROOT, default results/elcd/records/main). Submit from the repository root once the
# arrays' ids are known (python is taken from the PATH; add PYTHON=<interpreter> or ELCD_PAPER_ROOT=<dir> to --export
# to change the defaults):
#   mkdir -p logs
#   env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable --dependency=afterany:<JA>:<JB>:<JC> \
#       --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/elcd/slurm/summarize_records.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$PWD
R=$(dirname "${ELCD_PAPER_ROOT:-results/elcd/records/main}")/appendix
PY=${PYTHON:-python}
echo "== $(date) repo $REPO ($(git rev-parse --short HEAD)) root $R"
echo "== the appendix summary: summarize_records.py"
JAX_PLATFORMS=cpu $PY experiments/elcd/train/summarize_records.py "$R" --csv "$R/summary.csv"
echo "== exit $? at $(date)"
