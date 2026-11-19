#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH -c 2 --mem=24G --time=00:40:00
#SBATCH -J elcd_paper_figs
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_CPU_PARTITION (a CPU job).
#
# The table of training hours hours_table.csv from the main records (experiments/elcd/figures/table_training_time.py)
# under OUT (default results/elcd/check, so that the shipped results/elcd/hours_table.csv is not overwritten), the
# tables of hours, fallback steps and final Chamfer distances per seed in the log. The builder reads the run records
# under ELCD_PAPER_ROOT (default results/elcd/records/main; appendix/ sits beside it). Over main/ alone, line 99 of
# the CSV (pendulum, fp64, seed 2, scale-ELK) has the first run's 11.860 h, not the shipped 8.250 h of its rerun: for
# the shipped file, set ELCD_PAPER_ROOT to a copy of main/ in which npend-8-N5000/fp64/s2/scale-elk0.03.json is the
# record of rerun_slow/ (experiments/elcd/DETAILS.md, section 4).
# Submit from the repository root (python is taken from the PATH; add PYTHON=<interpreter>, ELCD_PAPER_ROOT=<dir> or
# an ELCD_*_DIR data directory to --export to change the defaults):
#   mkdir -p logs
#   env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable \
#       --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH[,OUT=<dir>] experiments/elcd/slurm/table_training_time.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$PWD
PY=${PYTHON:-python}
OUT=${OUT:-$REPO/results/elcd/check}
echo "== $(date) repo $REPO ($(git rev-parse --short HEAD)) out $OUT"
JAX_PLATFORMS=cpu PYTHONPATH="$REPO" $PY \
    -m experiments.elcd.figures.table_training_time --out "$OUT"
echo "== exit $? at $(date)"
