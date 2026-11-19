#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH -c 2 --mem=16G --time=00:20:00
#SBATCH -J elcd_seed_traj
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_CPU_PARTITION (a CPU job).
#
# The four ELCD appendix figures in the paper's style (experiments/elcd/figures/figure_appendix.py), with
# the rollouts of the ten seeds of each loss on the CPU, cached beside the figures. They are written over the tracked
# PDFs and seed_trajectories.pkl in figs/elcd/appendix; `git checkout -- figs/elcd/appendix` restores the committed
# files (tools/compare_pdf.py compares a redrawn PDF with them). The builder reads the run records
# under ELCD_PAPER_ROOT (default results/elcd/records/main). Submit from the repository root (python is taken from the
# PATH; add PYTHON=<interpreter>, ELCD_PAPER_ROOT=<dir> or an ELCD_*_DIR data directory to --export to change the
# defaults):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH experiments/elcd/slurm/figure_appendix.slurm
# Site setup (module load, environment activation) goes here if the job needs it.
cd "${SLURM_SUBMIT_DIR:-.}"
PY=${PYTHON:-python}
JAX_PLATFORMS=cpu PYTHONPATH=$PWD $PY \
    -m experiments.elcd.figures.figure_appendix \
    --out figs/elcd/appendix
echo "== exit $? at $(date)"
