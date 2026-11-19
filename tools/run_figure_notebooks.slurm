#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --mem=16G -c 4 --time=0:45:00
#SBATCH -J paper_figure_nbs
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Template: fill in YOUR_ACCOUNT and YOUR_CPU_PARTITION (a CPU job). `make figures` runs this file with bash instead,
# which reads the #SBATCH lines as comments (ONLY=lorenz, then ONLY=figure1).
#
# Build and execute the paper-figure notebooks, which write their PDFs and PNGs (ELCD: also the SVGs, the parts and
# the one composed figure) to figs/ at the repository root, and the first figure to experiments/fig1/output/. The
# executed copy of each notebook and its HTML twin go to figs/notebooks; the tracked notebooks stay without outputs.
#   experiments/lorenz/figure_bistable.ipynb       -> figs/lorenz/
#   experiments/elcd/figures/figure_elcd.ipynb     -> figs/elcd/
#   experiments/fig1/figure1_paper_figure.ipynb    -> experiments/fig1/output/
#       (no builder: that notebook is edited directly; its drawing code is experiments/fig1/*.py)
# ONLY=<part of a notebook's path: lorenz, elcd or figure1> runs that notebook alone. With --export=NONE the job
# sees only the listed variables, so ONLY (and CHECK_MEASURE=1) go inside --export: ...,PATH=$PATH,ONLY=figure1
# Fails when a build or nbconvert fails, a cell errored, or a code cell was not executed; prints the figure count.
# CHECK_MEASURE=1 also reruns experiments/lorenz/run_bistable_figure.py into CHECK_DIR (default
# results/lorenz/check/bistable_figure) and compares its numbers with the cache the Lorenz notebook reads.
# Submit from the repository root (python is taken from the PATH and jupyter from the same environment; add
# PYTHON=<interpreter>, ELCD_PAPER_ROOT=<dir> or an ELCD_*_DIR data directory to --export to change the defaults):
#   mkdir -p logs
#   sbatch --parsable --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH tools/run_figure_notebooks.slurm
set -o pipefail
# Site setup (module load, environment activation) goes here if the job needs it.
cd "${SLURM_SUBMIT_DIR:-.}"
REPO=$PWD
PY=${PYTHON:-python}
JP="$PY -m jupyter"
# JUPYTER_PATH puts the kernel of the interpreter's own environment before a user-level kernel of the same name
export JAX_PLATFORMS=cpu JUPYTER_PATH="$($PY -c 'import sys; print(sys.prefix)')/share/jupyter" PYTHONPATH="$REPO"
unset MPLBACKEND                # the kernel must keep the inline backend, or the figures are not embedded
echo "== job $SLURM_JOB_ID on $(hostname), $(date), repo $REPO ($(git rev-parse --short HEAD))"
rc_all=0
ran=0
EXECUTED=figs/notebooks          # the executed copies and their HTML twins
for pair in "experiments/lorenz/build_figure_bistable_notebook.py:experiments/lorenz/figure_bistable.ipynb" \
            "experiments/elcd/figures/build_figure_elcd_notebook.py:experiments/elcd/figures/figure_elcd.ipynb" \
            ":experiments/fig1/figure1_paper_figure.ipynb"; do
    SRC=${pair#*:}
    NB=$EXECUTED/$(basename "$SRC")
    if [ -n "$ONLY" ] && [[ "$SRC" != *"$ONLY"* ]]; then continue; fi
    ran=$((ran+1))
    [ -z "${pair%%:*}" ] || $PY ${pair%%:*} "$SRC" || exit 1
    $JP nbconvert --to notebook --execute --output-dir "$EXECUTED" --allow-errors --ExecutePreprocessor.timeout=-1 "$SRC" || exit 1
    $JP nbconvert --to html --output-dir "$EXECUTED" "$NB" || exit 1
    $PY - "$NB" <<'PYEOF'
import json, sys
nb = json.load(open(sys.argv[1]))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
missing = [i for i, c in enumerate(code) if c.get("execution_count") is None]
pngs = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
errs = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(code) for o in c.get("outputs", [])
        if o.get("output_type") == "error"]
print(sys.argv[1], "| code cells", len(code), "| not executed", len(missing), "| figures", pngs, "| errors", errs)
sys.exit(1 if (errs or missing) else 0)
PYEOF
    rc=$?; [ $rc -eq 0 ] || rc_all=$rc
done
[ $ran -gt 0 ] || { echo "ONLY=$ONLY matches no notebook"; exit 1; }
if [ "$CHECK_MEASURE" = "1" ]; then
    CHK=${CHECK_DIR:-results/lorenz/check/bistable_figure}
    $PY experiments/lorenz/run_bistable_figure.py --outdir "$CHK" || exit 1
    $PY - "$CHK/lorenz_bistable.json" results/lorenz/bistable_figure/lorenz_bistable.json <<'PYEOF'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:])
for k in ("lambda_h_guess", "lambda_h_truth", "gamma"):
    print(f"{k:16s} new {new[k]:.10g}  cache {old[k]:.10g}")
for t in ("undamped", "damped"):
    print(f"kappa {t:9s} new {new[t]['kappa']:.6g}  cache {old[t]['kappa']:.6g}")
for r in old["runs"]:
    print(f"{r:16s} iterations new {new['runs'][r]['iterations']}  cache {old['runs'][r]['iterations']}  "
          f"status new {new['runs'][r]['status']}  cache {old['runs'][r]['status']}")
PYEOF
fi
echo "== exit $rc_all at $(date)"
exit $rc_all
