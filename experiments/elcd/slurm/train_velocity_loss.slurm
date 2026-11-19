#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH -c 4 --mem=24G --time=04:00:00 --open-mode=append
#SBATCH -J elcd_appx_vel
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Template: fill in YOUR_ACCOUNT and YOUR_GPU_PARTITION; one GPU per task. The chain8z velocity arms of the paper ran
# on one NVIDIA B200 each (add your site's option for that GPU model, for example --constraint=<model>); the appendix
# velocity arm, which reports no time, ran on mixed GPU models.
#
# The velocity-loss arm of the appendix: the ELCD paper's recipe in our trainer, one run per line of CELLS, each line
# the full argument list of experiments/elcd/train/train_elcd.py including --out. Array task t runs the lines
# (t-1)*BLOCK+1 to t*BLOCK in sequence on one GPU and skips a line whose --out file already exists (the driver writes
# it at the end of a run). Submit from the repository root with a clean environment (inside an interactive allocation
# its SLURM_* variables would leak into the job; python is taken from the PATH, add PYTHON=<interpreter> or an
# ELCD_*_DIR data directory to --export to change the defaults):
#   mkdir -p logs
#   env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=$PATH sbatch --parsable --array=1-<n> \
#       --export=NONE,HOME=$HOME,USER=$USER,PATH=$PATH,CELLS=experiments/elcd/configs/cells/<name>.txt,BLOCK=<lines per task> \
#       experiments/elcd/slurm/train_velocity_loss.slurm
# Run lists of the paper:
#   appendix_vel.txt               186 lines  --array=1-6   BLOCK=31  the original ELCD tasks (30 LASA shapes, pend:8)
#   velocity_chain8z.txt            32 lines  --array=1-32  BLOCK=1   the velocity arms on chain8z
#   velocity_chain8z_cos_s3-9.txt    7 lines  --array=1-7   BLOCK=1   the cosine velocity arm on chain8z, seeds 3 to 9
# Site setup (module load, environment activation) goes here if the job needs it.
: "${CELLS:?set CELLS to the run list}" "${BLOCK:?set BLOCK to the lines per task}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$PWD
PY=${PYTHON:-python}
mkdir -p logs
unset JAX_PLATFORMS
first=$(( (SLURM_ARRAY_TASK_ID - 1) * BLOCK + 1 ))
last=$(( SLURM_ARRAY_TASK_ID * BLOCK ))
echo "== job $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID) on $(hostname), $(date), repo $REPO ($(git -C "$REPO" rev-parse --short HEAD)), lines $first-$last of $CELLS"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>/dev/null || true
rc=0
for line in $(seq "$first" "$last"); do
    ARGS=$(sed -n "${line}p" "$CELLS")
    [ -z "$ARGS" ] && continue
    OUT=$(echo "$ARGS" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    if [ -n "$OUT" ] && [ -s "$OUT" ]; then echo "== line $line: $OUT exists, skipped"; continue; fi
    [ -n "$OUT" ] && mkdir -p "$(dirname "$OUT")"
    echo "== $(date) line $line: $ARGS"
    $PY experiments/elcd/train/train_elcd.py $ARGS || rc=$?
done
echo "== exit $rc at $(date)"
exit $rc
