"""Load the GPU-scaling result files and compute the per-seed speedups and their mean and std over the seeds.

Importing the module does the work and prints a summary: it reads gpu_scaling_<gpu>.npz of the six GPU types and
gpu_scaling_mg_b200_<n>x.npz of n = 2, 4, 8 from results/gpu_scaling/, the shipped files, or from the folder that
the environment variable GPU_SCALING_RESULTS names (results/gpu_scaling/check for new runs), builds the frame `long`
(one row per GPU, number of devices, algorithm, N and seed; N is T in the code and the files) and aggregates it into
`agg` (mean, sample standard deviation and counts over the seeds). figure_single_gpu.py and table_multi_gpu.py
import it. The code was taken from the notebook that drew the paper's figure, which is not in the repository.

In `long` and `agg` the algorithm 'jacobi' is the Jacobi scan: the 'jacobi_scan' rows of the single-GPU files and
the rows of the multi-GPU files, each time rescaled to the sweeps needed (x iters_to_tol / 120). The 'jacobi' rows of
the single-GPU files (the while loop) give iteration counts only and are not used here.
"""
# ---- style and names
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The style of the paper's figure.
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.4,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
results_dir = os.environ.get('GPU_SCALING_RESULTS', os.path.join(ROOT, 'results', 'gpu_scaling'))
if not os.path.isdir(results_dir):
    raise RuntimeError(f'results dir not found: {results_dir}')
print('Loading from', results_dir)

GPU_ORDER = ['a40', 'a6000', 'l40s', 'rtx_pro_blackwell', 'h100', 'b200']
GPU_LABELS = {
    'a40': 'A40 (Ampere)',
    'a6000': 'RTX A6000 (Ampere)',
    'l40s': 'L40S (Lovelace)',
    'rtx_pro_blackwell': 'RTX Pro 6000 (Blackwell)',
    'h100': 'H100 (Hopper)',
    'b200': 'B200 (Blackwell)',
}
ALG_ORDER = ['jacobi', 'quasi_deer', 'full_deer']
ALG_LABELS = {
    'jacobi': 'Jacobi',
    'quasi_deer': 'Quasi-DEER',
    'full_deer': 'Full-DEER',
}
T_TARGETS = [100_000, 1_000_000, 10_000_000]
T_LABELS = {100_000: r'$N = 10^5$', 1_000_000: r'$N = 10^6$', 10_000_000: r'$N = 10^7$'}
MG_DEVICES = [1, 2, 4, 8]

# Methodology constants for the apples-to-apples scan rescaling.
SCAN_MAX_ITERS = 120  # all scan-based runs use lax.scan(length=120)

# ---- loading
def load_single_gpu(gpu: str) -> pd.DataFrame | None:
    path = os.path.join(results_dir, f'gpu_scaling_{gpu}.npz')
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    cols = {k: np.asarray(d[k]) for k in d.files}
    cols['algorithm'] = np.array([str(a) for a in cols['algorithm']])
    df = pd.DataFrame(cols)
    df['gpu'] = gpu
    df['n_devices'] = 1
    return df


def load_multi_gpu_b200(n_devices: int) -> pd.DataFrame | None:
    """Load scan-based multi-GPU sweep ``gpu_scaling_mg_b200_<n>x.npz`` (max_iters fixed; we rescale below)."""
    path = os.path.join(results_dir, f'gpu_scaling_mg_b200_{n_devices}x.npz')
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    keep = ['T', 'D', 'rho', 'tol', 'n_devices', 'seed',
            'iters_to_conv', 'wall_clock_ms', 'compile_time_ms', 'max_iters']
    cols = {k: np.asarray(d[k]) for k in keep if k in d.files}
    df = pd.DataFrame(cols)
    df = df.rename(columns={
        'wall_clock_ms': 'wall_clock_120iter_ms',
        'compile_time_ms': 'compile_ms',
        'iters_to_conv': 'iters_to_tol',
    })
    df['algorithm'] = 'jacobi'
    df['gpu'] = 'b200'
    df['oom'] = False
    df['source_file'] = os.path.basename(path)
    return df


single_dfs = {g: load_single_gpu(g) for g in GPU_ORDER}
single_dfs = {g: df for g, df in single_dfs.items() if df is not None}
print(f'Single-GPU files found: {sorted(single_dfs)}')
for g, df in single_dfs.items():
    seeds = sorted(int(s) for s in df['seed'].unique())
    print(f'  {g}: {len(df)} rows, seeds={seeds}')

mg_dfs = {n: load_multi_gpu_b200(n) for n in [2, 4, 8]}
mg_dfs = {n: df for n, df in mg_dfs.items() if df is not None}
print(f'\nMulti-GPU B200 files found: {sorted(mg_dfs)}')
for n, df in mg_dfs.items():
    seeds = sorted(int(s) for s in df['seed'].unique())
    Ts = sorted(df['T'].unique())
    src = df['source_file'].iloc[0]
    max_iters = int(df['max_iters'].iloc[0]) if 'max_iters' in df.columns else 'N/A'
    print(f'  {n}x: {len(df)} rows, seeds={seeds}, max_iters={max_iters}, source={src}')

# ---- per-seed speedups
# Build a unified long-form frame with per-seed speedup.
#
# Methodology:
#   - Jacobi (single-GPU): use the scan-based runner (algorithm='jacobi_scan'),
#     report effective_wall = scan_wall_ms × iters_to_tol / SCAN_MAX_ITERS.
#     Equivalent to lax.scan(length=iters_to_tol).
#   - Quasi-DEER, Full-DEER (single-GPU): use while_loop runner directly
#     (no scan equivalent; DEER's collective is on a different operator).
#   - Multi-GPU Jacobi: scan-based, same rescale as single-GPU Jacobi.
#     The B200 Jacobi bar of the figure therefore equals the 1x B200 row of
#     the multi-GPU table.
#
# We deliberately do NOT use the while_loop Jacobi rows: the multi-GPU runner
# has only the scan (lax.while_loop inside shard_map cannot pipeline
# collectives), and using scan everywhere gives uniform methodology across
# single-GPU and multi-GPU.

def _b200_seq_lookup() -> dict[tuple[int, int], float]:
    """(T, seed) -> seq_rollout_ms on single B200 (jacobi_scan rows)."""
    df_b200 = single_dfs.get('b200')
    if df_b200 is None:
        return {}
    j = df_b200[df_b200['algorithm'] == 'jacobi_scan']
    return {(int(r.T), int(r.seed)): float(r.seq_rollout_ms)
            for r in j.itertuples()}


_seq_b200 = _b200_seq_lookup()


def _b200_seq_for(T: int, seed: int) -> float:
    return _seq_b200[(T, seed)]


def _emit_singlegpu_rows(g: str, df: pd.DataFrame, rows: list) -> None:
    """For each algo, emit rows: jacobi -> rescaled scan; DEER -> while_loop."""
    # Jacobi: scan-based, rescaled.
    js = df[df['algorithm'] == 'jacobi_scan']
    for r in js.itertuples():
        if bool(r.oom) or not r.wall_ms or np.isnan(r.wall_ms):
            wall = float('nan')
            speedup = float('nan')
        else:
            iters = int(r.iters_to_tol) if r.iters_to_tol > 0 else SCAN_MAX_ITERS
            wall = float(r.wall_ms) * iters / SCAN_MAX_ITERS
            speedup = float(r.seq_rollout_ms) / wall if wall else float('nan')
        rows.append({
            'gpu': g, 'n_devices': 1, 'algorithm': 'jacobi',
            'T': int(r.T), 'seed': int(r.seed),
            'wall_ms': wall, 'wall_scan_ms': float(r.wall_ms),
            'seq_ms': float(r.seq_rollout_ms),
            'speedup': speedup, 'oom': bool(r.oom),
        })

    # DEER variants: while_loop, used directly.
    for alg in ('quasi_deer', 'full_deer'):
        ds = df[df['algorithm'] == alg]
        for r in ds.itertuples():
            if bool(r.oom) or not r.wall_ms or np.isnan(r.wall_ms):
                speedup = float('nan')
            else:
                speedup = float(r.seq_rollout_ms) / float(r.wall_ms)
            rows.append({
                'gpu': g, 'n_devices': 1, 'algorithm': alg,
                'T': int(r.T), 'seed': int(r.seed),
                'wall_ms': float(r.wall_ms), 'wall_scan_ms': float('nan'),
                'seq_ms': float(r.seq_rollout_ms),
                'speedup': speedup, 'oom': bool(r.oom),
            })


def _emit_multigpu_rows(df: pd.DataFrame, rows: list) -> None:
    for r in df.itertuples():
        seq_ms = _b200_seq_for(int(r.T), int(r.seed))
        max_iters = int(r.max_iters) if 'max_iters' in df.columns else SCAN_MAX_ITERS
        iters = int(r.iters_to_tol) if int(r.iters_to_tol) > 0 else max_iters
        wall_full = float(r.wall_clock_120iter_ms)
        wall = wall_full * iters / max_iters
        speedup = (seq_ms / wall if wall and not np.isnan(wall)
                   else float('nan'))
        rows.append({
            'gpu': 'b200', 'n_devices': int(r.n_devices), 'algorithm': 'jacobi',
            'T': int(r.T), 'seed': int(r.seed),
            'wall_ms': wall, 'wall_scan_ms': wall_full,
            'seq_ms': float(seq_ms),
            'speedup': speedup, 'oom': False,
        })


rows: list[dict] = []
for g, df in single_dfs.items():
    _emit_singlegpu_rows(g, df, rows)
for df in mg_dfs.values():
    _emit_multigpu_rows(df, rows)

long = pd.DataFrame(rows)
print(f'Combined long-form: {len(long)} rows.')

# ---- aggregation
# Aggregate to mean / std / count across seeds.  Keep all cells, including
# those where every seed OOMed (mean/std come out NaN, all_oom flags them).
agg = (
    long.groupby(['gpu', 'n_devices', 'algorithm', 'T'])
        .agg(
            speedup_mean=('speedup', 'mean'),
            speedup_std=('speedup', 'std'),
            n_seeds=('seed', 'nunique'),
            n_valid=('speedup', lambda s: int(np.sum(~np.isnan(s)))),
            any_oom=('oom', 'any'),
            all_oom=('oom', 'all'),
        )
        .reset_index()
)
agg['speedup_std'] = agg['speedup_std'].fillna(0.0)

print(f'Aggregated: {len(agg)} cells.')
print('Valid-seed counts:')
print(agg['n_valid'].value_counts().sort_index().to_string())
print('Cells with all seeds OOM:', int(agg['all_oom'].sum()))

# ---- helpers of the figure and the table
def _gpu_cmap(n: int) -> np.ndarray:
    return plt.cm.magma(np.linspace(0.15, 0.85, n))


def _lookup_cell(gpu: str, n_devices: int, alg: str, T: int):
    sel = agg[(agg['gpu'] == gpu) & (agg['n_devices'] == n_devices)
              & (agg['algorithm'] == alg) & (agg['T'] == T)]
    if len(sel) == 0:
        return None
    return sel.iloc[0]


