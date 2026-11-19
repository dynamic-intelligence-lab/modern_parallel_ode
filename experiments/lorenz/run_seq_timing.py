"""Wall-clock time of the sequential solver on the bistable Lorenz problem, the sequential row of tab:lorenz.

The problem of the table (run_bistable_figure.build: rho 10, x0 (1, 1, 1), explicit Euler dt 0.01, fp64) and the
table's timing protocol (run_benchmark.timed: one compile run, then the median of REPEATS runs): the jitted
lax.scan of the Euler step (parode.sequential.seq_integrate), the call with which the benchmark driver computes its
sequential solution. The solution's last state is checked against the truth_end column of the table's records
(cap200/summary.csv). The device column holds str(jax.devices()[0]), the JAX name of the device (cuda:0 in the
shipped file, written on an NVIDIA B200; TFRT_CPU_0 on a CPU), not the model name that the driver records.

  python experiments/lorenz/run_seq_timing.py OUT.csv
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from experiments.lorenz import run_bistable_figure as R        # noqa: E402  (sets jax_enable_x64)
from experiments.lorenz.run_benchmark import timed               # noqa: E402
from parode.sequential import seq_integrate                      # noqa: E402
import jax                                                       # noqa: E402

DT, REPEATS = 0.01, 10
OUT = sys.argv[1]
if os.path.exists(OUT):                  # the same job was submitted to two partitions; the first to finish is kept
    print("exists, not rerun:", OUT)
    sys.exit(0)
print("devices", jax.devices(), flush=True)
ref = {int(r["N"]): r for r in csv.DictReader(open(f"{ROOT}/results/lorenz/table_timing/cap200/summary.csv"))
       if r["method"] == "laser"}
rows = []
for N in (1_000, 10_000, 100_000):
    ode, _, x0, _, _, _, _ = R.build(N, DT)
    assert np.allclose(np.asarray(x0), [float(v) for v in ref[N]["x0"].split()]), (x0, ref[N]["x0"])
    fn = jax.jit(lambda a, ode=ode, N=N: seq_integrate(ode, a, DT, N, "euler"))
    x, wall, compile_s = timed(fn, (x0,), REPEATS)
    end = np.asarray(x[-1])
    truth_end = np.array([float(v) for v in ref[N]["truth_end"].split()])
    assert np.allclose(end, truth_end, rtol=0, atol=1e-12), (N, end, truth_end)
    rows.append(dict(N=N, method="seq", wall_s=wall, compile_s=compile_s, repeats=REPEATS, device=str(jax.devices()[0]),
                     end_minus_truth_end=float(np.max(np.abs(end - truth_end)))))
    print(f"  N {N}: {wall * 1e3:.3f} ms (median of {REPEATS}), compile {compile_s:.2f} s, "
          f"|end - truth_end| {rows[-1]['end_minus_truth_end']:.1e}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("wrote", OUT)
