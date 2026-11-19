"""The step-0 evaluation of a paper-run cell: the initial model of (system, dtype, seed), evaluated as the driver
evaluates a trained one, written as init_eval.json beside the cell's records.

Every run of a cell starts from the same model: train_elcd.init_theta draws it from --seed at the paper's sizes,
whatever the loss (suffix or velocity) and the solver. The driver's first evaluation comes at --eval-every, so no
record carries step 0; this script adds it once per cell, so that the curves of every solver and of both losses
start from one point and can be normalized by it.

  python experiments/elcd/train/run_init_eval.py --system chain8z --dtype fp32 --seed 0 \\
      --out <recdir>/chain8z/fp32/s0/init_eval.json

The record (one JSON, written atomically): name "init_eval"; config, the driver's parse of the same arguments (init
scratch, the paper's sizes); data; device, device_kind, jax, trainable; evals, one entry at step 0 with the rows,
the summary and the seconds the evaluation took; final, that summary; provenance (start and end). It has no log, so
the paper loaders (records) pass over it. An existing --out is kept.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NAME = "init_eval"
DTYPES = ("fp64", "fp32")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--system", required=True, help="the driver's --system (chain8z, npend:8-N5000, lasa:Sshape, pend:8)")
    p.add_argument("--dtype", choices=DTYPES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="the record's path, <cell>/init_eval.json")
    p.add_argument("--hidden", type=int, default=None, help="the field's width (default: the driver's)")
    p.add_argument("--flow-hidden", type=int, default=None, help="the flow conditioner's width (default: the driver's)")
    p.add_argument("--flow-steps", type=int, default=None, help="the flow steps (default: the driver's)")
    return p.parse_args(argv)


def driver_argv(a) -> List[str]:
    """The driver arguments whose parse is the record's config: the cell's system, precision and seed, and the
    sizes when given (init_theta reads hidden, flow_hidden, flow_steps and seed from them)."""
    argv = ["--system", a.system, "--dtype", a.dtype, "--seed", str(a.seed)]
    for flag, value in (("--hidden", a.hidden), ("--flow-hidden", a.flow_hidden), ("--flow-steps", a.flow_steps)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


def init_record(config: Dict, data: Dict, device: str, device_kind: str, jax_version: str, trainable: int,
                rows: List[Dict], summary: Dict, seconds: float, start: Dict, end: Dict) -> Dict:
    """The record's layout (module docstring): the driver's keys, evals with the one step-0 entry, no log."""
    return dict(name=NAME, config=dict(config), data=data, device=device, device_kind=device_kind, jax=jax_version,
                trainable=int(trainable), evals=[dict(step=0, rows=rows, summary=summary, seconds=float(seconds))],
                final=dict(summary), total_seconds=float(seconds), provenance=dict(start=start, end=end))


def main(argv=None) -> int:
    a = parse_args(argv)
    if os.path.exists(a.out):
        print(f"== {a.out} exists, skipped", flush=True)
        return 0
    from experiments.elcd.train.train_elcd import init_theta, parse_args as driver_parse, provenance, write_json_atomic
    dargs = driver_parse(driver_argv(a))
    import jax
    if a.dtype == "fp64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    from experiments.elcd.train import data as D
    from experiments.elcd.train import losses as L
    from experiments.elcd.train import metrics as M
    start = provenance()
    dtype = jnp.float64 if a.dtype == "fp64" else jnp.float32
    demos = D.load_system(a.system)
    device, device_kind = jax.devices()[0].platform, str(jax.devices()[0].device_kind)
    print(f"== {NAME}: {D.describe(demos)} seed {a.seed} {a.dtype} on {device} ({device_kind})", flush=True)
    theta = init_theta(dargs, demos, dtype)
    t0 = time.time()
    ev = M.evaluate(theta, demos, perturb_seed=a.seed)
    summary = M.summarize(ev["rows"])
    seconds = time.time() - t0
    record = init_record(vars(dargs), D.describe(demos), device, device_kind, jax.__version__,
                         L.count_trainable(theta), ev["rows"], summary, seconds, start, provenance())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_json_atomic(a.out, record)
    print("   step 0: " + ", ".join(f"{k} {v:.4g}" for k, v in summary.items() if k.endswith("_mean")), flush=True)
    print(f"done: {a.out} ({seconds:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
