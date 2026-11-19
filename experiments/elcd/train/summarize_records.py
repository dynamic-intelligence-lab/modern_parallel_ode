"""One row per run of the ELCD paper runs' cells (run_cell.py's records), for reading a phase as it finishes.

For every <root>/<system>/<dtype>/s<seed>/<item>.json (probes and snapshots excluded; unlike the loaders of
figures/records.py, a record set aside as <item>.stale-<time>.json gets a row too) the row holds:
- the steps, the compiling step's wall, the median wall of the other steps and the step hours (every wall summed, the
  compiling step included);
- the steps on which the loop re-solved the step and the failed parallel solves by cause;
- the mean iterations, the largest latent |z|, the peak device memory, the GPU and the host;
- the training Chamfer of the last evaluation.

The cell's probes (median wall each) are listed below its runs.

  python experiments/elcd/train/summarize_records.py <root> [--csv <file>]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.elcd.record_names import item_name  # noqa: E402

CAUSES = ("failed_nonfinite", "failed_max_iter", "failed_stalled")


def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_row(path: Path) -> Optional[Dict]:
    rec = _load(path)
    if not rec or not rec.get("log"):
        return None
    log = rec["log"]
    walls = [e["wall"] for e in log if not e.get("compile")]
    evals = rec.get("evals") or []
    chamfer = (rec.get("final") or {}).get("train_chamfer_native_mean")
    if chamfer is None and evals:
        chamfer = evals[-1]["summary"].get("train_chamfer_native_mean")
    host = ((rec.get("provenance") or {}).get("start") or {}).get("host")
    return dict(item=item_name(path.stem), steps=len(log), compile_s=sum(e["wall"] for e in log if e.get("compile")),
                median_s=statistics.median(walls) if walls else None, step_hours=sum(e["wall"] for e in log) / 3600,
                fallback_steps=sum(int(e.get("seq_fallback", 0)) for e in log),
                fallback_solves=sum(int(e.get("seq_fallback_solves", 0)) for e in log),
                **{c: sum(int(e.get(c, 0)) for e in log) for c in CAUSES},
                iters_mean=statistics.mean(e["iters_mean"] for e in log) if "iters_mean" in log[0] else None,
                latent_max=max((e.get("latent_max", 0.0) for e in log), default=None),
                peak_gb=max((e.get("peak_bytes", 0) for e in log), default=0) / 1e9,
                chamfer=chamfer, gpu=rec.get("device_kind"), host=host)


def collect(root: Path) -> List[Dict]:
    rows = []
    for cell in sorted(root.glob("*/*/s*")):
        system, dtype, seed = cell.parent.parent.name, cell.parent.name, cell.name
        for f in sorted(cell.glob("*.json")):
            if f.name.startswith(("cell", "probe")) or "_at" in f.stem:
                continue
            row = run_row(f)
            if row:
                rows.append(dict(system=system, dtype=dtype, seed=seed, **row))
        for summary in sorted(cell.glob("cell*.json")):
            for p in (_load(summary) or {}).get("probes", []):
                after = item_name(p["after"]) if p.get("after") else p.get("after")       # the item before it
                rows.append(dict(system=system, dtype=dtype, seed=seed, item=f"{p['name']} (after {after})",
                                 median_s=p.get("median_wall"), steps=p.get("steps")))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root")
    p.add_argument("--csv", default=None)
    a = p.parse_args(argv)
    rows = collect(Path(a.root))
    cols = ["system", "dtype", "seed", "item", "steps", "compile_s", "median_s", "step_hours", "fallback_steps",
            "fallback_solves", *CAUSES, "iters_mean", "latent_max", "peak_gb", "chamfer", "gpu", "host"]
    fmt = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    print("\t".join(cols))
    for r in rows:
        print("\t".join(fmt(r.get(c)) for c in cols))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
