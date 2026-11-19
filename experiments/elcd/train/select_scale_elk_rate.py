"""The rate g of the ELCD settings table: constant scale-ELK's rate per system and precision, from the tuning runs.

Reads the tuning cells that run_cell.py wrote under <tunedir>/<system>/<dtype>/s<seed>/scale-elk<g>.json (seeds 10
and 11, the first 2,000 steps of the paper's recipe, --time-only 2000 --save-at 2000). The rule was fixed before the
runs: per system and precision, the rate with the smallest summed step hours over the seeds, fallbacks included
(the sum of every logged wall, the compiling step included). A rate is a candidate only if every seed's
record is complete. The script also reports, per rate:
- the median step time over steps 1,000 to 2,000;
- the fallback steps and the failed solves by cause;
- the training Chamfer at step 2,000 (the snapshot's evaluation).

It flags a pick at an end of the grid: the protocol then adds one point three times further out.

`parse_extras` reads the KEY=ARGS lines (KEY a system of SYSTEMS or one precision of it) that select_chunk.py --out
writes: the pilot's chunk choices, which went into the main run lists.

  python experiments/elcd/train/select_scale_elk_rate.py <tunedir> [--out <tunedir>/selection.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

RATE = re.compile(r"^scale-elk(?P<g>[0-9.eE+-]+)\.json$")
CAUSES = ("failed_nonfinite", "failed_max_iter", "failed_stalled")


def run_summary(path: Path) -> Optional[Dict]:
    """One tuning record: step hours, the median step over steps 1,000 to 2,000, fallbacks, causes, Chamfer."""
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    log = rec.get("log") or []
    if not log or "end" not in rec.get("provenance", {}):
        return None
    late = [e["wall"] for e in log if 1000 <= e["step"] < 2000 and not e.get("compile")]
    snap = path.with_name(path.stem + "_at2000.json")
    chamfer = None
    if snap.exists():
        try:
            chamfer = json.loads(snap.read_text()).get("final", {}).get("train_chamfer_native_mean")
        except ValueError:
            chamfer = None
    return dict(steps=len(log), step_hours=sum(e["wall"] for e in log) / 3600.0,
                median_late=statistics.median(late) if late else None,
                fallback_steps=sum(int(e.get("seq_fallback", 0)) for e in log),
                **{c: sum(int(e.get(c, 0)) for e in log) for c in CAUSES},
                chamfer_2000=chamfer, device_kind=rec.get("device_kind"))


def select(tunedir: Path, seeds: List[int], steps: int) -> Dict:
    out = {}
    for sys_dir in sorted(p for p in tunedir.iterdir() if p.is_dir()):
        for dt_dir in sorted(p for p in sys_dir.iterdir() if p.is_dir()):
            rates: Dict[float, Dict] = {}
            for seed in seeds:
                cell = dt_dir / f"s{seed}"
                for f in sorted(cell.glob("scale-elk*.json")) if cell.exists() else []:
                    m = RATE.match(f.name)
                    if not m:
                        continue
                    g = float(m.group("g"))
                    rates.setdefault(g, {})[seed] = run_summary(f)
            table = []
            for g in sorted(rates):
                runs = rates[g]
                complete = all(runs.get(s) and runs[s]["steps"] == steps for s in seeds)
                table.append(dict(g=g, complete=complete, step_hours=sum(r["step_hours"] for r in runs.values() if r),
                                  median_late=[runs[s]["median_late"] if runs.get(s) else None for s in seeds],
                                  fallback_steps=sum(r["fallback_steps"] for r in runs.values() if r),
                                  **{c: sum(r[c] for r in runs.values() if r) for c in CAUSES},
                                  chamfer_2000=[runs[s]["chamfer_2000"] if runs.get(s) else None for s in seeds]))
            candidates = [row for row in table if row["complete"]]
            pick = min(candidates, key=lambda row: row["step_hours"]) if candidates else None
            grid = [row["g"] for row in table]
            out[f"{sys_dir.name}/{dt_dir.name}"] = dict(
                table=table, pick=pick["g"] if pick else None,
                at_edge=bool(pick) and (pick["g"] in (min(grid), max(grid))) and len(grid) > 1,
                all_complete=len(candidates) == len(table) and bool(table))
    return out


SYSTEMS = {"chain8z": ("chain8z", range(10), range(3)), "npend-8-N5000": ("npend:8-N5000", range(3), range(0))}


DTYPES = ("fp64", "fp32")


def parse_extras(items: List[str]) -> Dict[str, str]:
    """KEY=ARGS items (the lines of select_chunk.py --out) as {KEY: ARGS}; KEY is a system key of SYSTEMS or one
    precision of it ('chain8z/fp64'), and the ARGS of a repeated KEY are joined. ValueError for any other KEY."""
    keys = set(SYSTEMS) | {f"{k}/{d}" for k in SYSTEMS for d in DTYPES}
    out: Dict[str, str] = {}
    for item in items:
        key, sep, args = item.partition("=")
        if not sep or key not in keys:
            raise ValueError(f"{item!r}: want KEY=ARGS with KEY one of {sorted(keys)}")
        out[key] = f"{out[key]} {args}" if key in out else args
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tunedir")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    sel = select(Path(a.tunedir), a.seeds, a.steps)
    text = json.dumps(sel, indent=1)
    Path(a.out or Path(a.tunedir) / "selection.json").write_text(text + "\n")
    for key, v in sel.items():
        print(f"{key}: pick g = {v['pick']}{' (at the grid edge: add a point 3x further out)' if v['at_edge'] else ''}"
              f"{'' if v['all_complete'] else ' (incomplete cells)'}")
        for row in v["table"]:
            print(f"   g {row['g']:g}: step hours {row['step_hours']:.3f}, fallback steps {row['fallback_steps']}, "
                  f"causes {[row[c] for c in CAUSES]}, median step 1000-2000 {row['median_late']}, "
                  f"Chamfer@2000 {row['chamfer_2000']}, complete {row['complete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
