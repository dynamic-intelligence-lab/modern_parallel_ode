"""The chunk C of the ELCD settings table: the parallel solvers' chunk per system and precision, from the pilot runs.

The loop runs at its fastest chunk that fits: on the pendulum all 192 starts in one vmap, 3.06 s per step against
11.2 s (fp32) and 20.5 s (fp64) at the driver's automatic chunk (pilot); chain8z's automatic chunk is
already one vmap. The parallel solvers get the same treatment: one chunk for every parallel solver of a system and
precision, the fastest in the pilot.

A vmapped chunk iterates until its slowest solve stops, so a large chunk spends work on solves that have stopped (in
the pilot the chunks ran 3 to 7 times the mean iterations); a small one pays more launches. The rule, fixed before
the chunk sweep: per system and precision, among the chunks with complete DEER and LASER records at seed 0, the chunk
with the smallest wall over the late half of the steps (step >= half the steps), DEER's and LASER's summed. The late
half stands for the 10,000-step runs, most of whose steps are past the cold start. A pick at either end of the tested
chunks is flagged; it is still used.

Reads <root>/<system>/<dtype>/s0/{deer,laser}.json under every root given (the pilot and its chunk variants);
the chunk is the record's suffix.chunk (0: one vmap of all the step's solves). Prints the table. With --out it writes
the lines SYSTEM/DTYPE=--par-chunk K (select_scale_elk_rate.parse_extras reads them); the picks went into the main run lists
as --par-chunk.

  python experiments/elcd/train/select_chunk.py <root> [<root> ...] [--systems chain8z] [--out <file>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.elcd.record_names import item_name  # noqa: E402

SOLVERS = ("deer", "laser")


def _load(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def late_seconds(rec: Dict) -> Optional[float]:
    """The summed wall of the steps at or after half the steps run; None for a record that did not finish them."""
    log = rec.get("log") or []
    config = rec.get("config") or {}
    steps = int(config.get("time_only") or config.get("steps") or 0)
    if not log or len(log) != steps or "end" not in (rec.get("provenance") or {}):
        return None
    return float(sum(e["wall"] for e in log if e["step"] >= steps // 2))


def chunk_table(roots: Sequence[Path]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """{'<system>/<dtype>': {chunk: {solver: late seconds}}} over the complete seed-0 DEER and LASER records. A second
    record of the same system, precision, chunk and solver is reported and left out."""
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for root in roots:
        for f in sorted(Path(root).glob("*/*/s0/*.json")):
            solver = item_name(f.stem)
            if solver not in SOLVERS:
                continue
            rec = _load(f)
            sec = late_seconds(rec) if rec else None
            if sec is None:
                continue
            suffix = rec.get("suffix") or {}
            chunk = int(suffix.get("chunk") or 0) or int(suffix.get("solves_per_step") or 0)
            key = f"{f.parents[2].name}/{f.parents[1].name}"
            cell = out.setdefault(key, {}).setdefault(chunk, {})
            if solver in cell:
                print(f"note: {key} chunk {chunk}: a second {solver} record, {f}, left out")
                continue
            cell[solver] = sec
    return out


def pick(table: Dict[int, Dict[str, float]]) -> Dict:
    """The chunk with the smallest summed late wall of DEER and LASER among the chunks that have both (a tie goes to
    the larger chunk); at_edge when it is the smallest or the largest of those chunks."""
    full = {c: sum(v[s] for s in SOLVERS) for c, v in table.items() if all(s in v for s in SOLVERS)}
    if not full:
        return dict(pick=None, at_edge=False, seconds={})
    best = min(full, key=lambda c: (full[c], -c))
    return dict(pick=best, at_edge=len(full) > 1 and best in (min(full), max(full)), seconds=full)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("roots", nargs="+")
    p.add_argument("--systems", nargs="+", default=None, help="only these systems (directory names, e.g. chain8z)")
    p.add_argument("--out", default=None, help="write the lines SYSTEM/DTYPE=--par-chunk K here")
    a = p.parse_args(argv)
    table = chunk_table([Path(r) for r in a.roots])
    lines = []
    for key in sorted(table):
        if a.systems and key.split("/")[0] not in a.systems:
            continue
        res = pick(table[key])
        print(f"{key}: pick chunk {res['pick']}{' (at an end of the tested chunks)' if res['at_edge'] else ''}")
        for chunk in sorted(table[key]):
            row = table[key][chunk]
            total = res["seconds"].get(chunk)
            print(f"   chunk {chunk:4d}: " + ", ".join(f"{s} late {row[s]:.1f} s" if s in row else f"{s} missing"
                                                     for s in SOLVERS)
                  + (f", summed {total:.1f} s" if total is not None else "") + (" <- pick" if chunk == res["pick"] else ""))
        if res["pick"] is not None:
            lines.append(f"{key}=--par-chunk {res['pick']}")
    if a.out:
        Path(a.out).write_text("".join(line + "\n" for line in lines))
        print(f"wrote {a.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
