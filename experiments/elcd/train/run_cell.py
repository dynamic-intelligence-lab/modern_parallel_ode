"""One cell of the ELCD paper runs: one system, precision and seed, every solver on one GPU in one job.

Each run is its own train_elcd.py process with an explicit --out:

1. The probes, first and again after every item, on a fixed workload: chain8z, fp64, seed 0, the paper's recipe,
   20 steps (--time-only 20) from the initial model. Their median walls over steps 1 to 19 calibrate the GPU.
   - The loop probe is the sequential loop, which is latency-bound.
   - The DEER probe is plain DEER on the paper protocol, which is throughput-bound: its first 20 steps are the same
     work every time.
2. The items, in the order of a Williams design: row (seed + precision index) of the balanced Latin square of the
   items, so that across seeds no item always runs first and each follows every other equally often. The items:
   - the parallel solvers on the paper protocol (--fallback seq --max-iter 0 --tol <1e-7 fp64, 1e-5 fp32>);
   - the loop's timing, --solver seq --time-only <loop_time>.
3. The full loop run (--full-loop), last; it draws the loop's curves.

**Recipe.** The paper's, for every run: --loss suffix --starts 64 --lr-schedule cosine --steps <steps>, uniform
draws, no anchor. --time-only and --save-at apply to the parallel solvers: the pilot and the tuning of the constant.
--loop-chunk and --par-chunk set --start-chunk of the loop and of the parallel solvers (0: the driver's auto_chunk).

**GPU log.** While a run lasts, nvidia-smi logs the SM and memory clocks, the power, the temperature, the utilization
and the throttle reasons every 60 s to gpu_<run>.csv.

**Skips.** An item whose record is complete is skipped, because the driver cannot resume. Complete means:
- the JSON parses (the driver writes it last and atomically);
- its config is the item's: the driver's parse of the item's arguments, every key but out and outdir;
- it ran on the GPU (unless --allow-cpu) and holds the end provenance;
- its log has the item's steps (--time-only, else --steps);
- a full run also has its final evaluation and its npz.
Any other record at an item's path is moved aside to <name>.stale-<time>.json and the item runs again. Probes are
never skipped.

**Outputs.** Records go to <recdir>/<system>/<dtype>/s<seed>/<item>.json, probes to probe_<kind>_<k>.json (numbered
on, never overwritten). The cell's summary, cell.json, is rewritten after every run (--cell-tag T: cellT.json and
probeT_..., for a second job on the same cell). It holds:
- the order;
- every run's arguments, exit code and seconds;
- each probe's median;
- the runner's provenance.

**The code and the GPU.** The repository the runner lives in must have no modified tracked files (the cells run from
a snapshot worktree of one commit; --allow-dirty for tests). Runs get JAX_PLATFORMS=cuda, and a record whose device is
not the GPU stops the cell (--allow-cpu for tests).

  python experiments/elcd/train/run_cell.py --system chain8z --dtype fp32 --seed 0 \\
      --solvers deer,laser,scale-elk:0.3 --recdir <dir> [--full-loop] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DRIVER = HERE / "train_elcd.py"
PYTHON = sys.executable                       # the interpreter that runs this script also runs the driver
DTYPES = ("fp64", "fp32")
TOL = {"fp64": "1e-7", "fp32": "1e-5"}        # the tolerances of the paper runs, passed explicitly
PROBE_STEPS = 20
PROBE_KINDS = ("loop", "deer")
EVAL_EVERY = 500
GPU_LOG_SECONDS = 60
GPU_FIELDS = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
GPU_THROTTLE = ",clocks_throttle_reasons.active"   # dropped when this nvidia-smi does not know it


def paper_recipe(steps: int) -> List[str]:
    """The paper's recipe: the plain suffix loss, 64 uniform starts per demonstration, cosine over `steps`."""
    return ["--loss", "suffix", "--starts", "64", "--lr-schedule", "cosine", "--steps", str(steps)]


def paper_protocol(dtype: str) -> List[str]:
    """The parallel solvers' protocol: cap N, LASER's stopping rule and the sequential fallback, the tolerance TOL."""
    return ["--fallback", "seq", "--max-iter", "0", "--tol", TOL[dtype]]


def item_name(solver: str, loop_time: int = 0, full: bool = False) -> str:
    """The record's file stem: the solver without its colon ('scale-elk0.3'), loop_time<n> or loop_full."""
    if solver == "seq":
        return "loop_full" if full else f"loop_time{loop_time}"
    return solver.replace(":", "")


def cell_dir(recdir: str, system: str, dtype: str, seed: int) -> Path:
    return Path(recdir) / system.replace(":", "-").replace("+", "_") / dtype / f"s{seed}"


def williams(n: int) -> List[List[int]]:
    """The rows of a Williams design on n items: each row is an order, each item is in each position once per n rows
    and directly follows every other item equally often (n rows for even n, 2n for odd n)."""
    if n <= 1:
        return [list(range(n))]
    first = [0]
    lo, hi = 1, n - 1
    while len(first) < n:
        first.append(lo)
        lo += 1
        if len(first) < n:
            first.append(hi)
            hi -= 1
    rows = [[(x + r) % n for x in first] for r in range(n)]
    return rows if n % 2 == 0 else rows + [row[::-1] for row in rows]


def item_order(n: int, seed: int, dtype: str) -> List[int]:
    rows = williams(n)
    return rows[(int(seed) + DTYPES.index(dtype)) % len(rows)]


def plan_items(a) -> List[Dict]:
    """The cell's runs after the first probes, in order: the items in their Williams order, then the full loop run.
    Each item is {name, args (driver argv without --out), kind ('parallel', 'loop_time', 'loop_full'), time_only}."""
    common = ["--system", a.system, "--dtype", a.dtype, "--seed", str(a.seed)] + paper_recipe(a.steps)
    par_chunk = ["--start-chunk", str(a.par_chunk)] if a.par_chunk else []
    loop_chunk = ["--start-chunk", str(a.loop_chunk)] if a.loop_chunk else []
    items = []
    for s in a.solvers:
        extra = ["--solver", s, "--eval-every", str(a.eval_every)] + paper_protocol(a.dtype) + par_chunk
        if a.time_only:
            extra += ["--time-only", str(a.time_only)]
        if a.save_at:
            extra += ["--save-at"] + [str(x) for x in a.save_at]
        items.append(dict(name=item_name(s), args=common + extra, kind="parallel", time_only=a.time_only))
    if a.loop_time:
        items.append(dict(name=item_name("seq", a.loop_time), kind="loop_time", time_only=a.loop_time,
                          args=common + ["--solver", "seq", "--time-only", str(a.loop_time), "--eval-every", "0",
                                         "--tol", TOL[a.dtype]] + loop_chunk))
    items = [items[i] for i in item_order(len(items), a.seed, a.dtype)]
    if a.full_loop:
        items.append(dict(name=item_name("seq", full=True), kind="loop_full", time_only=0,
                          args=common + ["--solver", "seq", "--eval-every", str(a.eval_every), "--tol", TOL[a.dtype]]
                          + loop_chunk))
    return items


def probe_args(kind: str) -> List[str]:
    """The calibration probes: chain8z, fp64, seed 0, the paper's recipe, 20 steps; the loop or DEER on the
    protocol."""
    base = ["--system", "chain8z", "--dtype", "fp64", "--seed", "0"] + paper_recipe(10000)
    tail = ["--time-only", str(PROBE_STEPS), "--eval-every", "0"]
    if kind == "loop":
        return base + ["--solver", "seq", "--tol", TOL["fp64"]] + tail
    return base + ["--solver", "deer"] + paper_protocol("fp64") + tail


def load_record(path: Path) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def intended_config(args: Sequence[str]) -> Dict:
    """vars() of the driver's parse of an item's arguments (imports JAX, creates no array)."""
    sys.path.insert(0, str(REPO))
    from experiments.elcd.train.train_elcd import parse_args
    return vars(parse_args(list(args)))


def record_complete(path: Path, time_only: int, args: Optional[Sequence[str]] = None, gpu: bool = True) -> bool:
    """The record exists and is the item's finished run (see the module docstring); args None skips the config
    check (the old-record tests)."""
    rec = load_record(path)
    if not rec or not rec.get("log") or "end" not in rec.get("provenance", {}):
        return False
    if gpu and rec.get("device") != "gpu":
        return False
    if args is not None:
        want, got = intended_config(args), rec.get("config", {})
        if any(got.get(k) != v for k, v in want.items() if k not in ("out", "outdir")):
            return False
        if len(rec["log"]) != int(time_only or want["steps"]):
            return False
    elif time_only and len(rec["log"]) != int(time_only):
        return False
    if time_only:
        return True
    return bool(rec.get("final")) and path.with_suffix(".npz").exists()


def set_aside(path: Path) -> Optional[Path]:
    """Move a record that is not the item's finished run (and its npz) aside, so the item runs again."""
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.stem}.stale-{stamp}.json")
    os.replace(path, dest)
    if path.with_suffix(".npz").exists():
        os.replace(path.with_suffix(".npz"), dest.with_suffix(".npz"))
    return dest


def median_wall(rec: Dict) -> Optional[float]:
    """The median wall of the steps that did not compile (the probe's statistic), None without such steps."""
    walls = [e["wall"] for e in rec.get("log", []) if not e.get("compile")]
    return float(statistics.median(walls)) if walls else None


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def runner_provenance() -> Dict:
    """The job as the runner sees it: host, Slurm ids, the GPU (nvidia-smi), the repository's commit."""
    sys.path.insert(0, str(REPO))
    from experiments.elcd.train.train_elcd import provenance     # no JAX import: parse_args is not called
    return provenance()


def tree_is_clean(repo: Path) -> bool:
    """No modified tracked file in the repository (untracked files are allowed)."""
    done = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True, check=False)
    return done.returncode == 0 and not done.stdout.strip()


class GpuLog:
    """nvidia-smi logging every GPU_LOG_SECONDS to a CSV while a run lasts (nothing without nvidia-smi)."""

    def __init__(self, path: Path):
        self.proc = None
        for fields in (GPU_FIELDS + GPU_THROTTLE, GPU_FIELDS):
            try:
                out = open(path, "w")
                proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv",
                                         "-l", str(GPU_LOG_SECONDS)], stdout=out, stderr=subprocess.DEVNULL)
            except OSError:
                return
            time.sleep(2)
            if proc.poll() is None or proc.returncode == 0:
                self.proc = proc
                return

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Cell:
    def __init__(self, a):
        self.a = a
        self.dir = cell_dir(a.recdir, a.system, a.dtype, a.seed)
        self.summary_path = self.dir / f"cell{a.cell_tag}.json"
        old = load_record(self.summary_path) or {}
        self.summary = dict(system=a.system, dtype=a.dtype, seed=a.seed, solvers=a.solvers, order=None,
                            runs=old.get("runs", []), probes=old.get("probes", []), provenance=old.get("provenance", []))
        self.env = dict(os.environ, JAX_PLATFORMS="cpu" if a.allow_cpu else "cuda")

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.summary_path, self.summary)

    def run(self, name: str, args: Sequence[str], path: Path) -> Dict:
        cmd = [self.a.python, str(DRIVER)] + list(args) + ["--out", str(path)]
        print(f"== {time.strftime('%F %T')} {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        gpu = GpuLog(self.dir / f"gpu_{name}.csv") if not self.a.allow_cpu else None
        try:
            rc = subprocess.run(cmd, env=self.env, check=False).returncode
        finally:
            if gpu is not None:
                gpu.stop()
        rec = load_record(path)
        entry = dict(name=name, path=str(path), args=list(args), rc=rc, seconds=time.time() - t0,
                     device=rec.get("device") if rec else None, finished=time.strftime("%F %T"))
        print(f"== {name}: exit {rc} after {entry['seconds']:.0f} s, device {entry['device']}", flush=True)
        return entry

    def check_device(self, entry: Dict) -> None:
        if entry["rc"] == 0 and entry["device"] != "gpu" and not self.a.allow_cpu:
            self.save()
            raise SystemExit(f"{entry['name']} ran on {entry['device']}, not the GPU: the cell stops")

    def probe(self, after: str) -> None:
        k = sum(1 for p in self.summary["probes"] if p.get("kind") == PROBE_KINDS[0])
        for kind in PROBE_KINDS:
            name = f"probe{self.a.cell_tag}_{kind}_{k:02d}"
            path = self.dir / f"{name}.json"
            entry = self.run(name, probe_args(kind), path)
            rec = load_record(path) or {}
            entry.update(kind=kind, after=after, median_wall=median_wall(rec), steps=len(rec.get("log", [])))
            self.summary["probes"].append(entry)
            self.save()
            self.check_device(entry)
            if after == "start" and (entry["rc"] != 0 or entry["steps"] != PROBE_STEPS):
                raise SystemExit(3)      # the first probe failed: a machine problem, not a solver's; stop the cell

    def main(self) -> int:
        items = plan_items(self.a)
        self.summary["order"] = [it["name"] for it in items]
        if self.a.dry_run:
            print(json.dumps(dict(dir=str(self.dir), probes={k: probe_args(k) for k in PROBE_KINDS}, items=items),
                             indent=1))
            return 0
        if not self.a.allow_dirty and not tree_is_clean(REPO):
            raise SystemExit(f"{REPO}: git status --porcelain --untracked-files=no must succeed and print nothing (the "
                             f"cells run from a snapshot worktree of one commit). After make figures, git checkout -- "
                             f"figs experiments/fig1/output restores the redrawn PDFs; --allow-dirty skips this check")
        self.summary["provenance"].append(runner_provenance())
        self.save()
        if self.a.probes:
            self.probe("start")
        failed = 0
        for it in items:
            path = self.dir / f"{it['name']}.json"
            gpu = not self.a.allow_cpu
            if record_complete(path, it["time_only"], it["args"], gpu):
                print(f"== {it['name']}: complete record {path}, skipped", flush=True)
                continue
            stale = set_aside(path)
            if stale:
                print(f"== {it['name']}: {path} is not this item's finished run, moved to {stale}", flush=True)
            entry = self.run(it["name"], it["args"], path)
            entry.update(kind=it["kind"], complete=record_complete(path, it["time_only"], it["args"], gpu))
            failed += not entry["complete"]
            self.summary["runs"].append(entry)
            self.save()
            self.check_device(entry)
            if self.a.probes:
                self.probe(it["name"])
        self.summary["provenance"].append(runner_provenance())
        self.save()
        print(f"== cell {self.dir}: {failed} incomplete run(s)", flush=True)
        return 1 if failed else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--system", required=True, help="chain8z | npend:8-N5000 | any driver --system")
    p.add_argument("--dtype", choices=DTYPES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--solvers", default="deer,laser", help="comma-separated parallel solvers (scale-elk:<g> allowed)")
    p.add_argument("--recdir", required=True, help="records go to <recdir>/<system>/<dtype>/s<seed>/")
    p.add_argument("--steps", type=int, default=10000, help="the cosine schedule's length (every run)")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--loop-time", type=int, default=200, help="steps of the loop's timing item (0: none)")
    p.add_argument("--full-loop", action="store_true", help="also the loop's full run, last")
    p.add_argument("--loop-chunk", type=int, default=0, help="--start-chunk of the loop's runs (0: auto_chunk)")
    p.add_argument("--par-chunk", type=int, default=0, help="--start-chunk of the parallel solvers (0: auto_chunk)")
    p.add_argument("--time-only", type=int, default=0, help="parallel solvers: stop after this many steps (pilot, tuning)")
    p.add_argument("--save-at", type=int, nargs="*", default=[], help="parallel solvers: snapshot steps (tuning)")
    p.add_argument("--no-probes", dest="probes", action="store_false", help="skip the calibration probes")
    p.add_argument("--cell-tag", default="", help="suffix of cell.json and the probes' files, for a second job on the "
                                                   "same cell (the pendulum's full loop run: _loop)")
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--allow-cpu", action="store_true", help="tests: run on the CPU")
    p.add_argument("--allow-dirty", action="store_true", help="tests: run from a tree with modified tracked files")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = p.parse_args(argv)
    a.solvers = [s for s in a.solvers.split(",") if s]
    if "seq" in a.solvers or len(set(a.solvers)) != len(a.solvers):
        p.error(f"--solvers lists distinct parallel solvers (the loop is --loop-time / --full-loop), got {a.solvers}")
    if min(a.steps, a.eval_every, a.loop_time, a.time_only, a.loop_chunk, a.par_chunk) < 0:
        p.error("--steps, --eval-every, --loop-time, --time-only and the chunks must be >= 0")
    return a


def main(argv=None) -> int:
    return Cell(parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
