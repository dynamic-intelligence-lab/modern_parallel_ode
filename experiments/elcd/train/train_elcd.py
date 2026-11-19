"""Train one ELCD model on one system with one loss and one rollout solver; write a JSON record and the parameters.

  python experiments/elcd/train/train_elcd.py $(sed -n 1p experiments/elcd/configs/cells/velocity_chain8z.txt)
  python experiments/elcd/train/run_cell.py $(sed -n 1p experiments/elcd/configs/cells/main_chain8z_laser.txt)

The second line runs this driver once per solver of a cell of the paper (run_cell.py). Losses: "vel", the
velocity regression of Jaffe et al. over mini-batches of samples for --epochs epochs; "suffix", the multi-start
rollout loss for --steps optimizer steps. The initial model is the torch initialization drawn from --seed at the
sizes --hidden, --flow-hidden and --flow-steps (the paper's by default; saved in the npz, so that load_theta
rebuilds the shapes; non-default sizes add _h<w>, _fh<w>, _fs<k> to the run name). --solver: "seq" (lax.scan with
backpropagation) or a parallel-in-time solver of losses.SOLVERS through the implicit-function adjoint. Per step the
record keeps the loss, the gradient norm, the iterations and residuals of every solve, the converged count and the
wall time (the first step includes compilation and is marked). The final evaluation (metrics.evaluate) runs the
native and the paper's rollouts on every demonstration. dtype fp64 enables x64 before any array exists; fp32 sets
the matmul precision to highest. The record keeps an empty `diag` (no forward-only diagnostics), and its config keeps
init "scratch" and diag_every 0, the one value each of two options that were removed. One process per run. Run
through Slurm (experiments/elcd/slurm/).

--loss suffix rolls the model out from recorded states of every training demonstration to the end of the
demonstration and compares each rollout with the demonstration after its start (losses.suffix_loss). --starts all
(default) starts at every k in 0..N-2 each step; --starts K draws K distinct starts per demonstration per step from a
numpy Generator seeded by --seed. Each parallel solve starts from that start's last converged solution, held in a
buffer of every start's latest solution (WarmBuffer: device memory up to DEVICE_BUFFER_MAX_BYTES, host memory above),
and from the demonstration's own suffix in latent coordinates, padded with the goal, the first time; the sequential
solver takes no guess. --start-chunk sets how many starts one vmapped solve holds (0: automatic, auto_chunk). Run
names: <system>_suffix[-k<K>]_<init>_<solver>_<dtype>_s<seed> (loss_parts, option_tokens). Per step the log keeps the
loss, the starts, the converged fraction, the iterations per solve (mean, max, quantiles; warm and cold starts
apart), the dropped solves by cause (diverged, out of iterations), the fraction of solves guessed from the buffer
(warm_frac), the ages of the rows used (age_mean, age_max), the wall and the peak device memory. A host warm buffer
is checked against the Slurm job's memory at the start (check_host_buffer).

Further options, each off by default so that a run without them is the run above bit for bit. Loss: --x0-anchor a
adds a times the loss of the rollouts from x_0 (loss = (1 - a) suffix + a anchor), a rising linearly from 0 over
--anchor-ramp steps (default 500), and forces start 0 into every draw; --start-draw stratified draws one start per
window of (N - 1) / K indices (needs --starts K). These two are the recipe of the model of panel A of the ELCD
figure, which no run list of the paper uses. Optimizer: --lr-schedule cosine decays the learning rate from --lr to
--lr-final (default 1e-5) over the run's optimizer steps (--steps; the velocity loss: epochs times the steps per
epoch), logged per step as lr. --save-at S [S ...] writes, at those steps, a full evaluation and the model
(<name>_at<S>.npz with a record <name>_at<S>.json holding the log so far). Names: the loss parts gain -anc<a>,
-strat and -st0 (start 0 forced by the anchor), in that order after k<K>; the option tokens gain cos<steps>
('cos10k' for 10000 steps; the velocity loss cos<epochs>ep; f<lr_final> appended when --lr-final is not the
default).

The paper runs, every option off by
default so that a run without them is the run above bit for bit: --max-iter 0 caps every solve at its own length;
--fallback seq (suffix loss, a parallel solver) gives every solver LASER's stopping rule, gates on
its own residual and re-solves a step whose solves failed with the sequential loop (losses.solve_starts_fallback),
records `solve` (the resolved cap, tol, the protocol) and logs the causes before the fallback plus failed_stalled,
seq_fallback, seq_fallback_solves, n_used, latent_max, chunk_iters, retries and chunk_retries
(LASER's rejected trials per step and the retry bodies its chunks ran; suffix_log_entry); the name gains the token fb;
--solver scale-elk:<g> is constant scale-ELK at rate g on that protocol (named scale-elk<g>). Every record holds
`provenance` (host, Slurm ids, GPU clocks and temperature, git commit; at the start and at the end), and the JSON is
written last and atomically, after the npz.

Long runs: --log-every k > 1 on the suffix losses keeps one log entry per block of k consecutive steps
(thin_block: the block's last entry with the wall and the failure, fallback and retry counts summed, the maxima kept,
the rates and losses averaged, plus steps and step_first), so that a 10^6-step run's record stays the size of a
10^4-step one; the sum of wall over the thinned log equals the sum over the full one. Default 1: every step, as
before. run_cell's completeness check counts log entries, so its cells keep the default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUFFIX_LOSSES = ("suffix",)
LOSSES = ("vel",) + SUFFIX_LOSSES
DTYPES = ("fp64", "fp32")
DEFAULT_TOL = {"fp64": 1e-9, "fp32": 1e-5}
FALLBACKS = ("none", "seq")              # --fallback: the paper runs' sequential re-solve of a failed step
NVIDIA_QUERY = ("name", "uuid", "clocks.sm", "clocks.mem", "clocks.max.sm", "temperature.gpu", "power.draw", "pstate")
START_DRAWS = ("uniform", "stratified")
LR_SCHEDULES = ("const", "cosine")
ANCHOR_RAMP = 500                        # steps over which the anchor's weight rises linearly from 0 to --x0-anchor
LR_FINAL = 1e-5                          # the cosine schedule's final learning rate
FIELD_HIDDEN = 16              # model.HIDDEN, the paper's hidden_dim (not imported: model imports JAX)
FLOW_HIDDEN = 30               # flow_gatherfree.HIDDEN, the conditioner's hidden_features
FLOW_STEPS = 2                 # the paper's flow_steps
DEVICE_BUFFER_MAX_BYTES = 2 ** 30   # a warm buffer above 1 GiB lives in host memory (chain8z: 2.8 GB, npend:16: 19 GB)
HOST_RESERVE_BYTES = 8 * 2 ** 30    # host memory of the process besides a host warm buffer (2.3 to 5.4 GB measured,
                                    # plus the evaluation and the per-step copies)
CHUNK_SOLVE_STEPS = 2 ** 18         # auto_chunk's time cap: solve-steps per vmapped chunk of starts
CHUNK_BUDGET_BYTES = 2 ** 33        # auto_chunk's memory cap: 8 GiB per vmapped chunk of starts
SEQ_CHUNK_BUDGET_BYTES = 24 * 2 ** 30   # auto_chunk's memory cap for the sequential loop, which has no time cap
CHUNK_STEP_CONST, CHUNK_STEP_CUBIC = 1000, 6  # chunk_bytes_per_step, in values per solve-step (a fit to measured peaks)
# the options that run_options reads from a run's config, with their defaults; a record without a key has the default
# (a record written before the option existed)
OPTION_DEFAULTS = dict(hidden=FIELD_HIDDEN,
                       flow_hidden=FLOW_HIDDEN, flow_steps=FLOW_STEPS, starts="all",
                       # the anchor, the draw and the schedule (records written before them have these defaults)
                       x0_anchor=0.0, anchor_ramp=ANCHOR_RAMP, start_draw="uniform", start0=False,
                       lr_schedule="const", lr_final=LR_FINAL)


def parse_args(argv=None):
    from experiments.elcd.train.losses import SOLVERS
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--system", default="lasa:Sshape", help="lasa:<shape> | chain8z | pend:<n> | npend:<links>-N<n>; "
                                                         "data.load_system lists every form")
    p.add_argument("--loss", choices=LOSSES, default="vel")
    p.add_argument("--solver", type=_solver_arg, default="seq",
                   help=f"one of {SOLVERS}, scale-elk:<g> (constant scale-ELK at rate g per unit time) or elk:<s2> (ELK with "
                        "observation variance s2); the last two need --fallback seq")
    p.add_argument("--dtype", choices=DTYPES, default="fp64")
    p.add_argument("--epochs", type=int, default=100, help="velocity loss: epochs over the training samples")
    p.add_argument("--batch", type=int, default=100, help="velocity loss: samples per optimizer step")
    p.add_argument("--steps", type=int, default=3000, help="suffix loss: optimizer steps")
    p.add_argument("--hidden", type=int, default=FIELD_HIDDEN, help="the field's tanh width (P_s and P_a)")
    p.add_argument("--flow-hidden", type=int, default=FLOW_HIDDEN, help="the flow conditioner's width")
    p.add_argument("--flow-steps", type=int, default=FLOW_STEPS, help="the number of flow steps")
    p.add_argument("--starts", type=_starts_arg, default="all",
                   help="suffix losses: 'all' (every k in 0..N-2 each step) or K, random starts per demonstration per step")
    p.add_argument("--start-chunk", type=int, default=0,
                   help="suffix losses: starts per vmapped solve (0: automatic, auto_chunk's time and memory caps; seq: memory only)")
    p.add_argument("--x0-anchor", type=float, default=0.0,
                   help="suffix losses: weight a in [0, 1) of the loss of the rollouts from x_0 (start 0 forced into every draw)")
    p.add_argument("--anchor-ramp", type=int, default=ANCHOR_RAMP, help="steps over which the anchor's weight rises from 0 to --x0-anchor")
    p.add_argument("--start-draw", choices=START_DRAWS, default="uniform",
                   help="suffix losses with --starts K: uniform without replacement, or one start per window of (N - 1) / K indices")
    p.set_defaults(start0=False)          # start 0 in every draw: set by --x0-anchor (_check_anchor_and_schedule)
    p.set_defaults(init="scratch", diag_every=0)   # fixed: the records' config and the run names keep these values
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="const", help="constant, or cosine from --lr to --lr-final")
    p.add_argument("--lr-final", type=float, default=LR_FINAL, help="the cosine schedule's final learning rate")
    p.add_argument("--save-at", type=int, nargs="*", default=[],
                   help="step losses: also evaluate and save the model at these steps (<name>_at<S>.npz and .json)")
    p.add_argument("--clip", type=float, default=1.0, help="global gradient norm clip, every loss (0: off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=None, help="parallel solvers: residual tolerance (default by dtype)")
    p.add_argument("--max-iter", type=int, default=200, help="parallel solvers' iteration cap (0: each solve's own length)")
    p.add_argument("--fallback", choices=FALLBACKS, default="none",
                   help="suffix loss, parallel solvers: 'seq' gives every solver LASER's stopping rule and re-solves a step "
                        "whose solves failed with the sequential loop (the paper runs)")
    p.add_argument("--eval-every", type=int, default=0, help="steps between intermediate evaluations (0: final only)")
    p.add_argument("--log-every", type=int, default=1,
                   help="suffix losses: k > 1 keeps one log entry per block of k steps (thin_block)")
    p.add_argument("--time-only", type=int, default=0, help="if > 0: run this many steps, record wall times, skip evaluation")
    p.add_argument("--outdir", default=str(REPO_ROOT / "results" / "elcd" / "check" / "train"))
    p.add_argument("--out", default=None, help="JSON path (default: <outdir>/<auto name>.json)")
    args = p.parse_args(argv)
    if args.tol is None:
        args.tol = DEFAULT_TOL[args.dtype]
    _check_suffix_options(p, args)
    _check_anchor_and_schedule(p, args)
    _check_paper_options(p, args)
    for key in ("hidden", "flow_hidden", "flow_steps"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be >= 1, got {getattr(args, key)}")
    return args


def canonical_starts(value) -> str:
    """The --starts value in canonical form: 'all', or the decimal integer K >= 1 ('64'), from a command-line
    string, an int, or a CSV's float (64.0). Raises ValueError for anything else."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "all":
            return "all"
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"--starts must be 'all' or an integer K >= 1, got {text!r}") from None
    if not (isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
            and float(value) == int(value) and int(value) >= 1):
        raise ValueError(f"--starts must be 'all' or an integer K >= 1, got {value!r}")
    return str(int(value))


def _starts_arg(text: str) -> str:
    try:
        return canonical_starts(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _solver_arg(text: str) -> str:
    """A name of losses.SOLVERS as it is, or scale-elk:<g> / elk:<s2> with the value in canonical form: %g when that is
    exact ('scale-elk:0.3' for 0.30 or 3e-1), else repr, so that one value has one name."""
    from experiments.elcd.train.losses import ELK, SCALE_ELK, parse_solver
    try:
        base, value = parse_solver(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if base not in (SCALE_ELK, ELK):
        return base
    short = f"{value:g}"
    return f"{base}:{short if float(short) == value else repr(value)}"


def solver_token(solver: str) -> str:
    """The solver's part of a run name: the name with its colon dropped ('scale-elk:0.3' -> 'scale-elk0.3')."""
    return str(solver).replace(":", "")


def _check_suffix_options(p: argparse.ArgumentParser, args) -> None:
    """The suffix options apply to the suffix loss only (each start is rolled out to the end of its demonstration)."""
    if args.start_chunk < 0:
        p.error(f"--start-chunk must be >= 0, got {args.start_chunk}")
    if args.loss not in SUFFIX_LOSSES and (args.starts, args.start_chunk) != ("all", 0):
        p.error("--starts and --start-chunk apply to --loss suffix only")


def _check_anchor_and_schedule(p: argparse.ArgumentParser, args) -> None:
    """The anchor and draw options apply to the suffix losses only; the draw option needs K random starts; the anchor
    forces start 0 (args.start0 records the effective value); the schedule and --save-at apply to every loss but
    --save-at not to the velocity loss (epochs, not steps)."""
    suffix = args.loss in SUFFIX_LOSSES
    used = [flag for flag, on in (("--x0-anchor", args.x0_anchor != 0),
                                  ("--start-draw", args.start_draw != "uniform")) if on]
    if used and not suffix:
        p.error(f"{', '.join(used)}: apply to --loss suffix only")
    if not 0 <= args.x0_anchor < 1:
        p.error(f"--x0-anchor must be in [0, 1), got {args.x0_anchor}")
    if args.anchor_ramp < 0:
        p.error(f"--anchor-ramp must be >= 0, got {args.anchor_ramp}")
    if args.anchor_ramp != ANCHOR_RAMP and args.x0_anchor == 0:
        p.error("--anchor-ramp needs --x0-anchor")
    if suffix:
        if args.x0_anchor > 0 and args.starts != "all":
            args.start0 = True                                          # the anchor needs the x_0 rollout every step
        if args.start_draw != "uniform" and args.starts == "all":
            p.error("--start-draw stratified needs --starts K (every start is drawn with --starts all)")
    if args.log_every < 1:
        p.error("--log-every must be >= 1")
    if args.lr_final != LR_FINAL and args.lr_schedule == "const":
        p.error("--lr-final needs --lr-schedule cosine")
    if args.lr_schedule == "cosine" and not 0 < args.lr_final <= args.lr:
        p.error(f"--lr-final must be in (0, --lr], got {args.lr_final} with --lr {args.lr}")
    if args.save_at and args.loss == "vel":
        p.error("--save-at applies to the suffix loss")
    if any(s < 1 for s in args.save_at):
        p.error(f"--save-at steps must be >= 1, got {args.save_at}")


def _check_paper_options(p: argparse.ArgumentParser, args) -> None:
    """The paper runs' options: --max-iter 0 is each solve's length; --fallback seq needs the suffix
    loss and a parallel solver on LASER's stopping rule; scale-elk:<g> exists only on the shared loop of
    --fallback seq."""
    from experiments.elcd.train.losses import ELK, SCALE_ELK, parse_solver
    if args.max_iter < 0:
        p.error(f"--max-iter must be >= 0 (0: each solve's own length), got {args.max_iter}")
    base, _ = parse_solver(args.solver)
    if args.fallback == "seq":
        if args.loss != "suffix":
            p.error(f"--fallback seq applies to --loss suffix, got --loss {args.loss}")
        if base == "seq":
            p.error(f"--fallback seq needs a parallel solver on LASER's stopping rule (deer, laser, "
                    f"{SCALE_ELK}:<g>, {ELK}:<s2>), got {args.solver}")
    if base in (SCALE_ELK, ELK) and args.fallback != "seq":
        p.error(f"--solver {args.solver} runs on the shared loop of --fallback seq")


def _is_missing(value) -> bool:
    """None, pandas' NA, or a NaN of any float width (a numpy float32 NaN is not a Python float)."""
    if value is None or type(value).__name__ == "NAType":
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def canonical_bool(value) -> bool:
    """A flag from a config (bool), a record (bool) or a CSV cell ('True', 'False', 0, 1); bool('False') is True, so
    strings are read by name."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no", ""):
            return False
        raise ValueError(f"not a flag: {value!r}")
    return bool(value)


CANONICAL = dict(starts=canonical_starts, start0=canonical_bool)     # options whose canonical type is not their default's type


def run_options(cfg: Mapping) -> Dict[str, object]:
    """The options of OPTION_DEFAULTS in a run config (a record's `config`, vars(args), or a table row), each with
    its default when absent or missing (None, NaN: a CSV column that other runs have), with canonical types."""
    out = {}
    for key, default in OPTION_DEFAULTS.items():
        value = cfg.get(key) if hasattr(cfg, "get") else getattr(cfg, key, None)
        value = default if _is_missing(value) else value
        out[key] = CANONICAL.get(key, type(default))(value)
    return out


def _cfg_get(cfg, key: str, default=None):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def loss_parts(cfg: Mapping) -> List[str]:
    """The loss's name parts, which auto_name joins by '-': the loss; for the suffix loss k<K> (K random starts)
    when set."""
    loss = _cfg_get(cfg, "loss")
    parts = [loss]
    if loss in SUFFIX_LOSSES:
        o = run_options(cfg)
        if o["starts"] != "all":
            parts.append(f"k{o['starts']}")
        # loss-changing options: the anchor, stratified draws, start 0 forced
        if o["x0_anchor"] > 0:
            parts.append(f"anc{o['x0_anchor']:g}")
        if o["start_draw"] == "stratified":
            parts.append("strat")
        if o["starts"] != "all" and (o["start0"] or o["x0_anchor"] > 0):
            parts.append("st0")
    return parts


def option_tokens(cfg: Mapping) -> List[str]:
    """The name tokens of the non-default options, in the contract's order: the training token cos<steps>
    (the cosine schedule over the run's steps, any loss; 'cos10k' for 10000, plus f<lr_final> when the final rate is
    not the default), then h<hidden>, fh<flow_hidden>, fs<flow_steps>, then fb (--fallback seq, the paper runs,
    suffix losses). auto_name joins them with '_'."""
    o = run_options(cfg)
    loss = cfg.get("loss") if hasattr(cfg, "get") else getattr(cfg, "loss")
    tokens = []
    if o["lr_schedule"] == "cosine":
        tokens.append(cosine_token(cfg, o["lr_final"]))
    if o["hidden"] != FIELD_HIDDEN:
        tokens.append(f"h{o['hidden']}")
    if o["flow_hidden"] != FLOW_HIDDEN:
        tokens.append(f"fh{o['flow_hidden']}")
    if o["flow_steps"] != FLOW_STEPS:
        tokens.append(f"fs{o['flow_steps']}")
    if loss in SUFFIX_LOSSES and _cfg_get(cfg, "fallback", "none") == "seq":
        tokens.append("fb")                         # the paper runs' protocol; absent from older records
    return tokens


def schedule_steps(cfg: Mapping) -> int:
    """The optimizer steps a run's learning-rate schedule spans: --steps, or for the velocity loss epochs times the
    steps per epoch (the batch count of the training samples, known only to the driver: `n_vel_steps` in the config
    when the driver wrote it, else epochs)."""
    if _cfg_get(cfg, "loss") == "vel":
        n = _cfg_get(cfg, "n_vel_steps")
        return int(n) if n is not None and not _is_missing(n) else int(_cfg_get(cfg, "epochs"))
    return int(_cfg_get(cfg, "steps"))


def cosine_token(cfg: Mapping, lr_final: float) -> str:
    """'cos<steps>' with the step count in thousands when it is a multiple of 1000 ('cos10k', 'cos3k'); for the
    velocity loss, whose optimizer steps depend on the data, 'cos<epochs>ep' ('cos100ep'), so that a job line names
    its run without loading the system; plus 'f<lr_final>' (%g) when the final rate is not LR_FINAL."""
    if _cfg_get(cfg, "loss") == "vel":
        tok = f"cos{int(float(_cfg_get(cfg, 'epochs')))}ep"
    else:
        n = int(float(_cfg_get(cfg, "steps")))
        tok = f"cos{n // 1000}k" if n % 1000 == 0 else f"cos{n}"
    return tok + ("" if lr_final == LR_FINAL else f"f{lr_final:g}")


def auto_name(args) -> str:
    init = args.init
    loss = "-".join(loss_parts(vars(args)))
    opts = "".join("_" + t for t in option_tokens(vars(args)))
    return (f"{args.system.replace(':', '-').replace('+', '_')}_{loss}_{init}_{solver_token(args.solver)}_{args.dtype}"
            f"{opts}_s{args.seed}")


# ----------------------------------------------------------------------------- parameters

def model_sizes(elcd, flow) -> Dict[str, int]:
    """hidden (the field's tanh width), flow_hidden (the conditioner's width) and flow_steps of a model."""
    return dict(hidden=int(np.shape(elcd.W1s)[0]), flow_hidden=int(flow.hidden), flow_steps=int(flow.flow_steps))


def init_theta(args, demos, dtype):
    """The initial Theta in `dtype`: the torch initialization from args.seed at the sizes --hidden, --flow-hidden,
    --flow-steps (the paper's sizes by default)."""
    from experiments.elcd import flow_gatherfree as FG
    from experiments.elcd import model as model_mod
    from experiments.elcd.train.losses import Theta
    D = demos.D
    hidden = int(getattr(args, "hidden", FIELD_HIDDEN))
    flow_hidden = int(getattr(args, "flow_hidden", FLOW_HIDDEN))
    flow_steps = int(getattr(args, "flow_steps", FLOW_STEPS))
    flow = FG.random_flow(args.seed, D, flow_steps=flow_steps, hidden=flow_hidden)
    flow = flow._replace(linears=tuple(lu._replace(bias=np.zeros_like(np.asarray(lu.bias))) for lu in flow.linears))
    elcd = model_mod.init_synthetic(args.seed, D, gain=1.0, rate=0.0, hidden=hidden, z_star=np.asarray(demos.x_eq))
    return Theta(model_mod.cast(elcd, dtype), FG.cast_flow(flow, dtype))


def save_theta(path: str, theta, extra: Dict[str, np.ndarray]):
    """The field's leaves (elcd_<name>), the flow's torch state (flow_state.<key>), `extra`, and the model's sizes
    (hidden, flow_hidden, flow_steps, read from theta itself so that they cannot disagree with the arrays)."""
    from experiments.elcd import flow_gatherfree as FG
    from experiments.elcd.model import ELCDParams
    out = {f"elcd_{name}": np.asarray(leaf) for name, leaf in zip(ELCDParams._fields, theta.elcd)}
    out.update({f"flow_state.{k}": v for k, v in FG.state_from_flow(theta.flow).items()})
    out.update(extra)
    out.update({k: np.int64(v) for k, v in model_sizes(theta.elcd, theta.flow).items()})
    np.savez(path, **out)


def load_theta(path: str):
    """(elcd, flow) in float64 numpy from a save_theta npz. The flow's sizes come from its flow_steps and
    flow_hidden keys, with the paper's sizes when absent (an npz written before these keys existed, which loads as
    before); a stored `hidden` is checked against the field's arrays."""
    from experiments.elcd import flow_gatherfree as FG
    from experiments.elcd.model import ELCDParams
    with np.load(path, allow_pickle=False) as data:
        elcd = ELCDParams(*[np.asarray(data[f"elcd_{name}"], np.float64) for name in ELCDParams._fields])
        state = {k[len("flow_state."):]: np.asarray(data[k]) for k in data.files if k.startswith("flow_state.")}
        D = int(elcd.z_star.shape[0])
        flow_steps = int(data["flow_steps"]) if "flow_steps" in data.files else FLOW_STEPS
        flow_hidden = int(data["flow_hidden"]) if "flow_hidden" in data.files else FLOW_HIDDEN
        hidden = int(data["hidden"]) if "hidden" in data.files else None
    if hidden is not None and hidden != int(elcd.W1s.shape[0]):
        raise ValueError(f"{path}: hidden {hidden} but the field's W1s has {int(elcd.W1s.shape[0])} rows")
    return elcd, FG.flow_from_state(state, D, flow_steps, hidden=flow_hidden)


def load_model(path: str):
    """The saved model of a run as a float64 Theta."""
    import jax.numpy as jnp
    from experiments.elcd import flow_gatherfree as FG
    from experiments.elcd import model as model_mod
    from experiments.elcd.train.losses import Theta
    elcd, flow = load_theta(path)
    return Theta(model_mod.cast(elcd, jnp.float64), FG.cast_flow(flow, jnp.float64))


# ----------------------------------------------------------------------------- the suffix losses

def buffer_bytes(B: int, N: int, L: int, D: int, itemsize: int) -> int:
    """Bytes of the warm buffer: one (L, D) latent solution per start, N - 1 starts per demonstration, B
    demonstrations. fp64, L = N - 1, B 3: N 985 D 2 46.5 MB; chain8z (N 7689, D 2) 2.84 GB; npend:16-N5000 (D 32)
    19.2 GB; N 7880, D 8 11.9 GB (these two are tasks that no run list uses)."""
    return int(B) * (int(N) - 1) * int(L) * int(D) * int(itemsize)


def chunk_bytes_per_step(D: int, itemsize: int) -> int:
    """Peak device memory per solve-step of one vmapped chunk's value and gradient, itemsize x (CHUNK_STEP_CONST +
    CHUNK_STEP_CUBIC D^3). The D^3 term is forward-mode AD pushing D tangents through the matrix-valued field A(z).
    Fit to measured peaks (H100, fp64, the decode rematerialized): 6.2 to 8.1 KB per
    solve-step at D 2 (Sshape all starts, chain8z), 31 to 35 KB at D 8 (N 7880, a synthetic task that no run list
    uses), 1.44 to 1.54 MB at D 32
    (npend:16-N5000, 1 and 2 starts per chunk); the model gives 8.4 KB, 32.6 KB and 1.58 MB."""
    return int(itemsize) * (CHUNK_STEP_CONST + CHUNK_STEP_CUBIC * int(D) ** 3)


def auto_chunk(n_solves: int, L: int, D: int, itemsize: int, budget: Optional[int] = None,
               solve_steps: int = CHUNK_SOLVE_STEPS, sequential: bool = False) -> int:
    """Starts per vmapped solve (lax.map over the chunks), or 0 for one vmap over all starts.

    A chunk holds at most `solve_steps` solve-steps (time: the vmapped Newton loop runs until its slowest start
    converges, so smaller chunks stop sooner; per training step on Sshape with all starts, fp64, 3.71 s unchunked
    against 3.03 s at 123 starts per chunk on an H100, 13.8 s against 10.4 s on an L40S; chain8z with 64 starts, 30.0
    s against 18.6 s at 32 on the H100, 114.5 s against 60.5 s on the L40S) and at most
    `budget` bytes (chunk_bytes_per_step). The starts are split into the fewest chunks within both caps, balanced so
    that no chunk is mostly padding. fp64: Sshape all starts 12 chunks of 246; chain8z K 64 6 of 32; D 8, N 7880
    K 64 6 of 32; npend:16-N5000 1 start per chunk (memory). budget None: CHUNK_BUDGET_BYTES.

    sequential (the vmapped lax.scan with backpropagation: --solver seq, and every solver at L 1) has no time cap and
    the memory cap SEQ_CHUNK_BUDGET_BYTES (budget None). Its starts step in lockstep, so chunks only serialize them;
    a chunk is rematerialized in the backward pass (losses.solve_starts), so the chunks bound the memory, at one
    more forward pass per chunked step (before that, lax.map kept every chunk's activations: 61 GB at D 8, N 7880
    with 256 starts per demonstration). Measured before the rematerialization, on an L40S, fp64:
    Sshape all starts 14.5 s per step in 12 chunks of 246, 6.83 s in 3 of 984, 3.36 s in one vmap (peak 23.7 GB, 8.1 KB
    per solve-step); chain8z K 64 22.7 s in 6 chunks of 32, 7.9 s in one vmap (12.0 GB). The
    model then gives one vmap for both (22.7 GiB and 11.5 GiB)."""
    if budget is None:
        budget = SEQ_CHUNK_BUDGET_BYTES if sequential else CHUNK_BUDGET_BYTES
    time_cap = int(n_solves) if sequential else int(solve_steps) // int(L)
    largest = max(1, min(time_cap, int(budget) // (int(L) * chunk_bytes_per_step(D, itemsize))))
    if largest >= n_solves:
        return 0
    n_chunks = -(-int(n_solves) // largest)
    return -(-int(n_solves) // n_chunks)


def stratified_windows(n: int, K: int) -> np.ndarray:
    """(K + 1,) the window edges floor(i n / K), i = 0..K: K consecutive windows of n / K indices each (every window
    holds at least one index when K <= n)."""
    return (np.arange(K + 1, dtype=np.int64) * n) // K


def draw_starts(rng: np.random.Generator, B: int, N: int, K: Optional[int], draw: str = "uniform",
                start0: bool = False) -> np.ndarray:
    """(B, S) int32 start indices: every k in 0..N-2 for each demonstration (K None, the generator untouched), or
    K distinct k per demonstration drawn uniformly without replacement and sorted, K capped at N - 1 (so K >= N - 1
    draws every start, as K None does).

    With K, draw "stratified" takes one index uniformly from each of K windows of (N - 1) / K consecutive
    indices (stratified_windows), so the starts cover the demonstration evenly; start0 forces start 0 into the draw
    (uniform: 0 plus K - 1 draws from 1..N-2; stratified: the first window's draw is 0). The default (uniform, no
    start0) consumes the generator exactly as before."""
    n = int(N) - 1
    if K is None:
        return np.tile(np.arange(n, dtype=np.int32), (B, 1))
    K = min(int(K), n)
    if draw not in START_DRAWS:
        raise ValueError(f"draw must be one of {START_DRAWS}, got {draw!r}")
    if draw == "uniform" and not start0:
        return np.stack([np.sort(rng.choice(n, size=K, replace=False)) for _ in range(B)]).astype(np.int32)
    if draw == "uniform":
        return np.stack([np.sort(np.concatenate([[0], rng.choice(n - 1, size=K - 1, replace=False) + 1]))
                         for _ in range(B)]).astype(np.int32)
    edges = stratified_windows(n, K)
    out = np.stack([rng.integers(edges[:-1], edges[1:]) for _ in range(B)]).astype(np.int32)   # one per window, sorted
    if start0:
        out[:, 0] = 0
    return out


def _gather_rows(rows, filled, last, starts):
    import jax.numpy as jnp
    b = jnp.arange(rows.shape[0])[:, None]
    return rows[b, starts], filled[b, starts], last[b, starts]


def _scatter_rows(rows, filled, last, starts, new_rows, converged, step):
    import jax.numpy as jnp
    b = jnp.arange(rows.shape[0])[:, None]
    ok = converged > 0
    kept = jnp.where(ok[..., None, None], new_rows, rows[b, starts])
    stamped = jnp.where(ok, jnp.asarray(step, last.dtype), last[b, starts])
    return rows.at[b, starts].set(kept), filled.at[b, starts].set(filled[b, starts] | ok), last.at[b, starts].set(stamped)


def slurm_memory_bytes(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The job's host memory limit in bytes from Slurm's environment: SLURM_MEM_PER_NODE (MiB), or
    SLURM_MEM_PER_CPU (MiB) times SLURM_CPUS_ON_NODE. None outside a Slurm job or when Slurm does not say."""
    env = os.environ if env is None else env
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return int(env["SLURM_MEM_PER_NODE"]) * 2 ** 20
        if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_ON_NODE"):
            return int(env["SLURM_MEM_PER_CPU"]) * int(env["SLURM_CPUS_ON_NODE"]) * 2 ** 20
    except ValueError:
        return None
    return None


def check_host_buffer(nbytes: int, limit: Optional[int], reserve: int = HOST_RESERVE_BYTES) -> None:
    """Refuse, at the start of the run, a host warm buffer that the job's memory cannot hold next to the process.
    The buffer's pages fill as starts are drawn, so without this check a run with too little --mem is killed hours
    in, with no record written (npend:16-N5000: 19.2 GB under the array script's 24 GB). No known limit: no check."""
    if limit is not None and int(nbytes) + int(reserve) > int(limit):
        need = -(-(int(nbytes) + int(reserve)) // 2 ** 30)
        raise SystemExit(f"the warm buffer needs {nbytes / 1e9:.3g} GB of host memory and the process about "
                         f"{reserve / 2 ** 30:.0f} GiB more, but the job has {limit / 2 ** 30:.3g} GiB: submit with "
                         f"--mem={need}G or more")


class WarmBuffer:
    """The latest converged latent solution (L, D) of every start (N - 1 per demonstration) of B training
    demonstrations, whether a start has one yet (`filled`), and the step at which it was written (`last_step`,
    int32, -1 before the first; the ages of the rows used, in the log). In device memory when it holds at most
    DEVICE_BUFFER_MAX_BYTES (on_device None), else in host memory (numpy, after check_host_buffer against the Slurm
    job's memory); gather and scatter do the same in both."""

    def __init__(self, B: int, N: int, L: int, D: int, dtype, on_device: Optional[bool] = None):
        self.nbytes = buffer_bytes(B, N, L, D, np.dtype(dtype).itemsize)
        self.on_device = self.nbytes <= DEVICE_BUFFER_MAX_BYTES if on_device is None else bool(on_device)
        if not self.on_device:
            check_host_buffer(self.nbytes, slurm_memory_bytes())
        shape = (int(B), int(N) - 1, int(L), int(D))
        self._b = np.arange(B)[:, None]
        if self.on_device:
            import jax
            import jax.numpy as jnp
            self.rows, self.filled = jnp.zeros(shape, dtype), jnp.zeros(shape[:2], bool)
            self.last_step = jnp.full(shape[:2], -1, jnp.int32)
            self._gather = jax.jit(_gather_rows)
            self._scatter = jax.jit(_scatter_rows, donate_argnums=(0, 1, 2))
        else:
            self.rows, self.filled = np.zeros(shape, np.dtype(dtype)), np.zeros(shape[:2], bool)
            self.last_step = np.full(shape[:2], -1, np.int32)

    def gather(self, starts: np.ndarray):
        """(rows (B, S, L, D), filled (B, S), last_step (B, S)) of the drawn starts (B, S)."""
        if self.on_device:
            return self._gather(self.rows, self.filled, self.last_step, starts)
        return self.rows[self._b, starts], self.filled[self._b, starts], self.last_step[self._b, starts]

    def scatter(self, starts: np.ndarray, new_rows, converged, step: int = 0) -> None:
        """Store new_rows (B, S, L, D) of the drawn starts that converged (converged (B, S) > 0), mark them filled
        and stamp them with `step`; every other start keeps its row, its flag and its stamp."""
        if self.on_device:
            self.rows, self.filled, self.last_step = self._scatter(self.rows, self.filled, self.last_step, starts,
                                                                   new_rows, converged, np.int32(step))
            return
        ok = np.asarray(converged) > 0
        b, k = np.broadcast_to(self._b, starts.shape)[ok], np.asarray(starts)[ok]
        self.rows[b, k] = np.asarray(new_rows)[ok]
        self.filled[b, k] = True
        self.last_step[b, k] = np.int32(step)

    def block(self) -> None:
        if self.on_device:
            self.rows.block_until_ready()


def anchor_at(step: int, anchor: float, ramp: int) -> float:
    """The anchor's weight at `step`: anchor times min(1, step / ramp) (ramp 0: anchor from the first step)."""
    if anchor <= 0:
        return 0.0
    return float(anchor) * (min(1.0, step / ramp) if ramp > 0 else 1.0)


def make_suffix_step(args, static, tx, x_eq, pos_train, dt: float, chunk: int, warm: bool):
    """The jitted training step of a suffix loss, step(params, opt_state, starts[, rows, filled[, anchor]]) ->
    (params, opt_state, new_rows, out). Guesses: the drawn starts' demonstration suffixes at the current parameters
    (losses.demo_suffix_guesses), replaced by the buffer's rows where `filled` says so (warm: every row the buffer
    holds). new_rows (warm only, else None) holds each start's converged latent path without
    z_0, or its guess where it did not converge. The gradient guard is the rollout losses': no converged start, or a
    non-finite gradient, gives a zero gradient. With --x0-anchor the
    step takes the anchor's current weight as a traced scalar (anchor_at) and out gains loss_suffix, loss_anchor."""
    import jax
    import jax.numpy as jnp
    import optax
    from experiments.elcd.train import losses as L
    L_steps = L.suffix_length(int(pos_train.shape[1]))
    anchored = getattr(args, "x0_anchor", 0.0) > 0
    fallback = getattr(args, "fallback", "none") == "seq"

    def loss_fn(theta, starts, guesses, anchor):
        kw = dict(anchor=anchor)
        return L.suffix_loss(theta, x_eq, pos_train, starts, guesses, dt, args.solver, args.tol, args.max_iter,
                             0, chunk=chunk, fallback=fallback, **kw)

    def step(params, opt_state, starts, rows=None, filled=None, anchor=None):
        guesses = L.demo_suffix_guesses(L.combine(params, static), x_eq, pos_train, starts, L_steps)
        if warm:
            guesses = jnp.where(filled[..., None, None], rows, guesses)
        a = anchor if anchored else 0.0
        (loss, aux), grads = jax.value_and_grad(lambda p: loss_fn(L.combine(p, static), starts, guesses, a),
                                                has_aux=True)(params)
        gnorm = optax.tree.norm(grads)
        usable = (aux["n_ok"] > 0) & jnp.isfinite(gnorm)
        grads = jax.tree_util.tree_map(lambda g: jnp.where(usable, jnp.nan_to_num(g), jnp.zeros_like(g)), grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_rows = jnp.where(aux["mask"][..., None, None] > 0, aux["Z"][:, :, 1:], guesses) if warm else None
        out = dict(loss=loss, grad_norm=gnorm, it=aux["it"], res=aux["res"], mask=aux["mask"], n_pairs=aux["n_pairs"])
        if anchored:
            out.update(loss_suffix=aux["loss_suffix"], loss_anchor=aux["loss_anchor"])
        if fallback:
            out.update(ok_par=aux["ok_par"], seq_fallback=aux["seq_fallback"], latent_max=aux["latent_max"],
                       retries=aux["retries"])
        return params, opt_state, new_rows, out
    return jax.jit(step)


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if values.size else None


LOG_SUMS = ("wall", "seq_fallback", "seq_fallback_solves", "failed_nonfinite", "failed_max_iter", "failed_stalled",
            "failed_warm", "retries", "chunk_retries", "chunk_iters")
LOG_MAXES = ("iters_max", "residual_max", "latent_max", "age_max")
LOG_ANY = ("compile", "clipped")
LOG_ALL = ("finite_grad",)
LOG_LAST = ("step", "H", "lr", "n_starts", "n_solves", "n_pairs", "n_used", "converged", "anchor", "residual_p50",
            "residual_p10", "peak_bytes")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def thin_block(entries: Sequence[Mapping]) -> Dict:
    """One log entry for a block of consecutive steps (--log-every k > 1): the last step's entry with the keys of
    LOG_SUMS summed over the block (the wall, the failed solves, the fallbacks, the retries, the chunk iterations),
    LOG_MAXES at their maximum, compile and clipped true when any step's was, finite_grad true when every step's
    was, LOG_LAST as the last step's, every other numeric key averaged over the steps that have a value, plus
    steps (the block's length) and step_first (the first step's index)."""
    if not entries:
        raise ValueError("thin_block needs at least one entry")
    out = dict(entries[-1])
    for key in list(out):
        if key in LOG_LAST:
            continue
        vals = [e.get(key) for e in entries]
        nums = [v for v in vals if _is_number(v)]
        if key in LOG_SUMS:
            out[key] = sum(nums) if nums else out[key]
        elif key in LOG_MAXES:
            out[key] = max(nums) if nums else None
        elif key in LOG_ANY:
            out[key] = any(bool(v) for v in vals)
        elif key in LOG_ALL:
            out[key] = all(bool(v) for v in vals)
        elif nums and all(v is None or _is_number(v) for v in vals):
            out[key] = float(np.mean(nums))
    out["steps"] = len(entries)
    out["step_first"] = int(entries[0]["step"])
    return out


def suffix_log_entry(step: int, L_steps: int, out: Mapping, filled: Optional[np.ndarray], wall: float,
                     compile_: bool, clip: float, peak: Optional[int], max_iter: Optional[int] = None,
                     ages: Optional[np.ndarray] = None, fallback: Optional[np.ndarray] = None, chunk: int = 0) -> Dict:
    """One step of a suffix run's log: loss, gradient norm, starts per demonstration and solves, the pairs counted,
    the converged count and fraction, the accepted iterations per solve (mean, max, median, 90th percentile; mean
    over the warm starts, those guessed from the buffer, and over the cold ones), the largest residual of a
    converged solve, the wall, whether the step compiled, and the peak device memory so far (when known).

    `filled` is the (B, S) mask of the solves guessed from the buffer (warm_frac its mean); ages (B, S),
    the age in steps of every gathered row (meaningful where used), gives age_mean and age_max over the used rows
    (None when none); fallback (B, S), the filled rows left aside by the age limit, gives fallback_frac (0 without a
    limit).

    The dropped (unconverged) solves of a parallel solver by cause: failed_nonfinite (the solve diverged: its
    residual is NaN or inf; plain DEER then stops early, its update norm being NaN), failed_max_iter (finite
    residual above tol after max_iter iterations), and failed_warm (dropped solves that started from the buffer).
    The sequential loop has residual 0, so a dropped sequential start (a non-finite path) is in neither cause.

    The paper runs (out holds ok_par, --fallback seq): converged, residual_max and the causes describe the
    parallel solves before the fallback (ok_par), max_iter is the resolved cap, and the causes gain failed_stalled
    (finite residual above tol before the cap: a stalled step), so that the three partition the failed solves; the
    entry gains seq_fallback (1 when the loop re-solved the step), seq_fallback_solves (the failed solves it
    replaced), n_used (the starts in the loss after the fallback), latent_max (max|z| over their paths) and
    chunk_iters, the sum over the step's chunks (`chunk` solves each, in draw order; 0: one) of the chunk's largest
    iteration count: a vmapped chunk iterates until its slowest solve stops, so this is the iterations run;
    residual_p50, residual_p10, the median and 10th percentile of the converged solves' own residuals (a linearly
    converging solver lands just under tol, Newton far below it: the accuracy behind the gradients); and, when out
    holds `retries` (B, S) (the parallel solves' rejected trials: LASER's hidden scans, 0 for the other solvers),
    retries, their sum over the step's solves, and chunk_retries, the sum over the chunks of the chunk's largest
    count: a rejected trial runs the retry body for the whole vmapped chunk, so at least this many retry bodies
    ran (solves that retry at different iterations each run a body; the exact count is per iteration,
    n_trials of LASER's history, LaserConfig(return_history=True))."""
    it = np.asarray(out["it"])
    ok = np.asarray(out["mask"]) > 0
    res = np.asarray(out["res"])
    paper = "ok_par" in out
    retries = np.asarray(out["retries"]) if "retries" in out else np.zeros_like(it)
    if paper and not max_iter:
        raise ValueError("the paper runs' log needs the resolved cap (max_iter > 0)")
    par = np.asarray(out["ok_par"], bool) if paper else ok              # the parallel solves' own convergence
    cap = max_iter or np.iinfo(np.int64).max
    warm = np.zeros(ok.shape, bool) if filled is None else np.asarray(filled, bool)
    gnorm = float(out["grad_norm"])
    entry = dict(step=step, H=L_steps, loss=float(out["loss"]), grad_norm=gnorm, clipped=bool(clip > 0 and gnorm > clip),
                 finite_grad=bool(np.isfinite(gnorm)), n_starts=int(it.shape[1]), n_solves=int(it.size),
                 n_pairs=int(round(float(out["n_pairs"]))), converged=int(par.sum()), converged_frac=float(par.mean()),
                 iters_mean=float(it.mean()), iters_max=int(it.max()), iters_p50=float(np.percentile(it, 50)),
                 iters_p90=float(np.percentile(it, 90)), warm_frac=float(warm.mean()),
                 iters_mean_warm=_mean_or_none(it[warm]), iters_mean_cold=_mean_or_none(it[~warm]),
                 residual_max=float(res[par].max()) if par.any() else None, wall=wall, compile=compile_,
                 failed_nonfinite=int(np.sum(~par & ~np.isfinite(res))),
                 failed_max_iter=int(np.sum(~par & np.isfinite(res) & (it >= cap))),
                 failed_warm=int(np.sum(~par & warm)))
    if peak is not None:
        entry["peak_bytes"] = int(peak)
    used_ages = np.asarray(ages, np.int64)[warm] if ages is not None else np.zeros(0, np.int64)
    entry.update(age_mean=float(used_ages.mean()) if used_ages.size else None,
                 age_max=int(used_ages.max()) if used_ages.size else None,
                 fallback_frac=float(np.mean(np.asarray(fallback, bool))) if fallback is not None else 0.0)
    if paper:
        entry.update(failed_stalled=int(np.sum(~par & np.isfinite(res) & (it < cap))),
                     seq_fallback=int(bool(out["seq_fallback"])), seq_fallback_solves=int(np.sum(~par)),
                     n_used=int(ok.sum()), latent_max=float(out["latent_max"]),
                     chunk_iters=chunk_iterations(it.ravel(), chunk),
                     residual_p50=float(np.percentile(res[par], 50)) if par.any() else None,
                     residual_p10=float(np.percentile(res[par], 10)) if par.any() else None,
                     retries=int(retries.sum()), chunk_retries=chunk_iterations(retries.ravel(), chunk))
    return entry


def chunk_iterations(it: np.ndarray, chunk: int) -> int:
    """The sum over consecutive chunks of `chunk` solves (0 or >= n: one chunk) of each chunk's largest count."""
    it = np.asarray(it).ravel()
    size = it.size if chunk <= 0 or chunk >= it.size else int(chunk)
    return int(sum(int(it[i:i + size].max()) for i in range(0, it.size, size)))


def _run_text(cmd: Sequence[str], timeout: float = 15.0) -> Optional[str]:
    """stdout of a short command, or None when it is missing, fails or times out."""
    import subprocess
    try:
        done = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def nvidia_smi_state() -> Optional[List[Dict[str, str]]]:
    """One dict per visible GPU with the NVIDIA_QUERY fields as nvidia-smi prints them (inside a Slurm job only the
    allocated GPU is listed), or None without nvidia-smi (a CPU node)."""
    text = _run_text(["nvidia-smi", f"--query-gpu={','.join(NVIDIA_QUERY)}", "--format=csv,noheader,nounits"])
    if text is None:
        return None
    rows = [[v.strip() for v in line.split(",")] for line in text.strip().splitlines() if line.strip()]
    return [dict(zip(NVIDIA_QUERY, r)) for r in rows if len(r) == len(NVIDIA_QUERY)]


PROVENANCE_MODULES = ("parode.laser", "parode.explicit_parallel", "parode.sequential", "parode.sequential_integrators",
                      "parode.implicit", "experiments.elcd.train.losses", "experiments.elcd.model",
                      "experiments.elcd.flow_gatherfree", "experiments.elcd.train.train_elcd", "__main__")


def module_hashes() -> Dict[str, Dict[str, str]]:
    """The file and sha256 of every solver-path module this process has imported (PROVENANCE_MODULES), so that a
    record names the exact code it ran even if a tree changes between the runs of one cell."""
    import hashlib
    out = {}
    for name in PROVENANCE_MODULES:
        path = getattr(sys.modules.get(name), "__file__", None)
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                out[name] = dict(path=path, sha256=hashlib.sha256(f.read()).hexdigest())
    return out


def provenance() -> Dict:
    """Where and when this process runs: the time, the host, the Slurm job and array ids, the visible GPUs
    with their clocks, temperature and power (nvidia-smi), the repository's commit and whether its tracked files
    differ from it, the hashes of the imported solver modules, argv and the Python version. Never raises; a field it
    cannot read is None."""
    import platform
    import socket
    env = os.environ
    git = ["git", "--no-optional-locks", "-C", str(REPO_ROOT)]          # no index.lock: many cells run at once
    head = _run_text(git + ["rev-parse", "HEAD"])
    status = _run_text(git + ["status", "--porcelain", "--untracked-files=no"])
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), unix=time.time(), host=socket.gethostname(),
                slurm={k: env.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                               "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION")},
                cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"), gpus=nvidia_smi_state(),
                git_commit=head.strip() if head else None, git_dirty=None if status is None else bool(status.strip()),
                modules=module_hashes(), argv=list(sys.argv), python=platform.python_version())


def write_json_atomic(path: str, obj) -> None:
    """json.dump to a temporary file beside path, then rename: a JSON that exists is complete."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def time_fallback(theta, x_eq, pos_train, starts: np.ndarray, dt: float, L_steps: int, chunk: int,
                  reps: int = 3) -> Dict[str, Dict[str, float]]:
    """The paper runs' fallback cost: the forward-only loop (losses.seq_solve in one vmap) over every start of
    a step ("all") and over one chunk ("chunk"), at the given model and draw, the median wall of `reps` calls after
    the compiling one. A fallback step pays the "all" pass; the ratio tests the premise that the loop is latency-bound."""
    import jax
    import jax.numpy as jnp
    from experiments.elcd import flow_gatherfree as FG
    from experiments.elcd.model import make_ode
    from experiments.elcd.train import losses as L
    B, S = starts.shape
    x0s = jnp.asarray(np.asarray(pos_train)[np.arange(B)[:, None], starts].reshape(B * S, -1))
    elcd = L.with_equilibrium(theta, x_eq)
    z0s = jax.vmap(lambda x: FG.phi(theta.flow, x))(x0s)
    loop = jax.jit(jax.vmap(lambda z0: L.seq_solve(make_ode, elcd, z0, dt, L_steps, solver="rk4")))
    out = {}
    for key, width in (("all", B * S), ("chunk", chunk if 0 < chunk < B * S else B * S)):
        sub = z0s[:width]
        jax.block_until_ready(loop(sub))
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(loop(sub))
            walls.append(time.perf_counter() - t0)
        out[key] = dict(starts=int(width), seconds=float(np.median(walls)))
    return out


def peak_device_bytes(device) -> Optional[int]:
    """The device allocator's peak bytes in use so far (GPU), None where the device does not report it (CPU)."""
    try:
        stats = device.memory_stats()
    except Exception:                                   # a backend without allocator statistics
        return None
    return None if not stats or "peak_bytes_in_use" not in stats else int(stats["peak_bytes_in_use"])


def main(argv=None):
    args = parse_args(argv)
    import jax
    if args.dtype == "fp64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp
    import optax
    from experiments.elcd.train import data as D
    from experiments.elcd.train import losses as L
    from experiments.elcd.train import metrics as M

    dtype = jnp.float64 if args.dtype == "fp64" else jnp.float32
    t_start = time.time()
    demos = D.load_system(args.system)
    if args.loss == "vel":                                   # the velocity loss's optimizer steps (cosine_token reads it)
        args.n_vel_steps = args.epochs * math.ceil(demos.n_train * demos.N / args.batch)
    name = auto_name(args)
    out_json = args.out or os.path.join(args.outdir, name + ".json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    device = jax.devices()[0].platform
    device_kind = str(jax.devices()[0].device_kind)               # the GPU model (e.g. NVIDIA L40S): walls compare within one
    print(f"== {name}: {D.describe(demos)} on {device} ({device_kind})", flush=True)

    theta = init_theta(args, demos, dtype)
    params, static = L.partition(theta)                                  # the flow's index buffers are not trained
    x_eq = jnp.asarray(demos.x_eq, dtype)
    xs_all, vs_all = (jnp.asarray(a, dtype) for a in demos.samples())
    pos_train = jnp.asarray(demos.train()[0], dtype)                       # (B, N, D)
    B, N, Dd = pos_train.shape
    if args.lr_schedule == "cosine":
        lr_fn = optax.cosine_decay_schedule(args.lr, schedule_steps(vars(args)), alpha=args.lr_final / args.lr)
        lr_of = lambda t: float(lr_fn(t))
    else:
        lr_fn, lr_of = args.lr, (lambda t: float(args.lr))
    tx = optax.chain(optax.clip_by_global_norm(args.clip) if args.clip > 0 else optax.identity(), optax.adam(lr_fn))
    opt_state = tx.init(params)
    save_at = set(int(s) for s in args.save_at)
    rng = np.random.default_rng(args.seed)
    log: List[Dict] = []
    diag: List[Dict] = []
    evals: List[Dict] = []
    record = dict(name=name, config=vars(args), data=D.describe(demos), device=device, device_kind=device_kind, jax=jax.__version__,
                  trainable=L.count_trainable(theta), provenance=dict(start=provenance()))
    print(f"   trainable parameters {record['trainable']}", flush=True)

    def evaluate(step):
        t0 = time.time()
        ev = M.evaluate(theta, demos, perturb_seed=args.seed)
        summary = M.summarize(ev["rows"])
        evals.append(dict(step=step, rows=ev["rows"], summary=summary, seconds=time.time() - t0))
        # the run so far, in case the job dies (the driver cannot resume, but the log is evidence); removed at the end
        write_json_atomic(out_json + ".partial", dict(record, log=log, diag=diag, evals=evals, partial_at=step))
        print(f"   eval @ {step}: " + ", ".join(f"{k} {v:.4g}" for k, v in summary.items() if k.endswith("_mean")), flush=True)

    def model_extra():
        return dict(x_eq=np.asarray(demos.x_eq), pos_std=np.asarray(demos.pos_std), dt=np.float64(demos.dt),
                    system=np.array(args.system))

    def snapshot(step):
        """--save-at: the full evaluation at `step`, then the model and a record of the run so far under
        <name>_at<step>. The record is named like its files (name <name>_at<step>, so that tables keep it apart from
        the final model), `run` is the run's name and `saved_at` the step; its config is the run's."""
        evaluate(step)
        snap = dict(record, name=f"{name}_at{step}", run=name, log=log, diag=diag, evals=evals,
                    final=evals[-1]["summary"], saved_at=step, train_seconds=time.time() - t_start,
                    total_seconds=time.time() - t_start)
        stem = out_json[:-5] + f"_at{step}"
        save_theta(stem + ".npz", theta, model_extra())                  # the npz first: a JSON that exists has one
        write_json_atomic(stem + ".json", snap)
        print(f"   saved {stem}.npz", flush=True)

    def after_step(step):
        """The intermediate evaluation or snapshot after the (1-based) step `step` of a step loss."""
        if step in save_at and step < args.steps:
            snapshot(step)
        elif args.eval_every and step % args.eval_every == 0 and step != args.steps:
            evaluate(step)

    if args.loss == "vel":
        @jax.jit
        def step_fn(params, opt_state, xb, vb):
            loss, grads = jax.value_and_grad(lambda p: L.velocity_loss(L.combine(p, static), xb, vb, x_eq))(params)
            gnorm = optax.tree.norm(grads)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss, gnorm

        n = int(xs_all.shape[0])
        per_epoch = math.ceil(n / args.batch)
        step = 0
        for epoch in range(args.epochs):
            perm = rng.permutation(n)
            t0 = time.perf_counter()
            losses = []
            for b in range(per_epoch):
                idx = perm[b * args.batch: (b + 1) * args.batch]
                params, opt_state, loss, gnorm = step_fn(params, opt_state, xs_all[idx], vs_all[idx])
                losses.append(float(loss))
                step += 1
                if args.time_only and step >= args.time_only:
                    break
            jax.block_until_ready(params)
            theta = L.combine(params, static)
            log.append(dict(step=step, epoch=epoch, H=0, loss=float(np.mean(losses)), loss_sum=float(np.sum(losses)),
                            wall=time.perf_counter() - t0, compile=epoch == 0))
            if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
                print(f"   epoch {epoch}: mean loss {np.mean(losses):.4g} (sum {np.sum(losses):.4g}), {log[-1]['wall']:.2f} s", flush=True)
            if args.eval_every and (epoch + 1) % args.eval_every == 0 and epoch != args.epochs - 1:
                evaluate(step)
            if args.time_only and step >= args.time_only:
                break
    elif args.loss in SUFFIX_LOSSES:
        L_steps = L.suffix_length(N)
        K = None if args.starts == "all" else int(args.starts)
        S = N - 1 if K is None else min(K, N - 1)
        itemsize = np.dtype(dtype).itemsize
        parallel = args.solver != "seq" and L_steps >= 2                     # L 1: one RK4 step, solved sequentially
        chunk = args.start_chunk or auto_chunk(B * S, L_steps, Dd, itemsize, sequential=not parallel)
        warm = parallel                                                      # the sequential loop takes no guess
        buffer = WarmBuffer(B, N, L_steps, Dd, dtype) if warm else None
        record["suffix"] = dict(L=L_steps, starts_per_demo=S, solves_per_step=B * S, chunk=chunk, warm=warm,
                                buffer_bytes=buffer.nbytes if warm else 0, buffer_on_device=bool(warm and buffer.on_device))
        cap = L_steps if args.max_iter == 0 else args.max_iter               # --max-iter 0: each solve's length
        if args.fallback == "seq":                                           # a fresh generator: the draws stay the run's
            first = draw_starts(np.random.default_rng(args.seed), B, N, K, args.start_draw, args.start0)
            record["solve"] = dict(cap=cap, tol=args.tol, fallback=args.fallback, stop="laser",
                                   fallback_seconds=time_fallback(theta, x_eq, pos_train, first, demos.dt, L_steps,
                                                                  chunk))
            print(f"   paper protocol: cap {cap}, tol {args.tol:g}, fallback {record['solve']['fallback_seconds']}",
                  flush=True)
        print(f"   suffix: L {L_steps}, {S} starts per demonstration ({B * S} solves per step), chunk {chunk or 'none'}, "
              + (f"warm buffer {buffer.nbytes / 1e9:.3g} GB on {'device' if buffer.on_device else 'host'}" if warm
                 else "demonstration guesses"), flush=True)
        step_fn = make_suffix_step(args, static, tx, x_eq, pos_train, demos.dt, chunk, warm)
        anchored = args.x0_anchor > 0
        block: List[Dict] = []          # --log-every k > 1: the entries of the current block of k steps, kept as one
        for step in range(args.steps):
            starts = draw_starts(rng, B, N, K, args.start_draw, args.start0)
            extra = (jnp.asarray(anchor_at(step, args.x0_anchor, args.anchor_ramp), dtype),) if anchored else ()
            t0 = time.perf_counter()
            if warm:
                rows, filled, last = buffer.gather(starts)
                use = np.asarray(filled, bool)
                ages = step - np.asarray(last, np.int64)
                fallback = np.asarray(filled, bool) & ~use
                params, opt_state, new_rows, out = step_fn(params, opt_state, starts, rows, jnp.asarray(use), *extra)
                buffer.scatter(starts, new_rows, out["mask"], step)
                buffer.block()
            else:
                use = ages = fallback = None
                params, opt_state, new_rows, out = step_fn(params, opt_state, starts, None, None, *extra)
            jax.block_until_ready(out["loss"])
            wall = time.perf_counter() - t0
            theta = L.combine(params, static)
            entry = suffix_log_entry(step, L_steps, out, use, wall, step == 0, args.clip,
                                     peak_device_bytes(jax.devices()[0]), cap, ages, fallback, chunk)
            if anchored:
                entry.update(anchor=float(extra[0]), loss_suffix_term=float(out["loss_suffix"]),
                             loss_anchor=float(out["loss_anchor"]))
            if args.lr_schedule != "const":
                entry["lr"] = lr_of(step)
            if args.log_every > 1:
                block.append(entry)
                if len(block) >= args.log_every or step + 1 == args.steps:
                    log.append(thin_block(block))
                    block = []
            else:
                log.append(entry)
            if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
                print(f"   step {step} L {L_steps}: loss {entry['loss']:.4g}, grad {entry['grad_norm']:.3g}, iters mean "
                      f"{entry['iters_mean']:.2f} max {entry['iters_max']}, converged {entry['converged']}/{entry['n_solves']}, "
                      f"warm {entry['warm_frac']:.2f}, {wall:.3f} s", flush=True)
            after_step(step + 1)
            if args.time_only and step + 1 >= args.time_only:
                break
        if block:
            log.append(thin_block(block))                                  # the block a break left open

    record.update(log=log, diag=diag, train_seconds=time.time() - t_start)
    if not args.time_only:
        evaluate(len(log) if args.loss == "vel" else args.steps)
    record.update(evals=evals, final=evals[-1]["summary"] if evals else {}, total_seconds=time.time() - t_start)
    record["provenance"]["end"] = provenance()
    if not args.time_only:
        save_theta(out_json[:-5] + ".npz", theta, model_extra())        # the npz first: a JSON that exists has one
    write_json_atomic(out_json, record)
    if os.path.exists(out_json + ".partial"):
        os.remove(out_json + ".partial")
    print(f"done: {out_json} ({record['total_seconds']:.0f} s)", flush=True)
    return record


if __name__ == "__main__":
    main()
