"""The ELCD paper runs' training path: the suffix loss with the sequential fallback (losses.solve_starts_fallback),
its step log, and the driver's options, names and records. CPU, float64.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from experiments.elcd import flow_gatherfree as FG  # noqa: E402
from experiments.elcd import model as model_mod  # noqa: E402
from experiments.elcd.train import data as D  # noqa: E402
from experiments.elcd.train import losses as L  # noqa: E402
from experiments.elcd.train import train_elcd as T  # noqa: E402

TOL = 1e-12


def _synthetic_demos(seed=0, N=40, D_=2, rest=5):
    """Four demonstrations that decelerate into a common end point and rest there."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N - rest)[:, None]
    base = np.stack([np.cos(3 * t[:, 0]), np.sin(2 * t[:, 0])], axis=1) * (1 - t) ** 2
    pos = np.stack([base + 0.05 * rng.standard_normal((1, D_)) * (1 - t) ** 2 for _ in range(4)])
    pos = np.concatenate([pos, np.repeat(pos[:, -1:], rest, axis=1)], axis=1)
    dt = 0.01
    vel = np.zeros_like(pos)
    vel[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
    return D.normalize("synthetic", pos, vel, dt, n_train=3)


def _theta(seed: int, D_: int, x_eq: np.ndarray) -> L.Theta:
    flow = FG.random_flow(seed, D_, flow_steps=2)
    elcd = model_mod.init_synthetic(seed, D_, gain=1.0, rate=0.0, z_star=x_eq)
    return L.Theta(model_mod.cast(elcd, jnp.float64), FG.cast_flow(flow, jnp.float64))


def _case(seed: int = 11, N: int = 24):
    d = _synthetic_demos(N=N)
    theta = _theta(seed, 2, d.x_eq)
    return d, theta, jnp.asarray(d.x_eq), jnp.asarray(d.train()[0])


def _starts():
    return jnp.asarray([[0, 4, 9, 15], [2, 6, 11, 21], [1, 3, 8, 17]])


def _guesses(theta, x_eq, pos, starts):
    return jax.jit(lambda th, st: L.demo_suffix_guesses(th, x_eq, pos, st, pos.shape[1] - 1))(theta, starts)


def _loss_and_grad(theta, x_eq, pos, starts, guesses, dt, solver="deer", tol=TOL, max_iter=100, chunk=0, **kw):
    """The suffix loss, its aux and its gradient with respect to the trained leaves only, flattened."""
    params, static = L.partition(theta)

    def f(p):
        return L.suffix_loss(L.combine(p, static), x_eq, pos, starts, guesses, dt, solver, tol, max_iter, 0,
                             chunk=chunk, **kw)
    (loss, aux), grads = jax.jit(jax.value_and_grad(f, has_aux=True))(params)
    return float(loss), aux, np.concatenate([np.ravel(np.asarray(g)) for g in jax.tree_util.tree_leaves(grads)])


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture
def loop_calls(monkeypatch):
    """Count the runs of the fallback's loop branch: losses.seq_solve wrapped with a host callback (the callback runs
    only when lax.cond takes that branch)."""
    calls = []
    real = L.seq_solve

    def counted(*args, **kw):
        jax.debug.callback(lambda: calls.append(1))
        return real(*args, **kw)
    monkeypatch.setattr(L, "seq_solve", counted)
    return calls


# ----------------------------------------------------------------------------- the loss

@pytest.mark.parametrize("solver", ["deer", "laser"])
@pytest.mark.parametrize("chunk", [0, 5])
def test_fallback_without_failures_gives_the_loss_without_it(solver, chunk, loop_calls):
    """Every solve converges: the fallback path gives the loss and gradient of the path without it, keeps every
    start, and never runs the loop (in one vmap and in chunks of 5 of the 12 solves, the last one padded). aux
    carries the parallel solves' retries (int32, >= 0; 0 for DEER, which has none)."""
    d, theta, x_eq, pos = _case()
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    loss0, aux0, g0 = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, solver, chunk=chunk)
    loss1, aux1, g1 = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, solver, chunk=chunk, fallback=True)
    assert np.all(np.asarray(aux0["mask"]) == 1) and np.all(np.asarray(aux1["mask"]) == 1)
    assert np.all(np.asarray(aux1["ok_par"])) and not bool(aux1["seq_fallback"]) and not loop_calls
    assert abs(loss1 - loss0) <= 1e-10 * abs(loss0) and _rel(g1, g0) <= 1e-8
    assert float(jnp.max(jnp.abs(aux1["Z"] - aux0["Z"]))) <= 1e-10
    assert 0 < float(aux1["latent_max"]) == float(jnp.max(jnp.abs(aux1["Z"])))
    retries = np.asarray(aux1["retries"])
    assert retries.shape == aux1["it"].shape and retries.dtype == np.int32 and np.all(retries >= 0)
    assert solver != "deer" or np.all(retries == 0)
    assert "retries" not in aux0


@pytest.mark.parametrize("chunk", [0, 5])
def test_a_forced_failure_gives_the_loop_loss_and_gradient(chunk, loop_calls):
    """Cap 1 at tol 1e-12: the parallel solves fail (all but at most two starts near a demonstration's end, whose
    guess or first Newton step already passes), so the loop re-solves the step once (not once per chunk) and the loss and gradient are the
    sequential solver's (the IFT adjoint at the loop's path equals its BPTT; the converged start's parallel path is
    the loop's within tol)."""
    d, theta, x_eq, pos = _case(seed=18)
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    loss_s, aux_s, g_s = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "seq", chunk=chunk)
    loop_calls.clear()                                  # the seq solver's own rollouts call seq_solve too
    loss_f, aux_f, g_f = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "deer", max_iter=1, chunk=chunk,
                                        fallback=True)
    ok = np.asarray(aux_f["ok_par"])
    assert ok.sum() <= 2 and bool(aux_f["seq_fallback"]) and len(loop_calls) == 1
    assert np.all(np.asarray(aux_f["mask"]) == 1) and np.all(np.asarray(aux_f["it"]) <= 1)   # 0: its guess passed
    assert abs(loss_f - loss_s) <= 1e-10 * abs(loss_s) and _rel(g_f, g_s) <= 1e-8
    assert float(jnp.max(jnp.abs(aux_f["Z"] - aux_s["Z"]))) <= 1e-12


def test_a_nan_guess_falls_back_for_that_start_only():
    """A NaN guess makes one parallel solve non-finite: the loop re-solves the step, that start takes the loop's path
    and stays in the loss, the others keep their parallel paths, and the gradient is finite and equals the gradient
    with a clean guess (every path converged to the same fixed point)."""
    d, theta, x_eq, pos = _case(seed=18)
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    loss_c, aux_c, g_c = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, fallback=True)
    loss_b, aux_b, g_b = _loss_and_grad(theta, x_eq, pos, starts, guesses.at[0, 1].set(jnp.nan), d.dt, fallback=True)
    ok = np.asarray(aux_b["ok_par"])
    assert not ok[0, 1] and ok.sum() == ok.size - 1 and bool(aux_b["seq_fallback"])
    assert not np.isfinite(np.asarray(aux_b["res"])[0, 1]) and np.all(np.asarray(aux_b["mask"]) == 1)
    assert np.all(np.isfinite(g_b)) and abs(loss_b - loss_c) <= 1e-10 * abs(loss_c) and _rel(g_b, g_c) <= 1e-8


def test_scale_elk_and_the_cap_on_the_fallback_path():
    """scale-elk:<g> converges to the fixed point DEER finds; near-Jacobi scale-ELK (g 1e6) needs up to L sweeps and
    converges under --max-iter 0 (cap L = 23), while a cap of 5 sends the step to the loop."""
    d, theta, x_eq, pos = _case()
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    loss_d, aux_d, g_d = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, fallback=True)
    loss_e, aux_e, g_e = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "scale-elk:1", fallback=True)
    assert np.all(np.asarray(aux_e["ok_par"])) and abs(loss_e - loss_d) <= 1e-10 * abs(loss_d) and _rel(g_e, g_d) <= 1e-8
    assert float(np.mean(np.asarray(aux_e["it"]))) >= float(np.mean(np.asarray(aux_d["it"])))
    _, aux_j, _ = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "scale-elk:1000000", max_iter=0,
                                 fallback=True)
    assert np.all(np.asarray(aux_j["ok_par"])) and int(np.max(np.asarray(aux_j["it"]))) <= pos.shape[1] - 1
    _, aux_5, _ = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "scale-elk:1000000", max_iter=5,
                                 fallback=True)
    assert bool(aux_5["seq_fallback"]) and np.all(np.asarray(aux_5["mask"]) == 1)


def test_scale_elk_needs_the_laser_stop():
    with pytest.raises(ValueError):
        L.solver_kwargs("scale-elk:1")
    assert L.solver_kwargs("scale-elk:0.3", stop="laser") == {"scale_rate": 0.3, "stop": "laser"}
    assert L.solver_kwargs("deer", stop="laser") == {"stop": "laser"} and L.solver_kwargs("deer") == {}
    for bad in ("scale-elk", "scale-elk:", "scale-elk:-1", "scale-elk:nan", "scale-elk:x", "elk", "elk:0", "elk:-1",
                "elk:inf", "deer2"):
        with pytest.raises(ValueError):
            L.parse_solver(bad)
    assert L.parse_solver("elk:0.01") == ("elk", 0.01) and L.parse_solver("scale-elk:0") == ("scale-elk", 0.0)
    assert L.solver_kwargs("elk:100", stop="laser") == {"elk_sigma2": 100.0, "stop": "laser"}
    with pytest.raises(ValueError):
        L.solver_kwargs("elk:100")


# ----------------------------------------------------------------------------- the step log

def test_paper_log_entry_partitions_the_failed_solves():
    """Four solves: converged; non-finite; finite at the cap; finite and stalled before the cap. The loop re-solved
    the step, so every start is used."""
    out = dict(loss=1.0, grad_norm=0.5, n_pairs=40.0, it=np.array([[3, 2, 7, 4]]),
               res=np.array([[1e-13, np.inf, 1e-3, 2e-12]]), mask=np.ones((1, 4)),
               ok_par=np.array([[True, False, False, False]]), seq_fallback=True, latent_max=2.5)
    e = T.suffix_log_entry(0, 7, out, None, 0.1, False, 1.0, None, max_iter=7)
    assert (e["converged"], e["failed_nonfinite"], e["failed_max_iter"], e["failed_stalled"]) == (1, 1, 1, 1)
    assert (e["seq_fallback"], e["seq_fallback_solves"], e["n_used"], e["latent_max"]) == (1, 3, 4, 2.5)
    assert e["residual_max"] == 1e-13 and e["converged_frac"] == 0.25 and e["chunk_iters"] == 7
    e2 = T.suffix_log_entry(0, 7, out, None, 0.1, False, 1.0, None, max_iter=7, chunk=2)
    assert e2["chunk_iters"] == 3 + 7 and T.chunk_iterations(np.array([1, 5, 2, 2, 9]), 2) == 5 + 2 + 9
    assert (e["retries"], e["chunk_retries"]) == (0, 0)                # out without retries: a solver without any
    with_retries = dict(out, retries=np.array([[2, 0, 5, 1]], np.int32))
    e3 = T.suffix_log_entry(0, 7, with_retries, None, 0.1, False, 1.0, None, max_iter=7, chunk=2)
    assert (e3["retries"], e3["chunk_retries"]) == (8, 2 + 5) and type(e3["retries"]) is type(e3["chunk_retries"]) is int
    assert e["failed_nonfinite"] + e["failed_max_iter"] + e["failed_stalled"] == e["n_solves"] - e["converged"]
    assert e["residual_p50"] == e["residual_p10"] == 1e-13
    with pytest.raises(ValueError):                                # the paper log needs the resolved cap
        T.suffix_log_entry(0, 7, out, None, 0.1, False, 1.0, None, max_iter=0)
    old = T.suffix_log_entry(0, 7, {k: v for k, v in out.items() if k not in ("ok_par", "seq_fallback", "latent_max")},
                             None, 0.1, False, 1.0, None, max_iter=7)
    assert not {"failed_stalled", "seq_fallback", "seq_fallback_solves", "n_used", "latent_max"} & set(old)


# ----------------------------------------------------------------------------- the driver

def _run(tmp_path, monkeypatch, extra, sub="runs"):
    d = _synthetic_demos(N=20)
    monkeypatch.setattr(D, "load_system", lambda spec: d)
    argv = ["--system", "synthetic", "--outdir", str(tmp_path / sub)] + extra
    return T.main(argv), d


@pytest.mark.parametrize("solver", ["deer", "laser", "scale-elk:1", "elk:10000"])
def test_driver_paper_protocol(tmp_path, monkeypatch, solver):
    """Three steps of every paper solver converge every solve without a fallback; the log rows carry the integer
    retries and chunk_retries (>= 0; 0 for every solver but LASER, which alone retries)."""
    rec, d = _run(tmp_path, monkeypatch, ["--loss", "suffix", "--steps", "3", "--solver", solver, "--fallback", "seq",
                                          "--max-iter", "0", "--tol", "1e-10"])
    L_steps = d.N - 1
    assert rec["name"] == f"synthetic_suffix_scratch_{solver.replace(':', '')}_fp64_fb_s0"
    solve = dict(rec["solve"])
    timing = solve.pop("fallback_seconds")
    assert solve == dict(cap=L_steps, tol=1e-10, fallback="seq", stop="laser")
    assert set(timing) == {"all", "chunk"} and all(v["seconds"] > 0 for v in timing.values())
    assert rec["config"]["max_iter"] == 0 and rec["config"]["fallback"] == "seq"
    for e in rec["log"]:
        assert e["seq_fallback"] == 0 and e["seq_fallback_solves"] == 0 and e["failed_stalled"] == 0
        assert e["converged"] == e["n_used"] == e["n_solves"] and e["iters_max"] <= L_steps and e["latent_max"] > 0
        assert type(e["retries"]) is type(e["chunk_retries"]) is int and 0 <= e["chunk_retries"] <= e["retries"]
        assert solver.startswith("laser") or e["retries"] == 0
    prov = rec["provenance"]
    for when in ("start", "end"):
        assert {"time", "unix", "host", "slurm", "cuda_visible_devices", "gpus", "git_commit", "git_dirty", "argv",
                "python"} <= set(prov[when])
    assert prov["end"]["unix"] >= prov["start"]["unix"]
    path = tmp_path / "runs" / (rec["name"] + ".json")
    assert json.loads(path.read_text())["name"] == rec["name"] and (tmp_path / "runs" / (rec["name"] + ".npz")).exists()
    assert not glob.glob(str(tmp_path / "runs" / "*.tmp*")) and not glob.glob(str(tmp_path / "runs" / "*.partial"))


def test_driver_defaults_add_provenance_only(tmp_path, monkeypatch):
    rec, _ = _run(tmp_path, monkeypatch, ["--loss", "suffix", "--steps", "2", "--solver", "deer"])
    assert rec["name"] == "synthetic_suffix_scratch_deer_fp64_s0" and "solve" not in rec and "provenance" in rec
    assert rec["config"]["fallback"] == "none" and rec["config"]["max_iter"] == 200
    assert not {"failed_stalled", "seq_fallback", "seq_fallback_solves", "n_used", "latent_max"} & set(rec["log"][0])


@pytest.mark.parametrize("extra", [
    ["--loss", "roll", "--solver", "deer", "--fallback", "seq"],
    ["--loss", "mix-suffix", "--solver", "deer", "--fallback", "seq"],
    ["--loss", "suffix", "--solver", "seq", "--fallback", "seq"],
    ["--loss", "suffix", "--solver", "guard", "--fallback", "seq"],
    ["--loss", "suffix", "--solver", "deer", "--fallback", "seq", "--diag-every", "5"],    # no longer an option
    ["--loss", "suffix", "--solver", "scale-elk:1"],
    ["--loss", "suffix", "--solver", "scale-elk:-1", "--fallback", "seq"],
    ["--loss", "suffix", "--solver", "scale-elk:x", "--fallback", "seq"],
    ["--loss", "suffix", "--solver", "deer", "--max-iter", "-1"],
])
def test_driver_refuses_the_paper_options_where_they_do_not_apply(extra):
    with pytest.raises(SystemExit):
        T.parse_args(["--system", "synthetic"] + extra)


def test_names_and_canonical_solver():
    base = ["--system", "chain8z", "--loss", "suffix", "--starts", "64", "--lr-schedule", "cosine", "--steps",
            "10000"]
    a = T.parse_args(base + ["--solver", "scale-elk:0.30", "--fallback", "seq", "--dtype", "fp32", "--seed", "3"])
    assert a.solver == "scale-elk:0.3"
    assert T.auto_name(a) == "chain8z_suffix-k64_scratch_scale-elk0.3_fp32_cos10k_fb_s3"
    b = T.parse_args(base + ["--solver", "laser", "--fallback", "seq", "--max-iter", "0", "--tol", "1e-7"])
    assert T.auto_name(b) == "chain8z_suffix-k64_scratch_laser_fp64_cos10k_fb_s0"
    c = T.parse_args(base + ["--solver", "laser"])
    assert T.auto_name(c) == "chain8z_suffix-k64_scratch_laser_fp64_cos10k_s0"         # no fb without --fallback seq
    assert T.option_tokens(dict(vars(b), fallback=float("nan"))) == ["cos10k"]          # a CSV's missing value


def test_atomic_json_write(tmp_path):
    path = str(tmp_path / "r.json")
    T.write_json_atomic(path, {"a": 1})
    T.write_json_atomic(path, {"a": 2})
    assert json.load(open(path)) == {"a": 2} and os.listdir(tmp_path) == ["r.json"]


# ----------------------------------------------------------------------------- structure and the default program

def _eqns(jaxpr):
    """Every equation of a jaxpr and of its sub-jaxprs (branches, bodies, remat and custom-rule jaxprs)."""
    from jax.extend import core as jcore
    out = []
    for eqn in jaxpr.eqns:
        out.append(eqn)
        for v in eqn.params.values():
            for sub in (v if isinstance(v, (tuple, list)) else (v,)):
                if isinstance(sub, jcore.ClosedJaxpr):
                    out += _eqns(sub.jaxpr)
                elif isinstance(sub, jcore.Jaxpr):
                    out += _eqns(sub)
    return out


def test_the_gradient_program_holds_one_undifferentiated_fallback_cond():
    """The fallback's cond reads only stop_gradient values, so value_and_grad neither differentiates nor transposes
    it: the program holds exactly one cond with the loop's scan in a branch, and its only output is the loop's paths
    (a differentiated cond would also output the loop's residuals, and its transpose would be a second cond)."""
    d, theta, x_eq, pos = _case()
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    params, static = L.partition(theta)
    f = lambda p: L.suffix_loss(L.combine(p, static), x_eq, pos, starts, guesses, d.dt, "deer", TOL, 100, 0, chunk=5,
                                fallback=True)
    jaxpr = jax.make_jaxpr(jax.value_and_grad(f, has_aux=True))(params)
    conds = [e for e in _eqns(jaxpr.jaxpr) if e.primitive.name == "cond"]
    loops = [e for e in conds if any(x.primitive.name == "scan" for b in e.params["branches"] for x in _eqns(b.jaxpr))]
    assert len(loops) == 1 and len(loops[0].outvars) == 1


# The suffix loss, its aux and its gradient as the versions of implicit.py and losses.py before the paper options
# computed them (CPU, float64): one entry per solver and chunk size, written once from those versions (they are not
# in this repository, and no script of it writes the file).
SUFFIX_LOSS_REFERENCE = os.path.join(os.path.dirname(__file__), "data", "elcd_suffix_loss_reference.npz")


@pytest.mark.parametrize("solver", ["deer", "laser"])
def test_default_path_equals_the_stored_reference_bit_for_bit(solver):
    """At their defaults (stop 'parode', fallback off, max_iter 200) the current implicit.py and losses.py compute
    the suffix loss, its aux and its gradient exactly as their reference versions did (the stored values of
    SUFFIX_LOSS_REFERENCE), in one vmap and in chunks: the refactors (the adjoint shared with attach_adjoint, the
    decode and the chunking helpers) change no operation. The test reads the entries of deer and laser; the file
    also holds entries of guard and laser-plain, which it does not read."""
    d, theta, x_eq, pos = _case(seed=18)
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts).at[0, 1].set(jnp.nan)      # one dropped start on the old path too
    params, static = L.partition(theta)
    with np.load(SUFFIX_LOSS_REFERENCE) as ref:
        for chunk in (0, 5):
            fn = lambda p: L.suffix_loss(L.combine(p, static), x_eq, pos, starts, guesses, d.dt, solver, TOL, 100, 0,
                                         chunk=chunk)
            (ln, an), gn = jax.jit(jax.value_and_grad(fn, has_aux=True))(params)
            key = f"{solver}/chunk{chunk}"
            aux_old = {k[len(key) + 5:]: ref[k] for k in ref.files if k.startswith(f"{key}/aux/")}
            grad_old = [ref[k] for k in sorted(ref.files) if k.startswith(f"{key}/grad/")]
            grad_new = jax.tree_util.tree_leaves(gn)
            assert float(ref[f"{key}/loss"]) == float(ln) and set(aux_old) == set(an)
            for k in aux_old:
                assert np.array_equal(aux_old[k], np.asarray(an[k]), equal_nan=True), (chunk, k)
            assert len(grad_old) == len(grad_new)
            for a, b in zip(grad_old, grad_new):
                assert np.array_equal(a, np.asarray(b), equal_nan=True), chunk


def test_driver_fallback_trains_as_the_loop(tmp_path, monkeypatch):
    """Cap 1 at tol 1e-12: nearly every parallel solve fails at every step, the loop re-solves the step, and the run
    trains as the sequential solver's run does (loss and gradient norm at every step), using every start."""
    common = ["--loss", "suffix", "--steps", "3", "--time-only", "3", "--tol", "1e-12"]
    fb, d = _run(tmp_path, monkeypatch, common + ["--solver", "deer", "--fallback", "seq", "--max-iter", "1"], sub="fb")
    sq, _ = _run(tmp_path, monkeypatch, common + ["--solver", "seq"], sub="seq")
    for a, b in zip(fb["log"], sq["log"]):
        assert abs(a["loss"] - b["loss"]) <= 1e-9 * abs(b["loss"])
        assert abs(a["grad_norm"] - b["grad_norm"]) <= 1e-7 * abs(b["grad_norm"])
        assert a["seq_fallback"] == 1 and a["n_used"] == a["n_solves"] and a["converged"] < a["n_solves"]
        assert (a["seq_fallback_solves"] == a["n_solves"] - a["converged"]
                == a["failed_nonfinite"] + a["failed_max_iter"] + a["failed_stalled"])


def test_driver_fallback_fills_the_buffer_with_the_loop_paths(tmp_path, monkeypatch):
    """--lr 0 keeps the model: step 0 falls back, so the warm buffer holds the loop's paths (and the converged
    parallel ones); at step 1 every guess is its start's solution, so no solve iterates and nothing falls back."""
    rec, _ = _run(tmp_path, monkeypatch, ["--loss", "suffix", "--steps", "2", "--time-only", "2", "--tol", "1e-12",
                                          "--solver", "deer", "--fallback", "seq", "--max-iter", "1", "--lr", "0"])
    s0, s1 = rec["log"]
    assert s0["seq_fallback"] == 1 and s0["warm_frac"] == 0.0
    assert (s1["seq_fallback"], s1["warm_frac"], s1["iters_max"], s1["converged"]) == (0, 1.0, 0, s1["n_solves"])
    assert abs(s1["loss"] - s0["loss"]) <= 1e-12 * abs(s0["loss"])


@pytest.mark.parametrize("failure", ["missing", "timeout"])
def test_provenance_survives_missing_tools(monkeypatch, failure):
    """No nvidia-smi (a CPU node) or a hanging one, no git: provenance still returns, with those fields None."""
    import subprocess

    def broken(*args, **kw):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)
        raise FileNotFoundError(args[0][0])
    monkeypatch.setattr(subprocess, "run", broken)
    prov = T.provenance()
    assert prov["gpus"] is None and prov["git_commit"] is None and prov["git_dirty"] is None and prov["host"]
    assert "experiments.elcd.train.train_elcd" in prov["modules"]


def test_elk_on_the_fallback_path():
    d, theta, x_eq, pos = _case()
    starts = _starts()
    guesses = _guesses(theta, x_eq, pos, starts)
    loss_d, _, g_d = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, fallback=True)
    loss_e, aux_e, g_e = _loss_and_grad(theta, x_eq, pos, starts, guesses, d.dt, "elk:100", max_iter=0, fallback=True)
    assert np.all(np.asarray(aux_e["ok_par"])) and abs(loss_e - loss_d) <= 1e-10 * abs(loss_d) and _rel(g_e, g_d) <= 1e-8
