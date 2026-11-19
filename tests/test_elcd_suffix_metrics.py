"""The multi-start error (metrics.multistart_error).

The metric against a brute-force loop of separate rollouts on a tiny model; the rule that picks the starts; the
position slicing on a phase-space system; the existing metrics bit for bit against stored reference values. CPU,
float64 (JAX_PLATFORMS=cpu).
"""
import functools
import json
import math
import os
import resource
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from experiments.elcd import flow_gatherfree as FG  # noqa: E402
from experiments.elcd import model as model_mod  # noqa: E402
from experiments.elcd import solvers  # noqa: E402
from experiments.elcd.model import make_ode  # noqa: E402
from experiments.elcd.train import data as D  # noqa: E402
from experiments.elcd.train import losses as L  # noqa: E402
from experiments.elcd.train import metrics as M  # noqa: E402
from experiments.elcd.train.train_elcd import load_model, save_theta  # noqa: E402

# the output of an earlier metrics.py, without the multi-start error, on one small case (CPU, float64), written once
# from that version, which is not in this repository: the reference for the other metrics (.npz: the arrays, .json:
# the scalars, the rows and the summary)
METRICS_REFERENCE = os.path.join(os.path.dirname(__file__), "data", "elcd_metrics_reference")
TOL = 1e-12
NEW_ROW_KEYS = {"multistart_rmse", "multistart_starts"}
NEW_SUMMARY_KEYS = [f"{t}_multistart_rmse_{s}" for t in ("train", "heldout") for s in ("mean", "median")]


def _theta(seed: int, D_: int, x_eq: np.ndarray) -> L.Theta:
    flow = FG.random_flow(seed, D_, flow_steps=2)
    elcd = model_mod.init_synthetic(seed, D_, gain=1.0, rate=0.0, z_star=x_eq)
    return L.Theta(model_mod.cast(elcd, jnp.float64), FG.cast_flow(flow, jnp.float64))


def _synthetic_demos(seed=0, N=30, rest=4):
    """Four 2-D demonstrations (3 training, 1 held out) that decelerate into a common end point and rest there."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N - rest)[:, None]
    base = np.stack([np.cos(3 * t[:, 0]), np.sin(2 * t[:, 0])], axis=1) * (1 - t) ** 2
    pos = np.stack([base + 0.05 * rng.standard_normal((1, 2)) * (1 - t) ** 2 for _ in range(4)])
    pos = np.concatenate([pos, np.repeat(pos[:, -1:], rest, axis=1)], axis=1)
    return D.normalize("synthetic", pos, D.forward_differences(pos, 0.01), 0.01, n_train=3)


@functools.lru_cache(maxsize=None)
def _one_rollout(dt: float, H: int):
    """One start, exactly H latent RK4 steps (seq_scan, not metrics' code), decoded: (H + 1, D)."""
    def run(theta, x_eq, x0):
        Z = solvers.seq_scan(make_ode(L.with_equilibrium(theta, x_eq)), FG.phi(theta.flow, x0), dt, H, "rk4")
        return jax.vmap(lambda z: FG.psi(theta.flow, z))(Z)
    return jax.jit(run)


def _brute_force(theta, d, ks, p, lags=False):
    """The multi-start sums by separate rollouts: from every start x_k, exactly N - 1 - k steps, against
    x_{k+1..N-1} on the first p coordinates, in a Python loop. Returns (sse (n, S), pairs (S,), rmse (n,)), and with
    lags=True also sse_lag (n, N - 1): the squared error of step j after the start, summed over the starts."""
    x_eq = jnp.asarray(d.x_eq)
    N, n = d.N, d.pos.shape[0]
    sse = np.zeros((n, len(ks)))
    sse_lag = np.zeros((n, N - 1))
    for i in range(n):
        for c, k in enumerate(ks):
            X = np.asarray(_one_rollout(d.dt, int(N - 1 - k))(theta, x_eq, jnp.asarray(d.pos[i, k])))
            per_step = np.sum((X[1:, :p] - d.pos[i, k + 1:, :p]) ** 2, axis=1)       # steps j = 1..N-1-k
            sse[i, c] = np.sum(per_step)
            sse_lag[i, : per_step.size] += per_step
    pairs = N - 1 - np.asarray(ks)
    out = (sse, pairs, np.sqrt(sse.sum(axis=1) / (pairs.sum() * p)))
    return out + (sse_lag,) if lags else out


def _close(a, b, tol=TOL):
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=tol, atol=tol)


# ----------------------------------------------------------------------------- the start rule

def test_multistart_indices_rule():
    assert np.array_equal(M.multistart_indices(985), np.arange(984))          # LASA: every recorded state
    assert np.array_equal(M.multistart_indices(1000), np.arange(999))         # N 1000 (the pendula): every state
    assert M.multistart_indices(1001).size == M.MULTISTART_MAX                # N > 1000: 128 starts (the contract)
    for N in (1001, 1002, 4069, 7880, 20000):
        ks = M.multistart_indices(N)
        gaps = np.diff(ks)
        assert ks.size == M.MULTISTART_MAX == 128 and ks[0] == 0 and ks[-1] == N - 2
        assert np.all(gaps > 0) and gaps.max() - gaps.min() <= 1                # evenly spaced up to rounding
    assert np.array_equal(np.unique(np.diff(M.multistart_indices(7880))), [62, 63])
    assert M.multistart_indices(7880, max_starts=5000).size == 5000
    assert np.array_equal(M.multistart_indices(50, max_starts=100), np.arange(49))
    assert np.array_equal(M.multistart_indices(30, max_starts=7, all_max=10), [0, 5, 9, 14, 19, 23, 28])
    assert np.array_equal(M.multistart_indices(2), [0])
    with pytest.raises(ValueError):
        M.multistart_indices(1)
    with pytest.raises(ValueError):
        M.multistart_indices(100, max_starts=0)


# ----------------------------------------------------------------------------- the metric against brute force

@pytest.mark.parametrize("max_starts,all_max", [(M.MULTISTART_MAX, M.MULTISTART_ALL), (7, 10)], ids=["all", "seven"])
def test_multistart_matches_separate_rollouts(max_starts, all_max):
    d = _synthetic_demos()
    theta = _theta(3, 2, d.x_eq)
    ms = M.multistart_error(theta, d, max_starts, all_max)
    ks = M.multistart_indices(d.N, max_starts, all_max)
    assert np.array_equal(ms["k"], ks) and int(ms["pos_dims"]) == 2
    sse, pairs, rmse, sse_lag = _brute_force(theta, d, ks, 2, lags=True)
    assert np.array_equal(ms["pairs_start"], pairs)
    _close(ms["sse_start"], sse)
    _close(ms["rmse"], rmse)
    # the lag sums hold the same pairs: every start with k + j <= N - 1 at lag j, each lag at its own index (a shift
    # of the lags keeps the totals, so the totals alone would not catch it)
    assert np.array_equal(ms["pairs_lag"], [np.sum(ks <= d.N - 1 - j) for j in range(1, d.N)])
    assert ms["pairs_lag"].sum() == pairs.sum()
    _close(ms["sse_lag"], sse_lag)
    _close(ms["sse_lag"].sum(axis=1), sse.sum(axis=1))
    assert np.all(rmse > 0)
    # the horizon cap of the docstring (what the notebook's capped figure reads): the RMSE of separate rollouts of
    # min(H, N - 1 - k) steps from every start equals sqrt(sum_{j <= H} sse_lag / (p sum_{j <= H} pairs_lag))
    x_eq = jnp.asarray(d.x_eq)
    for H in (1, 5, d.N - 1):
        sse_h = np.zeros(d.pos.shape[0])
        for i in range(d.pos.shape[0]):
            for k in ks:
                h = int(min(H, d.N - 1 - k))
                X = np.asarray(_one_rollout(d.dt, h)(theta, x_eq, jnp.asarray(d.pos[i, k])))
                sse_h[i] += np.sum((X[1:, :2] - d.pos[i, k + 1: k + 1 + h, :2]) ** 2)
        n_h = np.sum(np.minimum(H, d.N - 1 - ks))
        capped = np.sqrt(ms["sse_lag"][:, :H].sum(axis=1) / (2 * ms["pairs_lag"][:H].sum()))
        assert ms["pairs_lag"][:H].sum() == n_h
        _close(capped, np.sqrt(sse_h / (2 * n_h)))


def test_multistart_blocks_and_padding_change_nothing():
    """Blocks of 3 starts (29 starts: 9 full blocks and one padded with a copy of its last start) give the sums of
    one block of all starts."""
    d = _synthetic_demos()
    theta = _theta(4, 2, d.x_eq)
    one = M.multistart_error(theta, d)
    small = M.multistart_error(theta, d, block=3 * d.N * d.D)
    for key in ("sse_start", "sse_lag", "rmse"):
        _close(small[key], one[key])
    assert np.array_equal(small["k"], one["k"]) and np.array_equal(small["pairs_lag"], one["pairs_lag"])


def test_multistart_keeps_rollouts_from_later_starts():
    """The kept rollouts (the notebook's overlay figure): 4 starts nearest to 20, 40, 60 and 80 percent of the
    demonstration, each path equal to a separate rollout from x_k on the position coordinates, indexed by the step
    after the start and NaN past the demonstration's end; the same paths in blocks of 3 starts; none with n_paths 0;
    and the other arrays unchanged by keeping them."""
    d = _synthetic_demos()
    theta = _theta(6, 2, d.x_eq)
    x_eq = jnp.asarray(d.x_eq)
    ms = M.multistart_error(theta, d)
    assert np.array_equal(ms["paths_k"], [6, 12, 17, 23])                     # nearest to 0.8 * 29 * (1, 2, 3, 4) / 4
    assert ms["paths"].shape == (d.pos.shape[0], 4, d.N, 2)
    for i in range(d.pos.shape[0]):
        for q, k in enumerate(ms["paths_k"]):
            X = np.asarray(_one_rollout(d.dt, int(d.N - 1 - k))(theta, x_eq, jnp.asarray(d.pos[i, k])))
            _close(ms["paths"][i, q, : d.N - k], X)
            assert np.all(np.isnan(ms["paths"][i, q, d.N - k:]))
    small = M.multistart_error(theta, d, block=3 * d.N * d.D)
    np.testing.assert_array_equal(small["paths_k"], ms["paths_k"])
    _close(np.nan_to_num(small["paths"], nan=7.0), np.nan_to_num(ms["paths"], nan=7.0))
    none = M.multistart_error(theta, d, n_paths=0)
    assert none["paths_k"].size == 0 and none["paths"].shape == (d.pos.shape[0], 0, d.N, 2)
    for key in ("k", "rmse", "sse_start", "pairs_start", "sse_lag", "pairs_lag"):
        np.testing.assert_array_equal(none[key], ms[key], err_msg=key)
    # evenly spaced starts (N > 1000): the kept starts are members of the metric's starts
    ks = M.multistart_indices(7880)
    kept = M.multistart_path_starts(ks, 7880)
    assert kept.size == 4 and set(kept) <= set(ks) and np.all(np.abs(kept - 0.8 * 7879 * np.arange(1, 5) / 4) <= 32)
    assert M.multistart_path_starts(ks, 7880, n_paths=0).size == 0


def test_multistart_start_zero_is_the_native_rollout():
    d = _synthetic_demos()
    theta = _theta(5, 2, d.x_eq)
    ev = M.evaluate(theta, d, keep_rollouts=True)
    native = ev["native"]
    _close(ev["multistart"]["sse_start"][:, 0], np.sum((native[:, 1:] - d.pos[:, 1:]) ** 2, axis=(1, 2)))
    for i, r in enumerate(ev["rows"]):
        assert r["multistart_starts"] == d.N - 1 and r["multistart_rmse"] == float(ev["multistart"]["rmse"][i])


def test_multistart_uses_the_positions_of_a_lifted_system():
    """phase: lift (state (x, v / s), D 4, pos_dims 2): the starts are full lifted states, the errors on x only."""
    d = D.lift_phase(_synthetic_demos())
    assert d.D == 4 and d.pos_dims == 2
    theta = _theta(6, 4, d.x_eq)
    ks = np.arange(d.N - 1)
    ms = M.multistart_error(theta, d)
    _, _, rmse = _brute_force(theta, d, ks, 2)
    assert int(ms["pos_dims"]) == 2
    _close(ms["rmse"], rmse)
    full = M.multistart_error(theta, d._replace(pos_dims=None))
    _, _, rmse_full = _brute_force(theta, d, ks, 4)
    assert int(full["pos_dims"]) == 4
    _close(full["rmse"], rmse_full)
    assert np.all(np.abs(full["rmse"] - ms["rmse"]) > 1e-6)


def test_multistart_marks_a_non_finite_pair():
    d = _synthetic_demos()
    theta = _theta(7, 2, d.x_eq)
    bad = d._replace(pos=d.pos.copy())
    bad.pos[3, 10] = np.nan                     # the held-out demonstration: one recorded state is not finite
    ms = M.multistart_error(theta, bad)
    assert np.all(np.isfinite(ms["rmse"][:3])) and np.isnan(ms["rmse"][3])
    s = M.summarize_multistart([dict(train_demo=i < 3, multistart_rmse=float(v)) for i, v in enumerate(ms["rmse"])])
    assert np.isfinite(s["train_multistart_rmse_mean"]) and s["heldout_multistart_rmse_mean"] == float("inf")


def test_summarize_multistart_means_and_medians_per_split():
    """The train and held-out mean and median of the per-demonstration values, in summarize's key order; the
    held-out keys absent without held-out demonstrations; nothing when a row lacks the value."""
    rows = [dict(train_demo=True, multistart_rmse=v) for v in (1.0, 2.0, 4.0)] + \
           [dict(train_demo=False, multistart_rmse=v) for v in (3.0, 6.0)]
    s = M.summarize_multistart(rows)
    assert list(s) == NEW_SUMMARY_KEYS
    assert s["train_multistart_rmse_mean"] == 7.0 / 3.0 and s["train_multistart_rmse_median"] == 2.0
    assert s["heldout_multistart_rmse_mean"] == 4.5 and s["heldout_multistart_rmse_median"] == 4.5
    assert list(M.summarize_multistart(rows[:3])) == NEW_SUMMARY_KEYS[:2]
    assert M.summarize_multistart(rows[:4] + [dict(train_demo=False)]) == {}
    assert M.summarize_multistart([]) == {}


def test_multistart_in_float32():
    """A float32 model (the driver's fp32 runs evaluate in their own dtype): the rollouts run in float32, the sums
    come back in float64, close to the float64 model's."""
    d = _synthetic_demos()
    theta = _theta(9, 2, d.x_eq)
    theta32 = L.Theta(model_mod.cast(theta.elcd, jnp.float32), FG.cast_flow(theta.flow, jnp.float32))
    ms64, ms32 = M.multistart_error(theta, d), M.multistart_error(theta32, d)
    assert ms32["sse_start"].dtype == np.float64 and ms32["rmse"].dtype == np.float64
    np.testing.assert_allclose(ms32["rmse"], ms64["rmse"], rtol=1e-4)
    assert not np.array_equal(ms32["rmse"], ms64["rmse"])          # the float32 path really ran in float32


def _long_demos(N=1100):
    """Four 2-D demonstrations of N samples (N - 1 > MULTISTART_ALL: the evenly spaced starts apply)."""
    t = np.linspace(0, 1, N)[:, None]
    rng = np.random.default_rng(1)
    pos = np.stack([np.concatenate([np.cos(3 * t), np.sin(2 * t)], axis=1) * (1 - t) ** 2
                    + 0.05 * rng.standard_normal((1, 2)) * (1 - t) ** 2 for _ in range(4)])
    return D.normalize("long", pos, D.forward_differences(pos, 0.01), 0.01, n_train=3)


def _write_run(indir, name, seed, d):
    indir.mkdir(exist_ok=True)
    save_theta(str(indir / f"{name}.npz"), _theta(seed, d.D, d.x_eq), {})
    cfg = dict(system="synthetic", loss="roll", curriculum=True, init="scratch", solver="deer", dtype="fp64", seed=seed,
               tag="", vel_source="fd")
    (indir / f"{name}.json").write_text(json.dumps(dict(name=name, config=cfg)))


def test_max_starts_picks_evenly_spaced_starts(tmp_path, monkeypatch):
    """The five starts multistart_indices picks on a demonstration of 1,099 steps; multistart_error at max_starts 5
    runs on a model written by save_theta and read back by load_model (its values are not checked here)."""
    d = _long_demos()
    monkeypatch.setattr(D, "load_system", lambda spec: d)
    ks = M.multistart_indices(d.N, 5)
    assert np.array_equal(ks, [0, 274, 549, 824, 1098])          # linspace 0..1098, halves rounded to even
    _write_run(tmp_path / "recs", "run_long_s0", 0, d)
    theta = load_model(str(tmp_path / "recs" / "run_long_s0.npz"))
    M.multistart_error(theta, d, 5)


# ----------------------------------------------------------------------------- the old metrics, bit for bit

def _stored(pair):
    """A scalar of METRICS_REFERENCE.json, stored as [name of its type, value] with floats in hex, as that type."""
    kind, value = pair
    if kind == "float":
        return float(value) if value in ("nan", "inf", "-inf") else float.fromhex(value)
    return {"int": int, "bool": bool}[kind](value)


def _base_metrics(lifted):
    """What the earlier metrics.py (without the multi-start error) returned for the case of the test below, read from
    METRICS_REFERENCE: its evaluate() as a dict with the scalars, the arrays and the rows, and its summarize()."""
    tag = "lifted" if lifted else "plain"
    with open(METRICS_REFERENCE + ".json") as f:
        stored = json.load(f)[tag]
    ev = {k: _stored(v) for k, v in stored["scalars"].items()}
    with np.load(METRICS_REFERENCE + ".npz") as arrays:
        ev.update({k[len(tag) + 1:]: arrays[k] for k in arrays.files if k.startswith(tag + "/")})
    ev["rows"] = [{k: _stored(v) for k, v in row} for row in stored["rows"]]
    return ev, {k: _stored(v) for k, v in stored["summary"]}


def _same(a, b) -> bool:
    """Equal values of the same kind; NaN equals NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("lifted", [False, True], ids=["plain", "lifted"])
def test_existing_metrics_are_bit_identical_to_the_stored_reference(lifted):
    ev_old, s_old = _base_metrics(lifted)
    d = _synthetic_demos(N=40)
    d = D.lift_phase(d) if lifted else d
    theta = _theta(8, d.D, d.x_eq)
    ev_new = M.evaluate(theta, d, perturb_seed=3, keep_rollouts=True)
    ev_off = M.evaluate(theta, d, perturb_seed=3, multistart=False)
    for k in ("dt", "N", "n_paper"):
        assert ev_new[k] == ev_old[k]
    for k in ("native", "settle", "paper", "perturbed", "vel_pred"):
        assert np.array_equal(ev_new[k], ev_old[k], equal_nan=True), k
    assert len(ev_old["rows"]) == len(ev_new["rows"]) == len(ev_off["rows"])
    for r_old, r_new, r_off in zip(ev_old["rows"], ev_new["rows"], ev_off["rows"]):
        assert list(r_new)[: len(r_old)] == list(r_old) and set(r_new) - set(r_old) == NEW_ROW_KEYS
        assert all(_same(r_new[k], r_old[k]) for k in r_old), (r_old, r_new)
        assert list(r_off) == list(r_old) and all(_same(r_off[k], r_old[k]) for k in r_old)
    s_new = M.summarize(ev_new["rows"])
    assert list(s_new)[: len(s_old)] == list(s_old) and list(s_new)[len(s_old):] == NEW_SUMMARY_KEYS
    assert all(_same(s_new[k], s_old[k]) for k in s_old)
    s_rows_old = M.summarize(ev_old["rows"])                    # rows without the new values: the old summary exactly
    assert list(s_rows_old) == list(s_old) and all(_same(s_rows_old[k], s_old[k]) for k in s_old)
    assert M.summarize_multistart(ev_old["rows"]) == {}


# ----------------------------------------------------------------------------- memory (a 32 GB job only)

@pytest.mark.skipif(os.environ.get("ELCD_MEMORY_PROBE") != "1", reason="set ELCD_MEMORY_PROBE=1 in a 32 GB CPU job")
def test_multistart_fits_a_32gb_job_at_N7880_D32():
    """The largest case: N 7880, 128 starts, D 32, 7 demonstrations; peak resident memory below 24 GB."""
    N, D_, n = 7880, 32, 7
    rng = np.random.default_rng(0)
    s = np.linspace(0, 1, N)[None, :, None]
    x0 = rng.standard_normal((n, 1, D_))
    pos = x0 * (1 - s) ** 2 * (1 + 0.2 * np.sin(6 * np.pi * s))
    d = D.normalize("probe", pos, D.forward_differences(pos, 0.005), 0.005, n_train=3)
    theta = _theta(0, D_, d.x_eq)
    t0 = time.time()
    ms = M.multistart_error(theta, d)
    wall = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"\nN {N}, D {D_}, {ms['k'].size} starts x {n} demonstrations: {wall:.1f} s, peak RSS {peak_gb:.2f} GB, "
          f"rmse {np.round(ms['rmse'], 4).tolist()}")
    assert ms["k"].size == 128 and np.all(np.isfinite(ms["rmse"]))
    assert peak_gb < 24
