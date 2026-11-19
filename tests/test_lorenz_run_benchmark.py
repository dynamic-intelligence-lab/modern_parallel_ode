"""Tests of the benchmark driver of the bistable Lorenz experiment (experiments/lorenz/run_benchmark.py).

The driver run of the paper's table at N = 1000 against the rows recorded on the GPU, the option --rho at its
default value against a run without it and at 15 (tag, label and row), and the foci of the system as fixed points
of its vector field. The last section holds a check of parode: LaserConfig rejects a window length that is not a
power of two.
"""
import os
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.lorenz.run_benchmark import lorenz_focus, main, make_system  # noqa: E402
from parode.laser import LaserConfig  # noqa: E402
from parode.ode_utils import prepare_ode_function  # noqa: E402
from parode.sequential import seq_integrate  # noqa: E402


# ----------------------------------------------------------------------------- Lorenz: the rows of the driver and rho

LORENZ_ARGV = ["run_benchmark.py", "--dt", "0.01", "--dtype", "fp64", "--repeats", "1"]

# The rows at N 1000 of the shipped tuning sweep (results/lorenz/tuning_sweep/stageA/summary.csv, and stageB for
# ELK), measured on an L40S: method -> (iterations, rejected trials, residual, status). Copied here so the test does
# not read a results file.
L40S_ROWS_N1000 = {
    "deer": (2, 0, float("nan"), "NaN"),
    "scale-elk:0.05": (90, 0, 6.034789378286942e-10, "ok"),
    "elk:10000": (33, 0, 9.609594890658268e-10, "ok"),
    "laser": (22, 10, 5.398459457239824e-15, "ok"),
}


def _main_rows(monkeypatch, outdir, extra):
    import csv as _csv
    monkeypatch.setattr(sys, "argv", LORENZ_ARGV + extra + ["--outdir", str(outdir)])
    main()
    return list(_csv.DictReader(open(outdir / "summary.csv")))


# Columns of a lorenz row of a run without --rho.
BASE_COLUMNS = ["system", "D", "N", "dt", "dtype", "guess", "seed", "method", "iters", "rejected", "wall_s", "compile_s",
                "residual", "err", "max_abs", "status", "device", "truth_max", "truth_tail_max", "x0_max"]
LORENZ_COLUMNS = ["rho", "lorenz_ic", "x0", "truth_end"]
REFERENCE_DEVICE = "NVIDIA L40S"


def test_lorenz_defaults_reproduce_the_l40s_tuning_sweep_at_N1000(tmp_path, monkeypatch):
    """Without --rho the driver runs the system of the tuning sweep: the file name of the paper's runs, the columns of
    BASE_COLUMNS, and per method the same iterations, rejected trials and status as the L40S rows. Every mismatch is collected, so
    one failing run prints all four methods.

    The residual is compared to 1e-12 relative on the reference device (both NaN for deer). On another device the
    iterates differ in round-off, and a converged residual max|X_t - X_{t-1} - dt f(X_{t-1})| is a difference of
    states of size up to max|X| = 14.7, so it is not defined below about eps max|X| = 3.3e-15. There the residual is
    held to 10 eps max|X| absolute instead. On CPU the absolute differences are 8.8e-16 (scale-elk:0.05),
    3.8e-16 (elk:10000) and 1.1e-16 (laser), relative 1.5e-6, 4.0e-7 and 2.1e-2, with every iteration
    and rejection count equal; the driver without --rho gives the same CPU residuals bit for bit.
    On the L40S all four residuals equal the reference bit for bit."""
    rows = _main_rows(monkeypatch, tmp_path, ["--N", "1000", "--methods", ",".join(L40S_ROWS_N1000)])
    assert (tmp_path / "lorenz_D3_N1000_dt0.01_fp64_zeros_s0.json").exists()
    assert [r["method"] for r in rows] == list(L40S_ROWS_N1000)
    assert list(rows[0].keys()) == BASE_COLUMNS
    report, bad = [], []
    for r in rows:
        it_ref, rej_ref, res_ref, status_ref = L40S_ROWS_N1000[r["method"]]
        it, rej, res = int(r["iters"]), int(r["rejected"]), float(r["residual"])
        both_nan = np.isnan(res) and np.isnan(res_ref)
        diff = 0.0 if both_nan else abs(res - res_ref)
        rel = 0.0 if both_nan else diff / abs(res_ref)
        floor = 10 * np.finfo(np.float64).eps * float(r["max_abs"]) if not both_nan else 0.0
        strict = r["device"] == REFERENCE_DEVICE
        res_ok = rel <= 1e-12 if strict else (both_nan or diff <= floor)
        report.append(f"{r['method']}: iters {it} (ref {it_ref}), rejected {rej} (ref {rej_ref}), status {r['status']} "
                      f"(ref {status_ref}), residual {res!r} (ref {res_ref!r}; absolute difference {diff:.3e}, relative "
                      f"{rel:.3e}; criterion {'1e-12 relative' if strict else f'{floor:.2e} absolute'}; device "
                      f"{r['device']})")
        if (it, rej, r["status"]) != (it_ref, rej_ref, status_ref) or not res_ok:
            bad.append(r["method"])
    print("\n".join(report))
    assert not bad, "mismatch in " + ", ".join(bad) + ":\n" + "\n".join(report)


def test_lorenz_rho_10_equals_the_default(tmp_path, monkeypatch):
    """rho=10.0 builds the same system as the default (label, x0, field bit for bit), and --rho 10 on the command
    line writes the same file names and the same rows up to the timings, plus the four Lorenz columns."""
    ode_d, x0_d, label_d = make_system()
    ode_r, x0_r, label_r = make_system(rho=10.0)
    assert label_d == label_r == "Lorenz sigma 10, rho 10, beta 8/3 (pre-chaotic: origin saddle, C+- attract)"
    np.testing.assert_array_equal(np.asarray(x0_d), np.asarray(x0_r))
    X = 20.0 * jax.random.normal(jax.random.PRNGKey(5), (50, 3))
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda x: ode_d(0.0, x))(X)),
                                  np.asarray(jax.vmap(lambda x: ode_r(0.0, x))(X)))
    common = ["--N", "300", "--methods", "scale-elk:0.05,laser"]
    rows_d = _main_rows(monkeypatch, tmp_path / "d", common)
    rows_r = _main_rows(monkeypatch, tmp_path / "r", common + ["--rho", "10"])
    assert sorted(os.listdir(tmp_path / "d")) == sorted(os.listdir(tmp_path / "r"))
    assert list(rows_d[0].keys()) == BASE_COLUMNS and list(rows_r[0].keys()) == BASE_COLUMNS + LORENZ_COLUMNS
    compared = [k for k in BASE_COLUMNS if k not in ("wall_s", "compile_s")]
    assert [[r[k] for k in compared] for r in rows_d] == [[r[k] for k in compared] for r in rows_r]
    for r in rows_r:
        assert (float(r["rho"]), r["lorenz_ic"], r["x0"]) == (10.0, "paper", "1.0 1.0 1.0")


def test_main_records_a_non_default_rho(tmp_path, monkeypatch):
    """A non-default rho enters the tag, the label and the row. The driver builds the Lorenz system from x0 = (1, 1, 1)
    only, so command lines that ask for another system or initial condition (--system, --D, --lorenz-ic) are refused."""
    rows = _main_rows(monkeypatch, tmp_path, ["--N", "200", "--methods", "scale-elk:0.1", "--rho", "15"])
    names = sorted(os.listdir(tmp_path))
    assert "lorenz_D3_N200_dt0.01_fp64_zeros_s0_rho15.json" in names
    assert "lorenz_D3_N200_dt0.01_fp64_zeros_s0_rho15_scale-elk0.1.npz" in names
    (r,) = rows
    assert list(r.keys()) == BASE_COLUMNS + LORENZ_COLUMNS
    assert (float(r["rho"]), r["lorenz_ic"]) == (15.0, "paper")
    np.testing.assert_array_equal(np.array(r["x0"].split(), float), [1.0, 1.0, 1.0])
    ode, x0, label = make_system(rho=15.0)
    assert label == "Lorenz sigma 10, rho 15, beta 8/3 (origin saddle, C+- stable)"
    truth = np.asarray(seq_integrate(prepare_ode_function(ode), x0, 0.01, 200, "euler"))
    np.testing.assert_allclose(np.array(r["truth_end"].split(), float), truth[-1], rtol=0, atol=1e-12)
    for bad in (["--system", "cubic", "--rho", "15"], ["--system", "cubic", "--rho", "10"],
                ["--system", "cubic", "--lorenz-ic", "mirror"],
                ["--system", "lorenz", "--D", "3", "--lorenz-ic", "box:x"]):
        monkeypatch.setattr(sys, "argv", ["run_benchmark.py"] + bad + ["--outdir", str(tmp_path / "bad")])
        with pytest.raises(SystemExit):
            main()


@pytest.mark.parametrize("rho", [5.0, 10.0, 15.0, 20.0, 23.0])
@pytest.mark.parametrize("s", [1, -1])
def test_lorenz_focus_is_a_fixed_point_of_the_vector_field(rho, s):
    ode, _, _ = make_system(rho=rho)
    C = lorenz_focus(rho, s)
    a = np.sqrt(8.0 / 3.0 * (rho - 1.0))
    np.testing.assert_allclose(np.asarray(C), [s * a, s * a, rho - 1.0], rtol=1e-15, atol=0)
    assert np.max(np.abs(np.asarray(ode(0.0, C)))) < 1e-12
    other, _, _ = make_system(rho=rho + 1.0)
    assert np.max(np.abs(np.asarray(other(0.0, C)))) > 1.0                # rho enters the field


# ----------------------------------------------------------------------------- a check of parode

def test_laser_config_rejects_a_non_power_of_two_min_block():
    with pytest.raises(ValueError):
        LaserConfig(min_block=12)
