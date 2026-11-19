"""The ELCD paper runs' tables (experiments/elcd/figures/records.py) on synthetic records with
known values: step hours, the loop's projection and the speedup, fallback hours and curves on the step-hours
clock, the evaluation of the initial model (load_init_evals); and the solver family and scale-ELK's rate of the main
figures (table_training_time.family, rate_of). numpy and pandas only; CPU."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.elcd.figures import records as PDATA  # noqa: E402
from experiments.elcd.record_names import PAPER_LASER  # noqa: E402


def _rec(path, walls, steps, time_only=0, evals=(), fallback_at=(), fb_seconds=None, final=None):
    log = [dict(step=i, wall=w, compile=i == 0, n_solves=192, iters_mean=5.0,
                seq_fallback=int(i in fallback_at), seq_fallback_solves=4 * int(i in fallback_at),
                failed_max_iter=4 * int(i in fallback_at)) for i, w in enumerate(walls)]
    rec = dict(config=dict(steps=steps, time_only=time_only), log=log, device_kind="NVIDIA B200",
               provenance=dict(start=dict(host="node01", gpus=[dict(uuid="GPU-1")])),
               evals=[dict(step=s, summary=dict(train_chamfer_native_mean=v)) for s, v in evals],
               final=final or {})
    if fb_seconds is not None:
        rec["solve"] = dict(fallback_seconds=dict(all=dict(starts=192, seconds=fb_seconds)))
    path.write_text(json.dumps(rec))


@pytest.fixture
def root(tmp_path):
    cell = tmp_path / "chain8z" / "fp32" / "s0"
    cell.mkdir(parents=True)
    # 4 steps planned: DEER compiles for 36 s then takes 0.5 s per step, one fallback (a 0.9 s forward loop)
    _rec(cell / "deer.json", [36.0, 0.5, 1.4, 0.5], 4, evals=[(2, 0.3), (4, 0.1)], fallback_at=(2,), fb_seconds=0.9,
         final=dict(train_chamfer_native_mean=0.1))
    # LASER's record under its name in the paper's records, which the loaders map to "laser"
    _rec(cell / f"{PAPER_LASER}.json", [40.0, 0.6, 0.6, 0.6], 4, evals=[(4, 0.12)],
         final=dict(train_chamfer_native_mean=0.12))
    _rec(cell / "deer_at2.json", [1.0], 4)                            # a snapshot: not a run
    _rec(cell / "loop_time3.json", [20.0, 3.0, 3.2], 4, time_only=3)   # the loop: compile 20 s, median 3.1 s
    (cell / "cell.json").write_text(json.dumps(dict(probes=[
        dict(kind="loop", after="start", median_wall=3.0, rc=0), dict(kind="deer", after="start", median_wall=15.0, rc=0),
        dict(kind="loop", after="deer", median_wall=3.3, rc=0), dict(kind="deer", after="deer", median_wall=15.0, rc=0)])))
    return tmp_path


def test_runs_and_the_loop_projection(root):
    runs = PDATA.load_runs(root)
    assert sorted(runs["item"]) == ["deer", "laser", "loop_time3"]
    deer = runs[runs["item"] == "deer"].iloc[0]
    assert abs(deer["step_hours"] - 38.4 / 3600) < 1e-12 and deer["compile_s"] == 36.0 and deer["median_s"] == 0.5
    assert (deer["fallback_steps"], deer["fallback_solves"], deer["failed_max_iter"]) == (1, 4, 4)
    assert abs(deer["fallback_hours"] - 0.9 / 3600) < 1e-12 and deer["host"] == "node01" and deer["gpu"] == "GPU-1"
    proj = PDATA.loop_projection(runs).iloc[0]
    assert proj["steps"] == 4 and abs(proj["loop_hours"] - (20.0 + 3.1 * 3) / 3600) < 1e-12


def test_speedups(root):
    sp = PDATA.speedups(PDATA.load_runs(root))
    assert sorted(sp["solver"]) == ["deer", "laser"]
    row = sp[sp["solver"] == "deer"].iloc[0]
    assert abs(row["speedup"] - (20.0 + 9.3) / 38.4) < 1e-12


def test_the_loop_s_hours_are_its_full_run_s_where_one_exists(root):
    proj = (20.0 + 3.1 * 3) / 3600
    assert PDATA.loop_hours(PDATA.load_runs(root)).to_dict() == pytest.approx({("chain8z", "fp32", 0): proj})
    _rec(root / "chain8z" / "fp32" / "s0" / "loop_full.json", [20.0, 3.0, 3.2, 3.4], 4)   # the loop in full: 29.6 s
    runs = PDATA.load_runs(root)
    assert PDATA.loop_hours(runs).to_dict() == pytest.approx({("chain8z", "fp32", 0): 29.6 / 3600})
    assert abs(PDATA.loop_projection(runs).iloc[0]["loop_hours"] - proj) < 1e-12          # the projection is kept
    sp = PDATA.speedups(runs)
    assert abs(sp[sp["solver"] == "deer"].iloc[0]["speedup"] - 29.6 / 38.4) < 1e-12


def test_curves_sit_on_the_step_hours_clock(root):
    cv = PDATA.curves(root)
    deer = cv[cv["item"] == "deer"].sort_values("step")
    assert deer["hours"].tolist() == pytest.approx([36.5 / 3600, 38.4 / 3600])        # before step 2, and the total
    assert deer["train_chamfer"].tolist() == [0.3, 0.1]


def test_family_of_the_solver_names(root):
    from experiments.elcd.figures import table_training_time as MF
    PDATA.load_runs(root)
    assert [MF.family(s) for s in ("scale-elk0.3", "elk1", "deer", "laser", "seq")] == ["scale-elk", "elk", "deer",
                                                                                       "laser", "seq"]


def test_rate_of_scale_elk_names():
    from experiments.elcd.figures import table_training_time as MF
    assert [MF.rate_of(s) for s in ("scale-elk0.03", "deer", "elk100")] == [0.03, None, None]


def test_load_init_evals_reads_the_initial_model_s_evaluation(root):
    cell = root / "chain8z" / "fp32" / "s0"
    assert PDATA.load_init_evals(root).empty
    (cell / "init_eval.json").write_text(json.dumps(dict(evals=[dict(step=0, seconds=12.0,
                                                                       summary=dict(train_chamfer_native_mean=1.5))])))
    init = PDATA.load_init_evals(root)
    assert init[["system", "dtype", "seed", "seconds", "train_chamfer"]].iloc[0].tolist() == ["chain8z", "fp32", 0, 12.0, 1.5]
    assert np.isnan(init["heldout_chamfer"].iloc[0])                  # a metric the summary lacks
    assert sorted(PDATA.load_runs(root)["item"]) == ["deer", "laser", "loop_time3"]   # init_eval is not a run
