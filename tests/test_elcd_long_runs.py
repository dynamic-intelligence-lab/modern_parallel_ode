"""The long-run tables, experiments/elcd/figures/long_runs.py: the evaluation frame (hours, the velocity
driver's final-evaluation step), the clean rows and the floor table, the solver and velocity tables. CPU, no JAX."""
import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.elcd.figures import long_runs as LG  # noqa: E402

KEYS = ("train_chamfer_paper_mean", "heldout_chamfer_paper_mean", "train_chamfer_native_mean", "heldout_chamfer_native_mean",
        "train_multistart_rmse_mean", "heldout_multistart_rmse_mean")


def _summary(y, diverged_heldout=0):
    s = {k: y for k in KEYS}
    if diverged_heldout:
        s["heldout_chamfer_paper_mean"] = 1e6
    s["train_chamfer_paper_diverged"] = 0
    s["heldout_chamfer_paper_diverged"] = diverged_heldout
    return s


def _suffix_rec(thinned=True, blocks=30, block=100, wall=1.0):
    if thinned:
        log = [dict(step=block * (i + 1) - 1, step_first=block * i, steps=block, loss=1.0 / (i + 1), lr=1e-3, wall=wall,
                    iters_mean=1.5, iters_max=3 if i else 5, seq_fallback=1 if i == 0 else 0, converged_frac=1.0)
               for i in range(blocks)]
        evals = [dict(step=1000, seconds=10.0, summary=_summary(0.20)), dict(step=2000, seconds=10.0, summary=_summary(0.12, 1)),
                 dict(step=3000, seconds=10.0, summary=_summary(0.10))]
    else:
        log = [dict(step=i, loss=1.0 / (i + 1), lr=1e-3, wall=wall / block, iters_mean=1.2, iters_max=2, seq_fallback=0,
                    converged_frac=1.0) for i in range(blocks * block)]
        evals = [dict(step=500, seconds=5.0, summary=_summary(0.3)), dict(step=1000, seconds=5.0, summary=_summary(0.22))]
    return dict(config=dict(loss="suffix"), log=log, evals=evals, total_seconds=3600.0)


def _vel_rec(epochs=10, per_epoch=231, wall=0.5):
    log = [dict(step=per_epoch * (i + 1), epoch=i, loss=5.0 if i == 2 else 0.1, wall=wall) for i in range(epochs)]
    evals = [dict(step=per_epoch * 5, seconds=3.0, summary=dict(_summary(0.5), train_chamfer_paper_diverged=0)),
             dict(step=epochs, seconds=3.0, summary=dict(_summary(0.4), train_chamfer_paper_mean=2e3, train_chamfer_paper_diverged=2))]
    return dict(config=dict(loss="vel"), log=log, evals=evals, total_seconds=100.0)


def test_evals_places_the_velocity_final_evaluation_at_the_last_epoch_and_sums_the_walls_before_each_evaluation():
    v = LG.evals(_vel_rec())
    assert v["step"].tolist() == [1155, 2310]                    # the final evaluation's step (the epoch count 10) is mapped
    assert v["hours"].tolist() == pytest.approx([5 * 0.5 / 3600, 10 * 0.5 / 3600])
    s = LG.evals(_suffix_rec())
    assert s["step"].tolist() == [1000, 2000, 3000]
    assert s["hours"].tolist() == pytest.approx([10 / 3600, 20 / 3600, 30 / 3600])   # blocks ending below the step
    r = LG.evals(_suffix_rec(thinned=False))
    assert r["hours"].tolist() == pytest.approx([500 * 0.01 / 3600, 1000 * 0.01 / 3600])
    assert "train_chamfer_paper_mean" in s and s["seconds"].tolist() == [10.0, 10.0, 10.0]
    assert len(LG.evals(dict(config={}, log=[], evals=[]))) == 0


def test_clean_rows_and_floor_table_skip_evaluations_with_a_diverged_rollout_and_find_the_margins():
    ev = {"suffix_long": LG.evals(_suffix_rec())}
    clean = LG.clean_rows(ev["suffix_long"], "heldout_chamfer_paper_mean")
    assert clean["step"].tolist() == [1000, 3000]                 # step 2000 had a diverged held-out rollout
    assert LG.clean_rows(ev["suffix_long"], "heldout_chamfer_native_mean")["step"].tolist() == [1000, 2000, 3000]
    ft = LG.floor_table(ev, last=2, margins=(1.0, 0.1)).set_index("metric")
    row = ft.loc["heldout_chamfer_paper"]
    assert row["evaluations"] == 3 and row["clean"] == 2 and row["floor"] == pytest.approx(0.15)
    assert row["minimum"] == pytest.approx(0.10) and row["at_step"] == 3000 and row["final"] == pytest.approx(0.10)
    assert row["within_100pct_step"] == 1000 and row["within_10pct_step"] == 3000
    assert row["within_10pct_hours"] == pytest.approx(30 / 3600)
    nat = ft.loc["train_chamfer_native"]
    assert nat["clean"] == 3 and nat["floor"] == pytest.approx(0.11) and nat["first"] == pytest.approx(0.20)
    assert LG.floor_table({"x": LG.evals(dict(config={}, log=[], evals=[]))}).empty


def test_solver_and_velocity_tables_read_the_logs():
    recs = {"suffix_long": _suffix_rec(), "suffix_ref": _suffix_rec(thinned=False), "vel_long": _vel_rec()}
    st = LG.solver_table(recs, late_from=1000).set_index("run")
    assert st.index.tolist() == ["suffix_long", "suffix_ref"]
    assert st.loc["suffix_long", "steps"] == 3000 and st.loc["suffix_long", "fallback_steps"] == 1
    assert st.loc["suffix_long", "iters_max"] == 5 and st.loc["suffix_long", "iters_mean_late"] == pytest.approx(1.5)
    assert st.loc["suffix_long", "wall_per_step_late"] == pytest.approx(0.01)
    assert st.loc["suffix_long", "train_hours"] == pytest.approx(30 / 3600) and st.loc["suffix_long", "eval_hours"] == pytest.approx(30 / 3600)
    assert st.loc["suffix_ref", "steps"] == 3000 and st.loc["suffix_ref", "fallback_steps"] == 0
    ev = {k: LG.evals(v) for k, v in recs.items()}
    vt = LG.velocity_table(recs, ev, last_epochs=5).set_index("run")
    assert vt.index.tolist() == ["vel_long"]
    assert vt.loc["vel_long", "epochs"] == 10 and vt.loc["vel_long", "steps"] == 2310
    assert vt.loc["vel_long", "epochs_loss_above_1"] == 1 and vt.loc["vel_long", "loss_last_median"] == pytest.approx(0.1)
    assert vt.loc["vel_long", "best_train_chamfer_paper"] == pytest.approx(0.5) and vt.loc["vel_long", "best_epoch"] == 5
    assert vt.loc["vel_long", "first_diverged_epoch"] == 10 and vt.loc["vel_long", "evals_diverged_train"] == 1
    assert vt.loc["vel_long", "final_train_chamfer_paper"] == pytest.approx(2e3)
