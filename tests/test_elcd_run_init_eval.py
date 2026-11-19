"""The step-0 evaluation of a paper cell, experiments/elcd/train/run_init_eval.py: the driver arguments it
derives, the record's layout, the skip of an existing record, and the end-to-end run with the evaluation stubbed
(no rollout). CPU, no JAX computation."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.elcd.train import run_init_eval as I  # noqa: E402


def test_driver_argv_is_the_cell_plus_the_given_sizes():
    a = I.parse_args(["--system", "chain8z", "--dtype", "fp32", "--seed", "3", "--out", "x.json"])
    assert I.driver_argv(a) == ["--system", "chain8z", "--dtype", "fp32", "--seed", "3"]
    b = I.parse_args(["--system", "pend:8", "--dtype", "fp64", "--seed", "0", "--out", "x.json", "--hidden", "8",
                      "--flow-steps", "1"])
    assert I.driver_argv(b)[-6:] == ["--seed", "0", "--hidden", "8", "--flow-steps", "1"]


def test_driver_parse_of_the_argv_is_the_paper_config():
    """The record's config is the driver's own parse: init scratch and the paper's sizes by default, the seed and
    precision of the cell."""
    from experiments.elcd.train.train_elcd import parse_args as driver_parse
    a = I.parse_args(["--system", "chain8z", "--dtype", "fp32", "--seed", "7", "--out", "x.json"])
    cfg = vars(driver_parse(I.driver_argv(a)))
    assert (cfg["system"], cfg["dtype"], cfg["seed"], cfg["init"]) == ("chain8z", "fp32", 7, "scratch")
    assert (cfg["hidden"], cfg["flow_hidden"], cfg["flow_steps"]) == (16, 30, 2)


def test_record_layout_has_one_step_zero_evaluation_and_no_log():
    rows = [dict(demo=0, train_demo=True, chamfer_paper=1.0)]
    summary = dict(train_chamfer_paper_mean=1.0)
    rec = I.init_record(dict(seed=1), dict(N=10), "cpu", "cpu", "0.6.2", 9608, rows, summary, 12.5, dict(a=1), dict(b=2))
    assert rec["name"] == "init_eval" and "log" not in rec
    assert rec["evals"] == [dict(step=0, rows=rows, summary=summary, seconds=12.5)]
    assert rec["final"] == summary and rec["trainable"] == 9608
    assert rec["provenance"] == dict(start=dict(a=1), end=dict(b=2))


def test_main_skips_an_existing_record(tmp_path):
    out = tmp_path / "init_eval.json"
    out.write_text("{}")
    assert I.main(["--system", "chain8z", "--dtype", "fp32", "--seed", "0", "--out", str(out)]) == 0
    assert out.read_text() == "{}"


def test_main_writes_the_evaluation_of_the_initial_model(tmp_path, monkeypatch):
    """End to end with metrics.evaluate stubbed: the initial model is built from the seed (init_theta), the stub
    sees a Theta and the demos, and the record carries its rows and summary at step 0."""
    from experiments.elcd.train import data as D
    from experiments.elcd.train import metrics as M
    if not os.path.exists(os.path.join(D.LASA_DIR, "Sshape.mat")):
        pytest.skip(f"no LASA dataset under {D.LASA_DIR}; see experiments/elcd/download_data.sh")
    seen = {}

    def fake_evaluate(theta, demos, perturb_seed=0, **kw):
        seen.update(seed=perturb_seed, D=demos.D, leaves=len(theta.flow.linears))
        return dict(rows=[dict(demo=i, train_demo=i < demos.n_train, chamfer_paper=0.5) for i in range(demos.pos.shape[0])])

    monkeypatch.setattr(M, "evaluate", fake_evaluate)
    monkeypatch.setattr(M, "summarize", lambda rows: dict(train_chamfer_paper_mean=0.5, n=len(rows)))
    out = tmp_path / "s4" / "init_eval.json"
    assert I.main(["--system", "lasa:Sshape", "--dtype", "fp32", "--seed", "4", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert seen == dict(seed=4, D=2, leaves=3)
    assert rec["config"]["seed"] == 4 and rec["config"]["dtype"] == "fp32" and rec["data"]["N"] == 985
    assert rec["evals"][0]["step"] == 0 and rec["final"] == dict(train_chamfer_paper_mean=0.5, n=7)
    assert rec["trainable"] > 0 and "start" in rec["provenance"] and "end" in rec["provenance"]
