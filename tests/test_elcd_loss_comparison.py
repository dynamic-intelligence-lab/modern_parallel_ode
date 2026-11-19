"""The velocity records' loader of experiments/elcd/figures/loss_comparison.py on synthetic records. CPU, no
JAX."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.elcd.figures import loss_comparison as LF  # noqa: E402


def _vel_record(epochs, per_epoch=10, eval_every=1, cosine=False):
    log = [dict(step=per_epoch * (e + 1), epoch=e, H=0, loss=1.0, loss_sum=10.0, wall=36.0, compile=e == 0) for e in range(epochs)]
    evals = [dict(step=per_epoch * (e + 1), rows=[], summary={v: 1.0 / (e + 1) for v in LF.METRICS.values()}, seconds=1.0)
             for e in range(epochs) if (e + 1) % eval_every == 0 or e == epochs - 1]
    return dict(name="x", config=dict(loss="vel", epochs=epochs, lr_schedule="cosine" if cosine else "const"), log=log, evals=evals,
                final=evals[-1]["summary"])


def test_load_velocity_reads_epochs_steps_hours_and_marks_the_final(tmp_path):
    cell = tmp_path / "chain8z" / "fp32" / "s0"
    cell.mkdir(parents=True)
    (cell / "vel-const100.json").write_text(json.dumps(_vel_record(3, eval_every=2)))
    (cell / "vel-const100.stale-x.json").write_text(json.dumps(_vel_record(2)))      # set aside: skipped
    (cell / "laser.json").write_text(json.dumps(dict(config=dict(loss="suffix"), log=[], evals=[])))
    v = LF.load_velocity(tmp_path)
    assert list(v["item"].unique()) == ["vel-const100"] and len(v) == 2
    assert list(v["epoch"]) == [2.0, 3.0] and list(v["step"]) == [20, 30]
    assert list(v["hours"]) == pytest.approx([0.02, 0.03]) and list(v["final"]) == [False, True]
    assert v["train_chamfer_paper"].iloc[1] == pytest.approx(1 / 3)


def test_load_velocity_maps_the_final_evaluation_s_epoch_count_to_the_last_step(tmp_path):
    cell = tmp_path / "chain8z" / "fp32" / "s0"
    cell.mkdir(parents=True)
    rec = _vel_record(3, eval_every=2)
    rec["evals"][-1]["step"] = 3                     # the driver's final evaluation logs the epoch count as its step
    (cell / "vel-const100.json").write_text(json.dumps(rec))
    v = LF.load_velocity(tmp_path)
    assert list(v["step"]) == [20, 30] and list(v["epoch"]) == [2.0, 3.0]
    assert list(v["hours"]) == pytest.approx([0.02, 0.03]) and list(v["final"]) == [False, True]
