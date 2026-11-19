"""The ELCD paper runs' cell runner, experiments/elcd/train/run_cell.py: the order and its rotation,
the driver arguments (parsed by the real driver's parser), the probe, the completeness rule, the median, and the
orchestration with a fake driver (no JAX run); also the refusal of an unknown key by select_scale_elk_rate.parse_extras. CPU.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.elcd.train import run_cell as P  # noqa: E402

BASE = ["--system", "chain8z", "--recdir", "/nonexistent", "--solvers", "deer,laser,scale-elk:0.3"]


def _names(argv):
    return [it["name"] for it in P.plan_items(P.parse_args(argv))]


def test_order_is_a_williams_design_row_by_seed_and_precision():
    d, l, e, t = "deer", "laser", "scale-elk0.3", "loop_time200"
    assert _names(BASE + ["--dtype", "fp64", "--seed", "0"]) == [d, l, t, e]          # row 0 of n = 4: 0 1 3 2
    assert _names(BASE + ["--dtype", "fp64", "--seed", "1"]) == [l, e, d, t]          # row 1: 1 2 0 3
    assert _names(BASE + ["--dtype", "fp32", "--seed", "0"]) == [l, e, d, t]          # fp32 shifts by one row
    assert _names(BASE + ["--dtype", "fp32", "--seed", "6"]) == [t, d, e, l]          # row 3: 3 0 2 1
    assert _names(BASE + ["--dtype", "fp64", "--seed", "2", "--full-loop"])[-1] == "loop_full"
    assert _names(BASE + ["--dtype", "fp64", "--seed", "0", "--loop-time", "0"]) == [d, l, e]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_williams_rows_balance_positions_and_carryover(n):
    """Every item is in every position equally often, and every ordered pair (a directly before b) occurs equally
    often: no solver always follows the same one."""
    from collections import Counter
    rows = P.williams(n)
    assert len(rows) == (n if n % 2 == 0 else 2 * n) and all(sorted(r) == list(range(n)) for r in rows)
    positions = Counter((pos, x) for r in rows for pos, x in enumerate(r))
    pairs = Counter((a, b) for r in rows for a, b in zip(r, r[1:]))
    assert len(set(positions.values())) == 1 and len(pairs) == n * (n - 1) and len(set(pairs.values())) == 1


def test_driver_arguments_parse_and_carry_the_protocol():
    """Every run's argv is accepted by the real driver's parser, with the paper's tolerance per precision, cap N, the
    fallback, no diagnostics and the paper's recipe."""
    from experiments.elcd.train import train_elcd as T
    for dtype, tol in (("fp64", 1e-7), ("fp32", 1e-5)):
        a = P.parse_args(BASE + ["--dtype", dtype, "--seed", "4", "--full-loop", "--time-only", "500"])
        for it in P.plan_items(a):
            args = T.parse_args(it["args"])
            assert (args.system, args.dtype, args.seed, args.tol) == ("chain8z", dtype, 4, tol)
            assert (args.loss, args.starts, args.lr_schedule, args.steps, args.diag_every) == ("suffix", "64", "cosine",
                                                                                              10000, 0)
            assert args.x0_anchor == 0.0 and args.start_draw == "uniform"
            if it["kind"] == "parallel":
                assert (args.fallback, args.max_iter, args.eval_every, args.time_only) == ("seq", 0, 500, 500)
            elif it["kind"] == "loop_time":
                assert (args.solver, args.time_only, args.eval_every) == ("seq", 200, 0)
            else:
                assert (args.solver, args.time_only, args.eval_every) == ("seq", 0, 500)
    for kind, solver in (("loop", "seq"), ("deer", "deer")):
        probe = T.parse_args(P.probe_args(kind))
        assert (probe.system, probe.dtype, probe.seed, probe.solver, probe.time_only, probe.tol) == ("chain8z", "fp64", 0,
                                                                                                    solver, 20, 1e-7)
    assert T.parse_args(P.probe_args("deer")).fallback == "seq" and T.parse_args(P.probe_args("deer")).max_iter == 0
    chunked = P.plan_items(P.parse_args(BASE + ["--dtype", "fp64", "--seed", "0", "--loop-chunk", "192",
                                                "--par-chunk", "48", "--full-loop"]))
    assert all(T.parse_args(it["args"]).start_chunk == (48 if it["kind"] == "parallel" else 192) for it in chunked)


def test_refusals():
    for extra in (["--solvers", "deer,seq"], ["--solvers", "deer,deer"], ["--time-only", "-1"]):
        with pytest.raises(SystemExit):
            P.parse_args(BASE[:4] + ["--dtype", "fp64", "--seed", "0"] + extra)


def _record(path, steps, final=True, npz=True, end=True, device="gpu"):
    rec = dict(device=device, log=[dict(step=i, wall=1.0 + i, compile=i == 0) for i in range(steps)],
               final={"x": 1} if final else {}, provenance=dict(start={}, **({"end": {}} if end else {})))
    path.write_text(json.dumps(rec))
    if npz:
        path.with_suffix(".npz").write_bytes(b"")
    return rec


def test_record_complete_and_median(tmp_path):
    p = tmp_path / "r.json"
    assert not P.record_complete(p, 0)
    _record(p, 3, npz=False)
    assert not P.record_complete(p, 0) and P.record_complete(p, 3) and not P.record_complete(p, 4)
    _record(p, 3)
    assert P.record_complete(p, 0)
    _record(p, 3, end=False)
    assert not P.record_complete(p, 0)
    p.write_text("{truncated")
    assert not P.record_complete(p, 0)
    assert P.median_wall(_record(p, 4)) == 3.0                     # steps 1..3 (walls 2, 3, 4); step 0 compiled


FAKE = r'''
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1]
steps = int(argv[argv.index("--time-only") + 1]) if "--time-only" in argv else int(argv[argv.index("--steps") + 1])
sys.path.insert(0, os.environ["FAKE_REPO"])
from experiments.elcd.train.train_elcd import parse_args
config = vars(parse_args(argv))
calls = os.environ["FAKE_CALLS"]
with open(calls, "a") as f:
    f.write(os.path.basename(out) + "\n")
if os.path.basename(out) == os.environ.get("FAKE_FAIL"):
    sys.exit(3)
rec = dict(device="gpu", config=config, log=[dict(step=i, wall=0.5, compile=i == 0) for i in range(steps)],
           final={} if "--time-only" in argv else {"chamfer": 1.0}, provenance=dict(start={}, end={}))
if "--time-only" not in argv:
    open(out[:-5] + ".npz", "wb").close()
json.dump(rec, open(out, "w"))
'''


def test_cell_runs_probes_and_skips_complete_records(tmp_path, monkeypatch):
    fake = tmp_path / "fake_driver.py"
    fake.write_text(FAKE)
    monkeypatch.setattr(P, "DRIVER", fake)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    monkeypatch.setenv("FAKE_REPO", str(P.REPO))
    monkeypatch.setenv("FAKE_FAIL", "laser.json")
    argv = ["--system", "chain8z", "--dtype", "fp64", "--seed", "1", "--solvers", "deer,laser",
            "--recdir", str(tmp_path / "rec"), "--python", sys.executable, "--allow-cpu", "--allow-dirty",
            "--loop-time", "5", "--steps", "3"]
    assert P.main(argv) == 1                                       # laser failed
    ran = calls.read_text().split()
    probes = lambda k: [f"probe_loop_{k:02d}.json", f"probe_deer_{k:02d}.json"]
    assert ran == (probes(0) + ["laser.json"] + probes(1) + ["loop_time5.json"] + probes(2) + ["deer.json"]
                   + probes(3))
    cell = json.loads((tmp_path / "rec" / "chain8z" / "fp64" / "s1" / "cell.json").read_text())
    assert cell["order"] == ["laser", "loop_time5", "deer"]
    assert [r["name"] for r in cell["runs"]] == ["laser", "loop_time5", "deer"]
    assert [r["complete"] for r in cell["runs"]] == [False, True, True] and cell["runs"][0]["rc"] == 3
    assert [p["after"] for p in cell["probes"]] == [a for a in ("start", "laser", "loop_time5", "deer") for _ in "ld"]
    assert [p["kind"] for p in cell["probes"]] == ["loop", "deer"] * 4 and all(p["median_wall"] == 0.5 for p in cell["probes"])
    # the resubmitted task reruns only the failed solver; probes are numbered on
    calls.write_text("")
    monkeypatch.setenv("FAKE_FAIL", "")
    assert P.main(argv) == 0
    assert calls.read_text().split() == probes(4) + ["laser.json"] + probes(5)
    cell = json.loads((tmp_path / "rec" / "chain8z" / "fp64" / "s1" / "cell.json").read_text())
    assert len(cell["probes"]) == 12 and len(cell["provenance"]) == 4 and cell["runs"][-1]["complete"]
    # a record at an item's path that is not that item's run (another tolerance) is moved aside and the item reruns
    cdir = tmp_path / "rec" / "chain8z" / "fp64" / "s1"
    rec = json.loads((cdir / "deer.json").read_text())
    rec["config"]["tol"] = 1e-9
    (cdir / "deer.json").write_text(json.dumps(rec))
    calls.write_text("")
    assert P.main(argv) == 0
    assert calls.read_text().split() == probes(6) + ["deer.json"] + probes(7)
    assert len(list(cdir.glob("deer.stale-*.json"))) == 1 and len(list(cdir.glob("deer.stale-*.npz"))) == 1


def test_dry_run_prints_the_plan(capsys):
    assert P.main(BASE + ["--dtype", "fp32", "--seed", "3", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dir"].endswith("chain8z/fp32/s3") and len(plan["items"]) == 4
    assert plan["probes"] == {k: P.probe_args(k) for k in P.PROBE_KINDS}


def test_a_tagged_cell_keeps_its_own_summary(tmp_path, monkeypatch):
    """The pendulum's full loop run is a second job on the seed-0 fp64 cell: --cell-tag _loop keeps its summary and
    probes apart from the cell's."""
    fake = tmp_path / "fake_driver.py"
    fake.write_text(FAKE)
    monkeypatch.setattr(P, "DRIVER", fake)
    monkeypatch.setenv("FAKE_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("FAKE_REPO", str(P.REPO))
    argv = ["--system", "npend:8-N5000", "--dtype", "fp64", "--seed", "0", "--solvers", "", "--loop-time", "0", "--steps", "3",
            "--full-loop", "--cell-tag", "_loop", "--recdir", str(tmp_path / "rec"), "--python", sys.executable,
            "--allow-cpu", "--allow-dirty"]
    assert P.main(argv) == 0
    cell_dir = tmp_path / "rec" / "npend-8-N5000" / "fp64" / "s0"
    probes = lambda k: [f"probe_loop_loop_{k:02d}.json", f"probe_loop_deer_{k:02d}.json"]
    assert (tmp_path / "calls.txt").read_text().split() == probes(0) + ["loop_full.json"] + probes(1)
    assert sorted(os.listdir(cell_dir)) == sorted(["cell_loop.json", "loop_full.json", "loop_full.npz"] + probes(0)
                                                  + probes(1))


def test_tuning_picks_the_smallest_summed_step_hours(tmp_path):
    """select_scale_elk_rate: per system and precision, the complete rate with the least summed wall over the seeds; a pick at
    an end of the grid is flagged; an incomplete rate is not a candidate."""
    from experiments.elcd.train import select_scale_elk_rate as PT

    def write(seed, g, wall, steps=4, fallbacks=0):
        d = tmp_path / "chain8z" / "fp32" / f"s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        log = [dict(step=i, wall=wall, compile=i == 0, seq_fallback=int(i < fallbacks), failed_max_iter=int(i < fallbacks))
               for i in range(steps)]
        (d / f"scale-elk{g}.json").write_text(json.dumps(dict(log=log, provenance=dict(start={}, end={}))))
    for seed in (10, 11):
        write(seed, "0.3", 2.0, fallbacks=2)
        write(seed, "1", 1.0)
        write(seed, "3", 1.5)
    write(11, "10", 0.1, steps=3)                                  # incomplete: not a candidate
    sel = PT.select(tmp_path, [10, 11], 4)["chain8z/fp32"]
    assert sel["pick"] == 1.0 and not sel["at_edge"] and not sel["all_complete"]
    row = {r["g"]: r for r in sel["table"]}
    assert row[0.3]["fallback_steps"] == 4 and row[0.3]["failed_max_iter"] == 4 and not row[10.0]["complete"]
    assert abs(row[1.0]["step_hours"] - 8.0 / 3600) < 1e-12


def test_parse_extras_refuses_an_unknown_key():
    """parse_extras takes system and system/precision keys and refuses an unknown key (what it returns is checked in
    test_chunk_pick_is_the_smallest_late_wall_of_deer_and_laser)."""
    from experiments.elcd.train import select_scale_elk_rate as PT
    PT.parse_extras(["chain8z/fp64=--par-chunk 8", "chain8z/fp32=--par-chunk 16"])
    with pytest.raises(ValueError):
        PT.parse_extras(["chain9=--par-chunk 8"])
    PT.parse_extras(["npend-8-N5000=--loop-chunk 192", "npend-8-N5000/fp64=--par-chunk 8"])


def test_chunk_pick_is_the_smallest_late_wall_of_deer_and_laser(tmp_path, capsys):
    """select_chunk: per system and precision, the chunk with the least wall over the late half of the steps, DEER's
    and LASER's summed, among the chunks with both records complete; the lines it writes parse with
    select_scale_elk_rate.parse_extras."""
    from experiments.elcd.record_names import PAPER_LASER
    from experiments.elcd.train import select_chunk as PC
    from experiments.elcd.train import select_scale_elk_rate as PT

    def write(root, solver, chunk, walls, time_only=None, dtype="fp32"):
        d = tmp_path / root / "chain8z" / dtype / "s0"
        d.mkdir(parents=True, exist_ok=True)
        log = [dict(step=i, wall=w, compile=i == 0) for i, w in enumerate(walls)]
        rec = dict(config=dict(steps=10000, time_only=time_only or len(walls)), log=log,
                   suffix=dict(chunk=chunk, solves_per_step=192), provenance=dict(start={}, end={}))
        (d / f"{solver}.json").write_text(json.dumps(rec))
    write("pilot", "deer", 32, [30, 9, 2, 2])                 # late half: steps 2 and 3, 4 s
    write("pilot", PAPER_LASER, 32, [30, 9, 1, 1])            # 2 s: chunk 32 sums 6 s (LASER as the pilot named it)
    write("pilot_c8", "deer", 8, [30, 99, 1, 1])              # early steps do not count: chunk 8 sums 4 s
    write("pilot_c8", PAPER_LASER, 8, [30, 99, 1, 1])
    write("pilot_c4", "deer", 4, [30, 1, 0.1, 0.1])           # no LASER record: not a candidate
    write("pilot_c16", "deer", 16, [1, 1, 0.1], time_only=4)  # stopped early: left out
    write("pilot_c16", PAPER_LASER, 16, [1, 1, 0.1, 0.1])
    write("pilot", "deer", 0, [5, 5, 3, 3], dtype="fp64")     # one vmap: chunk 192
    write("pilot", PAPER_LASER, 0, [5, 5, 3, 3], dtype="fp64")
    roots = [tmp_path / r for r in ("pilot", "pilot_c8", "pilot_c4", "pilot_c16")]
    table = PC.chunk_table(roots)
    assert sorted(table["chain8z/fp32"]) == [4, 8, 16, 32] and list(table["chain8z/fp64"]) == [192]
    res = PC.pick(table["chain8z/fp32"])
    assert res["pick"] == 8 and res["seconds"] == {8: 4.0, 32: 6.0} and res["at_edge"]
    assert PC.pick(table["chain8z/fp64"]) == dict(pick=192, at_edge=False, seconds={192: 12.0})
    out = tmp_path / "extra.txt"
    assert PC.main([str(r) for r in roots] + ["--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines == ["chain8z/fp32=--par-chunk 8", "chain8z/fp64=--par-chunk 192"]
    assert PT.parse_extras(lines) == {"chain8z/fp32": "--par-chunk 8", "chain8z/fp64": "--par-chunk 192"}
    assert "pick chunk 8 (at an end" in capsys.readouterr().out
    assert PC.main([str(r) for r in roots] + ["--systems", "npend-8-N5000", "--out", str(out)]) == 0
    assert out.read_text() == ""


def test_summary_rows(tmp_path):
    from experiments.elcd.train import summarize_records as PS
    cell = tmp_path / "chain8z" / "fp32" / "s0"
    cell.mkdir(parents=True)
    log = [dict(step=i, wall=10.0 if i == 0 else 0.5, compile=i == 0, iters_mean=3.0, seq_fallback=int(i == 1),
                seq_fallback_solves=4 * int(i == 1), failed_max_iter=4 * int(i == 1), latent_max=2.0 + i,
                peak_bytes=2e9) for i in range(3)]
    (cell / "deer.json").write_text(json.dumps(dict(log=log, final={"train_chamfer_native_mean": 0.1},
                                                    device_kind="NVIDIA B200", provenance=dict(start=dict(host="g1")))))
    (cell / "deer_at2000.json").write_text(json.dumps(dict(log=log)))       # a snapshot: not a run
    (cell / "cell.json").write_text(json.dumps(dict(probes=[dict(name="probe_00", after="start", median_wall=3.5,
                                                                 steps=20)])))
    rows = PS.collect(tmp_path)
    assert [r["item"] for r in rows] == ["deer", "probe_00 (after start)"]
    r = rows[0]
    assert (r["steps"], r["compile_s"], r["median_s"], r["fallback_steps"], r["fallback_solves"]) == (3, 10.0, 0.5, 1, 4)
    assert (r["failed_max_iter"], r["latent_max"], r["peak_gb"], r["chamfer"], r["host"]) == (4, 4.0, 2.0, 0.1, "g1")
    assert abs(r["step_hours"] - 11.0 / 3600) < 1e-12 and rows[1]["median_s"] == 3.5


def test_skip_rule_checks_the_config_steps_and_device(tmp_path):
    """record_complete with the item's arguments: the config must be the item's, the log its steps, the device the
    GPU; a time-only record needs no npz."""
    from experiments.elcd.train import train_elcd as T
    a = P.parse_args(BASE + ["--dtype", "fp32", "--seed", "2", "--time-only", "4"])
    item = next(it for it in P.plan_items(a) if it["kind"] == "parallel")
    assert item["args"][-2:] == ["--time-only", "4"]
    path = tmp_path / "x.json"
    rec = dict(device="gpu", config=vars(T.parse_args(item["args"])), log=[{}] * 4, provenance=dict(end={}))
    path.write_text(json.dumps(rec))
    assert P.record_complete(path, 4, item["args"])
    assert not P.record_complete(path, 4, item["args"][:-2] + ["--time-only", "5"])          # other steps
    path.write_text(json.dumps(dict(rec, device="cpu")))
    assert not P.record_complete(path, 4, item["args"]) and P.record_complete(path, 4, item["args"], gpu=False)
    path.write_text(json.dumps(dict(rec, config=dict(rec["config"], max_iter=200))))
    assert not P.record_complete(path, 4, item["args"])
