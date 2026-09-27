import json
from types import SimpleNamespace
import numpy as np
import pytest
from nsga2_sats.cli import main
from nsga2_sats.experiments import make_plan, ensure_empty, execute_plan, discover_fronts, analyze_records, RunTracker
from nsga2_sats.problem import SatelliteSchedulingProblem


def test_main_budget_and_one_based_seed_schedule():
    plan = make_plan("main")
    assert plan["run_count"] == 210
    assert not plan["execution_blockers"]
    first, last = plan["runs"][0], plan["runs"][-1]
    assert (first["population"], first["generations"], first["seed"], last["seed"]) == (100, 500, 1001, 7030)
    assert len(plan["source"]["sha256"]) == 64
    assert len(plan["input_instances"]) == 7


def test_ablation_budget_covers_five_distinct_variants():
    plan = make_plan("ablation")
    assert plan["run_count"] == 1050
    assert {r["variant"] for r in plan["runs"]} == {"full", "no_pdac", "no_lcam", "no_sats", "base"}
    assert {r["generations"] for r in plan["runs"]} == {200}


def test_large_and_cpu_require_explicit_missing_information():
    assert make_plan("large")["execution_blockers"]
    assert make_plan("cpu")["execution_blockers"]
    assert not make_plan("cpu", hv_target=.945, cpu_budget=3600)["execution_blockers"]


def test_large_rejects_incorrect_antenna_count(tmp_path, monkeypatch):
    (tmp_path / "reqlf19.dat").write_text("test")
    (tmp_path / "requp19.dat").write_text("test")
    monkeypatch.setattr("nsga2_sats.experiments.SatelliteSchedulingProblem", lambda **kw: SimpleNamespace(n_tasks=500, n_antennas=19))
    with pytest.raises(ValueError, match="supplement requires 500/24"):
        make_plan("large", scenarios=[8], data_dir=tmp_path)


def test_nonempty_output_never_overwritten(tmp_path):
    (tmp_path / "original.txt").write_text("retain")
    with pytest.raises(ValueError, match="not empty"):
        ensure_empty(tmp_path)
    assert (tmp_path / "original.txt").read_text() == "retain"


def test_dry_run_writes_nothing(tmp_path, capsys):
    output = tmp_path / "untouched"
    assert main(["--profile", "main", "--dry-run", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["run_count"] == 210
    assert not output.exists()


def test_sensitivity_settings_are_wired_to_actual_parameters():
    plan = make_plan("sensitivity", runs=1, sensitivity_set="all")
    assert plan["run_count"] == 23
    row = next(r for r in plan["runs"] if r["setting"] == "transfer_0p8")
    assert row["parameters"]["transfer_factor"] == .8
    assert row["parameters"]["load_weight"] == .5
    assert next(r for r in plan["runs"] if r["setting"] == "primary_3")["parameters"]["crossover_probability"] == .6


def test_demo_exports_feasible_reproducible_schedules(tmp_path):
    fronts = []
    for label in ("first", "second"):
        output = tmp_path / label
        plan = make_plan("demo", population=6, generations=2, output=output)
        assert plan["input_instances"]["1"]["tasks"] == 322
        assert plan["input_instances"]["1"]["antennas"] == 19
        execute_plan(plan, progress=lambda _: None)
        run_dir = output / "full" / "AFIT1" / "run_01"
        front = np.loadtxt(run_dir / "pareto.csv", delimiter=",", skiprows=1, ndmin=2)
        solutions = np.load(run_dir / "solutions.npy", allow_pickle=False)
        problem = SatelliteSchedulingProblem(1, data_dir=plan["profile"]["data_dir"])
        assert all(problem.is_feasible(x) for x in solutions)
        np.testing.assert_allclose(front, [problem.objectives(x) for x in solutions])
        meta = json.loads((run_dir / "metadata.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        assert meta["source_sha256"] == plan["source"]["sha256"]
        assert len(history) == 2 and "pre_repair_conflicts" in history[-1]
        assert meta["search_objective_calls"] >= meta["pymoo_evaluations"]
        fronts.append(front)
    np.testing.assert_array_equal(*fronts)


def test_analysis_only_uses_present_legacy_runs(tmp_path):
    source = tmp_path / "source" / "NSGA2-SATS" / "pareto_fronts" / "SRSP1"
    source.mkdir(parents=True)
    (source / "run_01.csv").write_text("f1,f2\n0.2,0.8\n0.5,0.5\n")
    records = discover_fronts(tmp_path / "source")
    assert len(records) == 1
    summary = analyze_records(records, tmp_path / "analysis")
    assert summary[0]["algorithm"] == "NSGA2-SATS" and summary[0]["runs"] == 1
    stats = json.loads((tmp_path / "analysis" / "statistics.json").read_text())
    assert stats["algorithms_present"] == ["NSGA2-SATS"]


def test_cpu_hits_after_budget_are_censored(monkeypatch):
    problem = SimpleNamespace(objective_calls=2)
    tracker = RunTracker(problem, (1.1, 1.1), hv_target=.5, cpu_budget=1)
    monkeypatch.setattr("nsga2_sats.experiments.time.process_time", lambda: tracker.cpu_start + 2)
    algorithm = SimpleNamespace(n_gen=1, record=[], evaluator=SimpleNamespace(n_eval=2),
                                pop=SimpleNamespace(get=lambda _: np.array([[.1, .1]])),
                                termination=SimpleNamespace(force_termination=False))
    tracker.notify(algorithm)
    assert all(value is None for value in tracker.thresholds.values())
    assert tracker.stop_reason == "cpu_budget"
    assert algorithm.termination.force_termination


def test_plan_honors_reference_and_rejects_irrelevant_sensitivity():
    assert make_plan("demo", reference=[1.2, 1.2])["profile"]["reference_point"] == [1.2, 1.2]
    with pytest.raises(ValueError, match="requires --profile sensitivity"):
        make_plan("demo", sensitivity_set="primary")


def test_changed_input_is_rejected_before_creating_output(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    from nsga2_sats.experiments import ROOT
    for name in ("reqlf12.dat", "requp12.dat"):
        source = ROOT / "benchmarks" / "afit" / name
        (inputs / name).write_bytes(source.read_bytes())
    output = tmp_path / "results"
    plan = make_plan("demo", data_dir=inputs, output=output)
    path = inputs / "reqlf12.dat"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Input files changed after planning"):
        execute_plan(plan)
    assert not output.exists()
