from copy import deepcopy
import math
import numpy as np
import pytest

from nsga2_sats.data_io import read
from nsga2_sats.problem import SatelliteSchedulingProblem


def task(task_id, duration, options, turnaround=0):
    return {"id": task_id, "duration": duration, "turnaround": turnaround,
            "chanceList": [[0, 0, 0]] + options,
            "antennaList": list(dict.fromkeys(option[0] for option in options))}


def test_profit_weighting_and_sample_cv():
    tasks = [task(1, 10, [[1, 0, 10]]), task(2, 3, [[1, 20, 23]]),
             task(3, 6, [[2, 0, 6]])]
    problem = SatelliteSchedulingProblem.from_data(tasks, [[1, "A"], [2, "B"]], profits=[2, 3, 5])
    np.testing.assert_allclose(problem.objectives([1, 0, 1]), [.3, math.sqrt(8) / 8])
    np.testing.assert_array_equal(problem.loads([1, 0, 1]), [10, 6])
    assert problem.objective_calls == 1
    assert problem.antennas == [[1, "A", [1, 2], 13.], [2, "B", [3], 6.]]


def test_default_profit_is_equal_and_zero_work_has_zero_cv():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]]), task(2, 3, [[1, 20, 23]])], [[1, "A"]])
    np.testing.assert_allclose(problem.objectives([1, 0]), [.5, 0])
    np.testing.assert_array_equal(problem.objectives([0, 0]), [1, 0])


@pytest.mark.parametrize("start, expected", [(12., 1), (13., 0), (14., 0)])
def test_turnaround_boundary_is_respected(start, expected):
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]], turnaround=9),
         task(2, 7, [[1, start, start + 7]], turnaround=3)], [[1, "A"]])
    assert problem.conflict_pairs([1, 1]) == expected
    assert problem.is_feasible([1, 1]) is (expected == 0)


def test_conflict_metric_counts_all_pairs_before_independent_repair():
    original = [task(i, 10, [[1, 0, 10]]) for i in range(1, 4)]
    snapshot = deepcopy(original)
    problem = SatelliteSchedulingProblem.from_data(original, [[1, "A"]])
    x = np.ones(3, dtype=int)
    repaired = problem.repair_solution(x)
    assert problem.conflict_pairs(x) == 3
    assert np.count_nonzero(x != repaired) == 2
    assert problem.conflict_pairs(repaired) == 0
    assert np.array_equal(x, [1, 1, 1]) and not np.shares_memory(x, repaired)
    assert original == snapshot
    with pytest.raises(ValueError, match="feasible"):
        problem.objectives(x)
    assert problem.objective_calls == 0


def test_repair_preserves_legacy_larger_option_count_priority():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]]), task(2, 10, [[1, 0, 10], [2, 0, 10]])],
        [[1, "A"], [2, "B"]])
    np.testing.assert_array_equal(problem.repair_solution([1, 1]), [0, 1])


def test_objectives_and_pymoo_evaluation_do_not_modify_candidate():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]]), task(2, 10, [[1, 0, 10]])], [[1, "A"]])
    raw = np.array([[1, 1]], dtype=int)
    result = problem.evaluate(raw)
    np.testing.assert_array_equal(raw, [[1, 1]])
    np.testing.assert_allclose(result, [[.5, 0]])
    assert problem.objective_calls == 1
    x = np.array([1, 0])
    expected = problem.objectives(x)
    np.testing.assert_array_equal(x, [1, 0])
    np.testing.assert_array_equal(expected, [.5, 0])


def test_can_place_excludes_itself_but_checks_other_assignments():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10], [2, 0, 10]]), task(2, 10, [[2, 0, 10]])],
        [[1, "A"], [2, "B"]])
    x = np.array([1, 1])
    assert problem.can_place(x, 0, 1)
    assert not problem.can_place(x, 0, 2)
    assert problem.can_place(x, 0, 0)
    np.testing.assert_array_equal(x, [1, 1])


def test_can_place_ignores_unrelated_existing_conflicts():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]]), task(2, 10, [[1, 0, 10]]), task(3, 10, [[2, 0, 10]])],
        [[1, "A"], [2, "B"]])
    assert problem.can_place([1, 1, 0], 2, 1)
    np.testing.assert_array_equal(problem.loads([1, 1, 0]), [20, 0])


def test_invalid_encoding_is_rejected_and_repair_returns_integer_copy():
    problem = SatelliteSchedulingProblem.from_data([task(1, 10, [[1, 0, 10]])], [[1, "A"]])
    assert not problem.is_feasible([-1])
    assert not problem.is_feasible([.4])
    with pytest.raises(ValueError, match="integer"):
        problem.conflict_pairs([2])
    np.testing.assert_array_equal(problem.repair_solution([float("nan")]), [0])
    np.testing.assert_array_equal(problem.repair_solution([.8]), [1])


def test_reader_groups_multiple_windows_and_counts_capacity_once(tmp_path):
    (tmp_path / "reqlf12.dat").write_text("5 A 0 3 3 1 100\n5 A 10 13 3 1 100\n")
    (tmp_path / "requp12.dat").write_text("5 B 1.2 4.2 2 1 200\n5 B 7.2 10.2 2 1 200\n")
    tasks, antennas = read(1, tmp_path)
    assert len(tasks) == 2 and len(set(task["id"] for task in tasks)) == 2
    assert [task["source_id"] for task in tasks] == [5, 5]
    assert tasks[0]["begin"] == 0 and tasks[0]["end"] == 13
    assert tasks[1]["chanceList"][1:] == [[2, 2., 4.], [2, 8., 10.]]
    assert [ant[3] for ant in antennas] == [3, 2]
    problem = SatelliteSchedulingProblem.from_data(tasks, antennas)
    np.testing.assert_array_equal(problem.profits, [1, 1])


def test_reader_rejects_inconsistent_repeated_task_attributes(tmp_path):
    (tmp_path / "reqlf12.dat").write_text("1 A 0 3 3 1\n1 B 10 14 4 1\n")
    (tmp_path / "requp12.dat").write_text("")
    with pytest.raises(ValueError, match="inconsistent duration"):
        read(1, tmp_path)


def test_reader_keeps_official_zero_duration_boundary(tmp_path):
    (tmp_path / "reqlf12.dat").write_text("97 BOSS-A 995 995 0 20\n97 BOSS-B 995 995 0 20\n")
    (tmp_path / "requp12.dat").write_text("")
    problem = SatelliteSchedulingProblem(1, tmp_path)
    assert problem.n_tasks == 1 and problem.n_antennas == 2
    assert problem.can_place([0], 0, 1)
    np.testing.assert_array_equal(problem.objectives([1]), [0, 0])


def test_supplied_capacity_is_validated_against_visibility_sum():
    with pytest.raises(ValueError, match="distinct visible tasks"):
        SatelliteSchedulingProblem.from_data([task(1, 10, [[1, 0, 10]])], [[1, "A", [1], 99]])
