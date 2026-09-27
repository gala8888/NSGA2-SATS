import numpy as np
import pytest

from nsga2_sats.problem import SatelliteSchedulingProblem
from nsga2_sats.operators import hierarchical_insertion, load_balance


def task(task_id, duration, options):
    return {"id": task_id, "duration": duration, "turnaround": 0,
            "chanceList": [[0, 0, 0]] + options,
            "antennaList": list(dict.fromkeys(option[0] for option in options))}


class StrategyRng:
    def __init__(self, strategy=0, chosen=None):
        self.strategy, self.chosen, self.calls = strategy, chosen, []

    def integers(self, high):
        self.calls.append(("integers", high))
        assert high == 3  # Generator.integers(3) is a uniform three-way draw.
        return self.strategy

    def choice(self, values, size, replace):
        self.calls.append(("choice", list(values), size, replace))
        assert replace is False
        return np.asarray(self.chosen if self.chosen is not None else values[:size])


@pytest.mark.parametrize("strategy, selected", [(0, 0), (1, 1), (2, 2)])
def test_hierarchical_ordering_uses_each_paper_heuristic(strategy, selected):
    tasks = [
        task(1, 5, [[1, 0, 5], [1, 10, 15], [1, 20, 25]]),
        task(2, 1, [[1, 30, 31], [1, 32, 33], [2, 30, 31], [2, 32, 33]]),
        task(3, 2, [[1, 40, 42], [2, 40, 42]]),
    ]
    problem = SatelliteSchedulingProblem.from_data(tasks, [[1, "A"], [2, "B"]])
    rng = StrategyRng(strategy)
    result = hierarchical_insertion([0, 0, 0], problem, rng)
    assert np.flatnonzero(result).tolist() == [selected]
    assert rng.calls == [("integers", 3)]


def test_insertion_prefers_low_load_before_first_window_in_data_order():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10]]), task(2, 5, [[1, 30, 35], [2, 30, 35]])],
        [[1, "A"], [2, "B"]])
    x = np.array([1, 0])
    result = hierarchical_insertion(x, problem, np.random.default_rng(1))
    np.testing.assert_array_equal(result, [1, 2])
    np.testing.assert_array_equal(x, [1, 0])
    assert not np.shares_memory(result, x)


def test_insertion_falls_back_after_all_low_load_windows_conflict():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 20, [[1, 0, 20]]), task(2, 5, [[2, 0, 5]]),
         task(3, 5, [[1, 50, 55], [2, 1, 6]])], [[1, "A"], [2, "B"]])
    result = hierarchical_insertion([1, 1, 0], problem, np.random.default_rng(2))
    np.testing.assert_array_equal(result, [1, 1, 1])
    assert problem.is_feasible(result)


def test_multiple_insertions_check_the_updated_candidate():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 5, [[1, 0, 5]]), task(2, 5, [[1, 0, 5]])], [[1, "A"]])
    result = hierarchical_insertion([0, 0], problem, np.random.default_rng(3), n_tasks=2)
    assert np.count_nonzero(result) == 1
    assert problem.is_feasible(result)


def test_migration_samples_tasks_instead_of_always_choosing_longest():
    tasks = [task(i + 1, i + 1, [[1, 10 * i, 11 * i + 1], [2, 10 * i, 11 * i + 1]])
             for i in range(4)]
    problem = SatelliteSchedulingProblem.from_data(tasks, [[1, "A"], [2, "B"]])
    x = np.ones(4, dtype=int)
    rng = StrategyRng(chosen=[1])
    result = load_balance(x, problem, rng)
    np.testing.assert_array_equal(result, [1, 2, 1, 1])
    assert rng.calls == [("choice", [0, 1, 2, 3], 1, False)]
    np.testing.assert_array_equal(x, [1, 1, 1, 1])
    assert not np.shares_memory(x, result)


def test_sequential_migration_does_not_collide_with_previously_moved_task():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 5, [[1, 0, 5], [2, 30, 35]]),
         task(2, 5, [[1, 10, 15], [2, 30, 35]])], [[1, "A"], [2, "B"]])
    result = load_balance([1, 1], problem, StrategyRng(chosen=[0, 1]), transfer_factor=2)
    np.testing.assert_array_equal(result, [2, 1])
    assert problem.is_feasible(result)


def test_unavailable_fixed_target_keeps_original_assignments():
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 10, [[1, 0, 10], [3, 0, 10]]), task(2, 5, [[1, 20, 25]])],
        [[1, "A"], [2, "B"], [3, "C"]])
    x = np.array([1, 1])
    result = load_balance(x, problem, StrategyRng(chosen=[0]))
    np.testing.assert_array_equal(result, x)
    assert not np.shares_memory(result, x)


@pytest.mark.parametrize("candidate", [[0, 0], [1, 2]])
def test_empty_or_balanced_migration_returns_independent_unchanged_copy(candidate):
    problem = SatelliteSchedulingProblem.from_data(
        [task(1, 5, [[1, 0, 5], [2, 0, 5]]), task(2, 5, [[1, 10, 15], [2, 10, 15]])],
        [[1, "A"], [2, "B"]])
    x = np.array(candidate)
    rng = StrategyRng()
    result = load_balance(x, problem, rng)
    np.testing.assert_array_equal(result, x)
    assert not np.shares_memory(result, x)
    assert rng.calls == []


def test_insertion_no_unscheduled_tasks_returns_independent_copy():
    problem = SatelliteSchedulingProblem.from_data([task(1, 5, [[1, 0, 5]])], [[1, "A"]])
    x = np.array([1])
    result = hierarchical_insertion(x, problem, StrategyRng())
    np.testing.assert_array_equal(result, x)
    assert not np.shares_memory(result, x)


def test_neighborhoods_preserve_feasibility_and_inputs_across_random_candidates():
    tasks = [task(i + 1, 5, [[1, i * 2, i * 2 + 5], [2, i * 2, i * 2 + 5]]) for i in range(8)]
    problem = SatelliteSchedulingProblem.from_data(tasks, [[1, "A"], [2, "B"]])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 3, 8)
        x = problem.repair_solution(raw)
        original = x.copy()
        inserted = hierarchical_insertion(x, problem, rng, n_tasks=3)
        balanced = load_balance(x, problem, rng)
        assert problem.is_feasible(inserted) and problem.is_feasible(balanced)
        np.testing.assert_array_equal(x, original)
        assert not np.shares_memory(inserted, x) and not np.shares_memory(balanced, x)
