"""Independent manuscript-alignment regressions for algorithm state transitions."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2

from nsga2_sats.algorithm import AdaptiveCrossover, SimulatedAnnealingTabuSearch, build_algorithm
from nsga2_sats.config import AlgorithmConfig
from nsga2_sats.problem import SatelliteSchedulingProblem


class CrossoverRng:
    def __init__(self):
        self.gates = 0

    def random(self):
        self.gates += 1
        return .5

    def choice(self, n, p):
        assert n == 3 and np.isclose(np.sum(p), 1)
        return 2


def independent_tasks():
    return SatelliteSchedulingProblem.from_data([
        {"id": 1, "duration": 5, "turnaround": 0, "chanceList": [[0, 0, 0], [1, 0, 5]]},
        {"id": 2, "duration": 5, "turnaround": 0, "chanceList": [[0, 0, 0], [1, 10, 15]]},
    ], [[1, "A"]])


def test_pdac_credits_both_children_and_retains_scores_between_batches():
    problem, rng = independent_tasks(), CrossoverRng()
    crossover = AdaptiveCrossover(AlgorithmConfig(), rng)
    X = np.array([[[1, 0]], [[0, 1]]])
    children = crossover._do(problem, X, algorithm=SimpleNamespace(n_gen=2))
    # Each full-schedule child dominates both half-schedule parents.
    np.testing.assert_array_equal(children, [[[1, 1]], [[1, 1]]])
    np.testing.assert_allclose(crossover.scores, [1, 1, 2])
    assert rng.gates == 1
    np.testing.assert_array_equal(X, [[[1, 0]], [[0, 1]]])
    # The next batch has one full and one empty parent. Each child only dominates
    # the empty parent, adding 0.3 twice to retained credit from the prior batch.
    Y = np.array([[[1, 1]], [[0, 0]]])
    crossover._do(problem, Y, algorithm=SimpleNamespace(n_gen=3))
    np.testing.assert_allclose(crossover.scores, [1, 1, 2.6])
    assert rng.gates == 2
    assert problem.objective_calls == 8


def test_pdac_failed_probability_gate_preserves_genes_and_does_no_credit_evaluation():
    problem, rng = independent_tasks(), CrossoverRng()
    crossover = AdaptiveCrossover(replace(AlgorithmConfig(), crossover_probability=.4), rng)
    X = np.array([[[1, 0]], [[0, 1]]])
    children = crossover._do(problem, X)
    np.testing.assert_array_equal(children, X)
    assert not np.shares_memory(children, X)
    np.testing.assert_array_equal(crossover.scores, [1, 1, 1])
    assert rng.gates == 1 and problem.objective_calls == 0


class AlwaysAcceptRng:
    def random(self):
        return 0.


class ContextObjectives:
    values = {(0, 0): (5., 5.), (1, 0): (4., 4.),
              (1, 1): (3., 6.), (0, 1): (2., 2.)}

    def objectives(self, x):
        return np.array(self.values[tuple(x)])

    def is_feasible(self, x):
        return tuple(x) in self.values


def test_sats_aspiration_accepts_contextual_inverse_move_without_updating_aspiration():
    next_states = iter([[1, 0], [1, 1], [0, 1]])
    inputs = []

    def neighbor(x, problem, rng):
        inputs.append(x.copy())
        return np.asarray(next(next_states))

    search = SimulatedAnnealingTabuSearch(
        replace(AlgorithmConfig(), local_search_iterations=3), AlwaysAcceptRng(), neighbor)
    source = np.array([0, 0])
    result = search.search(source, ContextObjectives())
    np.testing.assert_array_equal(inputs, [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(result, [0, 1])
    np.testing.assert_array_equal(search.last_current, [0, 1])
    np.testing.assert_array_equal(search.last_best, [0, 1])
    # Appendix C only updates s* for accepted NON-tabu moves.
    np.testing.assert_array_equal(search.last_aspiration, [1, 0])
    assert [row["reason"] for row in search.last_trace] == ["improvement", "annealing", "aspiration"]
    assert search.last_trace[-1]["tabu"] is True
    np.testing.assert_array_equal(source, [0, 0])
    result[:] = 99
    np.testing.assert_array_equal(search.last_best, [0, 1])


def test_exhausted_offspring_generation_does_not_repeat_previous_repair_statistics(monkeypatch):
    problem = independent_tasks()
    algorithm = build_algorithm(problem, population_size=2)
    algorithm._pending_counts = (7, 3, 2, 1)
    monkeypatch.setattr(NSGA2, "_infill", lambda self: None)
    assert algorithm._infill() is None
    assert all(value == 0 for value in algorithm._pending_counts)


class AllAssignedSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.ones((n_samples, problem.n_var), dtype=int)


def test_record_distinguishes_all_conflict_pairs_from_removed_tasks():
    problem = SatelliteSchedulingProblem.from_data([
        {"id": i, "duration": 5, "turnaround": 0, "chanceList": [[0, 0, 0], [1, 0, 5]]}
        for i in range(3)
    ], [[1, "A"]])
    algorithm = build_algorithm(problem, population_size=3)
    algorithm.initialization.sampling = AllAssignedSampling()
    minimize(problem, algorithm, ("n_gen", 1), seed=1001, copy_algorithm=False)
    assert algorithm.record[0]["pre_repair_conflicts"] == 3
    assert algorithm.record[0]["repair_count"] == 3
    assert algorithm.record[0]["removed_tasks"] == 2
