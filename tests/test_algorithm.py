from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from pymoo.optimize import minimize
from nsga2_sats.config import AlgorithmConfig
from nsga2_sats.algorithm import (IntegerLatinHypercube, AdaptiveCrossover, LoadCapacityMutation,
                                SimulatedAnnealingTabuSearch, build_algorithm)
from nsga2_sats.problem import SatelliteSchedulingProblem


def test_lhs_inclusive_integer_strata_and_nonzero_lower_bound():
    problem = SimpleNamespace(n_var=2, xl=np.array([0, 5]), xu=np.array([9, 14]))
    X = IntegerLatinHypercube(np.random.default_rng(7))._do(problem, 200)
    assert np.issubdtype(X.dtype, np.integer)
    assert np.array_equal(np.bincount(X[:, 0]), np.full(10, 20))
    assert np.array_equal(np.bincount(X[:, 1] - 5), np.full(10, 20))


def test_crossover_scores_persist_and_reset_only_once_per_generation():
    c = AdaptiveCrossover(AlgorithmConfig(), np.random.default_rng(8))
    c.scores[:] = [2, 3, 4]
    c.reset_for_generation(49)
    assert c.scores.tolist() == [2, 3, 4]
    c.reset_for_generation(50)
    assert c.scores.tolist() == [1, 1, 1]
    c.scores[0] += .5
    c.reset_for_generation(50)
    assert c.scores.tolist() == [1.5, 1, 1]
    c.scores[:] = [1e10, 0, 1]
    assert np.isfinite(c.probabilities()).all()
    assert c.probabilities().sum() == pytest.approx(1)


def test_structure_aware_inherits_scheduled_gene_and_copies_inputs():
    c = AdaptiveCrossover(AlgorithmConfig(), np.random.default_rng(8))
    a, b = np.array([0, 2, 1, 0]), np.array([3, 0, 4, 0])
    child1, child2 = c.cross_pair(a, b, 2)
    assert child1[:2].tolist() == child2[:2].tolist() == [3, 2]
    assert {child1[2], child2[2]} == {1, 4}
    child1[0] = 99
    assert a.tolist() == [0, 2, 1, 0] and b.tolist() == [3, 0, 4, 0]


def two_conflicting_tasks():
    return SatelliteSchedulingProblem.from_data([
        {'id': i, 'duration': 10, 'turnaround': 0, 'chanceList': [[0, 0, 0], [1, 0, 10]]}
        for i in range(2)], [[1, 'a']])


def test_lcam_never_inserts_conflicting_task_or_changes_source():
    p = two_conflicting_tasks()
    X = np.array([[1, 0]])
    m = LoadCapacityMutation(replace(AlgorithmConfig(), mutation_probability=1), np.random.default_rng(2))
    actual = m._do(p, X)
    assert np.array_equal(actual, X) and p.is_feasible(actual[0])
    actual[0, 0] = 0
    assert X.tolist() == [[1, 0]]


def test_lcam_uses_total_load_and_draws_from_all_antennas():
    class CapturingRng:
        def random(self): return 0.0
        def choice(self, n, p): self.p = np.array(p); return n - 1
    p = SatelliteSchedulingProblem.from_data([
        {'id': 0, 'duration': 10, 'turnaround': 0, 'chanceList': [[0, 0, 0], [1, 0, 10]]},
        {'id': 1, 'duration': 30, 'turnaround': 0, 'chanceList': [[0, 0, 0], [2, 0, 30]]}
    ], [[1, 'a'], [2, 'b'], [3, 'invisible']])
    rng = CapturingRng()
    m = LoadCapacityMutation(replace(AlgorithmConfig(), mutation_probability=1), rng)
    result = m._do(p, np.array([[1, 1]]))
    # LF=[.75,.25,1], CF=[0,0,0] (zero-capacity convention), w=.5*LF.
    assert rng.p == pytest.approx([.375, .125, .5])
    assert result.tolist() == [[1, 1]]


class ToyProblem:
    def objectives(self, x): return np.repeat(float(x[0]), 2)
    def is_feasible(self, x): return True


def test_sats_improving_chain_updates_current_and_best_independently():
    source = np.array([5])
    search = SimulatedAnnealingTabuSearch(replace(AlgorithmConfig(), local_search_iterations=2),
                                         np.random.default_rng(4), lambda x, p, r: x - 1)
    result = search.search(source, ToyProblem())
    assert result.tolist() == search.last_current.tolist() == [3]
    assert source.tolist() == [5]
    result[0] = 99
    assert search.last_best.tolist() == [3]


def test_sats_rejection_does_not_mutate_current():
    search = SimulatedAnnealingTabuSearch(replace(AlgorithmConfig(), local_search_iterations=2,
                                                 initial_temperature=1e-9),
                                         np.random.default_rng(4), lambda x, p, r: x + 1)
    assert search.search(np.array([5]), ToyProblem()).tolist() == [5]
    assert search.last_current.tolist() == [5]
    assert all(not row['accepted'] for row in search.last_trace)


def test_sats_inverse_tabu_blocks_annealing_and_cools_every_iteration():
    steps = iter([2, 3, 1])
    search = SimulatedAnnealingTabuSearch(replace(AlgorithmConfig(), local_search_iterations=3),
                                         np.random.default_rng(4), lambda x, p, r: np.array([next(steps)]))
    assert search.search(np.array([3]), ToyProblem()).tolist() == [1]
    assert search.last_trace[1]['reason'] == 'tabu'
    assert [r['temperature'] for r in search.last_trace] == pytest.approx([100, 90, 81])


@pytest.mark.parametrize('variant', ['full', 'no_pdac', 'no_lcam', 'no_sats', 'base'])
def test_seeded_end_to_end_solutions_and_objectives_are_consistent(variant):
    outputs = []
    for _ in range(2):
        p = SatelliteSchedulingProblem(1)
        config = replace(AlgorithmConfig(), local_search_iterations=2, local_search_probability=1)
        a = build_algorithm(p, config, population_size=6, seed=1001, variant=variant)
        assert a.problem is None
        minimize(p, a, ('n_gen', 3), seed=1001, copy_algorithm=False)
        X, F = a.pop.get('X'), a.pop.get('F')
        assert np.issubdtype(X.dtype, np.integer)
        assert all(p.is_feasible(x) for x in X)
        assert np.allclose(F, np.array([p.objectives(x) for x in X]))
        assert len(a.record) == 3
        assert a.record[0]['phase'] == 'initialization'
        assert a.record[-1]['objective_calls'] >= a.record[-1]['evaluations']
        if variant not in {'no_sats', 'base'}:
            assert a.record[1]['local_search_calls'] > 0
        outputs.append((X.copy(), F.copy(), a.record))
    assert np.array_equal(outputs[0][0], outputs[1][0])
    assert np.array_equal(outputs[0][1], outputs[1][1])
    assert outputs[0][2] == outputs[1][2]
