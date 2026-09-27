"""NSGA-II with PDAC, LCAM, and the Appendix C SATS state transitions.

Objective evaluations used by search/credit are pure and counted separately
from pymoo's population evaluator. See docs for implementation settings.
"""
from collections import deque
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from .config import AlgorithmConfig
from .operators import hierarchical_insertion, load_balance


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


class IntegerLatinHypercube(Sampling):
    """Map unit strata to inclusive integer bounds exactly once."""
    def __init__(self, rng):
        super().__init__()
        self.rng = rng

    def _do(self, problem, n_samples, **kwargs):
        result = np.empty((n_samples, problem.n_var), dtype=int)
        for j in range(problem.n_var):
            u = (self.rng.permutation(n_samples) + self.rng.random(n_samples)) / n_samples
            result[:, j] = np.floor(problem.xl[j] + u * (problem.xu[j] - problem.xl[j] + 1)).astype(int)
        return result


class AdaptiveCrossover(Crossover):
    """Cross-generation softmax credits; only one Pc gate per mating."""
    names = ('two_point', 'multi_point', 'structure_aware')

    def __init__(self, config, rng):
        super().__init__(2, 2, prob=1.0)
        self.config, self.rng = config, rng
        self.scores = np.ones(3, dtype=float)
        self._last_reset = None

    def reset_for_generation(self, generation):
        if generation > 0 and generation % self.config.score_reset_interval == 0 and generation != self._last_reset:
            self.scores[:] = 1.0
            self._last_reset = generation

    def probabilities(self):
        z = (self.scores - self.scores.max()) / self.config.softmax_temperature
        p = np.exp(z)
        return p / p.sum()

    def cross_pair(self, a, b, operator):
        c, d = np.array(a, dtype=int, copy=True), np.array(b, dtype=int, copy=True)
        n = len(c)
        if operator == 2:
            for j in range(n):
                if (a[j] == 0) != (b[j] == 0):
                    c[j] = d[j] = max(a[j], b[j])
                elif self.rng.random() < self.config.uniform_swap_probability:
                    c[j], d[j] = b[j], a[j]
        elif n >= 2:
            # Boundaries are sampled without replacement; alternate inherited segments.
            count = min(2 if operator == 0 else 4, n - 1)
            points = sorted(self.rng.choice(np.arange(1, n), count, replace=False))
            edges = [0] + points + [n]
            for k in range(1, len(edges) - 1, 2):
                lo, hi = edges[k], edges[k + 1]
                c[lo:hi], d[lo:hi] = b[lo:hi], a[lo:hi]
        return c, d

    def _do(self, problem, X, **kwargs):
        algorithm = kwargs.get('algorithm')
        generation = max(0, int(getattr(algorithm, 'n_gen', 1) or 1) - 1)
        self.reset_for_generation(generation)
        result = np.asarray(X, dtype=int).copy()
        for mating in range(X.shape[1]):
            if self.rng.random() >= self.config.crossover_probability:
                continue
            operator = int(self.rng.choice(3, p=self.probabilities()))
            parents = X[:, mating, :]
            children = self.cross_pair(parents[0], parents[1], operator)
            parent_f = [problem.objectives(problem.repair_solution(p)) for p in parents]
            child_f = [problem.objectives(problem.repair_solution(c)) for c in children]
            # Credit each child against both parents, without mutating either genotype.
            for f in child_f:
                count = sum(dominates(f, pf) for pf in parent_f)
                if count == 2:
                    self.scores[operator] += self.config.strong_reward
                elif count == 1:
                    self.scores[operator] += self.config.weak_reward
            result[:, mating, :] = children
        return result


class LoadCapacityMutation(Mutation):
    """Per-gene LCAM with current-solution load/capacity weights."""
    def __init__(self, config, rng):
        super().__init__(prob=1.0)
        self.config, self.rng = config, rng

    def _do(self, problem, X, **kwargs):
        result = np.asarray(X, dtype=int).copy()
        capacity = np.asarray([a[3] for a in problem.antennas], dtype=float)
        for x in result:
            for task_idx, task in enumerate(problem.tasks):
                if self.rng.random() >= self.config.mutation_probability:
                    continue
                loads = problem.loads(x)
                total_load = float(loads.sum())
                lf = 1.0 - loads / total_load if total_load > 0 else np.ones_like(loads)
                cf = np.divide(capacity - loads, capacity, out=np.zeros_like(loads), where=capacity > 0)
                weights = np.maximum(0, self.config.load_weight * lf + self.config.capacity_weight * cf)
                p = weights / weights.sum() if weights.sum() > 0 else np.full(problem.n_antennas, 1 / problem.n_antennas)
                target = int(problem.antennas[int(self.rng.choice(problem.n_antennas, p=p))][0])
                for win_idx, window in enumerate(task['chanceList'][1:], 1):
                    if int(window[0]) == target and problem.can_place(x, task_idx, win_idx):
                        x[task_idx] = win_idx
                        break
        return result


class SimulatedAnnealingTabuSearch:
    """Appendix C: accepted current, best, and aspiration are independent copies."""
    def __init__(self, config, rng, neighborhood=None):
        self.config, self.rng, self.neighborhood = config, rng, neighborhood
        self.last_trace = []

    @staticmethod
    def move_key(current, neighbor):
        # An undirected assignment change also marks its inverse tabu.
        return tuple((int(i), min(int(current[i]), int(neighbor[i])), max(int(current[i]), int(neighbor[i])))
                     for i in np.flatnonzero(current != neighbor))

    def _neighbor(self, x, problem):
        if self.neighborhood is not None:
            return self.neighborhood(x.copy(), problem, self.rng)
        if self.rng.random() < 0.5:
            return hierarchical_insertion(x, problem, self.rng, n_tasks=self.config.insertion_count)
        return load_balance(x, problem, self.rng, transfer_factor=self.config.transfer_factor)

    def search(self, x, problem):
        current = np.asarray(x, dtype=int).copy()
        current_f = problem.objectives(current)
        best, best_f = current.copy(), current_f.copy()
        aspiration, aspiration_f = current.copy(), current_f.copy()
        tabu = deque(maxlen=self.config.tabu_length)
        temperature = self.config.initial_temperature
        self.last_trace = []
        for iteration in range(self.config.local_search_iterations):
            neighbor = np.asarray(self._neighbor(current, problem), dtype=int).copy()
            if not problem.is_feasible(neighbor):
                raise ValueError('SATS neighborhood must return a feasible solution')
            move = self.move_key(current, neighbor)
            is_tabu, accepted, reason = move in tabu, False, 'no_change'
            if move:
                neighbor_f = problem.objectives(neighbor)
                if is_tabu:
                    accepted = dominates(neighbor_f, aspiration_f)
                    reason = 'aspiration' if accepted else 'tabu'
                else:
                    delta = float(np.mean(neighbor_f - current_f))
                    probability = float(np.exp(np.clip(-delta / max(temperature, np.finfo(float).tiny), -745, 0)))
                    improves = dominates(neighbor_f, current_f)
                    accepted = improves or self.rng.random() < probability
                    reason = 'improvement' if improves else ('annealing' if accepted else 'rejected')
                if accepted:
                    current, current_f = neighbor.copy(), neighbor_f.copy()
                    tabu.append(move)
                    if dominates(current_f, best_f):
                        best, best_f = current.copy(), current_f.copy()
                    if not is_tabu and dominates(current_f, aspiration_f):
                        aspiration, aspiration_f = current.copy(), current_f.copy()
            self.last_trace.append({'iteration': iteration + 1, 'accepted': accepted, 'tabu': is_tabu,
                                    'reason': reason, 'temperature': temperature})
            temperature *= self.config.cooling_rate
        self.last_current, self.last_best, self.last_aspiration = current.copy(), best.copy(), aspiration.copy()
        return best.copy()


class PaperNSGA2(NSGA2):
    def __init__(self, config, rng, population_size, variant):
        if variant not in {'full', 'no_pdac', 'no_lcam', 'no_sats', 'base'}:
            raise ValueError(f'unknown variant: {variant}')
        self.config, self.search_rng, self.variant = config, rng, variant
        crossover = (SBX(prob=config.crossover_probability, eta=config.sbx_eta, vtype=float, repair=RoundingRepair())
                     if variant in {'no_pdac', 'base'} else AdaptiveCrossover(config, rng))
        mutation = (PM(prob=1.0, prob_var=config.mutation_probability, eta=config.polynomial_eta,
                       vtype=float, repair=RoundingRepair())
                    if variant in {'no_lcam', 'base'} else LoadCapacityMutation(config, rng))
        self.adaptive_crossover = crossover if isinstance(crossover, AdaptiveCrossover) else None
        self.local_search = SimulatedAnnealingTabuSearch(config, rng)
        self.record = []
        self._pending_counts = (0, 0, 0, 0)
        super().__init__(pop_size=population_size, sampling=IntegerLatinHypercube(rng),
                         crossover=crossover, mutation=mutation, eliminate_duplicates=True)

    def _prepare(self, population, use_local_search):
        conflicts, removed, searched = 0, 0, 0
        for individual in population:
            raw = np.asarray(individual.X, dtype=int).copy()
            conflicts += self.problem.conflict_pairs(raw)
            x = self.problem.repair_solution(raw)
            removed += int(np.count_nonzero((raw > 0) & (x == 0)))
            if use_local_search and self.variant not in {'no_sats', 'base'} and self.search_rng.random() < self.config.local_search_probability:
                x = self.local_search.search(x, self.problem)
                searched += 1
            individual.X = x.copy()
        self._pending_counts = (int(conflicts), int(removed), searched, len(population))
        return population

    def _initialize_infill(self):
        return self._prepare(super()._initialize_infill(), use_local_search=False)

    def _infill(self):
        offspring = super()._infill()
        if offspring is None:
            self._pending_counts = (0, 0, 0, 0)
            return None
        return self._prepare(offspring, use_local_search=True)

    def _post_advance(self):
        conflicts, removed, searched, candidate_count = self._pending_counts
        self.record.append({'generation': int(self.n_gen), 'phase': 'initialization' if self.n_gen == 1 else 'offspring',
                            'evaluations': int(self.evaluator.n_eval), 'objective_calls': int(self.problem.objective_calls),
                            'pre_repair_conflicts': conflicts, 'repair_count': conflicts, 'removed_tasks': removed,
                            'candidate_count': candidate_count, 'mean_repair_count': conflicts / candidate_count if candidate_count else 0.0,
                            'local_search_calls': searched,
                            'operator_scores': [] if self.adaptive_crossover is None else self.adaptive_crossover.scores.tolist()})
        super()._post_advance()


def build_algorithm(problem, config=None, population_size=100, seed=1001, variant='full'):
    """Construct an uninitialized algorithm; pymoo minimize() performs setup."""
    if not isinstance(population_size, int) or population_size < 2:
        raise ValueError('population_size must be an integer >= 2')
    return PaperNSGA2(config or AlgorithmConfig(), np.random.default_rng(seed), population_size, variant)
