"""Paper parameters and explicitly documented implementation choices."""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class AlgorithmConfig:
    # Final manuscript settings and Supplement Appendix E.
    crossover_probability: float = 0.6
    mutation_probability: float = 0.3
    local_search_probability: float = 0.3
    local_search_iterations: int = 23
    tabu_length: int = 20
    initial_temperature: float = 100.0
    cooling_rate: float = 0.9
    strong_reward: float = 0.5
    weak_reward: float = 0.3
    load_weight: float = 0.5
    capacity_weight: float = 0.5
    transfer_factor: float = 0.5
    # Additional implementation settings.
    score_reset_interval: int = 50
    softmax_temperature: float = 1.0
    uniform_swap_probability: float = 0.5
    insertion_count: int = 1
    sbx_eta: float = 15.0
    polynomial_eta: float = 20.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        for key in ['crossover_probability', 'mutation_probability', 'local_search_probability',
                    'uniform_swap_probability', 'transfer_factor']:
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must lie in [0, 1]')
        for key in ['local_search_iterations', 'tabu_length', 'score_reset_interval', 'insertion_count']:
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ['initial_temperature', 'softmax_temperature', 'sbx_eta', 'polynomial_eta']:
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if not 0 < self.cooling_rate <= 1:
            raise ValueError('cooling_rate must lie in (0, 1]')
        if min(self.strong_reward, self.weak_reward, self.load_weight, self.capacity_weight) < 0:
            raise ValueError('rewards and weights must be nonnegative')
        if not self.strong_reward > self.weak_reward > 0:
            raise ValueError('crossover rewards must satisfy strong_reward > weak_reward > 0')
        if not math.isclose(self.load_weight + self.capacity_weight, 1.0):
            raise ValueError('load_weight and capacity_weight must sum to 1')
