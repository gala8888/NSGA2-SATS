"""NSGA-II-SATS paper-aligned implementation; see docs/PAPER_ALIGNMENT.md."""
__version__ = '0.2.1'
from .config import AlgorithmConfig
from .problem import SatelliteSchedulingProblem
from .algorithm import build_algorithm

__all__ = ['AlgorithmConfig', 'SatelliteSchedulingProblem', 'build_algorithm']
