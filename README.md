# NSGAII-SATS

Python implementation accompanying **A Physics-informed Memetic Framework for Multi-objective Satellite Range Scheduling**. It combines NSGA-II, performance-driven adaptive crossover (PDAC), load-capacity-aware mutation (LCAM), and simulated annealing tabu search (SATS).

## Quick start

Use Python 3.11:

```bash
python -m pip install -r requirements.txt
python run.py --profile demo --output results/demo
```

The demo uses AFIT1 with 10 individuals, 3 generations, and one run.

## Main experiment

```bash
python run.py --dry-run
python run.py --output results/main
```

The default configuration uses AFIT1–AFIT7, **100 individuals, 500 generations, and 30 runs per instance**. Seeds follow `1000 * scenario + run`, with runs numbered 1–30. Use a new output directory for each experiment.

| Parameter | Value |
|---|---:|
| Crossover / mutation / local-search probability | 0.6 / 0.3 / 0.3 |
| Local-search iterations / tabu length | 23 / 20 |
| SA initial temperature / cooling rate | 100 / 0.9 |
| Strong / weak crossover credit | 0.5 / 0.3 |
| Load / capacity weights | 0.5 / 0.5 |
| Task-transfer factor | 0.5 |
| HV reference point | (1.1, 1.1) |

## Files and documentation

- `nsga2_sats/`: algorithm and experiment tools.
- `configs/`: experiment configurations.
- `benchmarks/afit/`: 14 original AFIT input files; see [data sources and format](docs/DATA.md).
- `historical/`: archived experiment results; see [contents](historical/README.md).
- `tests/`: automated tests.

See [experiment commands](docs/EXPERIMENTS.md) and [implementation details](docs/PAPER_ALIGNMENT.md).

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
