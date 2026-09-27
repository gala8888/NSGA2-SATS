# Experiments

Run commands from the repository root. `--dry-run` prints the experiment plan. Choose a new output directory for each run.

## Demo and main experiment

```bash
python run.py --profile demo --output results/demo
python run.py --profile main --dry-run
python run.py --profile main --output results/main
```

| Profile | Instances | Population | Generations | Runs per instance |
|---|---|---:|---:|---:|
| demo | AFIT1 | 10 | 3 | 1 |
| main | AFIT1–AFIT7 | 100 | 500 | 30 |
| ablation | AFIT1–AFIT7, five variants | 100 | 200 | 30 |
| large | Three external instances, full/base variants | 100 | 200 | 30 |

Both `run.py` and `run_srsp.py` default to `main`. Seeds follow `1000*k+r`, with one-based run numbers. For a shorter run:

```bash
python run.py --profile main --scenario 1 --population 10 --generations 3 --runs 1 --output results/check
```

## Ablation

```bash
python run.py --profile ablation --output results/ablation
```

Variants are full NSGAII-SATS, no PDAC, no LCAM, no SATS, and base NSGA-II. The replacement operators are SBX crossover and polynomial mutation; the no-SATS variant disables local search.

## Optional large-scale experiments

Generate new synthetic inputs outside the repository, then run the large profile:

```bash
python -m nsga2_sats.generate_data --output ../nsga2-sats-large-inputs --seed 20260918
python run.py --profile large --data-dir ../nsga2-sats-large-inputs --output results/generated-large
```

The required task/antenna counts are 500/24, 750/30, and 1000/38. See [DATA.md](DATA.md).

## Sensitivity and timing

```bash
python run.py --profile sensitivity --sensitivity-set primary --dry-run
python run.py --profile sensitivity --sensitivity-set transfer --dry-run
python run.py --profile sensitivity --sensitivity-set local --dry-run
python run.py --profile cpu --hv-target 0.945 --cpu-budget 3600 --dry-run
```

Sensitivity runs use AFIT1, population 100, 500 generations, and 30 runs per setting. `primary` selects nine orthogonal combinations; `transfer` and `local` vary their respective parameters; `all` selects all supported sweeps.

The timing command above uses an example HV target of 0.945 and a 3600-second process CPU budget. Supply the target and budget for your experiment.

## Analysis

```bash
python run.py analyze --input historical/main --output results/history-analysis
python run.py analyze --input results/main --output results/main-analysis
```

Analysis reads all supported Pareto files under the selected input directory. Compare runs using the same input data and experimental settings. HV uses reference `(1.1,1.1)`; IGD uses the nondominated union of the supplied fronts for each scenario. Statistical output includes two-sided Wilcoxon comparisons of shared scenario means and Holm-adjusted p-values.

New run directories contain schedules, objective values, parameters, seeds, environment details, and input/source hashes. Archived results are listed in [historical/README.md](../historical/README.md).
