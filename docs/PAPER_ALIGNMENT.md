# Implementation details

Paths below are relative to `nsga2_sats/`.

| Component | Implementation |
|---|---|
| Objectives and scheduling constraints | `problem.py::SatelliteSchedulingProblem` |
| AFIT reader and integer representation | `data_io.py::read` |
| Latin hypercube initialization | `algorithm.py::IntegerLatinHypercube` |
| PDAC | `algorithm.py::AdaptiveCrossover` |
| LCAM | `algorithm.py::LoadCapacityMutation` |
| Hierarchical insertion | `operators.py::hierarchical_insertion` |
| Load balancing | `operators.py::load_balance` |
| SATS | `algorithm.py::SimulatedAnnealingTabuSearch` |
| NSGA-II evolution | `algorithm.py::PaperNSGA2` |

## Objectives and representation

Profit loss is `1 - scheduled_profit / total_profit`. Antenna workload is the sum of assigned task durations. Load imbalance is the sample standard deviation of antenna workloads divided by their mean; the zero-load and single-antenna cases return zero.

Each task has one integer gene: zero means unscheduled; a positive value selects an antenna/window option. Feasibility checks enforce task windows, antenna conflicts, and turnaround times.

## Settings

The main algorithm parameters are listed in the [README](../README.md) and experiment profiles in `configs/`. Additional settings are:

| Setting | Value |
|---|---|
| PDAC reset interval / softmax temperature | 50 / 1 |
| Uniform exchange probability | 0.5 |
| Multi-point crossover cuts | Four, limited by chromosome length |
| Hierarchical insertion | One attempted task; low-load antennas first |
| Load-balancing target | Least-loaded antenna |
| Migration count | `min(N_T, max(1, floor(lambda * rate_m * N_T)))` |
| SA objective difference | Mean of `F_new - F_current` |
| Tabu key | Task and unordered pair of old/new assignments |
| Neighborhood selection | Equal probability for insertion and balancing |
| Default task profits / high-orbit time grid | 1 / one time unit |
| Conflict repair order | Descending available-window count, stable task-index ties |
| Ablation SBX / polynomial mutation eta | 15 / 20 |

Repair diagnostics record conflicting task pairs before repair and the number of removed assignments separately. See [experiment commands](EXPERIMENTS.md) for configuration and outputs.
