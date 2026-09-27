"""Feasible, independent-copy neighborhoods from manuscript Algorithms 3 and 4."""
from __future__ import annotations

import math
import numpy as np


def hierarchical_insertion(x, problem, rng, n_tasks=1):
    """Try scarce-antenna, short-task or scarce-window insertion equally often.

    Every call chooses one ordering with probability 1/3. For each selected
    unscheduled task, windows on below-mean-load antennas are tried before
    all remaining antennas. Windows and equal-priority tasks retain data order.
    """
    if isinstance(n_tasks, bool) or int(n_tasks) != n_tasks or n_tasks < 0:
        raise ValueError("n_tasks must be a nonnegative integer")
    result = problem.repair_solution(x)
    unscheduled = list(np.flatnonzero(result == 0))
    if not unscheduled or n_tasks == 0:
        return result
    ordering = int(rng.integers(3))
    keys = (
        lambda i: len(problem.tasks[i]["antennaList"]),
        lambda i: problem.tasks[i]["duration"],
        lambda i: len(problem.tasks[i]["chanceList"]) - 1,
    )
    selected_tasks = sorted(unscheduled, key=keys[ordering])[:int(n_tasks)]
    for task_idx in selected_tasks:
        load = problem.loads(result)
        low_load = {problem.antenna_ids[i] for i in np.flatnonzero(load < load.mean())}
        options = problem.tasks[task_idx]["chanceList"]
        placed = False
        for low_stage in (True, False):
            for window_idx, (ant_id, _, _) in enumerate(options[1:], 1):
                if (ant_id in low_load) != low_stage:
                    continue
                if problem.can_place(result, task_idx, window_idx):
                    result[task_idx] = window_idx
                    placed = True
                    break
            if placed:
                break
    return result


def load_balance(x, problem, rng, transfer_factor=.5):
    """Randomly move tasks from the busiest antenna to a fixed least-loaded one.

    This operator fixes the target to the initially least-loaded antenna (stable input-order
    ties), floors lambda * rate_m * N_T, and clamps the migration count to
    [1, N_T]. A task keeps its assignment when the target has no feasible option.
    Balanced and zero-work solutions return independent unchanged copies.
    """
    if not math.isfinite(transfer_factor) or transfer_factor < 0:
        raise ValueError("transfer_factor must be finite and nonnegative")
    result = problem.repair_solution(x)
    load = problem.loads(result)
    total = float(load.sum())
    if total == 0 or np.allclose(load, load[0], rtol=1e-12, atol=1e-12):
        return result
    freedom = 1.0 - load / total
    busiest = int(np.argmin(freedom))
    target = int(np.argmax(freedom))
    source_id, target_id = problem.antenna_ids[busiest], problem.antenna_ids[target]
    source_tasks = [i for i, option in enumerate(result)
                    if option > 0 and problem.tasks[i]["chanceList"][option][0] == source_id]
    if not source_tasks:
        return result
    ratio = float((freedom.mean() - freedom.min()) / (freedom.max() - freedom.min()))
    count = min(len(source_tasks), max(1, math.floor(transfer_factor * ratio * len(source_tasks))))
    selected_tasks = rng.choice(np.asarray(source_tasks, dtype=int), size=count, replace=False)
    for task_idx in np.atleast_1d(selected_tasks):
        for window_idx, (ant_id, _, _) in enumerate(problem.tasks[task_idx]["chanceList"][1:], 1):
            if ant_id == target_id and problem.can_place(result, task_idx, window_idx):
                result[task_idx] = window_idx
                break
    return result
