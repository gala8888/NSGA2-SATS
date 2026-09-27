"""Paper-aligned biobjective satellite range scheduling model.

Objective evaluation never repairs or mutates its input. Repair is a separate,
explicit operation. The deterministic repair priority (more assignment options
first, stable task-index ties) is inherited from the identified implementation;
Turnaround is the setup time required
by the following task, matching the original AFIT reader/feasibility convention.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import numpy as np
from pymoo.core.problem import Problem

from .data_io import read


class SatelliteSchedulingProblem(Problem):
    def __init__(self, scenario_id=1, data_dir=None, profits=None):
        tasks, antennas = read(scenario_id, data_dir)
        self._initialize(tasks, antennas, profits, f"AFIT{scenario_id}")
        self.scenario_id = int(scenario_id)

    @classmethod
    def from_data(cls, tasks, antennas, profits=None):
        """Construct from legacy task dictionaries and antenna lists.

        Antennas may contain just [id, name], or the old four fields
        [id, name, supported_task_ids, Mj]. Supplied Mj must equal the sum of
        durations of distinct visible tasks. Profits are an aligned sequence
        or an id-keyed mapping; omitted profits explicitly mean p_i = 1.
        """
        self = cls.__new__(cls)
        self._initialize(tasks, antennas, profits, "custom")
        self.scenario_id = None
        return self

    def _initialize(self, tasks, antennas, profits, name):
        self.tasks, self.antennas = deepcopy(list(tasks)), deepcopy(list(antennas))
        self.n_tasks, self.n_antennas = len(self.tasks), len(self.antennas)
        if not self.n_tasks or not self.n_antennas:
            raise ValueError("at least one task and one antenna are required")
        self.antenna_ids = [int(ant[0]) for ant in self.antennas]
        if len(set(self.antenna_ids)) != self.n_antennas or any(i < 1 for i in self.antenna_ids):
            raise ValueError("antenna ids must be unique positive integers")
        self.antenna_index = {ant_id: idx for idx, ant_id in enumerate(self.antenna_ids)}
        task_ids = [task["id"] for task in self.tasks]
        if len(set(task_ids)) != self.n_tasks:
            raise ValueError("task ids must be unique")
        capacity = np.zeros(self.n_antennas)
        supported = [[] for _ in self.antennas]
        windows, offsets = [], []
        for task in self.tasks:
            duration, turnaround = float(task["duration"]), float(task["turnaround"])
            if not np.isfinite(duration) or duration < 0 or not np.isfinite(turnaround) or turnaround < 0:
                raise ValueError("task duration and turnaround must be nonnegative")
            task["duration"], task["turnaround"] = duration, turnaround
            options = task["chanceList"]
            if not options or len(options[0]) < 3 or options[0][0] != 0:
                raise ValueError("chanceList must begin with the zero (unscheduled) option")
            visible = []
            normalized = [[0, 0.0, 0.0]]
            for option in options[1:]:
                if len(option) != 3:
                    raise ValueError("assignment options must contain antenna id, start and end")
                ant_id, start, end = option
                if ant_id not in self.antenna_index or not np.isfinite(start) or not np.isfinite(end):
                    raise ValueError("assignment option has an unknown antenna or nonfinite time")
                if start < 0 or end < start or not np.isclose(end - start, duration, rtol=1e-9, atol=1e-9):
                    raise ValueError("each assignment option must have exactly the task duration")
                normalized.append([int(ant_id), float(start), float(end)])
                if ant_id not in visible:
                    visible.append(int(ant_id))
            if "antennaList" in task and set(task["antennaList"]) != set(visible):
                raise ValueError("antennaList must match the feasible assignment options")
            task["antennaList"] = visible
            task["chanceList"] = normalized
            if len(normalized) > 1:
                earliest = min(option[1] for option in normalized[1:])
                latest = max(option[2] for option in normalized[1:])
                if "begin" in task and earliest < float(task["begin"]) - 1e-9:
                    raise ValueError("assignment precedes the task's earliest visibility bound")
                if "end" in task and latest > float(task["end"]) + 1e-9:
                    raise ValueError("assignment exceeds the task's latest visibility bound")
                task.setdefault("begin", earliest)
                task.setdefault("end", latest)
            for ant_id in visible:
                ant_idx = self.antenna_index[ant_id]
                capacity[ant_idx] += duration
                supported[ant_idx].append(task["id"])
            offsets.append(len(windows))
            windows.extend([[self.antenna_index[ant], start, end] if ant else [-1, 0., 0.]
                            for ant, start, end in normalized])
        for idx, ant in enumerate(self.antennas):
            if len(ant) >= 4 and (not np.isfinite(ant[3]) or not np.isclose(ant[3], capacity[idx])):
                raise ValueError("antenna Mj must equal the duration sum of distinct visible tasks")
            self.antennas[idx] = [self.antenna_ids[idx], ant[1], supported[idx], float(capacity[idx])]
        self.capacities = capacity
        self.durations = np.array([task["duration"] for task in self.tasks])
        self.turnarounds = np.array([task["turnaround"] for task in self.tasks])
        self._window_table = np.asarray(windows, dtype=float)
        self._window_offsets = np.asarray(offsets, dtype=int)
        self._upper_bounds = np.array([len(task["chanceList"]) - 1 for task in self.tasks], dtype=int)
        self._repair_priority = sorted(range(self.n_tasks), key=lambda i: -self._upper_bounds[i])
        if profits is None:
            self.profits = np.ones(self.n_tasks)
        elif isinstance(profits, Mapping):
            self.profits = np.array([profits[task_id] for task_id in task_ids], dtype=float)
        else:
            self.profits = np.array(profits, dtype=float, copy=True)
        if self.profits.shape != (self.n_tasks,) or not np.isfinite(self.profits).all() or np.any(self.profits < 0) or self.profits.sum() <= 0:
            raise ValueError("profits must be finite nonnegative values with a positive total")
        self.problem_name, self.objective_calls = name, 0
        super().__init__(n_var=self.n_tasks, n_obj=2, n_ieq_constr=0,
                         xl=np.zeros(self.n_tasks, dtype=int), xu=self._upper_bounds.copy(),
                         vtype=int, elementwise=False)

    def _solution(self, x, *, repair=False):
        values = np.asarray(x, dtype=float)
        if values.shape != (self.n_tasks,):
            raise ValueError(f"expected a solution of shape ({self.n_tasks},)")
        if repair:
            values = np.where(np.isfinite(values), np.rint(values), 0)
            values = np.where((values >= 0) & (values <= self._upper_bounds), values, 0)
        elif (not np.isfinite(values).all() or np.any(values != np.rint(values))
              or np.any(values < 0) or np.any(values > self._upper_bounds)):
            raise ValueError("solution must contain valid bounded integer option indices")
        return values.astype(int, copy=True)

    def _selected(self, x):
        return self._window_table[self._window_offsets + x]

    def conflict_pairs(self, x):
        """Count every conflicting task pair before repair, not removed genes."""
        solution = self._solution(x)
        selected = self._selected(solution)
        count = 0
        for ant_idx in range(self.n_antennas):
            indices = np.flatnonzero((solution > 0) & (selected[:, 0] == ant_idx))
            starts = selected[indices, 1] - self.turnarounds[indices]
            ends = selected[indices, 2]
            overlaps = (starts[:, None] < ends[None, :]) & (starts[None, :] < ends[:, None])
            count += int(np.count_nonzero(np.triu(overlaps, k=1)))
        return count

    def is_feasible(self, x):
        try:
            return self.conflict_pairs(x) == 0
        except (ValueError, TypeError, OverflowError):
            return False

    def can_place(self, x, task_idx, window_idx):
        """Test a replacement against other tasks; the task never conflicts with itself.

        Existing conflicts between other tasks do not affect this local check.
        This lets LCAM act on raw offspring before the explicit repair stage.
        """
        solution = self._solution(x)
        if not isinstance(task_idx, (int, np.integer)) or not 0 <= task_idx < self.n_tasks:
            raise IndexError("task index out of range")
        if not isinstance(window_idx, (int, np.integer)) or not 0 <= window_idx <= self._upper_bounds[task_idx]:
            return False
        if window_idx == 0:
            return True
        ant_idx, start, end = self._window_table[self._window_offsets[task_idx] + window_idx]
        selected = self._selected(solution)
        other = (solution > 0) & (selected[:, 0] == ant_idx)
        other[task_idx] = False
        return not bool(np.any((start - self.turnarounds[task_idx] < selected[other, 2]) &
                               (selected[other, 1] - self.turnarounds[other] < end)))

    def repair_solution(self, x):
        """Return a feasible integer copy, preserving a stable legacy priority."""
        solution = self._solution(x, repair=True)
        selected = self._selected(solution)
        kept = [[] for _ in range(self.n_antennas)]
        for task_idx in self._repair_priority:
            if not solution[task_idx]:
                continue
            ant_idx, start, end = selected[task_idx]
            ant_idx = int(ant_idx)
            peers = np.asarray(kept[ant_idx], dtype=int)
            collision = bool(np.any((start - self.turnarounds[task_idx] < selected[peers, 2]) &
                                    (selected[peers, 1] - self.turnarounds[peers] < end)))
            if collision:
                solution[task_idx] = 0
            else:
                kept[ant_idx].append(task_idx)
        return solution

    def loads(self, x):
        """Return duration sums per antenna for any bounded integer candidate."""
        solution = self._solution(x)
        selected = self._selected(solution)
        scheduled = solution > 0
        return np.bincount(selected[scheduled, 0].astype(int),
                           weights=self.durations[scheduled], minlength=self.n_antennas)

    def objectives(self, x):
        """Evaluate feasible x: profit loss and sample-CV (ddof=1).

        The no-work and one-antenna cases use imbalance 0 as explicit conventions
        for the otherwise undefined coefficient of variation.
        """
        solution = self._solution(x)
        if not self.is_feasible(solution):
            raise ValueError("objectives requires a feasible solution; call repair_solution first")
        load = self.loads(solution)
        mean = float(load.mean())
        f1 = 1.0 - float(self.profits[solution > 0].sum() / self.profits.sum())
        f2 = float(np.std(load, ddof=1) / mean) if mean > 0 and self.n_antennas > 1 else 0.0
        self.objective_calls += 1
        return np.array([f1, f2], dtype=float)

    def _evaluate(self, X, out, *args, **kwargs):
        # Defensive public pymoo interface: F refers to a repaired copy. The
        # algorithm explicitly installs that copy in the population before eval.
        rows = np.atleast_2d(X)
        out["F"] = np.asarray([self.objectives(self.repair_solution(x)) for x in rows])

    def get_problem_name(self):
        return self.problem_name
