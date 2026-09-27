"""AFIT six-column reader and explicit finite-window encoding conventions.

Rows contain task id, antenna name, visibility start/end, duration and turnaround.
The official seventh column is unused orbit information, not task profit.
Extra columns are retained in the source files but are ignored by this model.
Task identities are grouped by (orbit, source id), so repeated visibility windows
remain options of one task. LEO uses each complete fixed window. GEO uses integer
start times on a one-time-unit grid, contained in the original visibility window.
The reader uses these discretization and equal-profit conventions.
"""
from __future__ import annotations

import math
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "afit"
GEO_TIME_STEP = 1.0


def _rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            location = f"{path}:{line_number}"
            if len(fields) < 6:
                raise ValueError(f"{location}: expected at least six data columns")
            try:
                source_id = int(fields[0])
                start, end, duration, turnaround = map(float, fields[2:6])
            except ValueError as error:
                raise ValueError(f"{location}: invalid task id or numeric field") from error
            if source_id < 1 or not all(map(math.isfinite, (start, end, duration, turnaround))):
                raise ValueError(f"{location}: ids must be positive and times finite")
            if start < 0 or end < start or duration < 0 or turnaround < 0:
                raise ValueError(f"{location}: invalid time bounds, duration or turnaround")
            if duration > end - start + 1e-9:
                raise ValueError(f"{location}: duration exceeds the visibility window")
            yield source_id, fields[1], start, end, duration, turnaround, location


def read(scenario_id=1, data_dir=None):
    """Read scenario k from reqlf(k+11).dat and requp(k+11).dat.

    AFIT1--7 therefore use file suffixes 12--18. Caller-supplied large datasets
    can use scenario ids 8--10 and suffixes 19--21.
    """
    if isinstance(scenario_id, bool) or int(scenario_id) != scenario_id or scenario_id < 1:
        raise ValueError("scenario_id must be a positive integer")
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    tasks, antennas = [], []
    antenna_ids, task_by_key = {}, {}
    for orbit, prefix in (("low", "reqlf"), ("high", "requp")):
        path = directory / f"{prefix}{int(scenario_id) + 11}.dat"
        for source_id, name, start, end, duration, turnaround, location in _rows(path):
            if orbit == "low" and not math.isclose(end - start, duration, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{location}: LEO requests must occupy the complete fixed window")
            if name not in antenna_ids:
                antenna_ids[name] = len(antennas) + 1
                antennas.append([antenna_ids[name], name, [], 0.0])
            antenna_id = antenna_ids[name]
            key = (orbit, source_id)
            if key not in task_by_key:
                task = {
                    "id": len(tasks) + 1, "source_id": source_id, "orbit": orbit,
                    "duration": duration, "turnaround": turnaround,
                    "begin": start, "end": end,
                    "chanceList": [[0, start, end]], "antennaList": [],
                }
                task_by_key[key] = task
                tasks.append(task)
            task = task_by_key[key]
            if not math.isclose(task["duration"], duration) or not math.isclose(task["turnaround"], turnaround):
                raise ValueError(f"{location}: inconsistent duration/turnaround for repeated task id")
            task["begin"] = min(task["begin"], start)
            task["end"] = max(task["end"], end)
            task["chanceList"][0] = [0, task["begin"], task["end"]]
            if orbit == "low":
                options = [[antenna_id, start, end]]
            else:
                # ceil/floor keep fractional input windows feasible; no int()
                # truncation that could place a task before its visibility start.
                first, last = math.ceil(start), math.floor(end - duration + 1e-9)
                options = [[antenna_id, float(t), t + duration] for t in range(first, last + 1)]
            task["chanceList"].extend(options)
            if options and antenna_id not in task["antennaList"]:
                task["antennaList"].append(antenna_id)
                ant = antennas[antenna_id - 1]
                ant[2].append(task["id"])
                ant[3] += duration
    if not tasks or not antennas:
        raise ValueError("the dataset must contain at least one task and one antenna")
    return tasks, antennas


read_data = read
