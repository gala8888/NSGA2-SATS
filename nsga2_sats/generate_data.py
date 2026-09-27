"""Generate new AFIT-shaped instances at the supplementary paper's scales.

Each dataset is generated from the AFIT1 template with recorded settings.
Uses Python's dedicated Random instance; importing this module has no side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_SCALES = ((500, 24), (750, 30), (1000, 38))
GENERATOR_VERSION = "afit-shaped-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_groups(path: Path) -> list[list[tuple[str, float, float, float, float]]]:
    groups: dict[int, list[tuple[str, float, float, float, float]]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 6:
            raise ValueError(f"{path.name}:{line_number}: expected six data fields")
        request_id = int(fields[0])
        antenna = fields[1]
        start, end, duration, turnaround = map(float, fields[2:6])
        if start < 0 or duration <= 0 or turnaround < 0 or end < start + duration:
            raise ValueError(f"{path.name}:{line_number}: invalid time window")
        groups.setdefault(request_id, []).append((antenna, start, end, duration, turnaround))
    if not groups:
        raise ValueError(f"{path.name}: no requests")
    return list(groups.values())


def generate_dataset(output_dir: str | Path, seed: int = 20260918,
                     template_dir: str | Path | None = None,
                     scales: tuple[tuple[int, int], ...] = DEFAULT_SCALES) -> dict:
    """Write reqlf19/requp19 etc. and a deterministic provenance manifest.

    Low/high proportions are rounded from AFIT1. Each new task samples a request
    with replacement from the matching orbit, preserves its durations and window
    offsets, and receives distinct synthetic antennas. Its shared temporal shift
    is drawn in [-120,120] and clipped to the template scheduling horizon. The
    first tasks cover every requested antenna. Files are never overwritten.
    """
    target = Path(output_dir)
    template = Path(template_dir) if template_dir else Path(__file__).resolve().parents[1] / "benchmarks" / "afit"
    scales = tuple((int(tasks), int(antennas)) for tasks, antennas in scales)
    if not scales or len(scales) > 3:
        raise ValueError("provide one to three scales")
    if any(tasks < antennas or antennas < 1 for tasks, antennas in scales):
        raise ValueError("task count must be at least the positive antenna count")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("output directory must be empty; choose a new directory")
    templates = {orbit: _read_groups(template / filename) for orbit, filename in
                 (("low", "reqlf12.dat"), ("high", "requp12.dat"))}
    all_groups = templates["low"] + templates["high"]
    max_degree = max(len({row[0] for row in group}) for group in all_groups)
    if any(antennas < max_degree for _, antennas in scales):
        raise ValueError(f"at least {max_degree} antennas are needed to preserve template visibility degree")
    horizon = max(row[2] for group in all_groups for row in group)
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR_VERSION,
        "data_status": "generated-paper-scale",
        "seed": int(seed),
        "template": "AFIT1",
        "template_files": [{"name": name, "sha256": _sha256(template / name)}
                           for name in ("reqlf12.dat", "requp12.dat")],
        "method": {
            "orbit_ratio": "AFIT1 low/high request-count proportions, nearest integer",
            "requests": "uniform sampling with replacement within each orbit",
            "visibility": "preserve each sampled request's antenna degree and relative windows",
            "antennas": "synthetic labels; distinct mapping within each request; first tasks ensure coverage",
            "time_shift": "uniform integer in [-120,120], clipped to fit the template horizon",
            "horizon": horizon,
            "profit": "equal profits; no profits generated from the ignored seventh source column"
        },
        "target_dimensions": "Task and antenna counts from Supplement Appendix J",
        "instances": []
    }
    target.mkdir(parents=True, exist_ok=True)
    for offset, (task_count, antenna_count) in enumerate(scales):
        scenario_id = 8 + offset
        rng = random.Random(int(seed) + 1000 * scenario_id)
        antennas = [f"SYN-{number:03d}" for number in range(1, antenna_count + 1)]
        rng.shuffle(antennas)
        low_count = round(task_count * len(templates["low"]) / len(all_groups))
        counts = {"low": low_count, "high": task_count - low_count}
        seen_antennas: set[str] = set()
        task_number = 0
        files = []
        for orbit, prefix in (("low", "reqlf"), ("high", "requp")):
            lines = []
            for request_id in range(1, counts[orbit] + 1):
                group = rng.choice(templates[orbit])
                source_antennas = list(dict.fromkeys(row[0] for row in group))
                first = antennas[task_number % antenna_count]
                pool = [antenna for antenna in antennas if antenna != first]
                assigned = [first] + rng.sample(pool, len(source_antennas) - 1)
                mapping = dict(zip(source_antennas, assigned))
                shift = max(-min(row[1] for row in group),
                            min(rng.randint(-120, 120), horizon - max(row[2] for row in group)))
                for old_antenna, start, end, duration, turnaround in group:
                    antenna = mapping[old_antenna]
                    seen_antennas.add(antenna)
                    values = (start + shift, end + shift, duration, turnaround)
                    lines.append(f"{request_id} {antenna} " + " ".join(f"{value:g}" for value in values))
                task_number += 1
            filename = f"{prefix}{scenario_id + 11}.dat"
            path = target / filename
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            files.append({"name": filename, "sha256": _sha256(path), "bytes": path.stat().st_size,
                          "orbit": orbit, "tasks": counts[orbit]})
        if len(seen_antennas) != antenna_count:
            raise RuntimeError("generator failed to cover the requested antenna count")
        manifest["instances"].append({"scenario_id": scenario_id, "tasks": task_count,
                                      "antennas": antenna_count, "seed": int(seed) + 1000 * scenario_id,
                                      "files": files})
    (target / "generation-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--template-dir", type=Path)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, args.seed, args.template_dir)
    print(json.dumps({"output": str(args.output), "data_status": manifest["data_status"],
                      "instances": [{"scenario_id": x["scenario_id"], "tasks": x["tasks"],
                                     "antennas": x["antennas"]} for x in manifest["instances"]]}, indent=2))


if __name__ == "__main__":
    main()
