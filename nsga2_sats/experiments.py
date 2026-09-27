"""Portable paper-aligned experiment plans and auditable run artifacts.

Experiment parameters follow the manuscript and supplement; each run records its inputs and outputs.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import numpy as np
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from .config import AlgorithmConfig
from .algorithm import build_algorithm
from .problem import SatelliteSchedulingProblem
from .metrics import as_front, hypervolume, igd, spread, reference_front, summarize, paired_tests

ROOT = Path(__file__).resolve().parents[1]
VERSION = "paper-aligned-implementation"
VARIANTS = ("full", "no_pdac", "no_lcam", "no_sats", "base")
LARGE_SHAPES = {8: (500, 24), 9: (750, 30), 10: (1000, 38)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + chr(10), encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    sources = sorted((ROOT / "nsga2_sats").glob("*.py"))
    sources += sorted((ROOT / "configs").glob("*.json"))
    sources += [p for p in (ROOT / "run.py", ROOT / "run_srsp.py", ROOT / "requirements.txt") if p.exists()]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": hashes}


def load_profile(name):
    if name not in ("demo", "main", "ablation", "large", "cpu", "sensitivity"):
        raise ValueError(f"Unknown profile: {name}")
    path = ROOT / "configs" / (name + ".json")
    result = json.loads(path.read_text(encoding="utf-8"))
    result["profile"] = name
    return result


def positive(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def make_plan(profile="main", *, population=None, generations=None, runs=None,
              scenarios=None, data_dir=None, output=None, variants=None,
              seed=None, config_overrides=None, hv_target=None, cpu_budget=None,
              sensitivity_set=None, reference=None):
    config = load_profile(profile)
    if reference is not None:
        if len(reference) != 2 or not np.isfinite(reference).all():
            raise ValueError("HV reference must contain two finite numbers")
        config["reference_point"] = list(reference)
    if sensitivity_set is not None and profile != "sensitivity":
        raise ValueError("--sensitivity-set requires --profile sensitivity")
    config["population"] = positive(population if population is not None else config["population"], "population", 2)
    config["generations"] = positive(generations if generations is not None else config["generations"], "generations")
    config["runs"] = positive(runs if runs is not None else config["runs"], "runs")
    config["scenarios"] = scenarios if scenarios is not None else config["scenarios"]
    if not config["scenarios"] or len(set(config["scenarios"])) != len(config["scenarios"]):
        raise ValueError("Supply at least one scenario without duplicates")
    allowed = set(LARGE_SHAPES) if profile == "large" else ({1} if profile == "demo" else set(range(1, 8)))
    if not set(config["scenarios"]).issubset(allowed):
        raise ValueError(f"{profile} supports scenarios {sorted(allowed)}")
    config["variants"] = variants if variants is not None else config["variants"]
    if not config["variants"] or len(set(config["variants"])) != len(config["variants"]) or any(v not in VARIANTS for v in config["variants"]):
        raise ValueError(f"Variants must be distinct members of {VARIANTS}")
    if seed is not None and (config["runs"] != 1 or len(config["scenarios"]) != 1):
        raise ValueError("--seed is only available for a single scenario and single run")
    if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32)")
    base_config = AlgorithmConfig(**(config.get("algorithm") or {}))
    base_config = replace(base_config, **(config_overrides or {}))
    config["algorithm"] = asdict(base_config)
    data = Path(data_dir).resolve() if data_dir is not None else (ROOT / config["data_dir"]).resolve() if config.get("data_dir") else None
    target = Path(output).resolve() if output else ROOT / "results" / profile
    config["data_dir"] = str(data) if data else None
    config["output"] = str(target)
    blockers = []
    if profile == "large" and data is None:
        blockers.append("Provide --data-dir with 500/24, 750/30, 1000/38 task/antenna instances")
    if profile == "cpu":
        config["hv_target"] = hv_target
        config["cpu_budget_seconds"] = cpu_budget
        if hv_target is None or cpu_budget is None:
            blockers.append("CPU execution requires explicit --hv-target and --cpu-budget (CPU seconds)")
        for value, name in ((hv_target, "hv_target"), (cpu_budget, "cpu_budget")):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and > 0")
    elif hv_target is not None or cpu_budget is not None:
        raise ValueError("CPU target/budget options require --profile cpu")
    settings = [{"name": "default", "algorithm": {}}]
    if profile == "sensitivity":
        selected = sensitivity_set or config.get("default_set", "primary")
        if selected != "all" and selected not in config["sweeps"]:
            raise ValueError("Sensitivity set must be primary, transfer, local, or all")
        settings = [s for key, values in config["sweeps"].items() if selected in (key, "all") for s in values]
        config["sensitivity_set"] = selected
    entries, inputs = [], {}
    for scenario in config["scenarios"]:
        if data is not None:
            paths = [data / f"{prefix}{scenario + 11}.dat" for prefix in ("reqlf", "requp")]
            missing = [p.name for p in paths if not p.is_file()]
            if missing:
                blockers.append(f"Scenario {scenario}: missing input files {', '.join(missing)}")
            else:
                problem = SatelliteSchedulingProblem(scenario_id=scenario, data_dir=data)
                shape = (problem.n_tasks, problem.n_antennas)
                if profile == "large" and shape != LARGE_SHAPES[scenario]:
                    raise ValueError(f"Large scenario {scenario} has {shape[0]} tasks/{shape[1]} antennas; supplement requires {LARGE_SHAPES[scenario][0]}/{LARGE_SHAPES[scenario][1]}")
                inputs[str(scenario)] = {"tasks": shape[0], "antennas": shape[1], "files_sha256": {p.name: sha256(p) for p in paths}}
                provenance = data / "generation-manifest.json"
                if provenance.is_file():
                    inputs[str(scenario)]["generation_manifest"] = json.loads(provenance.read_text(encoding="utf-8"))
                    inputs[str(scenario)]["generation_manifest_sha256"] = sha256(provenance)
        for setting in settings:
            parameters = asdict(replace(base_config, **setting["algorithm"]))
            for variant in config["variants"]:
                label = variant if setting["name"] == "default" else variant + "__" + setting["name"]
                for run_id in range(1, config["runs"] + 1):
                    run_seed = seed if seed is not None else 1000 * scenario + run_id
                    entries.append({"scenario": scenario, "algorithm": label, "variant": variant,
                                    "setting": setting["name"], "run": run_id, "seed": run_seed,
                                    "population": config["population"], "generations": config["generations"],
                                    "parameters": parameters})
    return {"schema_version": 1, "implementation": VERSION, "profile": config,
            "input_instances": inputs, "source": source_fingerprint(),
            "seed_policy": "1000 * scenario + run, run = 1..N; paired settings share seeds",
            "run_count": len(entries), "execution_blockers": blockers,
            "result_claim": "Results generated by this run using the recorded parameters, inputs, and implementation",
            "runs": entries}


class RunTracker(Callback):
    def __init__(self, problem, reference, hv_target=None, cpu_budget=None, fractions=None):
        super().__init__()
        self.problem = problem
        self.reference = reference
        self.hv_target = hv_target
        self.cpu_budget = cpu_budget
        self.fractions = fractions or [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        self.thresholds = {str(int(round(f * 100))): None for f in self.fractions}
        self.history = []
        self.cpu_start = time.process_time()
        self.wall_start = time.perf_counter()
        self.stop_reason = "generation_budget"

    def notify(self, algorithm):
        cpu = time.process_time() - self.cpu_start
        wall = time.perf_counter() - self.wall_start
        hv = hypervolume(algorithm.pop.get("F"), self.reference)
        record = dict(algorithm.record[-1]) if getattr(algorithm, "record", None) else {}
        record.update(generation=int(algorithm.n_gen), evaluations=int(algorithm.evaluator.n_eval),
                      objective_calls=int(self.problem.objective_calls), HV=hv, cpu_seconds=cpu, wall_seconds=wall)
        self.history.append(record)
        if self.hv_target is not None:
            # Hits measured after a generation exceeds its CPU budget are censored.
            within_budget = self.cpu_budget is None or cpu <= self.cpu_budget
            if within_budget:
                for fraction in self.fractions:
                    key = str(int(round(fraction * 100)))
                    if self.thresholds[key] is None and hv >= fraction * self.hv_target:
                        self.thresholds[key] = {"generation": int(algorithm.n_gen), "cpu_seconds": cpu, "wall_seconds": wall, "HV": hv}
            if self.cpu_budget is not None and cpu >= self.cpu_budget:
                self.stop_reason = "cpu_budget"
                algorithm.termination.force_termination = True
            elif hv >= self.hv_target:
                self.stop_reason = "hv_target"
                algorithm.termination.force_termination = True


def ensure_empty(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new --output directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def execute_plan(plan, progress=print):
    if plan["execution_blockers"]:
        raise ValueError("; ".join(plan["execution_blockers"]))
    if source_fingerprint()["sha256"] != plan["source"]["sha256"]:
        raise ValueError("Source/config files changed after planning; rebuild the run plan")
    for instance in plan["input_instances"].values():
        for filename, expected in instance["files_sha256"].items():
            if sha256(Path(plan["profile"]["data_dir"]) / filename) != expected:
                raise ValueError("Input files changed after planning; rebuild the run plan")
    output = ensure_empty(plan["profile"]["output"])
    write_json(output / "manifest.json", plan)
    run_records, diagnostics, traces = [], [], []
    for i, entry in enumerate(plan["runs"], 1):
        if source_fingerprint()["sha256"] != plan["source"]["sha256"]:
            raise ValueError("Source/config files changed during experiment; results already completed are retained")
        for filename, expected in plan["input_instances"][str(entry["scenario"])]["files_sha256"].items():
            if sha256(Path(plan["profile"]["data_dir"]) / filename) != expected:
                raise ValueError("Input files changed during experiment; results already completed are retained")
        run_dir = output / entry["algorithm"] / f"AFIT{entry['scenario']}" / f"run_{entry['run']:02d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "run_manifest.json", entry)
        problem = SatelliteSchedulingProblem(scenario_id=entry["scenario"], data_dir=plan["profile"]["data_dir"])
        algorithm = build_algorithm(problem, config=AlgorithmConfig(**entry["parameters"]),
                                    population_size=entry["population"], seed=entry["seed"], variant=entry["variant"])
        tracker = RunTracker(problem, plan["profile"]["reference_point"], plan["profile"].get("hv_target"),
                             plan["profile"].get("cpu_budget_seconds"), plan["profile"].get("threshold_fractions"))
        try:
            result = minimize(problem, algorithm, ("n_gen", entry["generations"]), seed=entry["seed"],
                              callback=tracker, copy_algorithm=False, save_history=False, verbose=False)
            elapsed_cpu = time.process_time() - tracker.cpu_start
            elapsed_wall = time.perf_counter() - tracker.wall_start
            if result.F is None or result.X is None:
                raise RuntimeError("Optimizer returned no Pareto solutions")
            front = as_front(result.F)
            solutions = np.atleast_2d(result.X).astype(int)
            if len(solutions) != len(front) or not all(problem.is_feasible(x) for x in solutions):
                raise RuntimeError("Export refused: infeasible or unmatched returned schedules")
            search_calls = int(problem.objective_calls)
            checked = np.asarray([problem.objectives(x.copy()) for x in solutions])
            if not np.allclose(front, checked, rtol=1e-9, atol=1e-12):
                raise RuntimeError("Export refused: schedules do not reproduce their objective values")
            np.savetxt(run_dir / "pareto.csv", front, delimiter=",", header="profit_loss,load_cv", comments="")
            np.savetxt(run_dir / "solutions.csv", solutions, delimiter=",", fmt="%d",
                       header=",".join(f"task_{k+1}" for k in range(problem.n_tasks)), comments="")
            np.save(run_dir / "solutions.npy", solutions, allow_pickle=False)
            write_json(run_dir / "history.json", tracker.history)
            timing = {"cpu_seconds": elapsed_cpu, "wall_seconds": elapsed_wall,
                      "timing_scope": "minimize call including initialization; excludes problem loading, post-run validation/export",
                      "clock": "time.process_time; current process including all its threads",
                      "sampling": "generation boundaries; CPU budget may be exceeded by one generation",
                      "machine_dependent": True, "stop_reason": tracker.stop_reason,
                      "threshold_hits": tracker.thresholds if tracker.hv_target is not None else {}}
            metadata = {**entry, "implementation": VERSION, "status": "completed", "search_objective_calls": search_calls,
                        "export_validation_calls": len(solutions), "pymoo_evaluations": int(algorithm.evaluator.n_eval),
                        "completed_generations": len(tracker.history), "nondominated_solutions": len(front),
                        "inputs": plan["input_instances"][str(entry["scenario"])], "source_sha256": plan["source"]["sha256"],
                        "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pymoo", "scipy")},
                        "finished_utc": datetime.now(timezone.utc).isoformat(), "timing": timing}
            write_json(run_dir / "metadata.json", metadata)
            offspring = [r for r in tracker.history if r.get("phase") == "offspring"]
            candidates = sum(r.get("candidate_count", 0) for r in offspring)
            pairs = sum(r.get("pre_repair_conflicts", 0) for r in offspring)
            diagnostics.append({"scenario": f"AFIT{entry['scenario']}", "algorithm": entry["algorithm"], "run": entry["run"],
                "seed": entry["seed"], "completed_generations": len(tracker.history),
                "cpu_seconds": elapsed_cpu, "wall_seconds": elapsed_wall, "stop_reason": tracker.stop_reason,
                "pymoo_evaluations": int(algorithm.evaluator.n_eval), "search_objective_calls": search_calls,
                "offspring_candidates": candidates, "repair_total_pairs": pairs,
                "repair_pairs_per_offspring": pairs / candidates if candidates else None,
                "last_generation_repair_pairs_per_offspring": offspring[-1].get("mean_repair_count") if offspring else None,
                "removed_tasks": sum(r.get("removed_tasks", 0) for r in offspring)})
            traces.append({"scenario": f"AFIT{entry['scenario']}", "algorithm": entry["algorithm"], "variant": entry["variant"],
                           "run": entry["run"], "final_HV": hypervolume(front, plan["profile"]["reference_point"]),
                           "history": tracker.history, "thresholds": tracker.thresholds})
            run_records.append({"scenario": f"AFIT{entry['scenario']}", "algorithm": entry["algorithm"], "run": entry["run"], "front": front, "source_file": (run_dir / "pareto.csv").relative_to(output).as_posix(), "source_sha256": sha256(run_dir / "pareto.csv")})
            progress(f"[{i}/{plan['run_count']}] AFIT{entry['scenario']} {entry['algorithm']} run {entry['run']:02d}: HV={hypervolume(front, plan['profile']['reference_point']):.6f}")
        except Exception as exc:
            write_json(run_dir / "failure.json", {"status": "failed", "error": str(exc), "entry": entry})
            raise
    summary = analyze_records(run_records, output / "analysis", plan["profile"]["reference_point"], provenance="new paper-aligned runs")
    export_diagnostics(diagnostics, traces, output / "analysis", plan["profile"]["profile"])
    return summary


def discover_fronts(directory):
    """Load actual fronts, never replace missing runs or algorithms with zeros.

    New: ALGORITHM/AFIT1/run_01/pareto.csv
    Historical: ALGORITHM/pareto_fronts/SRSP1/run_01.csv
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    records, keys = [], set()
    paths = sorted(set(directory.rglob("pareto.csv")) | set(directory.rglob("run_*.csv")))
    for path in paths:
        if path.name == "pareto.csv" and re.fullmatch(r"run_\d+", path.parent.name):
            run_id = int(path.parent.name.split("_")[-1])
            scenario, algorithm = path.parent.parent.name, path.parent.parent.parent.name
        elif path.parent.parent.name == "pareto_fronts" and re.fullmatch(r"run_\d+\.csv", path.name):
            run_id = int(path.stem.split("_")[-1])
            scenario, algorithm = path.parent.name, path.parent.parent.parent.name
        else:
            continue
        if not re.fullmatch(r"(?:AFIT|SRSP)\d+", scenario):
            continue
        scenario = "AFIT" + re.search(r"\d+$", scenario).group()
        key = (scenario, algorithm, run_id)
        if key in keys:
            raise ValueError(f"Duplicate result identity {key}; analyze distinct sources separately")
        keys.add(key)
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        front = as_front(values)
        if not len(front):
            raise ValueError(f"Empty Pareto CSV: {path}; failed runs cannot be treated as successful zero-valued runs")
        records.append({"scenario": scenario, "algorithm": algorithm, "run": run_id, "front": front,
                        "source_file": path.relative_to(directory).as_posix(), "source_sha256": sha256(path)})
    if not records:
        raise ValueError("No supported Pareto CSVs found; expected ALGORITHM/AFIT1/run_01/pareto.csv or ALGORITHM/pareto_fronts/SRSP1/run_01.csv")
    return records


def analyze_records(records, output, reference=(1.1, 1.1), provenance="existing result CSV files"):
    if not records:
        raise ValueError("At least one actual Pareto result is required")
    output = ensure_empty(output)
    references = {}
    for scenario in sorted({r["scenario"] for r in records}):
        references[scenario] = reference_front([r["front"] for r in records if r["scenario"] == scenario])
        np.savetxt(output / (scenario + "_reference_front.csv"), references[scenario], delimiter=",", header="profit_loss,load_cv", comments="")
    metrics = []
    for row in records:
        front, ref = row["front"], references[row["scenario"]]
        metrics.append({k: row[k] for k in ("scenario", "algorithm", "run")} | {
            "HV": hypervolume(front, reference), "IGD": igd(front, ref), "Spread": spread(front, ref)})
    summaries = summarize(metrics)
    write_csv(output / "metrics_by_run.csv", metrics)
    write_csv(output / "summary.csv", summaries)
    write_json(output / "statistics.json", paired_tests(summaries))
    write_json(output / "analysis_manifest.json", {
        "provenance": provenance, "HV_reference_point": list(reference), "objectives": "both minimized, no additional rescaling",
        "reference_front": "union of nondominated points from only supplied algorithms and runs, separately per scenario",
        "IGD": "mean nearest Euclidean distance from each reference point to approximation",
        "Spread": "Deb endpoint-aware spread; d_f/d_l reference-extreme distances; fewer than two points => 1",
        "std_ddof": 0, "algorithms_present": sorted({r["algorithm"] for r in records}),
        "result_files": [{k: v for k, v in r.items() if k != "front"} for r in records],
        "analysis_scope": "Metrics use the supplied fronts and recorded formulas. Base denotes the NSGA-II ablation control."})
    return summaries


def export_diagnostics(rows, traces, output, profile):
    """Export exact counters, CPU censoring, and measured time-to-target."""
    write_csv(output / "run_diagnostics.csv", rows)
    notes = {"repair_total_pairs": "Sum of pre-repair conflicting task pairs across offspring generations; initialization excluded",
             "repair_pairs_per_offspring": "Total pre-repair pair conflicts divided by total offspring candidates across all generations",
             "last_generation_repair_pairs_per_offspring": "Last offspring generation's mean number of pre-repair pair conflicts",
             "removed_tasks": "Genes changed to unassigned during repair; distinct from paper Repair Count",
             "timing": "Process CPU seconds, machine dependent, checked at generation boundaries; initialization included",
             "historical_comparison": "Diagnostics in this file are calculated from the present run; archived tables retain their source records"}
    if profile == "cpu":
        threshold_rows, summaries = [], []
        for trace in traces:
            for threshold, hit in trace["thresholds"].items():
                threshold_rows.append({"scenario": trace["scenario"], "algorithm": trace["algorithm"], "run": trace["run"],
                    "threshold_percent": int(threshold), "reached_within_budget": hit is not None,
                    "generation": hit["generation"] if hit else None,
                    "cpu_seconds": hit["cpu_seconds"] if hit else None,
                    "wall_seconds": hit["wall_seconds"] if hit else None})
        for key in sorted({(r["scenario"], r["algorithm"], r["threshold_percent"]) for r in threshold_rows}):
            matching = [r for r in threshold_rows if (r["scenario"], r["algorithm"], r["threshold_percent"]) == key]
            hits = [r["cpu_seconds"] for r in matching if r["reached_within_budget"]]
            summaries.append({"scenario": key[0], "algorithm": key[1], "threshold_percent": key[2], "runs": len(matching),
                "hits": len(hits), "censored": len(matching) - len(hits),
                "CPU_mean_among_hits": float(np.mean(hits)) if hits else None,
                "CPU_std_among_hits": float(np.std(hits, ddof=0)) if hits else None})
        write_csv(output / "cpu_thresholds_by_run.csv", threshold_rows)
        write_csv(output / "cpu_threshold_summary.csv", summaries)
        notes["CPU_censoring"] = "Unreached thresholds retain missing time, never zero. Summary means include successful hits only and report counts."
    if profile == "large":
        targets = {}
        for scenario in sorted({t["scenario"] for t in traces}):
            baseline = [t["final_HV"] for t in traces if t["scenario"] == scenario and t["variant"] == "base"]
            if baseline:
                targets[scenario] = float(np.mean(baseline))
        target_rows = []
        for trace in traces:
            if trace["variant"] != "full" or trace["scenario"] not in targets:
                continue
            target = targets[trace["scenario"]]
            hit = next((h for h in trace["history"] if h["HV"] >= target), None)
            target_rows.append({"scenario": trace["scenario"], "algorithm": trace["algorithm"], "run": trace["run"],
                "baseline_mean_final_HV": target, "target_reached": hit is not None,
                "generation": hit["generation"] if hit else None,
                "cpu_seconds": hit["cpu_seconds"] if hit else None,
                "wall_seconds": hit["wall_seconds"] if hit else None})
        if target_rows:
            write_csv(output / "large_time_to_target.csv", target_rows)
        notes["large_target"] = "Scenario-wise mean final HV of the actually supplied new base runs; target unavailable without base. Base is the NSGA-II ablation control in this run."
    write_json(output / "diagnostics_manifest.json", notes)
