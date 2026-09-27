"""Command line for paper-aligned experiments and CSV analysis."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from .config import AlgorithmConfig
from .experiments import make_plan, execute_plan, discover_fronts, analyze_records, VARIANTS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "analyze"), default="run")
    parser.add_argument("--profile", choices=("demo", "main", "ablation", "large", "cpu", "sensitivity"), default="main", help="Experiment profile (default: main, paper settings); use demo explicitly for a quick example")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete resolved run plan; do not optimize or write outputs")
    parser.add_argument("--population", "--pop", type=int)
    parser.add_argument("--generations", "--gen", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--scenario", type=int, nargs="+", dest="scenarios")
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", dest="variants")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, help="A new/empty directory; existing results are never overwritten")
    parser.add_argument("--seed", type=int, help="Override seed for a single scenario/run only")
    parser.add_argument("--algorithm-config", type=Path, help="JSON object overriding AlgorithmConfig fields")
    parser.add_argument("--hv-target", type=float, help="Absolute 100%% HV target for CPU profile")
    parser.add_argument("--cpu-budget", type=float, help="Explicit process CPU seconds budget")
    parser.add_argument("--sensitivity-set", choices=("primary", "transfer", "local", "all"))
    parser.add_argument("--input", type=Path, help="Existing result root for analyze")
    parser.add_argument("--reference", type=float, nargs=2, default=None, help="HV reference for run/analyze (default 1.1 1.1)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            if args.input is None or args.output is None:
                raise ValueError("analyze requires --input and --output")
            records = discover_fronts(args.input)
            if args.dry_run:
                print(json.dumps({"run_count": len(records), "algorithms_present": sorted({r["algorithm"] for r in records}),
                                  "scenarios": sorted({r["scenario"] for r in records}), "output": str(args.output)}, indent=2))
            else:
                summaries = analyze_records(records, args.output, reference=args.reference or [1.1, 1.1])
                print(f"Analyzed {len(records)} actual fronts into {len(summaries)} scenario/algorithm groups: {args.output}")
            return 0
        if args.input is not None:
            raise ValueError("--input is only used with analyze")
        overrides = {}
        if args.algorithm_config:
            overrides = json.loads(args.algorithm_config.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise ValueError("Algorithm override JSON must be an object")
            unknown = set(overrides) - {f.name for f in fields(AlgorithmConfig)}
            if unknown:
                raise ValueError(f"Unknown algorithm parameters: {sorted(unknown)}")
        plan = make_plan(args.profile, population=args.population, generations=args.generations, runs=args.runs,
                         scenarios=args.scenarios, data_dir=args.data_dir, output=args.output, variants=args.variants,
                         seed=args.seed, config_overrides=overrides, hv_target=args.hv_target,
                         cpu_budget=args.cpu_budget, sensitivity_set=args.sensitivity_set, reference=args.reference)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        execute_plan(plan)
        print(f"Results saved to {plan['profile']['output']}")
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
