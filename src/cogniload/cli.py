"""Stage runner.

    python -m cogniload.cli find_band
    python -m cogniload.cli single_cue [--limit N] [--force] [--model dev|prod]
    python -m cogniload.cli forced_demand
    python -m cogniload.cli retro_cue
    python -m cogniload.cli derived_state --arm derived|copyable
    python -m cogniload.cli state_tracking --arm derived|copyable
    python -m cogniload.cli hay_factorial
    python -m cogniload.cli hay_type --arm same|named|reworded
    python -m cogniload.cli report <experiment>

A stage skips any C_t whose shard already exists unless --force. `report`
reads parquet only.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

from cogniload import registry

EXPERIMENTS = ["single_cue", "forced_demand", "retro_cue", "derived_state", "state_tracking"]
#: Stages in a shared module: stage -> the module with its `run_<stage>`, `report_<stage>`.
SHARED = {"hay_factorial": "hay_load", "hay_type": "hay_load"}
ARMS = {"derived_state": ["derived", "copyable"], "state_tracking": ["derived", "copyable"],
        "hay_type": ["same", "named", "reworded"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["find_band", *EXPERIMENTS, *SHARED, "report"])
    parser.add_argument("experiment", nargs="?", choices=[*EXPERIMENTS, *SHARED],
                        help="what to report")
    parser.add_argument("--arm", choices=sorted({a for arms in ARMS.values() for a in arms}),
                        help="derived|copyable for derived_state and state_tracking; "
                             "same|named|reworded for hay_type")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--model", help="registry alias; overrides the config's")
    parser.add_argument("--out-dir", help="overrides the config's out_dir")
    parser.add_argument("--limit", type=int, help="streams per C_t, for a smoke run")
    parser.add_argument("--force", action="store_true", help="redo shards that exist")
    parser.add_argument("--top-tokens", type=int, metavar="N",
                        help="state_tracking: print top tokens for N items, write nothing")
    args = parser.parse_args(argv)
    if (args.stage == "report") != bool(args.experiment):
        parser.error("name an experiment after `report`, and only there")
    if args.arm not in ARMS.get(args.stage, [None]):
        parser.error(f"--arm for {args.stage} is one of {ARMS[args.stage]}" if args.stage in ARMS
                     else f"--arm is only for {', '.join(ARMS)}")
    if args.top_tokens and args.stage != "state_tracking":
        parser.error("--top-tokens is only for state_tracking")

    config = yaml.safe_load(Path(args.config).read_text())
    results = Path(args.out_dir or config["out_dir"])
    spec = registry.resolve(args.model or config["model"], results_dir=results)
    print(f"{args.stage}: {spec['alias']} -> {spec['hf_id']}  (results: {results / spec['alias']})")

    name = args.experiment or args.stage
    module = importlib.import_module(f"cogniload.{SHARED.get(name, name)}")
    if args.stage == "report":
        report = getattr(module, f"report_{name}") if name in SHARED else module.report
        report(spec, config, results)
    elif args.stage == "find_band":
        module.run(spec, config, results, force=args.force)
    else:
        arm = {"arm": args.arm} if args.arm else {}
        if args.top_tokens:
            arm["top_tokens"] = args.top_tokens
        run = getattr(module, f"run_{args.stage}") if args.stage in SHARED else module.run
        run(spec, config, results, limit=args.limit, force=args.force, **arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
