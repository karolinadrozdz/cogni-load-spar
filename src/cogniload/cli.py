"""Stage runner.

    python -m cogniload.cli find_band [--force] [--model dev|prod]
    python -m cogniload.cli confidence --set capitals|regions [--limit N] [--force]
    python -m cogniload.cli report <experiment>

`report` reads parquet only.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

from cogniload import registry

#: Experiment modules, each with `run` and `report`.
EXPERIMENTS = ["confidence"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["find_band", *EXPERIMENTS, "report"])
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS, help="what to report")
    parser.add_argument("--set", dest="stimuli_set", help="confidence only: the stimuli set")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--model", help="registry alias; overrides the config's")
    parser.add_argument("--out-dir", help="overrides the config's out_dir")
    parser.add_argument("--limit", type=int, help="items per condition, for a smoke run")
    parser.add_argument("--force", action="store_true", help="redo outputs that exist")
    args = parser.parse_args(argv)
    if (args.stage == "report") != bool(args.experiment):
        parser.error("name an experiment after `report`, and only there")
    if (args.stage == "confidence") != bool(args.stimuli_set):
        parser.error("--set is required for confidence, and only for it")

    config = yaml.safe_load(Path(args.config).read_text())
    results = Path(args.out_dir or config["out_dir"])
    spec = registry.resolve(args.model or config["model"], results_dir=results)
    print(f"{args.stage}: {spec['alias']} -> {spec['hf_id']}  (results: {results / spec['alias']})")

    module = importlib.import_module(f"cogniload.{args.experiment or args.stage}")
    if args.stage == "report":
        module.report(spec, config, results)
    elif args.stage == "find_band":
        module.run(spec, config, results, force=args.force)
    else:
        extra = {"stimuli_set": args.stimuli_set} if args.stimuli_set else {}
        module.run(spec, config, results, limit=args.limit, force=args.force, **extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
