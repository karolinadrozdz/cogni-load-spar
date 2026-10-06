"""Stage runner.

    python -m cogniload.cli find_band
    python -m cogniload.cli single_cue [--limit N] [--force] [--model dev|prod]
    python -m cogniload.cli forced_demand
    python -m cogniload.cli retro_cue
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

EXPERIMENTS = ["single_cue", "forced_demand", "retro_cue"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["find_band", *EXPERIMENTS, "report"])
    parser.add_argument("experiment", nargs="?", choices=EXPERIMENTS, help="what to report")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--model", help="registry alias; overrides the config's")
    parser.add_argument("--out-dir", help="overrides the config's out_dir")
    parser.add_argument("--limit", type=int, help="streams per C_t, for a smoke run")
    parser.add_argument("--force", action="store_true", help="redo shards that exist")
    args = parser.parse_args(argv)
    if (args.stage == "report") != bool(args.experiment):
        parser.error("name an experiment after `report`, and only there")

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
        module.run(spec, config, results, limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
