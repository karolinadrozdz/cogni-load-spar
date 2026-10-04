"""Stage runner.

    python -m cogniload.cli phase0  --config configs/exp1.yaml
    python -m cogniload.cli phase1  --config configs/exp1.yaml [--limit 20]
    python -m cogniload.cli analyze --config configs/exp1.yaml
    python -m cogniload.cli probe   --config configs/exp1.yaml

Stages are idempotent: a stage whose output exists is skipped unless --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cogniload import registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["phase0", "phase1", "analyze", "probe"])
    parser.add_argument("--config", default="configs/exp1.yaml")
    parser.add_argument("--model", help="override the config's registry alias")
    parser.add_argument("--out-dir", help="override the config's out_dir")
    parser.add_argument("--limit", type=int,
                        help="streams per C_t; use for the floor check before "
                             "committing to the full run")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    results = Path(args.out_dir or config["out_dir"])
    config["out_dir"] = str(results)
    spec = registry.resolve(args.model or config["model"], results_dir=results)
    print(f"{args.stage}: {spec.alias} -> {spec.hf_id}  (results: {results})")

    if args.stage == "phase0":
        from cogniload import phase0

        phase0.run(spec, config, results, force=args.force)
    elif args.stage == "phase1":
        from cogniload import phase1

        phase1.run(spec, config, results, limit=args.limit, force=args.force)
    elif args.stage == "probe":
        from cogniload import probe

        probe.run(spec, config, results)
    else:
        from cogniload import analyze

        analyze.run(spec, config, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
