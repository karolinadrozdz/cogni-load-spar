"""The workspace band, stored with the results.

It is discovered by `find_band`, and writing it into `configs/models.yaml`
would round-trip that file and strip its comments.
"""

from __future__ import annotations

import json
from pathlib import Path


def save(alias: str, band: tuple[int, int], results_dir: Path, **provenance) -> None:
    lo, hi = band
    if not 0 <= lo < hi:
        raise ValueError(f"band {band} is not a valid half-open range")
    p = Path(results_dir) / alias / "band.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"alias": alias, "band": [lo, hi], **provenance}, indent=2))


def load(alias: str, results_dir: Path) -> tuple[int, int] | None:
    p = Path(results_dir) / alias / "band.json"
    return tuple(json.loads(p.read_text())["band"]) if p.exists() else None
