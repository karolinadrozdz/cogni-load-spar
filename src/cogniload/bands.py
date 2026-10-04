"""Workspace band persistence.

The band is discovered by Phase 0 R1, not declared, so it lives with the
results rather than in `configs/models.yaml` — writing that file
programmatically would strip its comments (DECISIONS.md D8).
"""

from __future__ import annotations

import json
from pathlib import Path

Band = tuple[int, int]


def path(alias: str, results_dir: Path) -> Path:
    return Path(results_dir) / alias / "band.json"


def save(alias: str, band: Band, results_dir: Path, **provenance) -> Path:
    """Record the band, plus whatever justified it, for the manifest."""
    lo, hi = band
    if not 0 <= lo < hi:
        raise ValueError(f"band {band} is not a valid half-open range")
    p = path(alias, results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"alias": alias, "band": [lo, hi], **provenance}, indent=2))
    return p


def load(alias: str, results_dir: Path) -> Band | None:
    p = path(alias, results_dir)
    if not p.exists():
        return None
    return tuple(json.loads(p.read_text())["band"])


def require(alias: str, results_dir: Path) -> Band:
    band = load(alias, results_dir)
    if band is None:
        raise FileNotFoundError(
            f"no workspace band for alias {alias!r}; run Phase 0 R1 first "
            f"(expected {path(alias, results_dir)})"
        )
    return band
