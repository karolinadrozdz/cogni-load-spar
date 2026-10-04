"""Model + lens resolution from a registry alias.

The config names an alias (`dev`, `prod`); nothing downstream names a model.
`resolve()` touches no weights, so stimulus generation and tests run without a
GPU; `load()` does the heavy work.

`configs/models.yaml` is read-only here — the band is discovered rather than
declared, and lives in the results tree (`bands.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cogniload import bands

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """Fully-pinned description of one (model, lens) pair."""

    alias: str
    hf_id: str
    hf_revision: str
    n_layers: int
    d_model: int
    dtype: str
    lens_repo: str
    lens_revision: str
    lens_file: str
    enable_thinking: bool
    #: Set from the results tree by `resolve(..., results_dir=...)`, never from
    #: the registry file.
    band: tuple[int, int] | None = None

    @property
    def n_lens_matrices(self) -> int:
        """The lens transports layer l -> final, so the final layer has none."""
        return self.n_layers - 1

    def band_layers(self) -> list[int]:
        if self.band is None:
            raise ValueError(
                f"alias {self.alias!r} has no workspace band; run Phase 0 R1, then "
                f"resolve with results_dir= so the band is loaded"
            )
        lo, hi = self.band
        # Caught here rather than inside lens.apply, which reports it obscurely.
        if hi > self.n_lens_matrices:
            raise ValueError(
                f"band {self.band} reaches layer {hi - 1}, but the lens only fits "
                f"layers 0..{self.n_lens_matrices - 1} for {self.hf_id}"
            )
        return list(range(lo, hi))

    def provenance(self) -> dict[str, Any]:
        """Flat dict for the run manifest."""
        return {
            "model_alias": self.alias,
            "model_id": self.hf_id,
            "model_revision": self.hf_revision,
            "lens_repo": self.lens_repo,
            "lens_revision": self.lens_revision,
            "lens_file": self.lens_file,
            "enable_thinking": self.enable_thinking,
            "band": list(self.band) if self.band else None,
        }


def resolve(
    alias: str,
    *,
    path: Path = REGISTRY,
    results_dir: Path | None = None,
    allow_unpinned: bool = False,
) -> ModelSpec:
    """Build a ModelSpec from a registry alias. No weights are touched.

    Pass `results_dir` to attach the band discovered by Phase 0 R1.
    """
    doc = yaml.safe_load(path.read_text())
    try:
        entry = doc["aliases"][alias]
    except KeyError:
        known = ", ".join(sorted(doc["aliases"]))
        raise KeyError(f"unknown model alias {alias!r}; registry has: {known}") from None

    defaults = doc.get("defaults", {})
    lens_defaults = defaults.get("lens", {})

    spec = ModelSpec(
        alias=alias,
        hf_id=entry["hf_id"],
        hf_revision=entry.get("hf_revision", "main"),
        n_layers=entry["n_layers"],
        d_model=entry["d_model"],
        dtype=entry.get("dtype", defaults.get("dtype", "bfloat16")),
        lens_repo=lens_defaults["repo"],
        lens_revision=lens_defaults["revision"],
        lens_file=entry["lens_file"],
        # Required, never defaulted: see models.yaml.
        enable_thinking=entry["enable_thinking"],
        band=bands.load(alias, results_dir) if results_dir else None,
    )
    if not allow_unpinned and spec.hf_revision == "main":
        raise ValueError(f"alias {alias!r} is unpinned (hf_revision: main); pin a sha")
    return spec


def load(spec: ModelSpec) -> tuple[Any, Any]:
    """Load `(jlens model, lens)`. Imports torch lazily — needs a GPU.

    The tokenizer is reachable as `model.tokenizer`; it is deliberately not
    returned separately, so there is only one name for it.
    """
    import jlens
    import torch
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.hf_revision,
        dtype=getattr(torch, spec.dtype),
        device_map="auto",
    )
    tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.hf_revision)
    model = jlens.from_hf(hf, tok)

    lens = jlens.JacobianLens.from_pretrained(
        spec.lens_repo, filename=spec.lens_file, revision=spec.lens_revision
    )
    _assert_lens_matches(spec, model, lens)
    return model, lens


def _assert_lens_matches(spec: ModelSpec, model: Any, lens: Any) -> None:
    """Catch a mislabeled lens before it silently poisons every readout.

    Upstream exports have shipped with stale metadata, so trust shapes over
    filenames (DECISIONS.md D5).
    """
    if (model.n_layers, model.d_model) != (spec.n_layers, spec.d_model):
        raise ValueError(
            f"{spec.hf_id} is {model.n_layers}L/{model.d_model}d but registry "
            f"says {spec.n_layers}L/{spec.d_model}d"
        )
    if lens.d_model != spec.d_model:
        raise ValueError(
            f"lens {spec.lens_file} is d_model={lens.d_model}, {spec.hf_id} is "
            f"d_model={spec.d_model}. Wrong lens for this model."
        )
    if len(lens.source_layers) != spec.n_lens_matrices:
        raise ValueError(
            f"lens {spec.lens_file} fits {len(lens.source_layers)} layers; "
            f"{spec.hf_id} needs {spec.n_lens_matrices}."
        )

