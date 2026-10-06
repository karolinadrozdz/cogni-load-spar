"""Model + lens resolution from a registry alias.

The config names an alias (`dev`, `prod`); only `configs/models.yaml` names a
model. `resolve()` touches no weights, so tests run without a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from cogniload import bands

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def resolve(alias: str, *, path: Path = REGISTRY, results_dir: Path | None = None) -> dict:
    """The alias's registry entry over the defaults, plus the band from `results_dir`."""
    doc = yaml.safe_load(path.read_text())
    if alias not in doc["aliases"]:
        known = ", ".join(sorted(doc["aliases"]))
        raise KeyError(f"unknown model alias {alias!r}; registry has: {known}")
    spec = {**doc["defaults"], **doc["aliases"][alias], "alias": alias,
            "band": bands.load(alias, results_dir) if results_dir else None}
    # Never defaulted: the Qwen3 template turns thinking on when it is unset.
    if "enable_thinking" not in spec:
        raise KeyError(f"alias {alias!r} must set enable_thinking")
    for key in ("hf_revision", "lens_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", spec[key]):
            raise ValueError(f"alias {alias!r} is unpinned ({key}: {spec[key]}); pin a sha")
    return spec


def load(spec: dict) -> tuple[Any, Any]:
    """Load `(jlens model, lens)`; the tokenizer is `model.tokenizer`. Needs a GPU."""
    import jlens
    import torch
    import transformers

    kw = {"revision": spec["hf_revision"]}
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], dtype=getattr(torch, spec["dtype"]), device_map="auto", **kw)
    model = jlens.from_hf(hf, transformers.AutoTokenizer.from_pretrained(spec["hf_id"], **kw))
    lens = jlens.JacobianLens.from_pretrained(
        spec["lens_repo"], filename=spec["lens_file"], revision=spec["lens_revision"])
    _assert_lens_matches(spec, model, lens)
    return model, lens


def _assert_lens_matches(spec: dict, model: Any, lens: Any) -> None:
    """Check shapes, not names: upstream lens metadata has been wrong before."""
    if (model.n_layers, model.d_model) != (spec["n_layers"], spec["d_model"]):
        raise ValueError(f"{spec['hf_id']} is {model.n_layers}L/{model.d_model}d but the "
                         f"registry says {spec['n_layers']}L/{spec['d_model']}d")
    # The lens maps layer l to the final layer, so it has no matrix for the final layer.
    if (lens.d_model, len(lens.source_layers)) != (spec["d_model"], spec["n_layers"] - 1):
        raise ValueError(f"lens {spec['lens_file']} is d_model={lens.d_model} over "
                         f"{len(lens.source_layers)} layers; wrong lens for {spec['hf_id']}")
