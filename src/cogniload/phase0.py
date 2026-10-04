"""Phase 0 R1: find the layer band where lens readouts are legible.

Two-hop prompts whose bridge entity is never written down. If the lens works,
the bridge surfaces mid-network — that is the paper's spider->legs result, and
the band where it happens is what every downstream measure reads from.

The criterion is committed in `configs/exp1.yaml` before this runs and uses
only these readouts (DECISIONS.md D11): selecting the band by what makes R2 or
Phase 1 come out well would fit the band to its own answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from cogniload import bands, readout, registry, scoring

#: (prompt, unnamed bridge entity). Targets must be single tokens; `read`
#: raises if one is not. Kept deliberately small and plain — this measures the
#: lens, not the model's knowledge.
R1_ITEMS: list[tuple[str, str]] = [
    ("Fact: The currency used in the country shaped like a boot is", "Italy"),
    ("Fact: The language spoken in the country famous for the Eiffel Tower is", "France"),
    ("Fact: The capital of the country where the pyramids of Giza stand is", "Egypt"),
    ("Fact: The currency used in the country known for sushi and Mount Fuji is", "Japan"),
    ("Fact: The language spoken in the country famous for flamenco and paella is", "Spain"),
    ("Fact: The capital of the country that built the Great Wall is", "China"),
    ("Fact: The language spoken in the country famous for maple syrup and hockey is", "Canada"),
    ("Fact: The currency used in the country famous for the Taj Mahal is", "India"),
    ("Fact: The number of legs on the animal that spins a web is", "spider"),
    ("Fact: The sound made by the animal that fetches sticks and guards the house is", "dog"),
    ("Fact: The largest animal living in the ocean is the", "whale"),
    ("Fact: The animal that has a trunk and enormous ears is the", "elephant"),
    ("Fact: The colour of the fruit that keeps the doctor away is", "apple"),
    ("Fact: The instrument with eighty-eight black and white keys is the", "piano"),
    ("Fact: The tool used to drive a nail into wood is a", "hammer"),
    ("Fact: The body part used to grip a pen is the", "hand"),
]


def _sanity_check(model, lens, layers: list[int]) -> None:
    """Print the J-lens and logit-lens top tokens on one two-hop prompt.

    Eyeball check that weights, lens, layout and unembedding are wired up: the
    bridge entity should surface under the J-lens at layers where the logit lens
    is still noise.
    """
    prompt = "Fact: The currency used in the country shaped like a boot is"
    probe = [layers[0], layers[len(layers) // 2], layers[-1]]
    for use_jacobian, name in ((True, "jlens"), (False, "logit")):
        logits, _, _ = lens.apply(model, prompt, layers=probe, positions=[-1],
                                  use_jacobian=use_jacobian)
        for layer in probe:
            top = [model.tokenizer.decode([i]).strip()
                   for i in logits[layer][0].topk(6).indices.tolist()]
            print(f"  {name:5} L{layer:<3} {top}")


def run(spec: registry.ModelSpec, config: dict, results_dir: Path,
        *, force: bool = False) -> tuple[int, int]:
    """Measure every layer in the candidate range, pick the band, record it."""
    if not force and (band := bands.load(spec.alias, results_dir)):
        print(f"band already recorded: {band} (use --force to redo)")
        return band

    settings = config["phase0"]["r1_band"]
    lo_frac, hi_frac = settings["candidate_fraction"]
    # Every layer, not a stride: lens.apply returns all requested layers from
    # one forward pass, and choose_band refuses a gapped set.
    layers = [
        l for l in range(int(lo_frac * spec.n_layers), int(hi_frac * spec.n_layers))
        if l < spec.n_lens_matrices
    ]

    model, lens = registry.load(spec)
    _sanity_check(model, lens, layers)

    print(f"\nR1 over {len(R1_ITEMS)} items, layers {layers[0]}-{layers[-1]}:")
    per_item = []
    for prompt, bridge in R1_ITEMS:
        token_id = readout.single_token_id(model.tokenizer, bridge)
        r = readout.read(model, lens, prompt, [token_id], layers=layers, positions=[-1])
        ranks = {layer: int(r.rank[0, i, 0]) for i, layer in enumerate(r.layers)}
        per_item.append(ranks)
        best = min(ranks, key=ranks.get)
        print(f"  {bridge:>9}  best L{best} rank {ranks[best]}")

    medians = scoring.median_rank_by_layer(per_item)
    band = scoring.choose_band(
        medians, n_layers=spec.n_layers,
        candidate_fraction=tuple(settings["candidate_fraction"]),
        k_band=settings["k_band"], min_width=settings["min_width"],
    )
    bands.save(
        spec.alias, band, results_dir,
        criterion={"median_full_vocab_rank_at_most": settings["k_band"],
                   "min_width": settings["min_width"],
                   "candidate_fraction": settings["candidate_fraction"]},
        median_rank_by_layer={str(k): v for k, v in medians.items()},
        n_items=len(R1_ITEMS),
    )
    (results_dir / spec.alias / "r1_ranks.json").write_text(
        json.dumps({"items": [b for _, b in R1_ITEMS],
                    "median_rank_by_layer": {str(k): v for k, v in medians.items()}},
                   indent=2)
    )
    print(f"\nband = {band}  (median rank <= {settings['k_band']})")
    return band
