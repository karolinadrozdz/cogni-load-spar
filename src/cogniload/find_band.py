"""find_band: the layer band where lens readouts are legible.

Two-hop prompts whose bridge entity is never written down: if the lens works,
the bridge surfaces mid-network. Every experiment reads in this band. The
criterion is fixed in the config and uses only these readouts, so the band
cannot be fitted to an experiment's outcome.
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path
from statistics import median

from cogniload import bands, experiment, readout, registry

#: (prompt, unnamed bridge entity). Plain on purpose: this measures the lens,
#: not the model's knowledge.
ITEMS: list[tuple[str, str]] = [
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


def choose_band(medians: dict[int, float], *, k_band: int, min_width: int) -> tuple[int, int]:
    """Longest run of >= `min_width` layers with median rank <= `k_band`, as `[lo, hi)`."""
    layers = sorted(medians)
    # A gapped layer set would yield a band over layers that were never measured.
    assert layers == list(range(layers[0], layers[-1] + 1)), f"gapped layer set {layers}"
    runs = [list(g) for ok, g in groupby(layers, key=lambda l: medians[l] <= k_band) if ok]
    runs = [r for r in runs if len(r) >= min_width]
    if not runs:
        raise RuntimeError(
            f"no run of >= {min_width} layers with median rank <= {k_band} in layers "
            f"{layers[0]}-{layers[-1]}: the lens is not legible on this model under the "
            f"committed criterion")
    best = max(runs, key=len)
    return best[0], best[-1] + 1


def run(spec: dict, config: dict, results_dir: Path, *, force: bool = False) -> tuple[int, int]:
    if not force and (band := bands.load(spec["alias"], results_dir)):
        print(f"band already recorded: {band} (use --force to redo)")
        return band
    s = config["find_band"]
    # Every layer, not a stride: one forward pass returns them all.
    layers = experiment.layer_range(spec["n_layers"], s["candidate_fraction"])
    model, lens = registry.load(spec)

    print(f"{len(ITEMS)} items, layers {layers[0]}-{layers[-1]}:")
    per_item = []
    for prompt, bridge in ITEMS:
        rows, _ = readout.read(model, lens, prompt, [bridge],
                               [readout.token_id(model.tokenizer, bridge)],
                               layers=layers, positions=[-1])
        ranks = {r["layer"]: r["rank"] for r in rows}
        per_item.append(ranks)
        best = min(ranks, key=ranks.get)
        print(f"  {bridge:>9}  best L{best} rank {ranks[best]}")

    medians = {layer: median(r[layer] for r in per_item) for layer in layers}
    band = choose_band(medians, k_band=s["k_band"], min_width=s["min_width"])
    bands.save(spec["alias"], band, results_dir,
               criterion={"median_full_vocab_rank_at_most": s["k_band"],
                          "min_width": s["min_width"],
                          "candidate_fraction": s["candidate_fraction"]},
               median_rank_by_layer={str(k): v for k, v in medians.items()},
               n_items=len(ITEMS))
    print(f"\nband = {band}  (median rank <= {s['k_band']})")
    return band
