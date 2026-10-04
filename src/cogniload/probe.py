"""Diagnostic ladder: can this code reproduce a known-positive J-space readout?

Asking a model to remember a short sequence puts the items in the J-space —
visible in Neuronpedia's slice view. The keep-track run does not show that for
non-queried targets. Either our readout is wrong, or the paradigm differs.

Each rung changes one thing: content (digits vs words), length, how much text
follows the sequence, and raw vs chat rendering. Reads every layer at every
position, which is what the slice view shows, and reports the best rank each
item reaches anywhere.

    python -m cogniload.cli probe --model dev
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cogniload import readout, registry

DIGITS = ["4", "7", "2", "9", "1", "3"]
SHORT = ["cat", "pear", "red", "dog", "plum", "blue"]
LONG = ["cat", "pear", "red", "dog", "plum", "blue", "saw", "France", "piano",
        "hand", "bus", "goat", "lemon", "green", "axe", "Spain", "flute",
        "foot", "train", "fox", "grape", "grey", "drill", "Japan"]

#: (name, prompt, items). The keep-track rungs reproduce the real prompt's
#: shape so a difference points at prompt structure rather than at the lens.
LADDER: list[tuple[str, str, list[str]]] = [
    ("digits_bare",
     "Remember this sequence of numbers: " + ", ".join(DIGITS) + ".",
     DIGITS),
    ("words_bare",
     "Remember this sequence of words: " + ", ".join(SHORT) + ".",
     SHORT),
    ("words_long",
     "Remember this sequence of words: " + ", ".join(LONG) + ".",
     LONG),
    ("words_long_then_text",
     "Remember this sequence of words: " + ", ".join(LONG) + ". "
     "Remember the most recent word from each tracked category.",
     LONG),
    ("keep_track_shape",
     "Track these categories: animal, fruit. Here is a sequence of words: "
     + ", ".join(LONG) + ". Remember the most recent word from each tracked "
     "category. What was the most recent animal? Answer in one word.",
     LONG),
]


#: Distance from an item's own token, in tokens. Presence at 0 is the model
#: processing the word; presence at 11+ is the only thing that looks like
#: holding it. "anywhere after own" conflates the two, since it includes
#: distance 1.
BUCKETS: list[tuple[str, int, int]] = [
    ("0 (own)", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5),
    ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, 10**6),
]


def _decay(r, own: list[int], k: int = 25) -> list[dict]:
    """Present rate by distance from each item's own position.

    This is the persistence profile: how far past a word the J-space still
    carries it. Unconfounded by design -- the same item is compared with
    itself at increasing distance, so nothing depends on comparing words.
    """
    rows = []
    for label, lo, hi in BUCKETS:
        hits = total = 0
        for i, pos in enumerate(own):
            cols = [c for c in range(len(r.positions)) if lo <= c - pos <= hi]
            if not cols:
                continue
            total += 1
            hits += int(r.rank[i][:, cols].min() <= k)
        if total:
            rows.append({"bucket": label, "present": hits / total, "n_items": total})
    return rows


def _measure(model, lens, text, items, layers) -> dict:
    """Presence of each item, split by whether the model has moved past it.

    Every word is rank 1 at its own position -- the model is processing it --
    so a grid-wide "anywhere" is the diagonal and says nothing about held
    state. `after_own` starts one token past each item's own occurrence, which
    is the only region where presence means the item is still being carried.
    """
    resolved = [readout.resolve_token_id(model.tokenizer, w) for w in items]
    ids = [i for i, _ in resolved]
    forms = sorted({f for _, f in resolved})
    r = readout.read(model, lens, text, ids, layers=layers)  # every position

    own = _own_positions(model.tokenizer, text, items)
    best_all = r.rank.amin(dim=(1, 2))
    best_after, best_tail = [], []
    for i, pos in enumerate(own):
        after = r.rank[i, :, pos + 1:]
        best_after.append(int(after.min()) if after.numel() else 10**9)
        best_tail.append(int(r.rank[i, :, -1].min()))
    return {
        "n_items": len(items),
        "token_form": "+".join(forms),
        "present@25_incl_own_pos": float((best_all <= 25).float().mean()),
        "present@25_after_own_pos": sum(b <= 25 for b in best_after) / len(items),
        "present@25_at_last_pos": sum(b <= 25 for b in best_tail) / len(items),
        "n_positions": len(r.positions),
        "_decay": _decay(r, own),
    }


def _own_positions(tokenizer, text: str, items: list[str]) -> list[int]:
    """Token index where each item occurs in `text`, via character offsets."""
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=512, add_special_tokens=False)
    out = []
    for item in items:
        char = text.index(item)
        out.append(next((i for i, (a, b) in enumerate(enc["offset_mapping"])
                         if a <= char < b), 0))
    return out


def run(spec: registry.ModelSpec, config: dict, results_dir: Path, **_) -> None:
    lo, hi = config["readout"]["record_layer_fraction"]
    layers = [l for l in range(int(lo * spec.n_layers), int(hi * spec.n_layers))
              if l < spec.n_lens_matrices]
    model, lens = registry.load(spec)

    rows = []
    for name, raw, items in LADDER:
        chat = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}], tokenize=False,
            add_generation_prompt=True, enable_thinking=spec.enable_thinking,
        )
        for variant, text in (("raw", raw), ("chat", chat)):
            rows.append({"probe": name, "variant": variant,
                         **_measure(model, lens, text, items, layers)})
            r = rows[-1]
            print(f"  {name:<22}{variant:<6}"
                  f"own {r['present@25_incl_own_pos']:.2f} | "
                  f"after own {r['present@25_after_own_pos']:.2f} | "
                  f"last pos {r['present@25_at_last_pos']:.2f}")

    decay = pd.DataFrame([
        {"probe": r["probe"], "variant": r["variant"], **d}
        for r in rows for d in r.pop("_decay")
    ])
    out = pd.DataFrame(rows)
    print(f"\n{out.to_string(index=False, float_format='%.3f')}")

    print("\nPERSISTENCE — present@25 by distance from the item's own token")
    print("  distance 0 is the model processing the word; 11+ is holding it")
    print(decay.pivot_table(index=["probe", "variant"], columns="bucket",
                            values="present", sort=False)
          .reindex(columns=[b for b, _, _ in BUCKETS])
          .to_string(float_format="%.2f"))
    path = Path(results_dir) / spec.alias / "probe.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    decay.to_parquet(path.with_name("probe_decay.parquet"), index=False)
    print(f"\nwrote {path}")
    print("\n`after own` is the column that matters: `own` is ~1.00 everywhere "
          "because each word is rank 1 at its own position.")
