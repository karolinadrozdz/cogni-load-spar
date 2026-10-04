"""Category exemplar pools, filtered to single tokens per model.

The usable set is tokenizer-dependent, so it is built and cached per alias and
never shared. Pools are oversized; `build()` keeps the first `n_per_category`
that survive filtering.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "exemplars"

#: Order matters: filtering keeps a prefix, so append rather than insert if you
#: want older runs to stay reproducible.
#:
#: A word must belong to exactly one of these categories, or its role in a
#: stream is undefined. That rules out in-pool collisions ("orange", "olive")
#: and also words whose other sense is a category here but absent from its pool
#: — "organ", "bat", "date", "kiwi", "coral", "bass", "horn". These are
#: excluded on purpose, including past position 12, since which words a pool
#: reaches is tokenizer-dependent. `_assert_disjoint` covers only the first
#: case.
POOLS: dict[str, list[str]] = {
    "animal": ["cat", "dog", "horse", "bear", "wolf", "fox", "deer", "sheep", "goat",
               "mouse", "frog", "duck", "rabbit", "tiger", "lion", "snake", "eagle",
               "shark", "whale", "owl", "pig", "cow", "crab"],
    "fruit": ["apple", "pear", "plum", "peach", "grape", "lemon", "lime", "melon",
              "cherry", "banana", "mango", "berry", "fig", "strawberry",
              "raspberry", "apricot", "papaya", "guava", "coconut", "pineapple"],
    "color": ["red", "blue", "green", "yellow", "black", "white", "brown", "purple",
              "pink", "grey", "gold", "silver", "beige", "teal", "navy",
              "maroon", "amber", "ivory", "violet"],
    "tool": ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "axe", "knife",
             "shovel", "rake", "clamp", "brush", "ruler", "sander",
             "mallet", "crowbar", "scraper", "vise", "awl"],
    "vehicle": ["car", "truck", "bus", "train", "plane", "boat", "ship", "bike",
                "van", "jeep", "tram", "taxi", "ferry", "yacht", "scooter", "tractor",
                "wagon", "sled", "canoe", "kayak", "glider", "rocket"],
    "body part": ["hand", "foot", "head", "arm", "leg", "knee", "elbow", "finger",
                  "thumb", "wrist", "ankle", "shoulder", "neck", "chest",
                  "nose", "ear", "eye", "mouth", "chin", "hip", "heel"],
    "country": ["France", "Spain", "Italy", "Japan", "China", "Brazil", "Canada",
                "Egypt", "India", "Mexico", "Greece", "Poland", "Norway", "Sweden",
                "Chile", "Peru", "Kenya", "Ghana", "Nepal", "Cuba", "Iran", "Iraq"],
    "instrument": ["piano", "guitar", "violin", "drum", "flute", "trumpet", "sax",
                   "viola", "accordion", "piccolo", "bell", "whistle", "synth",
                   "cello", "harp", "banjo", "clarinet", "oboe", "tuba",
                   "fiddle", "lute", "sitar", "gong", "chime"],
}


#: Category labels to read out alongside the exemplars. The Neuronpedia slice
#: for these prompts is dominated by category vocabulary -- `animals`, `colors`,
#: `categories` at counts 300-670 -- while the actual remembered words sit at
#: 60-110. So the task's category structure is what the J-space appears to
#: carry, and it is measurable with the tracked/untracked contrast D6 already
#: built into every stream.
#:
#: "body part" is absent: it is two words, so it has no single token to read.
CATEGORY_LABELS: dict[str, list[str]] = {
    "animal": ["animal", "animals"],
    "fruit": ["fruit", "fruits"],
    "color": ["color", "colors"],
    "tool": ["tool", "tools"],
    "vehicle": ["vehicle", "vehicles"],
    "country": ["country"],
    "instrument": ["instrument", "instruments"],
}

#: Category names that never appear in a stream: the floor for label presence,
#: the same role absent words play for exemplars.
ABSENT_LABELS: list[str] = ["flower", "metal", "sport", "building",
                            "flowers", "metals", "sports", "buildings"]


def is_single_token(tokenizer: Any, word: str) -> bool:
    """True if `word` is one token in the form it appears in a stream.

    Tested space-prefixed, since many tokenizers split a bare word but not its
    space-prefixed form.
    """
    return len(tokenizer.encode(f" {word}", add_special_tokens=False)) == 1


def pools_sha() -> str:
    """Content hash of POOLS; stored in the cache so an edit forces a rebuild."""
    payload = json.dumps(POOLS, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build(
    tokenizer: Any, alias: str, *, n_per_category: int = 12, force: bool = False
) -> dict[str, list[str]]:
    """Filter POOLS to single-token exemplars for one model.

    Cached at `data/exemplars/{alias}.json`, keyed on the alias plus a hash of
    POOLS and `n_per_category`, so editing a pool cannot leave a stale cache in
    place for anyone who forgets `--force`.
    """
    cache = CACHE_DIR / f"{alias}.json"
    key = {"pools_sha": pools_sha(), "n_per_category": n_per_category}
    if cache.exists() and not force:
        cached = json.loads(cache.read_text())
        if all(cached.get(k) == v for k, v in key.items()):
            return cached["exemplars"]

    out: dict[str, list[str]] = {}
    for category, pool in POOLS.items():
        kept = [
            w for w in pool
            # No exemplar may equal any category name.
            if w.lower() not in POOLS and w.lower() != category
            and is_single_token(tokenizer, w)
        ]
        if len(kept) < n_per_category:
            raise ValueError(
                f"{alias}: category {category!r} has only {len(kept)} single-token "
                f"exemplars, need {n_per_category}. Extend POOLS[{category!r}]."
            )
        out[category] = kept[:n_per_category]

    _assert_disjoint(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({**key, "alias": alias, "exemplars": out}, indent=2,
                                sort_keys=True))
    return out


def _assert_disjoint(exemplars: dict[str, list[str]]) -> None:
    """No word may belong to two categories."""
    seen: dict[str, str] = {}
    for category, words in exemplars.items():
        for w in words:
            if w in seen:
                raise ValueError(f"{w!r} is in both {seen[w]!r} and {category!r}")
            seen[w] = category
