"""Category exemplar pools, filtered to single tokens for the model's tokenizer."""

from __future__ import annotations

from typing import Any

#: Filtering keeps a prefix of each pool, so append rather than insert, or
#: earlier runs stop being reproducible.
#:
#: A word must belong to exactly one of these categories, or its role in a
#: stream is undefined. That rules out in-pool collisions ("orange", "olive")
#: and words whose other sense is a category here but absent from its pool:
#: "organ", "bat", "date", "kiwi", "coral", "bass", "horn". They stay out even
#: past position 12, since how far into a pool filtering reaches depends on the
#: tokenizer. `_assert_disjoint` covers only the first case.
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

#: Category labels, read alongside the exemplars: Neuronpedia shows category
#: words dominating the J-space for these prompts. "body part" is two tokens,
#: so it has no label.
CATEGORY_LABELS: dict[str, list[str]] = {
    "animal": ["animal", "animals"],
    "fruit": ["fruit", "fruits"],
    "color": ["color", "colors"],
    "tool": ["tool", "tools"],
    "vehicle": ["vehicle", "vehicles"],
    "country": ["country"],
    "instrument": ["instrument", "instruments"],
}

#: Category names that never appear in a stream: the floor for label presence.
ABSENT_LABELS: list[str] = ["flower", "metal", "sport", "building",
                            "flowers", "metals", "sports", "buildings"]


def build(tokenizer: Any, n_per_category: int) -> dict[str, list[str]]:
    """The first `n_per_category` words of each pool that are one token after a space."""
    out: dict[str, list[str]] = {}
    for category, pool in POOLS.items():
        kept = [w for w in pool
                # No exemplar may equal any category name.
                if w.lower() not in POOLS and w.lower() != category
                and len(tokenizer.encode(f" {w}", add_special_tokens=False)) == 1]
        if len(kept) < n_per_category:
            raise ValueError(f"category {category!r} has only {len(kept)} single-token "
                             f"exemplars, need {n_per_category}; extend POOLS[{category!r}]")
        out[category] = kept[:n_per_category]
    _assert_disjoint(out)
    return out


def _assert_disjoint(exemplars: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for category, words in exemplars.items():
        for w in words:
            if w in seen:
                raise ValueError(f"{w!r} is in both {seen[w]!r} and {category!r}")
            seen[w] = category
