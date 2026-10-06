"""Keep-track streams (Yntema 1963): words from several categories, some of them tracked.

Pure and seeded, so streams are reproducible without a tokenizer or a GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    position: int
    word: str
    category: str
    role: str  # "target" | "replaced" | "untracked"
    #: Last occurrence of its category. A target is always final; an untracked
    #: final differs from a target only in not being tracked, so it is the
    #: matched control.
    is_category_final: bool = False


@dataclass(frozen=True)
class Stream:
    stream_id: int
    c_t: int
    tracked: tuple[str, ...]
    items: tuple[Item, ...]
    #: tracked category -> its final word (what the model must report).
    targets: dict[str, str]
    #: The category asked about, drawn from the stream's RNG so the
    #: queried/non-queried split is reproducible.
    queried_category: str

    @property
    def words(self) -> list[str]:
        return [i.word for i in self.items]

    def to_rows(self) -> list[dict]:
        """One row per item; recency 1 is the final item. Words are unique within a stream.

        `stream_pos` is 1-indexed; readout rows carry a 0-indexed `token_pos`.
        """
        n = len(self.items)
        return [
            {"stream_id": self.stream_id, "c_t": self.c_t,
             "queried_category": self.queried_category, "stream_pos": i.position,
             "recency": n - i.position + 1, "word": i.word, "category": i.category,
             "role": i.role, "is_category_final": i.is_category_final,
             "is_queried": i.category == self.queried_category}
            for i in self.items
        ]


def generate(exemplars: dict[str, list[str]], *, c_t: int, n_streams: int = 100, seed: int = 0,
             updates_per_tracked: int = 3, tail_guard: int = 3) -> list[Stream]:
    """`n_streams` seeded streams with `c_t` tracked categories.

    Every category appears `updates_per_tracked` times, tracked or not, so only
    the instruction varies with `c_t`. No target falls in the last `tail_guard`
    positions, so the answer is never just the most recent word.
    """
    n_categories = len(exemplars)
    n_distractor_items = (n_categories - c_t) * updates_per_tracked
    if c_t >= n_categories:
        raise ValueError(f"c_t={c_t} leaves no distractor categories out of {n_categories}; "
                         f"selectivity needs untracked items")
    if n_distractor_items < tail_guard:
        raise ValueError(f"c_t={c_t} of {n_categories} categories leaves {n_distractor_items} "
                         f"distractor items, fewer than tail_guard={tail_guard}")
    if any(len(w) < updates_per_tracked for w in exemplars.values()):
        raise ValueError(f"every category needs >= {updates_per_tracked} exemplars")

    rng = random.Random(seed)
    return [_one_stream(exemplars, rng, i, c_t, updates_per_tracked, tail_guard)
            for i in range(n_streams)]


def _one_stream(exemplars, rng, stream_id, c_t, updates_per_tracked, tail_guard) -> Stream:
    categories = rng.sample(sorted(exemplars), len(exemplars))
    tracked, distractor_cats = categories[:c_t], categories[c_t:]

    def draw(cats):
        return [(c, w) for c in cats for w in rng.sample(exemplars[c], updates_per_tracked)]

    tracked_pairs, distractor_pairs = draw(tracked), draw(distractor_cats)
    # A tracked item in the tail would be its category's last, hence a target,
    # so the tail is built from distractors directly. Rejection sampling would
    # accept ~1% of orderings at c_t=6.
    rng.shuffle(distractor_pairs)
    tail, rest = distractor_pairs[:tail_guard], distractor_pairs[tail_guard:]
    body = tracked_pairs + rest
    rng.shuffle(body)

    items = _assign_roles(body + tail, set(tracked))
    targets = {i.category: i.word for i in items if i.role == "target"}
    return Stream(stream_id, c_t, tuple(tracked), tuple(items), targets,
                  queried_category=rng.choice(tracked))


def _assign_roles(pairs: list[tuple[str, str]], tracked: set[str]) -> list[Item]:
    """Label tracked and untracked categories the same way; `tracked` only names the role."""
    last = {category: position for position, (category, _) in enumerate(pairs, 1)}
    items = []
    for position, (category, word) in enumerate(pairs, 1):
        final = last[category] == position
        role = ("target" if final else "replaced") if category in tracked else "untracked"
        items.append(Item(position, word, category, role, final))
    return items
