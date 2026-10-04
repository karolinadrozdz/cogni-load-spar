"""Keep-track stream generation (Yntema 1963; Miyake et al. 2000 "updating").

A stream of words from several categories; the model must report the most
recent word from each *tracked* category. Pure and seeded — takes exemplar
pools as input, so it runs without a tokenizer or a GPU.

Design is recorded in DECISIONS.md D6.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

Role = str  # "target" | "replaced" | "untracked"


@dataclass(frozen=True)
class Item:
    position: int
    word: str
    category: str
    role: Role
    #: Position of the item that superseded this one. Set for untracked items
    #: too, so the eviction contrast has a "went stale but was never
    #: task-relevant" baseline.
    replaced_at: int | None = None
    #: Last occurrence of its category. A target is always final; an untracked
    #: item is final 1 time in 3, so the untracked final is the control that
    #: differs from a target only in not having been named.
    is_category_final: bool = False


@dataclass(frozen=True)
class Stream:
    stream_id: int
    seed: int
    c_t: int
    tracked: tuple[str, ...]
    items: tuple[Item, ...]
    #: tracked category -> its final exemplar (what the model must report).
    targets: dict[str, str]
    #: The category Q1 asks about, drawn from this stream's RNG so the
    #: queried/non-queried split is reproducible and recorded. At the answer
    #: position the queried target is both task state and the literal next
    #: token, so only the other C_t - 1 targets discriminate (DECISIONS.md D15).
    queried_category: str

    @property
    def words(self) -> list[str]:
        return [i.word for i in self.items]

    def to_rows(self) -> list[dict]:
        """One row per item; recency 1 is the final item.

        Join to readout rows on `["stream_id", "word"]` — words are unique
        within a stream. The column is `stream_pos` (1-indexed) because readout
        rows carry `token_pos` (0-indexed), and one shared `position` name
        would merge the two silently.
        """
        n = len(self.items)
        return [
            {
                "stream_id": self.stream_id,
                "c_t": self.c_t,
                "queried_category": self.queried_category,
                "stream_pos": i.position,
                "recency": n - i.position + 1,
                "word": i.word,
                "category": i.category,
                "role": i.role,
                "replaced_at": i.replaced_at,
                "is_category_final": i.is_category_final,
                "is_queried": i.category == self.queried_category,
            }
            for i in self.items
        ]


def generate(
    exemplars: dict[str, list[str]],
    *,
    c_t: int,
    n_streams: int = 100,
    seed: int = 0,
    updates_per_tracked: int = 3,
    tail_guard: int = 3,
) -> list[Stream]:
    """Generate `n_streams` seeded keep-track streams at one tracked-set size.

    Every category appears exactly `updates_per_tracked` times whether tracked
    or not, so stream length and per-category frequency are the same at every
    `C_t` and only the instruction varies. Pass a subset of `exemplars` to use
    fewer categories.

    `tail_guard` keeps no target in the final N positions, so the answer is
    never just the most recent word.
    """
    n_categories = len(exemplars)
    n_distractor_items = (n_categories - c_t) * updates_per_tracked
    if c_t >= n_categories:
        raise ValueError(
            f"c_t={c_t} leaves no distractor categories out of {n_categories}; "
            f"selectivity needs untracked items"
        )
    if n_distractor_items < tail_guard:
        raise ValueError(
            f"c_t={c_t} of {n_categories} categories leaves {n_distractor_items} "
            f"distractor items, fewer than tail_guard={tail_guard}"
        )
    if any(len(w) < updates_per_tracked for w in exemplars.values()):
        raise ValueError(f"every category needs >= {updates_per_tracked} exemplars")

    rng = random.Random(seed)
    return [
        _one_stream(
            exemplars, rng, stream_id=i, seed=seed, c_t=c_t,
            updates_per_tracked=updates_per_tracked, tail_guard=tail_guard,
        )
        for i in range(n_streams)
    ]


def _one_stream(
    exemplars, rng, *, stream_id, seed, c_t, updates_per_tracked, tail_guard,
) -> Stream:
    categories = rng.sample(sorted(exemplars), len(exemplars))
    tracked, distractor_cats = categories[:c_t], categories[c_t:]

    def draw(cats):
        return [(c, w) for c in cats for w in rng.sample(exemplars[c], updates_per_tracked)]

    tracked_pairs, distractor_pairs = draw(tracked), draw(distractor_cats)

    # A tracked item in the tail would be its category's last occurrence, hence
    # a target — so "no target in the tail" means "the tail is all distractors".
    # Constructing that is uniform over admissible orderings and avoids
    # rejection sampling a ~1% acceptance region at c_t=6.
    rng.shuffle(distractor_pairs)
    tail, rest = distractor_pairs[:tail_guard], distractor_pairs[tail_guard:]
    body = tracked_pairs + rest
    rng.shuffle(body)

    items = _assign_roles(body + tail, set(tracked))
    targets = {i.category: i.word for i in items if i.role == "target"}
    return Stream(
        stream_id, seed, c_t, tuple(tracked), tuple(items), targets,
        queried_category=rng.choice(tracked),
    )


def _assign_roles(pairs: list[tuple[str, str]], tracked: set[str]) -> list[Item]:
    """Label each position from its category's occurrence list.

    Tracked and untracked categories are labelled identically — last occurrence
    is final, earlier ones point at their successor — and `tracked` only picks
    the role name. That symmetry is what makes the untracked final a matched
    control for a target.
    """
    occurrences: dict[str, list[int]] = {}
    for position, (category, _) in enumerate(pairs, 1):
        occurrences.setdefault(category, []).append(position)

    labels: dict[int, tuple[Role, int | None, bool]] = {}
    for category, positions in occurrences.items():
        is_tracked = category in tracked
        for earlier, later in zip(positions, positions[1:]):
            labels[earlier] = ("replaced" if is_tracked else "untracked", later, False)
        labels[positions[-1]] = ("target" if is_tracked else "untracked", None, True)

    return [
        Item(position, word, category, *labels[position])
        for position, (category, word) in enumerate(pairs, 1)
    ]

