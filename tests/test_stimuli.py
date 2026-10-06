"""Invariants that make streams interpretable. Pure: no model, no GPU."""

import json
from collections import Counter
from pathlib import Path

import pytest
from conftest import CONFIG, POOL

from cogniload import stimuli
from cogniload.exemplars import POOLS, _assert_disjoint

C_TS = [2, 4, 6]
N_CATEGORIES, UPDATES, TAIL = 8, 3, 3


def test_pools_are_disjoint_and_never_contain_a_category_name():
    _assert_disjoint(POOLS)
    assert not set(POOLS) & {w for ws in POOLS.values() for w in ws}


def test_every_pool_has_enough_single_token_words():
    """Catches a pool edit that breaks the tokenizer budget before a GPU run. The
    screening is vendored from the Qwen vocab, identical for dev and prod."""
    fixture = Path(__file__).parent / "fixtures" / "qwen_single_token.json"
    single = json.loads(fixture.read_text())["single_token"]
    unknown = {w for ws in POOLS.values() for w in ws} - set(single)
    assert not unknown, f"not in the vendored screening, re-generate it: {unknown}"
    need = CONFIG["stream"]["n_exemplars_per_category"]
    for category, words in POOLS.items():
        assert sum(single[w] for w in words) >= need, category


@pytest.mark.parametrize("c_t", C_TS)
def test_shape_and_equal_category_frequency(c_t):
    """Only the instruction may vary with C_t: every category appears equally
    often, and each tracked one is replaced the same number of times."""
    for s in stimuli.generate(POOL, c_t=c_t, n_streams=50, seed=0):
        assert len(set(s.words)) == len(s.items) == N_CATEGORIES * UPDATES
        assert set(Counter(i.category for i in s.items).values()) == {UPDATES}
        assert len(s.targets) == c_t
        for cat in s.tracked:
            roles = [i.role for i in s.items if i.category == cat]
            assert roles == ["replaced"] * (UPDATES - 1) + ["target"]


@pytest.mark.parametrize("c_t", C_TS)
def test_no_target_in_tail(c_t):
    """The answer is never just the most recent word."""
    for s in stimuli.generate(POOL, c_t=c_t, n_streams=50, seed=0):
        assert all(i.role == "untracked" for i in s.items[-TAIL:])
        assert min(r["recency"] for r in s.to_rows() if r["role"] == "target") > TAIL


@pytest.mark.parametrize("c_t", C_TS)
def test_untracked_items_overlap_targets_in_recency(c_t):
    """Selectivity matches targets to untracked items on recency, so the two must overlap."""
    rows = [r for s in stimuli.generate(POOL, c_t=c_t, n_streams=50, seed=0) for r in s.to_rows()]
    tgt = {r["recency"] for r in rows if r["role"] == "target"}
    unt = {r["recency"] for r in rows if r["role"] == "untracked"}
    assert len(tgt & unt) >= 10


def test_seeded_and_reproducible():
    a = stimuli.generate(POOL, c_t=4, n_streams=10, seed=7)
    assert [s.words for s in a] == [s.words for s in stimuli.generate(POOL, c_t=4, n_streams=10, seed=7)]
    assert [s.words for s in a] != [s.words for s in stimuli.generate(POOL, c_t=4, n_streams=10, seed=8)]


def test_refuses_impossible_configuration():
    with pytest.raises(ValueError, match="no distractor categories"):
        stimuli.generate(POOL, c_t=len(POOL), n_streams=1)
    with pytest.raises(ValueError, match="exemplars"):
        stimuli.generate({k: v[:2] for k, v in POOL.items()}, c_t=2, n_streams=1)


def test_queried_category_is_seeded_tracked_and_varies():
    """The queried/non-queried split is what the headline measure rests on."""
    for c_t in C_TS:
        a = stimuli.generate(POOL, c_t=c_t, n_streams=30, seed=1)
        b = stimuli.generate(POOL, c_t=c_t, n_streams=30, seed=1)
        assert [s.queried_category for s in a] == [s.queried_category for s in b]
        assert all(s.queried_category in s.tracked for s in a)
    picks = [s.queried_category == s.tracked[0]
             for s in stimuli.generate(POOL, c_t=6, n_streams=100, seed=1)]
    assert 0 < sum(picks) < len(picks)


def test_untracked_final_is_the_matched_control():
    """Exactly one final per category, at its last position; targets are the tracked finals."""
    for c_t in C_TS:
        for s in stimuli.generate(POOL, c_t=c_t, n_streams=30, seed=0):
            for category in {i.category for i in s.items}:
                items = [i for i in s.items if i.category == category]
                assert [i.is_category_final for i in items] == [False] * (UPDATES - 1) + [True]
            assert {i.category for i in s.items if i.role == "target"} == set(s.tracked)
