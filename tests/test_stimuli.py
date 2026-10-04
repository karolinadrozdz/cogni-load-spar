"""Invariants that make streams interpretable. Pure — no model, no GPU."""

from collections import Counter

import pytest

from cogniload import stimuli
from cogniload.exemplars import POOLS, _assert_disjoint

C_TS = [2, 4, 6]
N_CATEGORIES, UPDATES, TAIL = 8, 3, 3
LENGTH = N_CATEGORIES * UPDATES


@pytest.fixture(scope="module")
def ex():
    pools = {c: w[:10] for c, w in POOLS.items()}  # matches exp1.yaml
    _assert_disjoint(pools)
    return pools


def test_pools_are_disjoint_and_deep_enough():
    _assert_disjoint(POOLS)
    # Pools must survive single-token filtering with 12 to spare.
    assert all(len(w) >= 12 for w in POOLS.values())
    assert not set(POOLS) & {w for ws in POOLS.values() for w in ws}


@pytest.mark.parametrize("c_t", C_TS)
def test_shape(ex, c_t):
    for s in stimuli.generate(ex, c_t=c_t, n_streams=50, seed=0):
        assert len(s.items) == LENGTH
        assert len(s.targets) == c_t
        assert len(set(s.words)) == LENGTH, "a word may not repeat within a stream"


@pytest.mark.parametrize("c_t", C_TS)
def test_each_tracked_category_is_updated_the_same_number_of_times(ex, c_t):
    """Eviction (§4.3) compares replaced items across C_t, so the number of
    replacements per category must not itself vary with C_t."""
    for s in stimuli.generate(ex, c_t=c_t, n_streams=50, seed=0):
        for cat in s.tracked:
            occ = [i for i in s.items if i.category == cat]
            assert len(occ) == UPDATES
            assert occ[-1].role == "target"
            assert [i.role for i in occ[:-1]] == ["replaced"] * (UPDATES - 1)
            # every replaced item points at the item that superseded it
            for earlier, later in zip(occ, occ[1:]):
                assert earlier.replaced_at == later.position


@pytest.mark.parametrize("c_t", C_TS)
def test_no_target_in_tail(ex, c_t):
    """§1's mandatory rule: the answer is never just the most recent word."""
    for s in stimuli.generate(ex, c_t=c_t, n_streams=50, seed=0):
        tail = s.items[-TAIL:]
        assert all(i.role == "untracked" for i in tail)
        assert min(r["recency"] for r in s.to_rows() if r["role"] == "target") > TAIL


@pytest.mark.parametrize("c_t", C_TS)
def test_untracked_items_overlap_targets_in_recency(ex, c_t):
    """Selectivity (§4.4) matches targets to untracked items on recency, which
    requires the two to actually overlap."""
    rows = [r for s in stimuli.generate(ex, c_t=c_t, n_streams=50, seed=0)
            for r in s.to_rows()]
    tgt = {r["recency"] for r in rows if r["role"] == "target"}
    unt = {r["recency"] for r in rows if r["role"] == "untracked"}
    assert len(tgt & unt) >= 10


def test_seeded_and_reproducible(ex):
    a = stimuli.generate(ex, c_t=4, n_streams=10, seed=7)
    b = stimuli.generate(ex, c_t=4, n_streams=10, seed=7)
    c = stimuli.generate(ex, c_t=4, n_streams=10, seed=8)
    assert [s.words for s in a] == [s.words for s in b]
    assert [s.words for s in a] != [s.words for s in c]


def test_every_category_appears_equally_often(ex):
    """The selectivity contrast must be instruction, not category frequency:
    an untracked category may not appear more often than a tracked one."""
    for c_t in C_TS:
        for s in stimuli.generate(ex, c_t=c_t, n_streams=30, seed=0):
            counts = Counter(i.category for i in s.items)
            assert set(counts.values()) == {UPDATES}
            assert len(counts) == N_CATEGORIES


def test_stream_structure_is_identical_across_c_t(ex):
    """Only the tracked set may vary with C_t, not the stream's shape."""
    shapes = {
        c_t: Counter(
            len(s.items) for s in stimuli.generate(ex, c_t=c_t, n_streams=30, seed=0)
        )
        for c_t in C_TS
    }
    assert len({tuple(sorted(v.items())) for v in shapes.values()}) == 1


def test_refuses_impossible_configuration(ex):
    with pytest.raises(ValueError, match="no distractor categories"):
        stimuli.generate(ex, c_t=len(ex), n_streams=1)
    with pytest.raises(ValueError, match="exemplars"):
        stimuli.generate({k: v[:2] for k, v in ex.items()}, c_t=2, n_streams=1)


def test_queried_category_is_seeded_and_tracked(ex):
    """Picking the query at call time would leave the queried/non-queried split
    unrecoverable, and that split is what makes the headline measure mean
    anything."""
    for c_t in C_TS:
        a = stimuli.generate(ex, c_t=c_t, n_streams=30, seed=1)
        b = stimuli.generate(ex, c_t=c_t, n_streams=30, seed=1)
        assert [s.queried_category for s in a] == [s.queried_category for s in b]
        for s in a:
            assert s.queried_category in s.tracked
            assert sum(r["is_queried"] for r in s.to_rows()) == UPDATES
    # and it varies across streams rather than always being tracked[0]
    picks = [s.queried_category == s.tracked[0]
             for s in stimuli.generate(ex, c_t=6, n_streams=100, seed=1)]
    assert 0 < sum(picks) < len(picks)


def test_untracked_final_is_the_matched_control(ex):
    """A target is its category's last occurrence by definition; an untracked
    item is final only 1 time in 3. Without the flag, target-vs-untracked
    conflates `tracked` with `not-superseded`."""
    for c_t in C_TS:
        for s in stimuli.generate(ex, c_t=c_t, n_streams=30, seed=0):
            by_cat = {}
            for i in s.items:
                by_cat.setdefault(i.category, []).append(i)
            for category, items in by_cat.items():
                finals = [i for i in items if i.is_category_final]
                assert len(finals) == 1
                assert finals[0].position == max(i.position for i in items)
                assert finals[0].replaced_at is None
                # every non-final item points at its successor, tracked or not
                for i in items:
                    if not i.is_category_final:
                        assert i.replaced_at is not None
            # targets are exactly the tracked finals
            assert ({i.category for i in s.items if i.role == "target"}
                    == set(s.tracked))


def test_no_column_name_collides_with_readout_rows(ex):
    """Stream rows and Readout rows get merged; a shared `position` name would
    join a 1-indexed stream index against a 0-indexed token index."""
    row = stimuli.generate(ex, c_t=2, n_streams=1, seed=0)[0].to_rows()[0]
    assert "position" not in row and "stream_pos" in row
    assert "word" in row and "stream_id" in row  # the actual join key


def test_every_pool_has_enough_single_token_words():
    """Catches a pool edit that breaks the tokenizer budget here, rather than
    after a GPU round trip. The screening is vendored from the real Qwen vocab
    (identical for dev and prod); `exemplars.build` is the live check.
    """
    import json
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "qwen_single_token.json").read_text()
    )
    single = fixture["single_token"]
    unknown = {w for ws in POOLS.values() for w in ws} - set(single)
    assert not unknown, f"not in the vendored screening, re-generate it: {unknown}"

    need = 10  # configs/exp1.yaml phase1.stream.n_exemplars_per_category
    for category, words in POOLS.items():
        survivors = [w for w in words if single[w]]
        assert len(survivors) >= need, (
            f"{category!r} has {len(survivors)} single-token words, needs {need}"
        )
