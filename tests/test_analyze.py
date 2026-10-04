"""Analysis over synthetic rows with the exact phase1 schema.

Plants a known contrast — non-queried targets present in band, everything else
absent — and checks each entry point recovers it. Catches a broken groupby or
filter before a run is paid for.
"""

import sys
import types

import pandas as pd
import pytest
import torch

from cogniload import analyze, stimuli
from cogniload.exemplars import POOLS

BAND = (23, 27)
LAYERS = list(range(8, 27))
ABSENT = ["zebra", "turnip"]


@pytest.fixture(autouse=True)
def fake_jlens(monkeypatch):
    vis = types.ModuleType("jlens.vis")
    vis._meaningful_token_mask = lambda t, v, d: torch.ones(v, dtype=torch.bool, device=d)
    pkg = types.ModuleType("jlens")
    pkg.vis = vis
    monkeypatch.setitem(sys.modules, "jlens", pkg)
    monkeypatch.setitem(sys.modules, "jlens.vis", vis)


def _rank(info, lens, layer) -> int:
    """In-band J-lens: targets at rank 2, everything else out of reach."""
    if lens == "logit" or not BAND[0] <= layer < BAND[1]:
        return 900
    return 2 if info["role"] == "target" else 800


@pytest.fixture
def df():
    pool = {c: w[:12] for c, w in POOLS.items()}
    rows = []
    for c_t in (2, 3, 4):
        for stream in stimuli.generate(pool, c_t=c_t, n_streams=4, seed=c_t):
            meta = {r["word"]: r for r in stream.to_rows()}
            blank = {
                "stream_id": stream.stream_id, "c_t": stream.c_t,
                "queried_category": stream.queried_category, "stream_pos": None,
                "recency": None, "category": None, "role": "absent",
                "replaced_at": None, "is_category_final": False, "is_queried": False,
            }
            for lens in ("jlens", "logit"):
                for word in stream.words + ABSENT:
                    info = meta.get(word, {**blank, "word": word})
                    for layer in LAYERS:
                        for pos, at in ((70, "in_stream"), (110, "answer")):
                            rank = _rank(info, lens, layer)
                            rows.append({
                                **info, "word": word, "token_id": 1, "layer": layer,
                                "token_pos": pos, "rank": rank, "rank_wordlike": rank,
                                "coefficient": 0.0, "lens": lens, "readout_at": at,
                                "in_band": BAND[0] <= layer < BAND[1],
                                "q2_parsed": True, "q2_correct": True,
                            })
    return pd.DataFrame(rows)


def test_capacity_counts_only_non_queried_targets(df):
    cap = analyze.capacity(df)
    jlens = cap[(cap.lens == "jlens") & (cap.readout_at == "in_stream")]
    # every non-queried target planted present, so the mean is c_t - 1
    for _, row in jlens.iterrows():
        assert row["mean"] == pytest.approx(row["c_t"] - 1)


def test_selectivity_separates_targets_from_the_matched_control(df):
    sel = analyze.selectivity(df)
    sel = sel[(sel.lens == "jlens") & (sel.readout_at == "in_stream")]
    assert (sel[sel.group == "target_non_queried"]["mean"] > 0.99).all()
    assert (sel[sel.group == "untracked_final"]["mean"] < 0.01).all()
    assert "absent" in set(sel.group), "the chance floor must be reported"


def test_matched_control_thins_out_as_c_t_grows(df):
    """Documents a real limit: untracked categories shrink as 8 - C_t, and
    tail_guard eats their final occurrences, so the matched control nearly
    vanishes at high C_t."""
    sel = analyze.selectivity(df)
    counts = (sel[(sel.lens == "jlens") & (sel.readout_at == "in_stream")
                  & (sel.group == "untracked_final")]
              .set_index("c_t")["count"])
    assert counts.loc[2] > counts.loc[4]


def test_presence_takes_band_min_over_the_band_only(df):
    """Out-of-band layers are recorded but must not enter the headline number."""
    narrow = analyze.presence(df, band_only=True)
    wide = analyze.presence(df, band_only=False)
    assert len(narrow) == len(wide), "row count must not depend on band_only"
    jl = narrow[(narrow.lens == "jlens") & (narrow.role == "target")]
    assert jl["present"].all()


def test_by_layer_localises_presence(df):
    """The per-layer profile is what would reveal a band in the wrong place."""
    profile = analyze.by_layer(df)
    eos = profile[profile.readout_at == "in_stream"]
    assert eos[eos.in_band]["target_non_queried"].mean() > 0.99
    assert eos[~eos.in_band]["target_non_queried"].mean() < 0.01
    assert set(LAYERS) == set(profile.layer.unique())


def test_floor_check_reports_both_rank_columns(df, capsys):
    """Which column carries the signal is model-dependent: on Qwen3.6-27B the
    queried target reads 0.29 on full-vocab rank and 0.74 on word-like, so
    gating on one column alone raised a false alarm."""
    analyze.floor_check(df)
    out = capsys.readouterr().out
    assert "[rank " in out and "[rank_wordlike" in out
    assert out.count("queried target at answer") == 2
    assert "absent-word floor" in out


def test_floor_check_warns_only_when_both_columns_are_low(df, capsys):
    low = df.copy()
    low["rank"] = 900
    analyze.floor_check(low)                     # rank_wordlike still fine
    assert "BELOW 50% ON BOTH" not in capsys.readouterr().out
    low["rank_wordlike"] = 900
    analyze.floor_check(low)
    assert "BELOW 50% ON BOTH" in capsys.readouterr().out


def test_headline_reports_both_rank_columns(df):
    """rank_wordlike is what Neuronpedia displays; the full-vocab column can
    hide a concept behind punctuation on Qwen. Both must be reported."""
    h = analyze.headline(df)
    assert {"rank", "rank_wordlike"} <= set(h["present"].columns)
    assert "absent" in set(h["group"]), "the chance floor must be in the table"
    assert set(h["readout_at"]) <= {"in_stream", "stream_end", "answer"}


def test_wordlike_can_differ_from_full_vocab(df):
    """Make the two columns disagree and check the analysis follows."""
    d = df.copy()
    d["rank_wordlike"] = 2  # everything word-like-present, nothing full-vocab
    d.loc[d.lens == "jlens", "rank"] = 900
    full = analyze.presence(d, rank_col="rank")
    word = analyze.presence(d, rank_col="rank_wordlike")
    assert not full[full.lens == "jlens"]["present"].any()
    assert word["present"].all()


# --- probe persistence profile -----------------------------------------------


def test_decay_recovers_a_known_persistence_window():
    """The persistence profile is the eviction measure done cleanly: the same
    item compared with itself at increasing distance. Plant a 3-token window
    and check the buckets follow."""
    from cogniload import probe, readout

    n_pos = 30
    rank = torch.full((2, 1, n_pos), 900)
    for i, own in enumerate((5, 10)):
        for p in range(own, min(own + 4, n_pos)):
            rank[i, 0, p] = 1
    r = readout.Readout(
        token_ids=[1, 2], layers=[23], positions=list(range(n_pos)),
        rank=rank, rank_wordlike=rank, coefficient=torch.zeros(2, 1, n_pos),
        model_logits=torch.zeros(1, 10),
    )
    by_bucket = {d["bucket"]: d["present"] for d in probe._decay(r, [5, 10])}
    assert by_bucket["0 (own)"] == 1.0
    assert by_bucket["1-2"] == 1.0
    assert by_bucket["6-10"] == 0.0
    assert by_bucket["11-20"] == 0.0
