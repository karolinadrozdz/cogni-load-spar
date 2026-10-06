"""single_cue on a fake model: read positions, row assembly, scoring, and the
analysis run over rows the real row builder made."""

import pandas as pd
import pytest
import torch
from conftest import CONFIG, POOL

from cogniload import experiment, prompts, single_cue

K = CONFIG["readout"]["primary_k"]
TAIL = CONFIG["stream"]["tail_guard"]


def test_positions_land_on_the_delimiters_and_rows_carry_roles(ctx):
    stream = experiment.streams(POOL, CONFIG, 4, 1)[0]
    rows, summary = single_cue.run_stream(ctx, stream)
    tok = ctx.model.tokenizer
    ids = tok.encode(prompts.build_recent(tok, stream.words, stream.tracked,
                                          stream.queried_category, enable_thinking=False))
    assert "," in tok.decode([ids[summary["in_stream_pos"]]])
    assert "." in tok.decode([ids[summary["stream_end_pos"]]])
    assert summary["answer_pos"] == len(ids) - 1

    df = pd.DataFrame(rows)
    assert set(df.readout_at) == {"in_stream", "stream_end", "answer"}
    assert set(df.lens) == {"jlens", "logit"} and df.role.notna().all()
    assert set(df[df.role == "target"].word) == set(stream.targets.values())
    assert {"absent", "label_tracked", "label_untracked", "label_absent"} <= set(df.role)


class _Tok:
    """Five tokens, space-prefixed as stream words appear."""

    _V = [" van", " Van", " bus", ",", " the"]

    def decode(self, ids):
        return "".join(self._V[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self._V.index(text)] if text in self._V else [0, 1]


@pytest.mark.parametrize("logits,expected,correct,rank", [
    ([5.0, 4.0, 3.0, 2.0, 1.0], "van", True, 1),
    ([5.0, 4.0, 3.0, 2.0, 1.0], "bus", False, 3),
    ([4.0, 5.0, 3.0, 2.0, 1.0], "van", True, 2),   # case and leading space are not errors
    ([4.0, 1.0, 1.0, 5.0, 1.0], "van", False, 2),  # a formatting token wins the argmax
])
def test_score_q1(logits, expected, correct, rank):
    s = experiment.score_q1(_Tok(), torch.tensor(logits), expected)
    assert (s["correct"], s["expected_rank"]) == (correct, rank)
    assert s["top5"].split(" | ")[0].startswith(s["answer"])


@pytest.mark.parametrize("shape", ["colon", "bulleted", "bare"])
def test_q2_parses_every_output_shape(shape):
    s = experiment.streams(POOL, CONFIG, 4, 1)[0]
    text = {"colon": ", ".join(f"{c}: {s.targets[c]}" for c in s.tracked),
            "bulleted": "\n".join(f"- {c}: {s.targets[c]}" for c in s.tracked),
            "bare": ", ".join(s.targets[c] for c in s.tracked)}[shape]
    assert all(r["parsed"] and r["correct"]
               for r in single_cue.score_q2(text, s.tracked, s.targets, s.words))


def test_q2_separates_unparseable_from_wrong():
    s = experiment.streams(POOL, CONFIG, 4, 1)[0]
    unreadable = single_cue.score_q2("I don't recall.", s.tracked, s.targets, s.words)
    assert not any(r["parsed"] or r["correct"] for r in unreadable)
    wrong = ", ".join(f"{c}: zebra" for c in s.tracked)
    assert all(r["parsed"] and not r["correct"]
               for r in single_cue.score_q2(wrong, s.tracked, s.targets, s.words))


# --- analysis -------------------------------------------------------------------


@pytest.fixture
def df(ctx):
    """Real rows with a planted contrast: in-band J-lens targets at rank 2, all else 900."""
    rows = [r for c_t in (2, 3, 4) for s in experiment.streams(POOL, CONFIG, c_t, 4)
            for r in single_cue.run_stream(ctx, s)[0]]
    df = pd.DataFrame(rows)
    planted = (df.role == "target") & (df.lens == "jlens") & df.in_band
    df["rank"] = df["rank_wordlike"] = planted.map({True: 2, False: 900})
    return df


def test_capacity_counts_only_non_queried_targets(df):
    cap = single_cue.capacity(df, K)
    for _, row in cap[(cap.lens == "jlens") & (cap.readout_at == "in_stream")].iterrows():
        assert row["mean"] == pytest.approx(row["c_t"] - 1)


def test_selectivity_separates_targets_from_the_matched_control(df):
    sel = single_cue.selectivity(df, K, TAIL)
    sel = sel[(sel.lens == "jlens") & (sel.readout_at == "in_stream")]
    assert (sel[sel.group == "target_non_queried"]["mean"] > 0.99).all()
    assert (sel[sel.group == "untracked_final"]["mean"] < 0.01).all()
    assert "absent" in set(sel.group)


def test_presence_takes_band_min_over_the_band_only(df):
    narrow = experiment.presence(df, K, band_only=True)
    assert len(narrow) == len(experiment.presence(df, K, band_only=False))
    assert narrow[(narrow.lens == "jlens") & (narrow.role == "target")]["present"].all()
    profile = single_cue.by_layer(df, K)
    eos = profile[profile.readout_at == "in_stream"]
    assert eos[eos.in_band]["target_non_queried"].mean() > 0.99
    assert eos[~eos.in_band]["target_non_queried"].mean() < 0.01


def test_floor_check_warns_only_when_both_rank_columns_are_low(df, capsys):
    """On Qwen3.6-27B the two columns read 0.29 vs 0.74; gating on one raised a false alarm."""
    low = df.assign(rank=900)
    single_cue.floor_check(low, K)
    out = capsys.readouterr().out
    assert "[rank " in out and "[rank_wordlike" in out and "BELOW 50%" not in out
    single_cue.floor_check(low.assign(rank_wordlike=900), K)
    assert "BELOW 50% ON BOTH" in capsys.readouterr().out


def test_headline_reports_both_rank_columns_and_every_group(df):
    h = single_cue.headline(df, K)
    assert {"rank", "rank_wordlike"} <= set(h["present"].columns)
    assert {"absent", "label_tracked", "label_absent"} <= set(h["group"])
    assert set(h["readout_at"]) == {"in_stream", "stream_end", "answer"}
