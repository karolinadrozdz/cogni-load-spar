"""End-to-end plumbing, with a fake model. No GPU, no weights, no jlens.

Covers the seams that only break at run time: locating the end-of-stream
token, merging readout rows against stream metadata, scoring Q1 from the
readout's own `model_logits`, and parsing every Q2 output shape.
"""

import re
import sys
import types

import pandas as pd
import pytest
import torch

from cogniload import align, prompts, readout, scoring, stimuli
from cogniload.exemplars import POOLS

from test_registry_and_prompts import TEMPLATE, _Tokenizer


@pytest.fixture(autouse=True)
def fake_jlens(monkeypatch):
    """readout imports jlens.vis lazily; stub the vocab mask."""
    vis = types.ModuleType("jlens.vis")
    vis._meaningful_token_mask = lambda tok, vocab, device: torch.ones(
        vocab, dtype=torch.bool, device=device
    )
    pkg = types.ModuleType("jlens")
    pkg.vis = vis
    monkeypatch.setitem(sys.modules, "jlens", pkg)
    monkeypatch.setitem(sys.modules, "jlens.vis", vis)


class _FakeTokenizer(_Tokenizer):
    """Renders the real chat template; tokenises on whitespace with offsets."""

    def __init__(self):
        super().__init__(TEMPLATE.read_text())
        self.vocab: dict[str, int] = {}
        self.eos_token_id = 0

    def _spans(self, text):
        out = []
        for m in re.finditer(r"\S+|\s+", text):
            self.vocab.setdefault(m.group(), len(self.vocab) + 1)
            out.append((self.vocab[m.group()], m.start(), m.end()))
        return out

    def __call__(self, text, **kw):
        spans = self._spans(text)
        return {"input_ids": [i for i, _, _ in spans],
                "offset_mapping": [(s, e) for _, s, e in spans]}

    def encode(self, text, add_special_tokens=False):
        return [i for i, _, _ in self._spans(text)]

    def decode(self, ids, **kw):
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse.get(i, "?") for i in ids)


@pytest.fixture
def tok():
    pytest.importorskip("jinja2")
    return _FakeTokenizer()


@pytest.fixture
def stream():
    pool = {c: w[:12] for c, w in POOLS.items()}
    return stimuli.generate(pool, c_t=4, n_streams=1, seed=0)[0]


@pytest.fixture
def q1_text(tok, stream):
    return prompts.build_q1(tok, stream.words, stream.tracked,
                            stream.queried_category, enable_thinking=False)


def test_delimiter_after_lands_on_the_delimiter(tok, stream, q1_text):
    """Reading at the last stream word instead would make that word trivially
    top-ranked, and tail_guard guarantees it is always untracked."""
    index = align.delimiter_after(tok, q1_text, stream.words[-1])
    ids = tok.encode(q1_text)
    assert "." in tok.decode([ids[index]])
    assert index < len(ids) - 1, "the question text must follow the stream"


def test_delimiter_after_refuses_an_ambiguous_word(tok, stream, q1_text):
    with pytest.raises(ValueError, match="0 times"):
        align.delimiter_after(tok, q1_text, "notinthestream")


def _readout(stream, positions, n_items=3):
    shape = (n_items, 2, len(positions))
    return readout.Readout(
        token_ids=list(range(1, n_items + 1)), layers=[10, 11], positions=positions,
        rank=torch.randint(1, 50, shape), rank_wordlike=torch.randint(1, 50, shape),
        coefficient=torch.randn(shape), model_logits=torch.randn(len(positions), 100),
    )


def test_rows_join_on_word_without_a_position_collision(stream):
    """Both row sets used to emit `position`, meaning a 1-indexed stream index
    in one and a 0-indexed token index in the other."""
    meta = pd.DataFrame(stream.to_rows())
    rows = pd.DataFrame(_readout(stream, [74, 112]).to_rows(
        words=stream.words[:3], lens="jlens"))
    assert "position" not in meta.columns and "position" not in rows.columns

    merged = rows.merge(meta, on="word", how="left", validate="many_to_one")
    assert merged["role"].notna().all()
    assert {"stream_pos", "token_pos"} <= set(merged.columns)


def test_readout_rows_need_per_item_words(stream):
    with pytest.raises(ValueError, match="words for"):
        _readout(stream, [1]).to_rows(words=["only-one"])


def test_q1_is_scored_from_the_readouts_own_logits(tok, stream):
    """Scoring from a second forward pass would not be guaranteed consistent
    with the measured ranks."""
    result = _readout(stream, [74, 112])
    score = scoring.score_q1(tok, result.model_logits[-1],
                             stream.targets[stream.queried_category])
    assert score.expected == stream.targets[stream.queried_category]
    assert len(score.top5) == 5


@pytest.mark.parametrize("shape", ["colon", "bulleted", "bare"])
def test_q2_parses_every_output_shape(stream, shape):
    tracked, targets = stream.tracked, stream.targets
    text = {
        "colon": ", ".join(f"{c}: {targets[c]}" for c in tracked),
        "bulleted": "\n".join(f"- {c}: {targets[c]}" for c in tracked),
        "bare": ", ".join(targets[c] for c in tracked),
    }[shape]
    rows = scoring.score_q2(text, tracked, targets, stream.words)
    assert all(r["parsed"] and r["correct"] for r in rows)


def test_unparseable_q2_is_not_scored_as_wrong(stream):
    """v1 scores Q2 descriptively, so a parse failure must be distinguishable
    from a wrong answer."""
    rows = scoring.score_q2("I don't recall any of those.", stream.tracked,
                            stream.targets, stream.words)
    assert not any(r["parsed"] for r in rows)
    assert not any(r["correct"] for r in rows)


def test_q2_scores_a_genuinely_wrong_answer_as_parsed_but_incorrect(stream):
    wrong = ", ".join(f"{c}: zebra" for c in stream.tracked)
    rows = scoring.score_q2(wrong, stream.tracked, stream.targets, stream.words)
    assert all(r["parsed"] and not r["correct"] for r in rows)
