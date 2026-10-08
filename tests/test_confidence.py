"""confidence: the stimuli file, the prompt, and run_item + table on a fake model."""

import importlib.util

import pandas as pd
import pytest
from conftest import CONFIG

from cogniload import confidence

STIMULI = confidence.ROOT / CONFIG["confidence"]["stimuli"]
K = CONFIG["readout"]["primary_k"]


def test_stimuli_file_matches_its_script():
    """The csv is generated; an edit to the lists must be followed by a re-run."""
    spec = importlib.util.spec_from_file_location("capitals", STIMULI.with_suffix(".py"))
    capitals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capitals)
    assert confidence.load_items(STIMULI) == capitals.items()


def test_stimuli_are_balanced_and_only_real_items_have_an_answer():
    items = confidence.load_items(STIMULI)
    assert [i["id"] for i in items] == list(range(100))
    for condition, has_answer in (("real", True), ("fictitious", False)):
        group = [i for i in items if i["condition"] == condition]
        assert len(group) == 50 and all(bool(i["answer"]) == has_answer for i in group)


def test_prompt_ends_in_thinking_off_plus_prefill(tok):
    out = confidence.build(tok, "France", enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")
    assert "What is the capital of France? Answer in one word." in out


@pytest.mark.parametrize("text,answer,expected", [
    (" Paris", "Paris", "correct"), (" paris.", "Paris", "correct"),
    (" Lyon", "Paris", "guess"), (" Lisbon", "", "guess"),
    (" Unknown", "", "abstain"), (" I don't know", "Paris", "abstain"), ("", "", "abstain"),
])
def test_output_type(text, answer, expected):
    assert confidence.output_type(text, answer) == expected


def test_runner_and_table_on_a_fake_model(ctx):
    items = confidence.load_items(STIMULI)
    items = items[:3] + items[50:53]
    word_meta = confidence.words(ctx.model.tokenizer)
    assert {m["group"] for m in word_meta.values()} == set(confidence.GROUPS)
    rows, summaries = [], []
    for item in items:
        r, s = confidence.run_item(ctx, item, word_meta)
        rows += r
        summaries.append(s)
    df, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    assert set(df.lens) == {"jlens", "logit"} and set(df.readout_at) == {"answer"}
    # The capital is read for real items only.
    assert set(df[df.group == "answer"].id) == {0, 1, 2}
    assert summary[summary.condition == "real"].expected_rank.notna().all()

    t = confidence.table(df, summary, K)
    assert t.n.sum() == len(items) and set(t.condition) == {"real", "fictitious"}
    assert {"uncertain", "nonexistent", "control", "answer", "uncertain_any"} <= set(t.columns)
