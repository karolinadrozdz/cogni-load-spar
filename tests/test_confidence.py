"""confidence: the stimuli files, the prompt, and run_item + tables on a fake model."""

import importlib.util

import pandas as pd
import pytest
from conftest import CONFIG

from cogniload import confidence

SETS = {name: confidence.ROOT / p for name, p in CONFIG["confidence"]["sets"].items()}
K = CONFIG["readout"]["primary_k"]


@pytest.fixture(scope="module")
def items():
    return [i for path in SETS.values() for i in confidence.load_items(path)]


def _sample(items):
    """Three items of each condition."""
    return [i for c in ("real", "fictitious", "region")
            for i in [x for x in items if x["condition"] == c][:3]]


def _run(ctx, items):
    word_meta = confidence.words(ctx.model.tokenizer)
    out = [confidence.run_item(ctx, i, word_meta) for i in items]
    return (pd.DataFrame([r for rows, _, _ in out for r in rows]),
            pd.DataFrame([r for _, top, _ in out for r in top]),
            pd.DataFrame([s for _, _, s in out]))


def test_capitals_file_matches_its_script():
    """The csv is generated; an edit to the lists must be followed by a re-run."""
    spec = importlib.util.spec_from_file_location("capitals", SETS["capitals"].with_suffix(".py"))
    capitals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capitals)
    assert confidence.load_items(SETS["capitals"]) == [{**i, "links": None}
                                                       for i in capitals.items()]


def test_stimuli_conditions_and_answers(items):
    by = {c: [i for i in items if i["condition"] == c] for c in ("real", "fictitious", "region")}
    assert len(by["real"]) == len(by["fictitious"]) == 50 and len(by["region"]) >= 100
    assert sum(map(len, by.values())) == len(items) == len({i["id"] for i in items})
    assert all(i["answer"] for i in by["real"] + by["region"])
    assert not any(i["answer"] for i in by["fictitious"])
    # Regions carry how well known they are, and no capital is asked twice.
    assert all(isinstance(i["links"], int) for i in by["region"])
    answers = [i["answer"] for i in by["real"] + by["region"]]
    assert len(set(answers)) == len(answers)


def test_prompt_ends_in_thinking_off_plus_prefill(tok):
    out = confidence.build(tok, "France", enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")
    assert "What is the capital of France? Answer in one word." in out


@pytest.mark.parametrize("text,answer,expected", [
    (" Paris", "Paris", "correct"), (" paris.", "Paris", "correct"),
    (" Lyon", "Paris", "guess"), (" Lisbon", "", "guess"),
    (" N'Djamena", "Bol", "guess"), (" N/A", "Bol", "abstain"),
    (" I'm not sure", "Paris", "abstain"), (" There's no such country", "", "abstain"),
    (" 'Paris'", "Paris", "correct"), ("\nRabaul\n", "Buka", "guess"),
    (" Unknown", "", "abstain"), (" I don't know", "Paris", "abstain"), ("", "", "abstain"),
])
def test_output_type(text, answer, expected):
    assert confidence.output_type(text, answer) == expected


@pytest.mark.parametrize("text,answer,entity,expected", [
    (" Jufra", "Hun", "Jufra, Libya", "echo"),
    (" New York City", "Albany", "New York, United States", "echo"),
    (" Jorvan", "", "Jorvania", "echo"),
    (" Kumasi", "Wa", "Upper West Region, Ghana", "guess"),
    (" Accra", "Wa", "Upper West Region, Ghana", "guess"),  # the country is not the entity
    (" La Paz", "Tela", "Atlantida, Honduras", "guess"),    # "la" is inside the name, not a word of it
])
def test_output_type_separates_an_echo_of_the_entity(text, answer, entity, expected):
    assert confidence.output_type(text, answer, entity) == expected


def test_table_values_on_hand_built_rows():
    """Two items, band 23-26. Item 1 has "unknown" at rank 3 in layer 26 only; item 2 never."""
    def rows(item, concept, group, ranks):
        return [{"id": item, "group": group, "concept": concept, "word": concept, "lens": "jlens",
                 "layer": layer, "in_band": 23 <= layer < 27, "rank": r, "rank_wordlike": r}
                for layer, r in ranks.items()]
    far = {layer: 900 for layer in (18, 22, 26, 27)}
    df = pd.DataFrame(
        rows(1, "unknown", "uncertain", {**far, 26: 3}) + rows(1, "maybe", "uncertain", far)
        + rows(2, "unknown", "uncertain", {**far, 27: 1, 22: 1}) + rows(2, "maybe", "uncertain", far)
        + [r for i in (1, 2) for g in ("nonexistent", "control") for r in rows(i, "x", g, far)])
    summary = pd.DataFrame([
        {"id": 1, "condition": "region", "output_type": "guess", "output_rank_uncertain": 25,
         "output_rank_nonexistent": 26, "output_rank_control": 500},
        {"id": 2, "condition": "region", "output_type": "correct", "output_rank_uncertain": 90,
         "output_rank_nonexistent": 90, "output_rank_control": 500}])
    t = confidence.table(df, summary, K).set_index("output_type")
    # One of two uncertainty concepts present on item 1; layers 22 and 27 are outside the band.
    assert t.loc["guess", "uncertain"] == 0.5 and t.loc["correct", "uncertain"] == 0.0
    assert t.loc["guess", "uncertain_any"] == 1.0 and t.loc["correct", "uncertain_any"] == 0.0
    assert t.loc["guess", "out_uncertain"] == 1.0 and t.loc["guess", "out_nonexistent"] == 0.0
    # The half-open window (18, 23) holds layer 22 and not 26.
    early = confidence.table(df, summary, K, layers=(18, 23)).set_index("output_type")
    assert early.loc["guess", "uncertain"] == 0.0 and early.loc["correct", "uncertain"] == 0.5


def test_runner_and_tables_on_a_fake_model(ctx, items):
    sample = _sample(items)
    df, top, summary = _run(ctx, sample)
    assert set(df.lens) == {"jlens", "logit"} and set(df.readout_at) == {"answer"}
    # The capital is read wherever there is one.
    assert set(df[df.group == "answer"].id) == {i["id"] for i in sample if i["answer"]}
    assert summary[summary.condition != "fictitious"].expected_rank.notna().all()
    assert summary.top1_prob.between(0, 1).all()
    # Top tokens: k per layer and lens, for every item.
    k = CONFIG["confidence"]["top_k"]
    assert len(top) == len(sample) * 2 * len(ctx.layers) * k and set(top["rank"]) == set(range(1, k + 1))

    t = confidence.table(df, summary, K)
    assert t.n.sum() == len(sample) and set(t.condition) == {"real", "fictitious", "region"}
    assert {"uncertain", "nonexistent", "control", "answer", "uncertain_any",
            "out_uncertain", "out_nonexistent", "out_control"} <= set(t.columns)
    # A layer window outside the band, and the per-layer profile.
    assert confidence.table(df, summary, K, layers=(10, 11)).n.sum() == len(sample)
    profile = confidence.by_layer(df, summary, K)
    assert list(profile.index) == ctx.layers
    assert ("fictitious", "uncertain") in profile.columns and ("region", "answer") in profile.columns


def test_enriched_ranks_a_token_only_one_set_has():
    top = pd.DataFrame([
        {"id": 1, "lens": "jlens", "layer": 10, "rank": 1, "token": " Unknown"},
        {"id": 2, "lens": "jlens", "layer": 10, "rank": 1, "token": " unknown"},
        {"id": 3, "lens": "jlens", "layer": 10, "rank": 1, "token": " Paris"},
        {"id": 1, "lens": "jlens", "layer": 10, "rank": 2, "token": " the"},
        {"id": 3, "lens": "jlens", "layer": 10, "rank": 2, "token": " the"},
        {"id": 3, "lens": "jlens", "layer": 20, "rank": 1, "token": " unknown"},  # outside
    ])
    out = confidence.enriched(top, pd.Series([1, 2]), pd.Series([3]), layers=(10, 11))
    first = out.iloc[0]
    assert (first.token, first.a, first.b) == ("unknown", 1.0, 0.0)
    assert out[out.token == "the"]["diff"].item() == pytest.approx(-0.5)


def test_report_prints_windows_profiles_and_top_tokens(ctx, items, tmp_path, capsys):
    df, top, summary = _run(ctx, _sample(items))
    paths = confidence._paths("dev", tmp_path, "capitals", False)
    paths[0].parent.mkdir(parents=True)
    for frame, path in zip((df, top, summary), paths):
        frame.to_parquet(path, index=False)
    spec = {**ctx.spec, "band": ctx.band}
    confidence.report(spec, CONFIG, tmp_path)
    printed = capsys.readouterr().out
    assert "capitals FULL" in printed and "regions" not in printed.replace("region ", "")
    assert "in-band" in printed and "early layers 8-18" in printed and "per layer" in printed
    assert "top tokens, fictitious (a) vs real (b)" in printed
    # A run stored before top tokens existed still reports its fixed-list tables.
    paths[1].unlink()
    confidence.report(spec, CONFIG, tmp_path)
    printed = capsys.readouterr().out
    assert "in-band" in printed and "top tokens" not in printed


def test_report_types_outputs_again_from_the_stored_text(ctx, items, tmp_path, capsys):
    """A stored type from older code must not survive into the tables."""
    df, top, summary = _run(ctx, _sample(items))
    summary["generated"], summary["output_type"] = " None", "guess"
    paths = confidence._paths("dev", tmp_path, "regions", True)
    paths[0].parent.mkdir(parents=True)
    for frame, path in zip((df, top, summary), paths):
        frame.to_parquet(path, index=False)
    confidence.report({**ctx.spec, "band": ctx.band}, CONFIG, tmp_path)
    printed = capsys.readouterr().out
    assert "regions SMOKE" in printed and "abstain" in printed and "guess" not in printed


def test_cli_requires_a_stimuli_set_for_confidence_only(capsys):
    from cogniload import cli

    for argv in (["confidence"], ["find_band", "--set", "regions"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
    assert "--set is required" in capsys.readouterr().err
