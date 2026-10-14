"""self_check: answer normalisation, the three renders, the AUROC, and a fake end-to-end run
of all three arms on fake state_tracking and state_tracking_thinking runs, then the report."""

import json

import pandas as pd
import pytest
from test_state_tracking_thinking import fake_stage

from cogniload import cli, experiment, state_tracking as st
from cogniload import self_check as sc

END = "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"


def test_shown_word_is_the_one_object_the_token_starts():
    objects, dropped = ["vase", "bell", "bag", "key", "kite"], ["vase", "kite"]
    assert sc.shown_word(" V", objects, dropped) == ("vase", True)
    assert sc.shown_word(" Bell", objects, dropped) == ("bell", True)
    assert sc.shown_word(" key", objects, dropped) == ("key", True)
    # ` Key` is one token, so ` K` can only start `kite`.
    assert sc.shown_word(" K", objects, dropped) == ("kite", True)
    assert sc.shown_word(" Be", objects, dropped) == ("Be", False)
    assert sc.shown_word(" **", objects, dropped) == ("**", False)
    assert sc.shown_word(" ", objects, dropped) == ("", False)


def renders(tok):
    names, objects = st.pools(tok)
    item = st.make_items(2, 1, 1, names, objects)[0]
    off, _ = st.build(tok, item, "derived", "Answer:", enable_thinking=False)
    user = sc.user_turn(tok, item, off)
    reasoning = "\nAnn has the bag.\nSo the bag.\n"
    return user, sc.build(tok, user, "bag"), sc.build(tok, user, "bag", reasoning), reasoning


def test_every_arm_ends_thinking_off_and_prefilled_with_the_answer_as_history(tok):
    user, answer_only, with_trace, reasoning = renders(tok)
    for text in (answer_only, with_trace):
        assert text.endswith(END) and text.count("<|im_start|>user") == 2
        assert f"{user}<|im_end|>" in text and f"{sc.QUESTION}<|im_end|>" in text
    assert "<|im_start|>assistant\nAnswer: bag<|im_end|>" in answer_only


def test_with_trace_is_answer_only_plus_the_trace_block(tok):
    user, answer_only, with_trace, reasoning = renders(tok)
    block = sc.TRACE.format(reasoning=reasoning.strip())
    assert with_trace.replace(block, "", 1) == answer_only
    trace = "<think>\nAnn has the bag.\nSo the bag.\n</think>\n\n"
    assert f"<|im_start|>assistant\n{trace}Answer: bag<|im_end|>" in with_trace
    # The same text the template gives a turn whose reasoning it keeps.
    kept = tok.apply_chat_template(
        [{"role": "user", "content": user},
         {"role": "assistant", "content": "Answer: bag", "reasoning_content": reasoning},
         {"role": "user", "content": sc.QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
        preserve_thinking=True)
    assert kept + "Answer:" == with_trace


def test_wrong_is_the_poi_previous_object():
    for k in (1, 3):
        item = st.make_items(k, 1, 1, ["Ann", "Ben", "Tom"], st.OBJECTS[:21])[0]
        assert sc.wrong_object(item) == item["chains"][item["poi"]][-2]


def test_corruption_rewrites_unquoted_target_mentions_in_their_case(tok):
    reasoning = ("The person holding the pen swaps it for the cup. After a lot of careful "
                 "thought here, Ann has the pen.\nPen it is, not pens or a pen-like thing.")
    out, counts = sc.corrupt(tok, reasoning, "pen", "map")
    assert out == ("The person holding the pen swaps it for the cup. After a lot of careful "
                   "thought here, Ann has the map.\nMap it is, not pens or a map-like thing.")
    assert counts == {"n_replaced": 3, "n_quoted_kept": 1, "n_replaced_tail": 1,
                      "corruption_empty": False}
    assert sc.corrupt(tok, "Nothing here.", "pen", "map") == (
        "Nothing here.", {"n_replaced": 0, "n_quoted_kept": 0, "n_replaced_tail": 0,
                          "corruption_empty": True})


def test_trace_contradicted_is_with_trace_but_for_the_answer_line(tok):
    user, _, with_trace, reasoning = renders(tok)
    contradicted = sc.build(tok, user, "map", reasoning)
    assert contradicted.endswith(END)
    assert with_trace.replace("Answer: bag<|im_end|>", "Answer: map<|im_end|>") == contradicted


def test_answer_forms_drop_forms_sharing_an_id(tok):
    # The fake tokenizer splits on whitespace, so `yes` is ` yes`.
    assert sc.answer_forms(tok) == ({"yes": [" yes", " Yes"], "no": [" no", " No"]},
                                    ["yes", "no"])


def test_auroc_on_ranks_with_ties_averaged():
    assert sc.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert sc.auroc([1, 1, 0, 2], [1, 0, 0, 1]) == pytest.approx(3.5 / 4)
    assert sc.auroc([3, 2, 1], [0, 0, 0]) != sc.auroc([3, 2, 1], [0, 0, 0])  # NaN


def fake_sources(args, tmp_path):
    """Thinking-off and thinking-on runs, one item per cell, with answers set so the shown
    answer is right on, right or stale off by item, and once not an object."""
    cli.main(["state_tracking", "--arm", "derived", "--limit", "1", *args])
    cli.main(["state_tracking_thinking", "--limit", "1", *args])
    for path in sorted((tmp_path / "dev").glob("state_tracking_*summary_k*_limit.parquet")):
        s = pd.read_parquet(path)
        chains = [json.loads(c)[p] for c, p in zip(s.chains, s.poi)]
        if "thinking" in path.name:
            s["answer"] = [" **" if "k1_h1" in path.name else " " + c[-1].capitalize()
                           for c in chains]
        else:
            s["answer"] = [" " + (c[-1] if c[0] < c[-1] else c[0]) for c in chains]
        s.to_parquet(path, index=False)


def test_end_to_end_three_arms_then_report(ctx, monkeypatch, tmp_path, capsys):
    args = fake_stage(ctx, monkeypatch, tmp_path)
    fake_sources(args, tmp_path)
    for arm in ("answer_only", "no_reasoning", *sc.ARMS[2:]):
        cli.main(["self_check", "--arm", arm, "--limit", "1", *args])
    out = tmp_path / "dev"
    for arm in sc.ARMS:
        assert len(list(out.glob(f"self_check_{arm}_k*_limit.parquet"))) == 14
        assert len(list(out.glob(f"self_check_{arm}_summary_k*_limit.parquet"))) == 14
    manifest = json.loads((out / "self_check_no_reasoning_manifest.json").read_text())
    assert manifest["templates"]["answer_forms_dropped"] == ["yes", "no"]
    assert manifest["templates"]["sources"] == {"state_tracking_derived": "limit",
                                                "state_tracking_thinking": "limit"}
    s = pd.concat(experiment.load(f"self_check_{arm}", "dev", tmp_path, limit=True,
                                  kind="_summary", cell="k") for arm in sc.ARMS)
    injected = s[s.arm.isin(sc.WRONG_ANSWER)]
    assert (injected.word == injected.wrong).all() and not injected.correct.any()
    corrupted = s[s.arm == "trace_corrupted"]
    # The fake trace writes the target capitalised and unquoted after the update it quotes.
    assert (corrupted.n_replaced > 0).all() and not corrupted.corruption_empty.any()
    assert not corrupted.apply(lambda r: f"Then the {r.target.capitalize()}." in r.prompt,
                               axis=1).any()
    on = s[s.arm.isin(["answer_only", "with_trace"])]
    assert not on[(on.k == 1) & (on.h == 1)].answer_is_object.any()
    assert on[(on.k != 1) | (on.h != 1)].correct.all()
    nr = s[s.arm == "no_reasoning"]
    assert nr.correct.any() and not nr.correct.all()
    # Copied exactly where the two prompts match, which is where the shown words match.
    assert (nr.copied_from_answer_only == nr.shared_with_answer_only).all()
    assert nr.shared_with_answer_only.sum() == (nr.correct & ((nr.k != 1) | (nr.h != 1))).sum()
    trace = s[s.arm == "with_trace"].set_index(["k", "h"])
    assert (trace.n_tokens > on[on.arm == "answer_only"].set_index(["k", "h"]).n_tokens).all()
    assert s.prompt.str.endswith(END).all()
    rows = pd.read_parquet(out / "self_check_with_trace_k3_h6_limit.parquet")
    assert set(rows.readout_at) == {"judge"} and set(rows.layer) == {10, 11, 12}
    assert {"yes", "no", "target", "floor"} <= set(rows.role)
    assert set(rows[rows.is_shown].base) == {rows[rows.role == "target"].base.iloc[0]}
    capsys.readouterr()
    cli.main(["report", "self_check", *args])
    text = capsys.readouterr().out
    for table in ("judge behaviour", "discrimination", "calibration",
                  "b = answer_only vs a = no_reasoning", "b = with_trace vs a = answer_only",
                  "judge-position contents"):
        assert table in text
    for column in ("auroc", "sign_p", "target_not_shown_ll", "closed=True", "off wrong, on right",
                   "wrong_not_shown_ll", "trace_corrupted vs trace_contradicted",
                   "answer_injected vs answer_only", "corruption_empty=False", "n_replaced "):
        assert column in text


def test_cli_requires_a_self_check_arm(ctx, monkeypatch, tmp_path):
    args = fake_stage(ctx, monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["self_check", *args])
