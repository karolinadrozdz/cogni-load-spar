"""state_tracking_thinking: same items as state_tracking, thinking on, the appended close,
reasoning positions, and a fake end-to-end run with its report."""

import json

import pandas as pd
import pytest
import torch
from conftest import CONFIG

from cogniload import cli, experiment, readout, registry, state_tracking as st
from cogniload import state_tracking_thinking as stt


def fake_reasoning(ctx, text):
    """Make the fake model 'generate' `text`, and log every text the lens reads."""
    ids = torch.tensor([ctx.model.tokenizer.encode(text)])
    ctx.model._hf_model.generate = lambda prompt, **kw: torch.cat([prompt, ids], dim=1)
    read, apply = [], ctx.lens.apply
    ctx.lens.apply = lambda model, prompt, **kw: read.append(prompt) or apply(model, prompt, **kw)
    return read


def first_item(tok):
    names, objects = st.pools(tok)
    return st.make_items(2, 1, 1, names, objects)[0], stt.capitals(tok, objects)[0]


def test_prompt_is_state_tracking_derived_with_thinking_on(ctx):
    item, caps = first_item(ctx.model.tokenizer)
    fake_reasoning(ctx, "Fine. </think>")
    _, summary = stt.run_item(ctx, item, caps)
    assert summary["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    off, _ = st.build(ctx.model.tokenizer, item, "derived", "Answer:", enable_thinking=False)
    user = lambda t: t.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    assert user(summary["prompt"]) == user(off)


def test_unclosed_reasoning_gets_the_close_and_the_prefill_appended(ctx):
    item, caps = first_item(ctx.model.tokenizer)
    read = fake_reasoning(ctx, "Ann has the bag")
    rows, unclosed = stt.run_item(ctx, item, caps)
    assert not unclosed["closed"] and unclosed["n_reasoning_tokens"] == 4
    assert read[0] == unclosed["prompt"] + " Ann has the bag\n</think>\n\nAnswer:"
    assert {r["closed"] for r in rows} == {False} and unclosed["retok_match"]
    read = fake_reasoning(ctx, "Ann has the bag. </think>")
    rows, closed = stt.run_item(ctx, item, caps)
    assert closed["closed"] and closed["n_reasoning_tokens"] == 4
    assert read[0] == closed["prompt"] + " Ann has the bag.</think>\n\nAnswer:"
    assert {r["closed"] for r in rows} == {True}
    table = stt.behaviour_table(pd.DataFrame([unclosed, closed]), None)
    assert table.closed.tolist() == [False, True] and (table.n == 1).all()
    assert "retok_match" in table


def test_traces_past_jlens_truncation_are_read_whole(ctx):
    item, caps = first_item(ctx.model.tokenizer)
    fake_reasoning(ctx, " ".join(f"w{i}" for i in range(600)) + " </think>")
    rows, summary = stt.run_item(ctx, item, caps)
    assert summary["n_tokens"] > 600 and summary["answer_pos"] == summary["n_tokens"] - 1
    assert {r["token_pos"] for r in rows if r["readout_at"] == "answer"} == {
        summary["answer_pos"]}


def test_read_refuses_positions_past_the_truncated_prompt_and_ids_past_the_vocab(ctx):
    prompt = " ".join(f"w{i}" for i in range(600))
    ids = [readout.token_id(ctx.model.tokenizer, "w1")]
    read = lambda positions, ids=ids, **kw: readout.read(
        ctx.model, ctx.lens, prompt, ["w1"], ids, layers=[11], positions=positions, **kw)
    with pytest.raises(ValueError, match="longer than the 512-token lens limit"):
        read([599])
    assert read([599], max_seq_len=600)[0][0]["token_pos"] == 599
    with pytest.raises(ValueError, match="outside the 600-token encoded prompt"):
        read([600], max_seq_len=600)
    with pytest.raises(ValueError, match="unembedding"):
        read([-1], ids=[10**6], max_seq_len=600)


def test_prewrite_lands_on_the_token_before_each_object_in_either_case(tok):
    text = "<think>\n Ann had the key, then Bag.\nbag again; keyring is not key. Kim got the bag"
    enc = tok(text, return_offsets_mapping=True)
    span = (len("<think>\n"), len(text))
    writes = stt.prewrite(enc, text, span, ["key", "bag"])
    assert [(o, n) for _, o, n, _, _ in writes] == [("key", 0), ("bag", 0), ("bag", 1),
                                                    ("key", 1), ("bag", 2)]
    for index, o, _, char, _ in writes:
        a, b = enc["offset_mapping"][index + 1]
        assert text[a:b].strip(".,;").casefold() == o
        assert text[char:char + len(o)].casefold() == o


def test_quoted_marks_mentions_copied_from_an_update_sentence(tok):
    text = ("<think>\n The person holding the key swaps it for the bag. Ann holds the pen. "
            "Then, after all of that, so Ann now has the cup.")
    enc = tok(text, return_offsets_mapping=True)
    writes = stt.prewrite(enc, text, (len("<think>\n"), len(text)), ["key", "bag", "pen", "cup"])
    assert [(o, quoted) for _, o, _, _, quoted in writes] == [
        ("key", True), ("bag", True), ("pen", True), ("cup", False)]


def test_sentence_ends():
    text = "x One. Two? 3.5 is fine!\nThree.\n\nFour"
    tokens = ["x", " One", ".", " Two", "?", " 3", ".", "5", " is", " fine", "!\n", "Three",
              ".\n\n", "Four"]
    offsets, char = [], 0
    for t in tokens:
        offsets.append((char, char + len(t)))
        char += len(t)
    ends = stt.sentence_ends({"offset_mapping": offsets}, text, (1, len(text)))
    assert [tokens[i] for i in ends] == [".", "?", "!\n", ".\n\n"]


def test_sentence_positions_are_evenly_spaced_under_the_cap(ctx):
    assert stt.even(list(range(10)), 4) == [(0, 0), (3, 3), (6, 6), (9, 9)]
    assert stt.even(list(range(3)), 4) == [(0, 0), (1, 1), (2, 2)]
    item, caps = first_item(ctx.model.tokenizer)
    fake_reasoning(ctx, " ".join(f"Step{i} done." for i in range(250)) + " </think>")
    rows, summary = stt.run_item(ctx, item, caps)
    cap = CONFIG["state_tracking_thinking"]["max_sentences"]
    # 250 ends over a cap of 120: every third, from the first.
    kept = list(range(0, 250, 3))
    assert summary["n_sentences"] == 250 and len(json.loads(summary["sentence_pos"])) == len(kept)
    assert len(kept) <= cap
    sentence_rows = pd.DataFrame(rows).query("readout_at == 'sentence'")
    assert sorted(set(sentence_rows.sentence)) == kept
    assert set(sentence_rows.layer) == {11, 12}


def test_prewrites_are_capped_evenly_keeping_mention_index_and_control(ctx):
    item, caps = first_item(ctx.model.tokenizer)
    a, b = item["chains"][item["poi"]][:2]
    fake_reasoning(ctx, " ".join(f"the {a} then the {b}" for _ in range(150)) + " </think>")
    rows, summary = stt.run_item(ctx, item, caps)
    cap = CONFIG["state_tracking_thinking"]["max_prewrites"]
    writes = pd.DataFrame(rows).query("readout_at == 'prewrite'")
    # 300 mentions over a cap of 120: every third, from the first; a, b alternate.
    kept = list(range(0, 300, 3))
    assert json.loads(summary["prewrite_kept"]) == kept
    assert len(json.loads(summary["prewrite"])) == 300
    assert sorted(set(writes.mention)) == kept and len(kept) <= cap
    by = writes.drop_duplicates("mention").set_index("mention")
    assert by.loc[3, "writing"] == b and by.loc[3, "control"] == a  # the mention before is 2
    assert pd.isna(by.loc[0, "control"])
    assert min(summary[t] for t in ("t_generate", "t_read_answer", "t_read_reasoning")) >= 0


def test_both_case_forms_are_read_with_their_role():
    item = st.make_items(1, 1, 1, ["Ann", "Ben", "Tom"], st.OBJECTS[:21])[0]
    meta = stt.words_and_meta(item, {o: o.capitalize() for o in st.OBJECTS[:20]})
    target = item["chains"][item["poi"]][-1]
    assert meta[target.capitalize()] | {"form": "lower"} == meta[target]
    assert meta[target.capitalize()]["base"] == target
    assert st.OBJECTS[20].capitalize() not in meta and "Ann" in meta


def fake_stage(ctx, monkeypatch, tmp_path) -> list[str]:
    """A fake model whose reasoning quotes an update, then writes the PoI's chain and a floor
    object; returns the CLI arguments for a run on it."""
    tok = ctx.model.tokenizer
    names, objects = st.pools(tok)

    def generate(ids, **kw):
        text = tok.decode(ids[0].tolist())
        item = next(it for k in CONFIG["state_tracking"]["ks"] for h in st.hays(k)
                    for it in st.make_items(k, h, 1, names, objects)
                    if all(f"{p} holds the {c[0]}." in text for p, c in it["chains"].items()))
        chain = item["chains"][item["poi"]]
        words = (f"The person holding the {chain[0]} swaps it for the {chain[1]}. "
                 f"{item['poi']} starts with the {chain[0]}. "
                 + " ".join(f"Then the {o.capitalize()}." for o in chain[1:])
                 + f" Not the {item['floor'][0]}.")
        return torch.cat([ids, torch.tensor([tok.encode(words + " </think>")])], dim=1)
    ctx.model._hf_model.generate = generate
    monkeypatch.setattr(registry, "load", lambda spec: (ctx.model, ctx.lens))
    monkeypatch.setattr(experiment, "recorded_layers", lambda spec, config: [10, 11, 12])
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "band.json").write_text('{"band": [11, 13]}')
    return ["--model", "dev", "--config", str(registry.REGISTRY.parent / "experiments.yaml"),
            "--out-dir", str(tmp_path)]


def test_cell_runs_one_cell_under_its_usual_name(ctx, monkeypatch, tmp_path, capsys):
    args = fake_stage(ctx, monkeypatch, tmp_path)
    cli.main(["state_tracking_thinking", "--limit", "1", "--cell", "3,6", *args])
    assert sorted(p.name for p in (tmp_path / "dev").glob("*.parquet")) == [
        "state_tracking_thinking_k3_h6_limit.parquet",
        "state_tracking_thinking_summary_k3_h6_limit.parquet"]
    assert "peak RSS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["state_tracking", "--arm", "derived", "--cell", "3,6", *args])
    with pytest.raises(ValueError, match="not in the grid"):
        cli.main(["state_tracking_thinking", "--cell", "3,5", *args])


def test_end_to_end_writes_shards_on_state_tracking_items_and_reports(ctx, monkeypatch,
                                                                      tmp_path, capsys):
    names, objects = st.pools(ctx.model.tokenizer)
    args = fake_stage(ctx, monkeypatch, tmp_path)
    cli.main(["state_tracking_thinking", "--limit", "1", *args])
    out = tmp_path / "dev"
    assert len(list(out.glob("state_tracking_thinking_k*_limit.parquet"))) == 14
    manifest = json.loads((out / "state_tracking_thinking_manifest.json").read_text())
    assert manifest["spec"]["enable_thinking"]
    assert not manifest["templates"]["registry_enable_thinking"]
    assert manifest["templates"]["capitals_dropped"] == []
    s = experiment.load("state_tracking_thinking", "dev", tmp_path, limit=True, kind="_summary",
                        cell="k")
    for it in s.itertuples():
        same = st.make_items(it.k, it.h, 1, names, objects)[0]
        assert json.loads(it.chains) == same["chains"] and json.loads(it.updates) == same["updates"]
    rows = pd.read_parquet(out / "state_tracking_thinking_k3_h6_limit.parquet")
    assert set(rows.readout_at) == {"answer", "sentence", "prewrite"}
    assert set(rows[rows.readout_at == "answer"].layer) == {10, 11, 12}
    assert set(rows[rows.readout_at != "answer"].layer) == {11, 12}
    assert set(rows.form) == {"lower", "cap"} and set(rows.closed) == {True}
    assert set(rows[rows.readout_at == "prewrite"].quoted) == {True, False}
    assert manifest["templates"]["sentence_sampling"] == "even"
    # An off run on the same items, for top1_off.
    s.assign(correct=True).to_parquet(out / "state_tracking_derived_summary_k1_h1.parquet")
    capsys.readouterr()
    cli.main(["report", "state_tracking_thinking", *args])
    text = capsys.readouterr().out
    for table in ("T1 behaviour", "answer contents", "pre-write", "sentence ends"):
        assert table in text
    assert "top1_off" in text and "poi_final" in text and "any_in_prompt_ll" in text
    assert "retok_match" in text and "floor_ll" in text and "not comparable" in text
    for table in ("paired, pooled", "OLS n_reasoning_tokens", "pre-write by k",
                  "floor-object writes", "every incorrect item", "trace state claims",
                  "correct_prefix"):
        assert table in text


def test_mcnemar_is_the_doubled_binomial_tail():
    assert stt.mcnemar(0, 0) == 1.0 and stt.mcnemar(3, 3) == 1.0
    assert stt.mcnemar(5, 0) == pytest.approx(2 / 32) and stt.mcnemar(0, 5) == stt.mcnemar(5, 0)


def summaries(**columns):
    base = {"k": [1, 1, 2, 2], "h": [1, 1, 1, 1], "item_id": [0, 1, 0, 1]}
    return pd.DataFrame(base | columns)


def test_paired_counts_items_per_cell_and_pooled():
    on = summaries(correct=[True, True, True, False], last_is_needle=[False, True, False, True])
    off = summaries(correct=[False, True, True, True])
    on["correct_prefix"], off["correct_prefix"] = on.correct, off.correct
    cells, pooled = stt.paired(on, off)
    assert cells[["on_only", "off_only", "both", "neither"]].values.tolist() == [
        [1, 0, 1, 0], [0, 1, 1, 0]]
    assert pooled.set_index("items").loc["all", ["n", "on_only", "off_only"]].tolist() == [4, 1, 1]
    assert pooled.set_index("items").loc["hay_last", "n"] == 2


def test_correct_prefix_counts_the_start_of_a_multi_token_capital_only():
    s = pd.DataFrame({"correct": [True, False, False, False, False],
                      "answer": [" vase", " V", " K", " S", " "],
                      "expected": ["vase", "vase", "kite", "bag", "vase"]})
    assert stt.correct_prefix(s, ["vase", "kite"]).tolist() == [True, True, True, False, False]


def test_lengths_gives_median_and_iqr_per_cell():
    s = summaries(n_reasoning_tokens=[10, 20, 30, 50], n_sentences=[1, 3, 2, 2],
                  prewrite=['[[0, "a", 0, 0, false]]', "[]", "[]", "[]"])
    out = stt.lengths(s).set_index("k")
    assert out.loc[1, "med_tokens"] == 15 and out.loc[2, "iqr_tokens"] == 10
    assert out.loc[1, "med_mentions"] == 0.5


def test_length_regression_recovers_an_exact_line():
    updates = [json.dumps([{"is_needle": n} for n in u]) for u in
               ([True, False], [False, True], [True], [False, True, False, False], [True, True])]
    s = pd.DataFrame({"k": [1, 1, 1, 1, 2], "updates": updates})
    s = st.with_hay_split(s)
    s["n_reasoning_tokens"] = 100 + 10 * s.k + 3 * s.pre_hays + 7 * s.tail_hays
    out = stt.length_regression(s.drop(columns=["pre_hays", "tail_hays"])).set_index("term")
    assert out.coef.round(6).tolist() == [100, 10, 3, 7]


def test_spearman_is_on_ranks():
    assert stt.spearman(pd.Series([1, 2, 30]), pd.Series([5, 6, 7])) == pytest.approx(1)
    assert stt.spearman(pd.Series([1, 2, 3]), pd.Series([9, 4, 1])) == pytest.approx(-1)


def prewrite_rows(rows):
    return pd.DataFrame([dict(zip(["lens", "writing", "base", "role", "quoted", "present",
                                   "mention"], r), k=1, h=1, item_id=0) for r in rows])


def test_floor_writes_counts_unquoted_floor_writes_per_word_and_prose():
    p = prewrite_rows([(lens, w, w, "floor", q, present, 0) for lens in ("jlens", "logit")
                       for w, q, present in (("chain", False, True), ("chain", False, False),
                                             ("map", False, True), ("map", True, True))]
                      + [("jlens", "bag", "bag", "target", False, True, 0)])
    out = stt.floor_writes(p).set_index("writing")
    assert out.n.to_dict() == {"chain": 2, "map": 1} and out.loc["chain", "present"] == 0.5
    assert stt.floor_writes(p, "prose").set_index("prose").n.to_dict() == {True: 2, False: 1}


def test_state_claims_finds_each_pattern_in_either_case_and_markdown():
    text = ("Ann holds the bag. **Ben**: Cup, Tom -> key, Ann → cup, Ben=bag. Tom now has the "
            "pen and Ann is holding the KEY. Ann swaps the bag. the bag holds Ann.")
    claims = stt.state_claims(text, ["Ann", "Ben", "Tom"], ["bag", "cup", "key", "pen"])
    assert [(n, o) for _, n, o in claims] == [
        ("Ann", "bag"), ("Ben", "cup"), ("Tom", "key"), ("Ann", "cup"), ("Ben", "bag"),
        ("Tom", "pen"), ("Ann", "key")]
    assert all(text[i:i + len(o)].casefold() == o for i, _, o in claims)


def claim_item(reasoning, correct=True):
    return pd.DataFrame([{"k": 2, "h": 1, "item_id": 0, "poi": "Ann", "correct": correct,
                          "answer": " cup", "closed": True, "reasoning": reasoning,
                          "prompt": "P" * 10, "floor": '["pen"]',
                          "chains": '{"Ann": ["bag", "cup"], "Ben": ["key"]}',
                          "prewrite": json.dumps([[0, "bag", 0, 10 + reasoning.find("bag"),
                                                   False]])}])


def test_claim_table_classes_against_the_chains():
    c = stt.claim_table(claim_item("Ann: cup. Ann: bag. Ben: pen. Ann: key."))
    assert c.claim.tolist() == ["true-final", "true-ever", "never", "never"]
    assert stt.claim_table(claim_item("nothing")).empty


def test_claim_summary_counts_items_with_a_never_claim():
    s = pd.concat([claim_item("Ben: pen."), claim_item("Ann: cup.", False).assign(item_id=1)])
    by_k, by_correct = stt.claim_summary(s, stt.claim_table(s))
    assert by_k[["true-final", "never", "items", "with_never"]].values.tolist() == [[1, 1, 2, 1]]
    assert by_correct.set_index("correct").share_never.to_dict() == {False: 0.0, True: 1.0}


def test_claim_presence_reads_the_written_and_final_object_at_the_claim():
    s = claim_item("Ann: bag.")
    p = prewrite_rows([(lens, "bag", base, "", False, present, 0.0) for lens in ("jlens", "logit")
                       for base, present in (("bag", True), ("cup", False))]).assign(k=2)
    out = stt.claim_presence(p, s, stt.claim_table(s)).set_index("claim")
    assert out.loc["true-ever", ["n", "written", "final", "written_ll"]].tolist() == [1, 1, 0, 1]
