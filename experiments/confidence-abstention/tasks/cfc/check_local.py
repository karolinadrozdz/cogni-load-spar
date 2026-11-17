"""Local end-to-end check of the cfc task: no GPU, no vLLM. TASK.config and TASK.run on a fake Context with the six
checkpoints' REAL chat templates (jinja2 sandbox, as transformers renders them), the real Olmo-3 tokenizer, the REAL
reference runs (copies of the volume's olmo32b-popqa-<key>/ dirs: phases.csv, start_format.csv, params.json) and a
scripted engine whose answers have known metric values.

    python tasks/cfc/check_local.py --ref <dir with instruct-sft/ ... think-rl31-main/> --tpl <cache dir> --out <dir>

--tpl: chat_template.jinja / tokenizer_config.json per checkpoint (<tpl>/<repo name>/) and tokenizer.json
(<tpl>/tokenizer-<line>/); missing files are downloaded from the Hub (public repos).
--out: scratch dir for the fake runs' outputs (cfc.csv, params.json; analyze_cfc.py is run on them at the end).
"""

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules["vllm"] = types.SimpleNamespace(SamplingParams=lambda **kw: types.SimpleNamespace(**kw))

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True)
ap.add_argument("--tpl", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
A = ap.parse_args()

import abstention_modal as am  # noqa: E402
import tasks  # noqa: E402
import tasks.cfc as cfc  # noqa: E402
from tasks.cfc import build_pairs as bp  # noqa: E402

real_read, real_exists, real_read_text = pd.read_csv, Path.exists, Path.read_text


def loc(s):
    s = str(s).replace("\\", "/")
    if s.startswith("/data/"):
        return ROOT / "data" / s[6:]
    if s.startswith("/results/"):
        return A.ref / s[len("/results/"):].replace(cfc.REF_PREFIX, "")
    return None


pd.read_csv = lambda p, *a, **k: real_read(loc(p) or p, *a, **k)
Path.exists = lambda self: real_exists(loc(self) or self)
Path.read_text = lambda self, *a, **k: real_read_text(loc(self) or self, *a, **k)

RUNS = {  # key -> model, thinking, dtype, template repo
    "instruct-sft": ("allenai/Olmo-3.1-32B-Instruct-SFT", "off", "bfloat16"),
    "instruct-dpo": ("allenai/Olmo-3.1-32B-Instruct-DPO", "off", "bfloat16"),
    "instruct-rl": ("allenai/Olmo-3.1-32B-Instruct", "off", "auto"),
    "think-sft-main": ("allenai/Olmo-3-32B-Think-SFT", "native", "auto"),
    "think-dpo": ("allenai/Olmo-3-32B-Think-DPO", "native", "auto"),
    "think-rl31-main": ("allenai/Olmo-3.1-32B-Think", "native", "auto"),
}
TOKENIZER = {"off": "allenai/Olmo-3.1-32B-Instruct-DPO", "native": "allenai/Olmo-3-32B-Think-DPO"}


# ---- templates and tokenizer -------------------------------------------------------------------------

def fetch(repo, names, sub):
    """The first of `names` that the repo has, cached in <tpl>/<sub>/ (each repo in its own directory)."""
    d = A.tpl / sub
    for name in names:
        if (d / name).exists():
            return d / name
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo)
    name = next(n for n in names if n in files)
    return Path(hf_hub_download(repo, name, local_dir=str(d)))


def template(repo):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    f = fetch(repo, ["chat_template.jinja", "tokenizer_config.json"], repo.split("/")[1])
    src = f.read_text(encoding="utf-8")
    if f.name == "tokenizer_config.json":  # Olmo-3-32B-Think-SFT@main keeps its template here
        src = json.loads(src)["chat_template"]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(src)


class Tok:
    """The checkpoint's tokenizer (tokenizer.json via `tokenizers`) with the two methods the task uses."""

    def __init__(self, repo):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(fetch(repo, ["tokenizer.json"], "tokenizer-" + repo.split("/")[1])))

    def encode(self, s, add_special_tokens=False):
        return self.t.encode(s, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.t.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def one(self, s):
        ids = self.encode(s)
        assert len(ids) == 1, (s, ids)
        return ids[0]


# ---- independent reference tables (not the task's loader) ---------------------------------------------

full, arr = am.mc_items(str(ROOT / "data" / "popqa_phase1_questions.csv"))
QIDX = {q: i for i, q in enumerate(full.question)}


def ref_table(key):
    """first answer, correctness, raw log-prob per option, digit mass and arranged options, read straight from CSVs."""
    d = A.ref / key
    p1 = real_read(d / "phases.csv").query("phase == 1").set_index("question_index").reindex(range(1000))
    t = pd.DataFrame(index=range(1000))
    t["gtp"] = p1.gt_position
    for k in range(1, 5):
        t[f"opt{k}"] = p1[f"choice_{k}_arranged"]
    if RUNS[key][1] == "off":
        t["first"] = pd.to_numeric(p1.final_answer)
        t["mass"] = p1.digit_mass
        for k in range(1, 5):
            t[f"lp{k}"] = p1[f"logprob_choice_{k}"]
    else:
        sf = real_read(d / "start_format.csv")
        sf = sf[(sf.phase == 1) & (sf.prefix == "Answer: ")].set_index("question_index").reindex(range(1000))
        t["first"] = pd.to_numeric(sf.forced_answer)
        t["mass"] = sf.digit_mass
        for k in range(1, 5):
            t[f"lp{k}"] = np.log(sf[f"prob_{k}"]) + np.log(sf.digit_mass)
    t["correct"] = (t["first"] == t.gtp).astype(float).where(t["first"].notna())
    return t


REF = {k: ref_table(k) for k in RUNS}
for k, t in REF.items():  # same items, option order and answer position as the arrangement (seed 1042)
    for i in range(1000):
        c, gt, _ = am.arranged_choices(full, arr, i)
        assert [t.at[i, f"opt{q}"] for q in range(1, 5)] == list(c) and gt == t.gtp[i], (k, i)


def expected_prompt(cond, a, b, shown):
    """The PLAN.md prompt, written out independently of the task's builder."""
    def block(lab, x):
        o = [REF["instruct-dpo"].at[x, f"opt{q}"] for q in range(1, 5)]
        s = f"Question {lab}: {full.question[x]}\n1) {o[0]}\n2) {o[1]}\n3) {o[2]}\n4) {o[3]}\n"
        return s + (f"Answer given: {shown[x]}) {o[shown[x] - 1]}\n" if cond != "P" else "")
    if cond == "P":
        return ("You will be shown two multiple choice questions. Do not answer them.\n\n" + block("A", a) + "\n"
                + block("B", b) + "\nIf you were asked both questions, which one would you be more likely to answer "
                "correctly? Respond only with A or B.")
    return ("You will be shown two multiple choice questions, together with answers that were given to them.\n\n"
            + block("A", a) + "\n" + block("B", b) + "\nWhich of the two answers is more likely to be correct? "
            "Respond only with A or B.")


# ---- fake engine ---------------------------------------------------------------------------------------

class FakeCtx:
    def __init__(self, cfg, judge):
        model, thinking, _ = RUNS[cfc.tag_key(cfg["ref_tag"])]
        self.cfg, self.judge, self.key = cfg, judge, cfc.tag_key(cfg["ref_tag"])
        self.tpl, self.tok = template(model), Tok(TOKENIZER[thinking])
        self.thinking_active = thinking == "native"
        self.events, self.texts = [], None

    def save(self, name, rows):
        self.events.append(("save", name, len(rows)))

    def render(self, prompt, prefix=""):
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.tpl.render(messages=msgs, add_generation_prompt=True, bos_token="",
                               eos_token="<|endoftext|>") + prefix

    @property
    def llm(self):
        ctx = self

        class LLM:
            def generate(self, texts, sp):
                assert sp.max_tokens == 1 and sp.temperature == 0 and sp.logprobs == 20
                ctx.events.append(("generate", len(texts)))
                ctx.texts = list(texts)
                outs = []
                for text in texts:
                    top = ctx.judge(ctx, text)
                    best = max(top, key=top.get)
                    lps = {ctx.tok.one(s): NS(decoded_token=s, logprob=lp) for s, lp in top.items()}
                    outs.append(NS(outputs=[NS(token_ids=[ctx.tok.one(best)], finish_reason="length",
                                               logprobs=[lps])]))
                return outs
        return LLM()


def parse(text):
    """Items labelled A and B, and the shown answers (None in P), read back from a rendered text."""
    qa = QIDX[re.search(r"Question A: (.*)\n", text).group(1)]
    qb = QIDX[re.search(r"Question B: (.*)\n", text).group(1)]
    shown = [int(x) for x in re.findall(r"Answer given: ([1-4])\) ", text)]
    return qa, qb, (shown if shown else None)


def emit(ctx, p_a, mass=0.95):
    """Top-20 with P(A) = p_a among the labels, a "Question" and a digit token beside them."""
    sp = " " if ctx.thinking_active else ""
    top = {sp + "Question": math.log(0.03), "1": math.log(0.02)}
    if p_a > 0:
        top[sp + "A"] = math.log(mass * p_a)
    if p_a < 1:
        top[sp + "B"] = math.log(mass * (1 - p_a))
    return top


def internal_judge(ctx, text):
    """Prefers the item whose answer the judge's own log-prob ranks higher (the internal choice)."""
    a, b, shown = parse(text)
    R = REF[ctx.key]
    ans = shown or [int(R["first"][a]), int(R["first"][b])]
    sa, sb = R.at[a, f"lp{ans[0]}"], R.at[b, f"lp{ans[1]}"]
    return emit(ctx, 0.9 if sa > sb else 0.1 if sa < sb else 0.5)


def position_judge(ctx, text):
    return emit(ctx, 0.8)


def oracle_judge(ctx, text):
    """Prefers the item whose (shown, or in P the judge's own) answer is right."""
    a, b, shown = parse(text)
    R = REF[ctx.key]
    ans = shown or [int(R["first"][a]), int(R["first"][b])]
    ra, rb = ans[0] == R.gtp[a], ans[1] == R.gtp[b]
    return emit(ctx, 0.5 if ra == rb else 0.97 if ra else 0.03)


def no_label_judge(ctx, text):
    return {"Question": math.log(0.6), "The": math.log(0.3)}


def config(key, cross=True, n_pairs=0, **extra):
    model, thinking, dtype = RUNS[key]
    opts = f"ref_tag={cfc.REF_PREFIX}{key},n_pairs={n_pairs}"
    if cross:
        opts += f",cross_ref_tag={cfc.REF_PREFIX}{cfc.PARTNER[key]}"
    f = dict(model=model, thinking=thinking, n=1000, budget=4096, max_model_len=8192, temperature=-1.0, seed=42,
             tp=1, revision="", dtype=dtype, questions="popqa", tag=f"cfc-{key}", task_opts=opts)
    f.update(extra)
    return cfc.TASK.config(f)


def run(key, judge, n_pairs=0, save_dir=None):
    cfg = config(key, n_pairs=n_pairs)
    ctx = FakeCtx(cfg, judge)
    orig = cfc.CFC.score

    def score(self, *a, **k):
        ctx.events.append(("score",))
        return orig(self, *a, **k)
    cfc.CFC.score = score
    try:
        files, params = cfc.TASK.run(ctx)
    finally:
        cfc.CFC.score = orig
    assert "score_error" not in params, params.get("score_error")
    t = pd.DataFrame(files["cfc.csv"])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        t.to_csv(save_dir / "cfc.csv", index=False)
        pd.DataFrame(files["raw_generations.csv"]).to_csv(save_dir / "raw_generations.csv", index=False)
        t = pd.read_csv(save_dir / "cfc.csv")  # as analyze reads it
        (save_dir / "params.json").write_text(json.dumps({**cfg, **params}, indent=2))  # as run(): no default=
    return cfg, ctx, files, params, t


# ---- 1. the pair set ----------------------------------------------------------------------------------

cfc.check_pair_module()
bp.check_pairs(full, cfc.pairs_data.PAIRS)
P = pd.DataFrame(cfc.pairs_data.PAIRS, columns=["pair_id", "i", "j", "kind", "round", "ddec"])
assert not P.duplicated(["i", "j"]).any()
# subjects re-derived without the templates: the frame shared by all questions of a relation is cut off
subj = {}
for prop, g in full.groupby("prop"):
    qs = list(g.question)
    if len(qs) == 1:  # one item ("color"): no frame to derive
        subj[g.index[0]] = bp.subjects(g)[0]
        continue
    pre = os.path.commonprefix(qs)
    pre = pre[:pre.rfind(" ") + 1]  # the frame ends at a word boundary
    suf = os.path.commonprefix([q[::-1] for q in qs])[::-1]
    while suf and not (suf[0] in " '" or suf == "?"):  # ... and resumes at one (" the capital of?", "'s occupation?")
        suf = suf[1:]
    for x, q in zip(g.index, qs):
        subj[x] = q[len(pre):len(q) - len(suf)]
mine = [subj[i] for i in range(1000)]
assert mine == bp.subjects(full), [(a, b) for a, b in zip(mine, bp.subjects(full)) if a != b][:5]
norm = lambda s: str(s).strip().lower()
opts = [{norm(full.at[i, f"choice_{k}"]) for k in range(1, 5)} for i in range(1000)]
dec = pd.qcut(full.s_pop, 10, labels=False).to_numpy()


def word_in(needle, hay):
    """needle occurs in hay with no word character on either side (str.find scan, not the builder's regex)."""
    k = hay.find(needle)
    while k >= 0:
        before = hay[k - 1] if k else " "
        after = hay[k + len(needle)] if k + len(needle) < len(hay) else " "
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return True
        k = hay.find(needle, k + 1)
    return False


ques = [q.lower() for q in full.question]
for r in P.itertuples():
    si, sj = norm(mine[r.i]), norm(mine[r.j])
    assert si != sj and si not in opts[r.j] and sj not in opts[r.i], r
    for x, y in ((r.i, r.j), (r.j, r.i)):  # fuzzy leaks: an option inside the other question, a subject in an option
        assert not any(word_in(o, ques[y]) for o in opts[x]), (r, x, y)
        assert not any(word_in(norm(mine[y]), o) for o in opts[x]), (r, x, y)
    assert r.ddec == abs(dec[r.i] - dec[r.j]), r
    if r.kind == "matched":
        assert full.prop[r.i] == full.prop[r.j] and r.ddec <= 1, r
leaks = sum(norm(mine[i]) in opts[j] for i in range(1000) for j in range(1000) if i != j)
never = 1000 - len(set(P[P.kind == "matched"].i) | set(P[P.kind == "matched"].j))
print(f"pairs: {len(P)} ({P.kind.value_counts().to_dict()}; matched by decile difference "
      f"{P[P.kind == 'matched'].ddec.value_counts().sort_index().to_dict()}), constraints hold (incl. fuzzy leaks); "
      f"{leaks} exact leaking (subject, other item) combinations exist and none is paired; {never} items in no "
      f"matched pair")
# choice_4 is the type-mismatched distractor: never an answer of its own relation, unlike choices 1-3
pool = {p: set(g.choice_1.map(norm)) for p, g in full.groupby("prop")}
share = {k: np.mean([norm(full.at[i, f"choice_{k}"]) in pool[full.prop[i]] - {norm(full.choice_1[i])}
                     for i in range(1000)]) for k in range(1, 5)}
assert share[4] == 0 and min(share[k] for k in (2, 3)) > 0.15, share
print("share of options that are the answer of another item of the same relation, by choice column:",
      {k: round(v, 3) for k, v in share.items()}, "-> choice_4 is the wrong-type distractor")

# ---- 2. every checkpoint, internal judge: prompts, references, dedup, scoring ----------------------------

summary = []
for key in RUNS:
    model, thinking, dtype = RUNS[key]
    cfg, ctx, files, params, t = run(key, internal_judge, save_dir=A.out / f"cfc-{key}")
    texts = ctx.texts
    assert ctx.events[:3] == [("generate", len(texts)), ("save", "raw_generations.csv", len(texts)), ("score",)], \
        ctx.events  # raw readout saved before any scoring
    raw = pd.DataFrame(files["raw_generations.csv"])
    assert list(raw.text_id) == list(range(len(texts)))
    # the local template renders the reference's own Phase 1 prompt exactly as the reference run did
    rp = json.loads(real_read_text(A.ref / key / "params.json"))
    p1 = real_read(A.ref / key / "phases.csv").query("phase == 1").set_index("question_index")
    assert ctx.render(p1.at[0, "prompt"]) == rp["rendered_example"], key
    sys_default = ctx.render("X").split("<|im_start|>user")[0]
    end = "<|im_start|>assistant\n<think>\n</think>\n\nAnswer:" if thinking == "native" else "<|im_start|>assistant\n"
    prefix = "\n</think>\n\nAnswer:" if thinking == "native" else ""
    assert params["prefix"] == prefix, (key, params["prefix"])
    for x in texts:
        assert x.startswith(sys_default) and x.endswith(end) and not re.search(r"\bnan\b", x), x[-300:]
        assert x.count("<|im_start|>user") == 1 and x.count("Respond only with A or B.") == 1
    # every row, both orders: the rendered text is the expected prompt with the target's answers, in reference order
    M, N = REF[key], REF[cfc.PARTNER[key]]
    for r in t.itertuples():
        T = M if r.condition in "PS" else N
        shown = {r.i: int(T["first"][r.i]), r.j: int(T["first"][r.j])}
        assert r.target_answer_i == shown[r.i] and r.target_answer_j == shown[r.j]
        assert texts[r.text_id_ij] == ctx.render(expected_prompt(r.condition, r.i, r.j, shown), prefix), (key, r)
        assert texts[r.text_id_ji] == ctx.render(expected_prompt(r.condition, r.j, r.i, shown), prefix), (key, r)
    # pairs: all selected; skipped exactly where the target lacks a first answer
    for cond, T in (("P", M), ("S", M), ("X", N)):
        has = T["first"].notna().to_numpy()
        want = P[has[P.i] & has[P.j]].pair_id.tolist()
        assert t[t.condition == cond].pair_id.tolist() == want, (key, cond)
        assert params["skipped_pairs"].get(cond, 0) == len(P) - len(want)
    # S and X: same template, differ only in the answers; identical texts deduplicated
    s, x = t[t.condition == "S"].set_index("pair_id"), t[t.condition == "X"].set_index("pair_id")
    common = s.index.intersection(x.index)
    for pid in common:
        for col in ("text_id_ij", "text_id_ji"):
            u, v = texts[s.at[pid, col]], texts[x.at[pid, col]]
            assert re.sub(r"Answer given: .*\n", "", u) == re.sub(r"Answer given: .*\n", "", v)
    same = ((M["first"][s.loc[common].i].to_numpy() == N["first"][s.loc[common].i].to_numpy())
            & (M["first"][s.loc[common].j].to_numpy() == N["first"][s.loc[common].j].to_numpy()))
    assert (s.loc[common].sx_identical.astype(bool).to_numpy() == same).all()
    assert (s.loc[common].text_id_ij.to_numpy()[same] == x.loc[common].text_id_ij.to_numpy()[same]).all()
    assert len(texts) == 2 * len(t) - 2 * int(same.sum()), "dedup"
    # correctness, internal scores, distractor flag, all from the independent tables
    for cond, T in (("P", M), ("S", M), ("X", N)):
        u = t[t.condition == cond]
        assert (u.correct_i.to_numpy() == T.correct[u.i].to_numpy()).all()
        assert (u.decile_diff.to_numpy() == P.set_index("pair_id").ddec[u.pair_id].to_numpy()).all()
        si = np.array([M.at[i, f"lp{a}"] for i, a in zip(u.i, u.target_answer_i)])
        sj = np.array([M.at[j, f"lp{a}"] for j, a in zip(u.j, u.target_answer_j)])
        ok = (M.mass[u.i].to_numpy() >= 0.5) & (M.mass[u.j].to_numpy() >= 0.5) & np.isfinite(si) & np.isfinite(sj)
        ic = np.where(ok, np.where(si > sj, 1.0, np.where(si < sj, 0.0, 0.5)), np.nan)
        assert np.allclose(u.int_choice_i, ic, equal_nan=True), (key, cond)
        sc = u.scored.to_numpy() == 1
        want = np.where(sc & ok, ic * u.correct_i + (1 - ic) * u.correct_j, np.nan)
        assert np.allclose(u.int_correct, want, equal_nan=True)
        d4 = np.array([[q for q in range(1, 5) if T.at[i, f"opt{q}"] == full.choice_4[i]][0] for i in u.i])
        assert (u.distractor_pos_i.to_numpy() == d4).all()
        assert (u.target_distractor.to_numpy() == ((u.target_answer_i.to_numpy() == d4)
                                                    | (u.target_answer_j.to_numpy() == u.distractor_pos_j.to_numpy()))).all()
        # this judge follows the internal ranking: stated = internal wherever the internal choice is defined
        both = sc & ok
        assert np.allclose(u.chosen_correct[both], u.int_correct[both]) and (u.choice_i[ok] == ic[ok]).all()
    rep = cfc.TASK.analyze(cfg, A.out / f"cfc-{key}")
    assert "analyze_error" not in rep, rep.get("analyze_error")
    for cond in "PSX":
        assert abs(rep[f"{cond}_stated_on_internal_set"] - rep[f"{cond}_internal_all"]) < 1e-4, (key, cond, rep)
        assert rep[f"{cond}_agreement"] >= 0.999 and abs(rep[f"{cond}_ab_mass"] - 0.95) < 1e-6
        assert rep[f"{cond}_stated_all_lo"] <= rep[f"{cond}_stated_all"] <= rep[f"{cond}_stated_all_hi"]
    summary.append(dict(run=key, rows=len(t), prompts=2 * len(t), unique_texts=len(texts), P_skipped=params[
        "skipped_pairs"].get("P", 0), X_skipped=params["skipped_pairs"].get("X", 0),
        sx_identical=round(float(same.mean()), 3), P_internal_matched=rep["P_internal_matched"],
        P_internal_all=rep["P_internal_all"], S_internal=rep["S_internal_all"], X_internal=rep["X_internal_all"],
        first_token=raw.first_token.value_counts().index[0], label_tokens=params["label_tokens"]))
pd.set_option("display.width", 250)
print(pd.DataFrame(summary).drop(columns="label_tokens").to_string(index=False))
for s_ in summary[:1] + summary[3:4]:
    print(f"label tokens after the prefix ({s_['run']}): {s_['label_tokens']}")
for key in ("instruct-dpo", "think-dpo"):
    p = json.loads((A.out / f"cfc-{key}" / "params.json").read_text())
    print(f"\n======== {key}: rendered P / X example ========\n{p['rendered_P']}\n--------\n{p['rendered_X'][-420:]}")
print("\nINTERNAL-JUDGE RUNS OK (6 checkpoints, every prompt checked)")

# ---- 3. scripted judges with known values ---------------------------------------------------------------

for key in ("instruct-dpo", "think-dpo"):
    _, _, _, _, t = run(key, position_judge, save_dir=A.out / "position")
    rep = cfc.TASK.analyze(None, A.out / "position")
    for cond in "PSX":
        assert rep[f"{cond}_stated_all"] == 0.5 and rep[f"{cond}_ties"] == 1.0, (key, cond, rep)
        assert abs(rep[f"{cond}_position_bias"] - 0.8) < 1e-9
    _, _, _, _, t = run(key, oracle_judge, save_dir=A.out / "oracle")
    rep = cfc.TASK.analyze(None, A.out / "oracle")
    for cond in "PSX":
        assert rep[f"{cond}_stated_all"] == 1.0 and rep[f"{cond}_n_scored_all"] > 300, (key, cond, rep)
    _, _, _, params, t = run(key, no_label_judge, n_pairs=200, save_dir=A.out / "nolabel")
    rep = cfc.TASK.analyze(None, A.out / "nolabel")
    assert params["n_pairs_selected"] == 200 and set(t.kind) == {"matched", "random"}
    assert rep["P_ab_mass"] < 1e-15 and rep["P_ties"] == 1.0 and rep["P_stated_all"] == 0.5 and \
        rep["P_share_first_question"] == 1.0
print("SCRIPTED JUDGES OK: position judge 0.5 (bias 0.8), oracle 1.0, no-label judge 0.5 with A/B mass 0; pilot "
      "subset of 200 pairs mixes matched and random")
sel = cfc.select_pairs(200)
assert sel == cfc.select_pairs(200) and len({p[0] for p in sel}) == 200 and sel[:50] != cfc.pairs_data.PAIRS[:50]

# ---- 4. one-shot safety ----------------------------------------------------------------------------------

cfg = config("think-dpo")
ctx = FakeCtx(cfg, internal_judge)
orig = cfc.CFC.score
cfc.CFC.score = lambda *a, **k: 1 / 0
try:
    files, params = cfc.TASK.run(ctx)
finally:
    cfc.CFC.score = orig
assert set(files) == {"raw_generations.csv"} and "ZeroDivisionError" in params["score_error"]
assert ctx.events == [("generate", len(ctx.texts)), ("save", "raw_generations.csv", len(ctx.texts))]
empty = A.out / "empty"
empty.mkdir(parents=True, exist_ok=True)
assert "analyze_error" in cfc.TASK.analyze(cfg, empty)
print("ONE-SHOT SAFETY OK: a scoring failure returns the raw readout + score_error; analyze on an empty dir returns "
      "analyze_error")

# ---- 5. references and pairing are checked ---------------------------------------------------------------

def fails(fn):
    try:
        fn()
    except (AssertionError, SystemExit):
        return True
    return False


assert fails(lambda: config("instruct-dpo", task_opts="ref_tag=olmo32b-popqa-instruct-dpo,"
                                                      "cross_ref_tag=olmo32b-popqa-think-sft-main"))
assert fails(lambda: config("think-dpo", task_opts="ref_tag=olmo32b-popqa-think-dpo,"
                                                   "cross_ref_tag=olmo32b-popqa-instruct-rl"))
assert fails(lambda: config("instruct-dpo", model="allenai/Olmo-3.1-32B-Instruct-SFT"))   # ref is another model
assert fails(lambda: config("instruct-dpo", task_opts="ref_tag=olmo32b-popqa-instruct-dpo,conditions=PQ"))
assert fails(lambda: config("instruct-dpo", questions="simpleqa"))
assert fails(lambda: config("instruct-dpo", temperature=0.6))
assert "X" not in config("instruct-dpo", cross=False)["conditions"]
good = json.loads(real_read_text(A.ref / "think-dpo" / "params.json"))
c0 = config("instruct-dpo")
cfc.check_partner(c0, good)
assert fails(lambda: cfc.check_partner(c0, json.loads(real_read_text(A.ref / "think-sft-main" / "params.json"))))
assert fails(lambda: cfc.check_partner(dict(c0, cross_ref_tag="olmo32b-popqa-think-rl31-main"), good))
inst = json.loads(real_read_text(A.ref / "instruct-dpo" / "params.json"))
from tasks.second_chance import check_reference  # noqa: E402
check_reference(dict(c0, think=0), inst, "phases")
assert fails(lambda: check_reference(dict(c0, think=0, dtype="auto"), inst, "phases"))      # Instruct DPO is bf16
# a wrong reference at run time (here: a cross reference whose params name another checkpoint) stops before generating
bad = copy.deepcopy(good)
bad["model"] = "allenai/Olmo-3-32B-Think-SFT"
ctx = FakeCtx(c0, internal_judge)
real_rt = Path.read_text
Path.read_text = lambda self, *a, **k: (json.dumps(bad) if str(self).replace("\\", "/").endswith("think-dpo/params.json")
                                        else real_rt(self, *a, **k))
try:
    assert fails(lambda: cfc.TASK.run(ctx)) and not ctx.events
finally:
    Path.read_text = real_rt
print("REFERENCE AND PAIRING CHECKS OK: wrong partner, model, dtype, conditions, questions and temperature fail")

# ---- 6. registry, sweep hooks, cross-run analysis --------------------------------------------------------

assert tasks.module("cfc").TASK is cfc.TASK and am.load_task("cfc") is cfc.TASK
it = cfc.items(A.out / "cfc-instruct-dpo")
m = cfc.measures(it)
assert set(cfc.show) <= set(m) and cfc.gates(m, m)["gate_format"] is True
dirs = [str(A.out / f"cfc-{k}") for k in RUNS]
res = subprocess.run([sys.executable, str(Path(__file__).parent / "analyze_cfc.py"), *dirs, "--n-boot", "500",
                      "--json", str(A.out / "analyze_cfc.json")], capture_output=True, text=True)
print(res.stdout[-6000:])
assert res.returncode == 0, res.stderr[-3000:]
rj = json.loads((A.out / "analyze_cfc.json").read_text())
# the internal-following judge: stated and internal self-advantage coincide in every stage
for stage, o in rj["all"]["H4_stages"].items():
    assert abs(o["diff"]) < 1e-12, (stage, o)

# ---- 7. H4 with a random (seeded) judge, against an independent implementation of the formula -------------

import hashlib  # noqa: E402

import tasks.cfc.analyze_cfc as ac  # noqa: E402


def random_judge(ctx, text):
    """A fixed pseudo-random preference per (judge, text): stated and internal self-advantage now differ."""
    h = int(hashlib.sha256((ctx.key + text).encode()).hexdigest()[:8], 16) / 16 ** 8
    return emit(ctx, 0.05 + 0.9 * h)


h4runs = {k: dict(t=run(k, random_judge, n_pairs=1500, save_dir=A.out / "h4" / k)[4]) for k in RUNS}


def h4_independent(runs, m, n):
    """1/2{[acc(M->M) - acc(N->M)] + [acc(N->N) - acc(M->N)]}, row by row from the four tables."""
    rows = {(key, r.condition, r.pair_id): r for key in (m, n) for r in runs[key]["t"].itertuples()
            if r.condition in ("S", "X")}
    pids = sorted({p for _, _, p in rows if all((key, c, p) in rows for key in (m, n) for c in "SX")})
    keep = [p for p in pids if all(
        not np.isnan(rows[(key, c, p)].int_choice_i) and not any(np.isnan(getattr(rows[(key, c, p)], f)) for f in (
            "self_first_i", "self_first_j", "partner_first_i", "partner_first_j")) for key in (m, n) for c in "SX")]
    for p in keep:  # who is judged on whose answers
        assert rows[(n, "X", p)].target_tag == cfc.REF_PREFIX + m and rows[(m, "X", p)].target_tag == cfc.REF_PREFIX + n
    tm = [p for p in keep if rows[(m, "S", p)].scored == 1]   # target M
    tn = [p for p in keep if rows[(n, "S", p)].scored == 1]   # target N
    out = {}
    for what, col in (("stated", "chosen_correct"), ("internal", "int_correct")):
        acc_ = lambda key, c, ps: float(np.mean([getattr(rows[(key, c, p)], col) for p in ps]))
        MM, NM, NN, MN = acc_(m, "S", tm), acc_(n, "X", tm), acc_(n, "S", tn), acc_(m, "X", tn)
        out[what] = 0.5 * ((MM - NM) + (NN - MN))
        # a wrong pairing (S and X of the same run, i.e. the same judge): must differ from the right one
        wrong = lambda key, c, ps: float(np.nanmean([getattr(rows[(key, c, p)], col) for p in ps]))
        out[what + "_swapped"] = 0.5 * ((MM - wrong(m, "X", tm)) + (NN - wrong(n, "X", tn)))
    return out, len(keep)


C = cfc.boot_counts(1000, 50, 1)
teeth = []
for stage, m, n in cfc.STAGES:
    o, _ = ac.h4_stage(h4runs, m, n, C, drop=False)
    ind, nk = h4_independent(h4runs, m, n)
    assert o["n_base"] == nk, (stage, o["n_base"], nk)
    assert abs(o["sa_stated"] - ind["stated"]) < 1e-12 and abs(o["sa_internal"] - ind["internal"]) < 1e-12, (o, ind)
    assert abs(o["diff"] - (ind["stated"] - ind["internal"])) < 1e-12
    teeth.append((abs(o["diff"]), abs(ind["stated_swapped"] - ind["stated"])))
    print(f"H4 {stage}: n {nk}, stated SA {o['sa_stated']:+.4f}, internal SA {o['sa_internal']:+.4f}, "
          f"diff {o['diff']:+.4f} (a swapped term pairing would give stated SA {ind['stated_swapped']:+.4f})")
assert max(d for d, _ in teeth) > 0.005 and min(s for _, s in teeth) > 1e-6, teeth
print("H4 FORMULA OK: h4_stage equals an independent row-by-row implementation under a random judge")
print("ALL CHECKS PASSED")
