"""Confidence forced choice (CFC; Mamassian & de Gardelle 2022) across Olmo-3 training stages, on the PopQA Phase 1
items. Spec: tasks/cfc/PLAN.md. Launch lines: tasks/cfc/LAUNCH.md. Cross-run tests: tasks/cfc/analyze_cfc.py.

The model M sees two items (a fixed pair set, tasks/cfc/pairs_data.py, built by build_pairs.py) and says which one it
is more likely to get right. Conditions (each pair in both orders, A/B labels follow position):
    P  prospective: two questions, no answers                      judge M, target M
    S  retrospective, self: M's first answers shown                 judge M, target M
    X  retrospective, cross: the same-stage sibling N's answers      judge M, target N
The target's first answers, their correctness and the internal signal come from the confidence_abstention reference
runs (--task-opts ref_tag=<M's run>, cross_ref_tag=<N's run>), each read in its own no-trace mode, chosen from ITS OWN
params.json thinking:
    off    (Instruct) phases.csv Phase 1: final_answer, is_correct, logprob_choice_k, digit_mass
    native (Think)    start_format.csv, prefix "Answer: ", phase 1: forced_answer, correct = forced_answer == gt_position
                      (phases.csv), raw log-prob log(prob_k) + log(digit_mass); items without a readout (trace cut in the
                      reference run) have no first answer and their pairs are skipped
Readout: every prompt rendered with the chat template; Instruct (--thinking off) after "assistant\\n", Think (--thinking
native) after a closed think block + "Answer:" (no trailing space: the letter is the token " A"). One greedy token, top-20
log-probs at position 0 (second_chance.start_readout), each unique text once (S and X often coincide). Label log-prob:
max over "A", " A", "A)", " A)", "A.", " A." (-50 when absent from the top 20); P(A) renormalized over A/B.
Per item preference: pref(i) = (P(A | i first) + P(B | i second)) / 2; the choice is the item with pref > 0.5, a tie
(pref = 0.5) counts half. A pair is scored when the target has exactly one of the two items right.
Internal choice: the judge's raw log-prob of the target's answer (judge = M; for X that is M's log-prob of N's answer),
higher wins; items whose judge readout put < 0.5 of its mass on the digits are excluded.

Outputs: raw_generations.csv (one row per unique text, saved right after the single generate call), cfc.csv (one row
per condition x pair), analysis.json (per-condition metrics with item-bootstrap intervals).
"""

import hashlib
import json
import math
import random
import re
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import QUESTION_FILES, arranged_choices, default_tag, mc_items, parse_task_opts, probe_text
from tasks.second_chance import START_PREFIX, check_reference, first_answers, start_readout

from . import build_pairs as bp
from . import pairs_data

REF_PREFIX = "olmo32b-popqa-"
# reference tag (without REF_PREFIX) -> (model, revision, thinking) of the checkpoint it was run on
CHECKPOINTS = {
    "instruct-sft": ("allenai/Olmo-3.1-32B-Instruct-SFT", None, "off"),
    "instruct-dpo": ("allenai/Olmo-3.1-32B-Instruct-DPO", None, "off"),
    "instruct-rl": ("allenai/Olmo-3.1-32B-Instruct", None, "off"),
    "think-sft-main": ("allenai/Olmo-3-32B-Think-SFT", None, "native"),
    "think-dpo": ("allenai/Olmo-3-32B-Think-DPO", None, "native"),
    "think-rl31-main": ("allenai/Olmo-3.1-32B-Think", None, "native"),
}
# same-stage partner across the two lines (judge M <-> sibling N)
PARTNER = {"instruct-sft": "think-sft-main", "instruct-dpo": "think-dpo", "instruct-rl": "think-rl31-main"}
PARTNER.update({v: k for k, v in list(PARTNER.items())})
STAGES = [("sft", "instruct-sft", "think-sft-main"), ("dpo", "instruct-dpo", "think-dpo"),
          ("rl", "instruct-rl", "think-rl31-main")]

LABELS = ("A", "B")
LABEL_VARIANTS = ("{}", " {}", "{})", " {})", "{}.", " {}.")
THINK_PREFIX = "Answer:"  # after the closed think block; no trailing space (letters tokenize as " A")
MIN_DIGIT_MASS = 0.5      # internal ranking only where the judge's answer token was a digit
PILOT_SEED = 1042         # fixed shuffled order of the pair ids; n_pairs > 0 takes its first n
TIE_EPS = 1e-9

P_HEAD = "You will be shown two multiple choice questions. Do not answer them.\n\n"
P_TAIL = ("If you were asked both questions, which one would you be more likely to answer correctly? "
          "Respond only with A or B.")
R_HEAD = "You will be shown two multiple choice questions, together with answers that were given to them.\n\n"
R_TAIL = "Which of the two answers is more likely to be correct? Respond only with A or B."
ANSWER_LINE_RE = re.compile(r"^Answer given: [1-4]\) .*\n", re.M)


# ---- prompts and readout ---------------------------------------------------------------------------

def item_block(label, question, options, shown=None):
    """One item of the pair prompt; `shown` is the answer position (1-4) for S/X, None for P."""
    s = f"Question {label}: {question}\n" + "".join(f"{k}) {options[k - 1]}\n" for k in range(1, 5))
    if shown is not None:
        s += f"Answer given: {shown}) {options[shown - 1]}\n"
    return s


def pair_prompt(cond, a, b):
    """a, b = (question, options, shown) of the items labelled A and B. S and X share one template."""
    head, tail = (P_HEAD, P_TAIL) if cond == "P" else (R_HEAD, R_TAIL)
    return head + item_block("A", *a) + "\n" + item_block("B", *b) + "\n" + tail


def letter_logprobs(top, labels=LABELS):
    """Log-prob of each label at the first token: the max over its variants; -50 when none is in the top 20."""
    return {lab: max([top[v.format(lab)] for v in LABEL_VARIANTS if v.format(lab) in top], default=-50.0)
            for lab in labels}


def ab_readout(top):
    """P(A) renormalized over A/B, the A/B probability mass, and both log-probs."""
    lp = letter_logprobs(top)
    a, b = lp["A"], lp["B"]
    m = max(a, b)
    pa = math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))
    return pa, math.exp(a) + math.exp(b), a, b


def tag_key(tag):
    assert tag.startswith(REF_PREFIX) and tag[len(REF_PREFIX):] in CHECKPOINTS, (
        f"reference tag {tag!r} is not one of {[REF_PREFIX + k for k in CHECKPOINTS]}")
    return tag[len(REF_PREFIX):]


def select_pairs(n_pairs):
    """All pairs, or the first n_pairs of a fixed shuffled order (mixes matched and random), in pair_id order."""
    pairs = pairs_data.PAIRS
    if not n_pairs:
        return list(pairs)
    order = list(range(len(pairs)))
    random.Random(PILOT_SEED).shuffle(order)
    return [pairs[k] for k in sorted(order[:n_pairs])]


def check_pair_module():
    assert bp.pairs_sha256(pairs_data.PAIRS) == pairs_data.PAIRS_SHA256, "pairs_data.py was edited: rebuild it"


# ---- reference runs --------------------------------------------------------------------------------

def reference(ref_dir, n=1000):
    """A reference run's params, readout source and per-item table: question, gt_position, arranged options, the first
    answer (NaN: no readout), its correctness, the raw log-prob of every option (lp_1..4) and the digit mass."""
    params = json.loads((ref_dir / "params.json").read_text())
    source = {"native": "start_format", "off": "phases"}[params["thinking"]]
    if source == "start_format":
        assert (ref_dir / "start_format.csv").exists(), f"{ref_dir.name} has no start_format.csv"
    ref = first_answers(ref_dir, n, source)
    ph = pd.read_csv(ref_dir / "phases.csv")
    p1 = ph[ph.phase == 1].set_index("question_index")
    for k in range(1, 5):
        ref[f"option_{k}"] = p1[f"choice_{k}_arranged"].reindex(ref.index)
    if source == "phases":
        for k in range(1, 5):
            ref[f"lp_{k}"] = p1[f"logprob_choice_{k}"].reindex(ref.index)
    else:  # prob_k is renormalized over the digits: back to the raw log-prob of the token
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(1, 5):
                ref[f"lp_{k}"] = np.log(ref[f"base_p{k}"]) + np.log(ref.base_digit_mass)
    ref["correct"] = (ref["first"] == ref.gt_position).astype(float).where(ref["first"].notna())
    if source == "phases":
        ok = ref["first"].notna()
        assert (ref.correct[ok] == p1.is_correct.reindex(ref.index)[ok]).all(), f"{ref_dir.name}: is_correct differs"
    ref["mass"] = ref.base_digit_mass
    return params, source, ref


def check_partner(cfg, params):
    """The cross reference is the fixed same-stage sibling of this run's reference, in its own settings."""
    m_key, n_key = tag_key(cfg["ref_tag"]), tag_key(cfg["cross_ref_tag"])
    assert PARTNER[m_key] == n_key, f"cross_ref_tag must be {REF_PREFIX + PARTNER[m_key]}, got {cfg['cross_ref_tag']}"
    got = (params.get("model"), params.get("revision") or None, params.get("thinking"))
    assert got == CHECKPOINTS[n_key], f"{cfg['cross_ref_tag']} is {got}, expected {CHECKPOINTS[n_key]}"
    assert params.get("questions") == cfg["questions"] and params.get("n", 1000) >= cfg["n"]


def check_items(full, arr, ref, name):
    """The reference shows every item with the questions, options and option order of the arrangement (seed 1042)."""
    assert (ref.question.to_numpy() == full.question.to_numpy()[:len(ref)]).all(), f"{name}: different items"
    for i in range(len(ref)):
        c, gt, _ = arranged_choices(full, arr, i)
        assert gt == ref.gt_position.iat[i], f"{name} item {i}: answer position differs"
        assert [ref[f"option_{k}"].iat[i] for k in range(1, 5)] == list(c), f"{name} item {i}: option order differs"


# ---- the task --------------------------------------------------------------------------------------

class CFC:
    name = "cfc"
    options = {"ref_tag": "", "cross_ref_tag": "", "conditions": "PSX", "n_pairs": 0, "pairs_sha": ""}

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        assert f["thinking"] in ("off", "native"), "cfc runs Instruct with --thinking off, Think with --thinking native"
        assert f["questions"] == "popqa", "the pair set is over the PopQA Phase 1 items: pass --questions popqa"
        assert f["n"] == 1000, "the pair set indexes all 1000 Phase 1 items; use n_pairs to run fewer pairs"
        assert f["temperature"] <= 0, "the readout is one greedy token"
        assert opts["ref_tag"], "--task-opts ref_tag=<confidence_abstention run of this checkpoint> is required"
        m_key = tag_key(opts["ref_tag"])
        assert CHECKPOINTS[m_key] == (f["model"], f["revision"] or None, f["thinking"]), (
            f"ref_tag {opts['ref_tag']} is {CHECKPOINTS[m_key]}; this run is {f['model']} {f['revision']} {f['thinking']}")
        conds = opts["conditions"].upper()
        assert conds and set(conds) <= set("PSX") and len(set(conds)) == len(conds), "conditions: letters of PSX"
        if opts["cross_ref_tag"]:
            n_key = tag_key(opts["cross_ref_tag"])
            assert PARTNER[m_key] == n_key, (f"cross_ref_tag must be the same-stage sibling {REF_PREFIX + PARTNER[m_key]}"
                                             f", got {opts['cross_ref_tag']}")
        elif "X" in conds:
            print("cfc: no cross_ref_tag, so no X condition")
            conds = conds.replace("X", "")
            assert conds, "no condition left"
        # the pair set: unedited, over these items, every constraint holds (checked locally, before any GPU time)
        check_pair_module()
        local = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / QUESTION_FILES["popqa"].format(1))
        assert bp.items_sha256(local) == pairs_data.ITEMS_SHA256, "data/popqa_phase1_questions.csv changed"
        bp.check_pairs(local, pairs_data.PAIRS)
        if opts["pairs_sha"]:
            assert pairs_data.PAIRS_SHA256.startswith(opts["pairs_sha"]), "pairs_sha does not match pairs_data.py"
        assert 0 <= opts["n_pairs"] <= len(pairs_data.PAIRS)
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=0.0, seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"],
                    ref_tag=opts["ref_tag"], cross_ref_tag=opts["cross_ref_tag"], conditions=conds,
                    n_pairs=opts["n_pairs"], pairs_sha256=pairs_data.PAIRS_SHA256,
                    items_sha256=pairs_data.ITEMS_SHA256, tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        full, arr = mc_items("/data/" + QUESTION_FILES[cfg["questions"]].format(1))
        assert bp.items_sha256(full) == pairs_data.ITEMS_SHA256, "/data/popqa_phase1_questions.csv changed"
        check_pair_module()
        pairs = select_pairs(cfg["n_pairs"])
        m_params, m_source, M = reference(Path("/results") / cfg["ref_tag"])
        check_reference(dict(cfg, think=0), m_params, m_source)
        check_items(full, arr, M, cfg["ref_tag"])
        N, n_source = None, None
        if cfg["cross_ref_tag"]:
            n_params, n_source, N = reference(Path("/results") / cfg["cross_ref_tag"])
            check_partner(cfg, n_params)
            check_items(full, arr, N, cfg["cross_ref_tag"])
            for col in ["question", "gt_position"] + [f"option_{k}" for k in range(1, 5)]:
                assert (M[col].to_numpy() == N[col].to_numpy()).all(), f"references differ in {col}"
        assert ctx.thinking_active == (cfg["thinking"] == "native")

        # every prompt is built and checked before the one generate call
        targets = {"P": M, "S": M, "X": N}
        uniq, meta, skipped = {}, [], Counter()
        for cond in cfg["conditions"]:
            T = targets[cond]
            for pid, i, j, kind, rnd, _ in pairs:
                if pd.isna(T["first"].iat[i]) or pd.isna(T["first"].iat[j]):
                    skipped[cond] += 1  # never rendered with a missing answer
                    continue
                shown = {x: None if cond == "P" else int(T["first"].iat[x]) for x in (i, j)}
                item = {x: (full.question.iat[x], [M[f"option_{k}"].iat[x] for k in range(1, 5)], shown[x])
                        for x in (i, j)}
                assert all(isinstance(o, str) for x in (i, j) for o in item[x][1]), (cond, pid)  # no NaN option
                assert all(s in (None, 1, 2, 3, 4) for s in shown.values()), (cond, pid)
                ids = []
                for a, b in ((i, j), (j, i)):
                    p = pair_prompt(cond, item[a], item[b])
                    assert p.count("Question A: ") == 1 and p.count("Question B: ") == 1, (cond, pid)
                    assert len(ANSWER_LINE_RE.findall(p)) == (0 if cond == "P" else 2), (cond, pid)
                    ids.append(uniq.setdefault(p, len(uniq)))
                meta.append((cond, pid, i, j, kind, rnd, ids[0], ids[1]))
        prompts = list(uniq)
        r0 = ctx.render(prompts[0])
        prefix = probe_text(r0, "")[len(r0):] + THINK_PREFIX if ctx.thinking_active else ""
        texts = [ctx.render(p, prefix) for p in prompts]
        assert all(t.endswith(prefix) for t in texts) and len(set(texts)) == len(texts)
        by_key = {(c, pid): (a, b) for c, pid, _, _, _, _, a, b in meta}
        for (c, pid), (a, b) in by_key.items():  # S and X differ only in the shown answers
            if c == "X" and ("S", pid) in by_key:
                for u, v in zip((a, b), by_key[("S", pid)]):
                    assert ANSWER_LINE_RE.sub("", prompts[u]) == ANSWER_LINE_RE.sub("", prompts[v]), pid
        examples = {}
        for c, pid, i, j, kind, rnd, a, b in meta:
            examples.setdefault(f"rendered_{c}", texts[a])
        params = dict(prefix=prefix, outcome_source=m_source, cross_outcome_source=n_source,
                      n_pairs_selected=len(pairs), n_prompts=2 * len(meta), n_unique_texts=len(texts),
                      skipped_pairs=dict(skipped), **examples, label_tokens=label_tokens(ctx, prefix))

        gens = start_readout(ctx, texts)
        # one-shot safety: the raw readout reaches the volume first, and nothing after this point raises (run() has
        # Modal retries, which would redo the whole GPU job)
        uses = [[] for _ in texts]
        for c, pid, i, j, kind, rnd, a, b in meta:
            uses[a].append(f"{c}:{pid}:ij")
            uses[b].append(f"{c}:{pid}:ji")
        raw = [dict(text_id=k, text_sha256=hashlib.sha256(t.encode()).hexdigest()[:16], first_token=g["content"],
                    finish_reason=g["finish_reason"], top=json.dumps(g["top"]), uses=";".join(uses[k]))
               for k, (t, g) in enumerate(zip(texts, gens))]
        ctx.save("raw_generations.csv", raw)
        try:
            rows = self.score(cfg, meta, gens, full, arr, M, N)
            return {"cfc.csv": rows, "raw_generations.csv": raw}, params
        except Exception:
            return {"raw_generations.csv": raw}, dict(params, score_error=traceback.format_exc())

    def score(self, cfg, meta, gens, full, arr, M, N):
        read = [ab_readout(g["top"]) for g in gens]
        first_tok = [g["content"] for g in gens]
        tags = {"P": cfg["ref_tag"], "S": cfg["ref_tag"], "X": cfg["cross_ref_tag"]}
        dist = [arr[x][0].index(3) + 1 for x in range(len(full))]  # arranged position of choice_4 (wrong type)
        ddec = {p[0]: p[5] for p in pairs_data.PAIRS}  # |s_pop decile difference| of each pair
        s_ids = {pid: (a, b) for c, pid, _, _, _, _, a, b in meta if c == "S"}
        x_ids = {pid: (a, b) for c, pid, _, _, _, _, a, b in meta if c == "X"}
        rows = []
        for cond, pid, i, j, kind, rnd, a, b in meta:
            T = M if cond in "PS" else N
            ans = {x: int(T["first"].iat[x]) for x in (i, j)}
            pa_ij, mass_ij, lpa_ij, lpb_ij = read[a]
            pa_ji, mass_ji, lpa_ji, lpb_ji = read[b]
            pref_i = 0.5 * (pa_ij + (1.0 - pa_ji))
            choice_i = 1.0 if pref_i > 0.5 + TIE_EPS else 0.0 if pref_i < 0.5 - TIE_EPS else 0.5
            ci, cj = float(T.correct.iat[i]), float(T.correct.iat[j])
            scored = int(ci + cj == 1)
            # internal: the judge's (M's) raw log-prob of the target's answer; excluded where M's digit mass < 0.5
            si, sj = M[f"lp_{ans[i]}"].iat[i], M[f"lp_{ans[j]}"].iat[j]
            int_ok = (M.mass.iat[i] >= MIN_DIGIT_MASS and M.mass.iat[j] >= MIN_DIGIT_MASS
                      and np.isfinite(si) and np.isfinite(sj))
            int_choice = (1.0 if si > sj else 0.0 if si < sj else 0.5) if int_ok else np.nan
            sp_i, sp_j = full.s_pop.iat[i], full.s_pop.iat[j]
            mine = {x: M["first"].iat[x] for x in (i, j)}
            other = {x: N["first"].iat[x] if N is not None else np.nan for x in (i, j)}
            any_dist = any(v == dist[x] for d in (mine, other) for x, v in d.items() if pd.notna(v))
            sx = float(s_ids[pid] == x_ids[pid]) if pid in s_ids and pid in x_ids else np.nan
            rows.append(dict(
                pair_id=pid, i=i, j=j, kind=kind, round=rnd, decile_diff=ddec[pid], condition=cond,
                judge_tag=cfg["ref_tag"], target_tag=tags[cond], question_i=full.question.iat[i], question_j=full.question.iat[j],
                target_answer_i=ans[i], target_answer_j=ans[j],
                shown_i=np.nan if cond == "P" else ans[i], shown_j=np.nan if cond == "P" else ans[j],
                shown_text_i="" if cond == "P" else M[f"option_{ans[i]}"].iat[i],
                shown_text_j="" if cond == "P" else M[f"option_{ans[j]}"].iat[j],
                self_first_i=mine[i], self_first_j=mine[j], partner_first_i=other[i], partner_first_j=other[j],
                p_a_ij=pa_ij, p_a_ji=pa_ji, lp_a_ij=lpa_ij, lp_b_ij=lpb_ij, lp_a_ji=lpa_ji, lp_b_ji=lpb_ji,
                mass_ij=mass_ij, mass_ji=mass_ji, first_token_ij=first_tok[a], first_token_ji=first_tok[b],
                pref_i=pref_i, choice_i=choice_i, choice=i if choice_i == 1 else j if choice_i == 0 else -1,
                correct_i=ci, correct_j=cj, scored=scored,
                chosen_correct=choice_i * ci + (1 - choice_i) * cj if scored else np.nan,
                int_score_i=si, int_score_j=sj, int_mass_i=M.mass.iat[i], int_mass_j=M.mass.iat[j],
                int_choice_i=int_choice,
                int_correct=int_choice * ci + (1 - int_choice) * cj if scored and int_ok else np.nan,
                cue_choice_i=1.0 if sp_i > sp_j else 0.0 if sp_i < sp_j else np.nan,
                s_pop_i=sp_i, s_pop_j=sp_j, prop_i=full.prop.iat[i], prop_j=full.prop.iat[j],
                qlen_i=len(full.question.iat[i]), qlen_j=len(full.question.iat[j]),
                distractor_pos_i=dist[i], distractor_pos_j=dist[j],
                target_distractor=int(ans[i] == dist[i] or ans[j] == dist[j]), any_distractor=int(any_dist),
                sx_identical=sx, text_id_ij=a, text_id_ji=b))
        return rows

    def analyze(self, cfg, out):
        try:  # never raise in the container (see run); everything can be redone locally from cfc.csv
            t = pd.read_csv(out / "cfc.csv")
            return dict(run=out.name, n_rows=len(t), **run_summary(t, n_boot=2000))
        except Exception:
            return dict(run=out.name, analyze_error=traceback.format_exc())


def label_tokens(ctx, prefix):
    """How the labels tokenize after this run's prefix (a diagnostic for params.json)."""
    try:
        enc = lambda s: [ctx.tok.decode([t]) for t in ctx.tok.encode(s, add_special_tokens=False)]
        tail = prefix[-12:] if prefix else "assistant\n"
        return {repr(s): enc(s) for s in ("A", " A", "B", " B", tail + "A", tail + " A")}
    except Exception as e:
        return f"unavailable: {e!r}"


TASK = CFC()


# ---- metrics (used in the container and by analyze_cfc.py) ------------------------------------------

def boot_counts(n_items, n_boot, seed=0):
    """Item resample counts, shape (n_boot, n_items)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot).astype(np.float32)


def wmean_boot(C, i, j, y, chunk=1000):
    """Bootstrap distribution of the mean of y over pairs (i, j) under item resampling: in each resample a pair weighs
    the product of its two items' counts, so a pair drops out when one of its items is not drawn (the scored set is
    recomputed per resample) and pairs sharing an item move together. NaN y = pair not in the set."""
    y = np.asarray(y, dtype=float)
    keep = ~np.isnan(y)
    i, j, y = np.asarray(i)[keep], np.asarray(j)[keep], y[keep]
    out = np.full(len(C), np.nan)
    if not len(y):
        return out
    for b in range(0, len(C), chunk):
        W = C[b:b + chunk][:, i] * C[b:b + chunk][:, j]
        den = W.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[b:b + chunk] = np.where(den > 0, (W @ y) / den, np.nan)
    return out


def agree(x, y):
    """Agreement of two choices coded 1 / 0 / 0.5 (tie): 1, 0, or 0.5 when one is a tie."""
    return 1 - np.abs(np.asarray(x, float) - np.asarray(y, float))


def condition_metrics(s):
    """Point estimates for one condition's rows of cfc.csv."""
    sc = s[s.scored == 1]
    m = dict(n_pairs=len(s))
    for name, sub in (("matched", sc[sc.kind == "matched"]), ("random", sc[sc.kind == "random"]), ("all", sc)):
        m[f"n_scored_{name}"] = len(sub)
        m[f"stated_{name}"] = sub.chosen_correct.mean()
        m[f"internal_{name}"] = sub.int_correct.mean()
    both = sc[sc.int_correct.notna()]
    m["n_scored_internal"] = len(both)
    m["stated_on_internal_set"] = both.chosen_correct.mean()
    gap = both.int_correct.mean() - 0.5
    m["readout_efficiency"] = (both.chosen_correct.mean() - 0.5) / gap if len(both) and gap > 0 else np.nan
    has = s[s.int_choice_i.notna()]
    m["agreement"] = agree(has.choice_i, has.int_choice_i).mean()
    conflict = has[has.cue_choice_i.notna() & (has.int_choice_i != 0.5) & (has.int_choice_i != has.cue_choice_i)]
    m["n_cue_conflict"] = len(conflict)
    m["agreement_cue_conflict"] = agree(conflict.choice_i, conflict.int_choice_i).mean()
    cue = s[s.cue_choice_i.notna()]
    m["follows_cue"] = agree(cue.choice_i, cue.cue_choice_i).mean()
    pa = np.concatenate([s.p_a_ij, s.p_a_ji])
    mass = np.concatenate([s.mass_ij, s.mass_ji])
    tok = pd.Series(np.concatenate([s.first_token_ij.fillna(""), s.first_token_ji.fillna("")]).astype(str))
    m.update(position_bias=pa.mean() if len(pa) else np.nan, ab_mass=mass.mean() if len(mass) else np.nan,
             share_ab_mass_ge_09=(mass >= 0.9).mean() if len(mass) else np.nan,
             share_first_letter=tok.isin(["A", "B"]).mean(), share_first_digit=tok.str.fullmatch(r"\d").mean(),
             share_first_question=tok.str.startswith("Question").mean(), ties=(s.choice_i == 0.5).mean(),
             share_target_distractor=s.target_distractor.mean())
    return m


def run_summary(t, n_boot=2000, seed=0):
    """Per-condition metrics of one run, with 95% item-bootstrap intervals for the accuracies."""
    out = {}
    C = boot_counts(int(max(t.i.max(), t.j.max())) + 1, n_boot, seed) if n_boot and len(t) else None
    for cond, s in t.groupby("condition", sort=False):
        m = condition_metrics(s)
        if C is not None:
            sc = s.scored.to_numpy() == 1
            for name, sel in (("matched", sc & (s.kind == "matched").to_numpy()),
                              ("random", sc & (s.kind == "random").to_numpy()), ("all", sc)):
                for what, col in (("stated", "chosen_correct"), ("internal", "int_correct")):
                    y = np.where(sel, s[col].to_numpy(float), np.nan)
                    b = wmean_boot(C, s.i.to_numpy(), s.j.to_numpy(), y)
                    m[f"{what}_{name}_lo"], m[f"{what}_{name}_hi"] = np.nanpercentile(b, [2.5, 97.5]) \
                        if np.isfinite(b).any() else (np.nan, np.nan)
        out.update({f"{cond}_{k}": (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in m.items()})
    return out


# ---- sweep hooks (analyze_sweep.py; resamples pairs, not items: analyze_cfc.py has the item bootstrap) -----

def items(d):
    t = pd.read_csv(Path(d) / "cfc.csv")
    base = t.drop_duplicates("pair_id").set_index("pair_id")[["i", "j", "kind"]].copy()
    base["question"] = base.i.astype(str) + "-" + base.j.astype(str)
    for cond, s in t.groupby("condition"):
        s = s.set_index("pair_id")
        base[f"{cond}_stated"] = s.chosen_correct
        base[f"{cond}_internal"] = s.int_correct
        base[f"{cond}_mass"] = (s.mass_ij + s.mass_ji) / 2
        base[f"{cond}_pos"] = (s.p_a_ij + s.p_a_ji) / 2
    return base


def measures(t):
    m = {}
    nanmean = lambda x: float(np.nanmean(x)) if np.isfinite(np.asarray(x, float)).any() else np.nan
    for cond in "PSX":
        has = f"{cond}_stated" in t
        for name, sel in (("matched", t.kind == "matched"), ("random", t.kind == "random"), ("all", t.kind.notna())):
            m[f"{cond}_stated_{name}"] = nanmean(t.loc[sel, f"{cond}_stated"]) if has else np.nan
        m[f"{cond}_internal_all"] = nanmean(t[f"{cond}_internal"]) if has else np.nan
        m[f"{cond}_ab_mass"] = nanmean(t[f"{cond}_mass"]) if has else np.nan
        m[f"{cond}_position_bias"] = nanmean(t[f"{cond}_pos"]) if has else np.nan
    return m


def gates(m, lo):
    """The stated choice is readable only where the first token is a label."""
    return dict(gate_format=bool(m["P_ab_mass"] >= 0.9) if not np.isnan(m["P_ab_mass"]) else None)


show = ["P_stated_matched", "P_stated_random", "P_internal_all", "S_stated_all", "X_stated_all", "P_ab_mass",
        "P_position_bias"]
