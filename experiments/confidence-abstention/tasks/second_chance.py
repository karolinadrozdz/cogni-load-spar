"""Second Chance (Ackerman 2025, "Evidence for Limited Metacognition in LLMs", arXiv 2509.21545, §2.5, Fig. 5;
code github.com/cma1114/self_awareness metacog/second_chance_game.py), on the PopQA Phase 1 items.

Each item is a three-turn conversation. The model's first answer is not shown ("[redacted]", as in the paper):
    user       the reference run's Phase 1 prompt (her stored prompt string, the prompt that gave the first answer)
    assistant  [redacted]
    user       game:    "Your answer was incorrect. Choose a different answer." + the same prompt
               neutral: "Your response was lost in transmission. Choose again." + the same prompt
Every item is played in both conditions, whether its first answer was right or wrong (the paper pools both).
A model that knows what it answered avoids that answer in the game more than in the neutral control:
    normalized lift = (game change rate - neutral change rate) / (1 - neutral change rate)      (paper §2.5)

Adaptations to OLMo and to Artemis's runs (each keeps the comparison with her reference run):
- Options are 1-4, worded as in her Phase 1 prompt (Ackerman: A-D in his own format).
- Ackerman's game-only system line ("Your previous answer was incorrect. Respond ONLY with your new answer") opens
  the game's second user turn: a system message would replace Olmo's default system prompt, which the reference
  runs keep (TASKS.md). His "Output ONLY the letter of your choice" line (sent in both conditions) is covered by
  her prompt's "single number". His turn 2 does not repeat the setup sentence; here turn 2 repeats her whole prompt.
- The neutral condition has no game line at all (his code sends the literal system text "None" there: a bug; the
  paper says the line is removed).
- The first answer and its option probabilities come from the reference run (same checkpoint and settings: checked
  against its params.json), and the second answer is read the same way, so both answers are in one mode:
  think=0 (default), no trace:
    Instruct (--thinking off): Phase 1 of phases.csv, 4 tokens, greedy, digit_readout.
    Think (--thinking native): the empty-think-block readout (start_format.csv, prefix "Answer: "), read exactly as
      abstention_modal.start_probe reads it: one greedy token, log-probs at position 0.
    (Ackerman: argmax of the first answer token's log-probs.)
  think=1, Think checkpoints with their trace (--thinking native --budget 8192 --max-model-len 12288, sampled at
    the runner's T 0.6, as her runs): her with-trace Phase 1 answer (phases.csv). The answer is parsed from the text
    after the trace (her parse_answer). Items whose trace hit the budget, in her run or here, are not scored.
    Ackerman ran thinking models (Gemini 2.5 Flash / Flash Lite "T") the same way: only the final answer is scored.
- Option entropy is over the four answer digits, renormalized (Ackerman: raw top-4 tokens, not renormalized).
  Probability-based measures (lift_lp, entropy, the runner-up) use only answers whose first token put >= 0.5 of its
  mass on the digits (with a trace the answer is often prose, and its digit probabilities are then flat); a runner-up
  tied with another option is undefined.
- Ackerman's Test 1 (AccIncor) is the accuracy of the changed answers on items first answered wrong, against chance
  1/3; Test 2 (SecChoice) the share of changes to the first answer's runner-up, against 1/3; Test 3 game entropy
  minus first-answer entropy (and, his Fig. 20, the same for the neutral condition).

Launch lines (tag sc-<ref> without think, sct-<ref> with think; ref = olmo32b-popqa-<ref>):
  think=0, as predict_before:
    --thinking native: Olmo-3-32B-Think-SFT (think-sft-main), Olmo-3-32B-Think-DPO (think-dpo),
                       Olmo-3.1-32B-Think (think-rl31-main)
    --thinking off:    Olmo-3.1-32B-Instruct-SFT --dtype bfloat16 (instruct-sft),
                       Olmo-3.1-32B-Instruct-DPO --dtype bfloat16 (instruct-dpo), Olmo-3.1-32B-Instruct (instruct-rl)
  think=1, --thinking native --budget 8192 --max-model-len 12288:
    Olmo-3-32B-Think-SFT --revision 1e-4-step2000 (think-sft-1e-4-step2000), --revision 1e-4-step6000
    (think-sft-1e-4-step6000), main (think-sft-main); Olmo-3-32B-Think-DPO (think-dpo);
    Olmo-3-32B-Think (think-rl-main); Olmo-3.1-32B-Think (think-rl31-main)
  e.g. modal run --detach abstention_modal.py::main --task second_chance --questions popqa --gpu H200
         --model allenai/Olmo-3-32B-Think-DPO --thinking native --budget 8192 --max-model-len 12288
         --task-opts ref_tag=olmo32b-popqa-think-dpo,think=1 --tag sct-think-dpo
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import (QUESTION_FILES, arranged_choices, default_tag, digit_logprobs, digit_readout, mc_items,
                              parse_task_opts, probe_text, softmax)

# Ackerman, second_chance_game.py L326-341 and Fig. 5 (verbatim; the game's system line + "\n\n")
GAME = ("Your previous answer was incorrect. Respond ONLY with your new answer\n\n"
        "Your answer was incorrect. Choose a different answer.\n\n")
NEUTRAL = "Your response was lost in transmission. Choose again.\n\n"
REDACTED = "[redacted]"
START_PREFIX = "Answer: "  # the start_format readout used as the no-trace answer of Think checkpoints
VALID = [1, 2, 3, 4]
MIN_DIGIT_MASS = 0.5  # probability-based measures need an answer token that is a digit
# descriptive only (Ackerman did not read traces): does the trace guess what its first answer was? (rough)
PREV_RE = re.compile(r"\bI (?:probably|likely|must have|might have|would have|may have) "
                     r"(?:chose|chosen|picked|answered|said|selected|gone with|gave)\b"
                     r"|\b(?:my|the) (?:previous|first|earlier|original) (?:answer|choice|response) "
                     r"(?:was|must have been|might have been|is) (?:probably |likely )?(?:option |answer )?\(?\d", re.I)


def entropy2(p):
    """Entropy in bits of an option distribution (Ackerman: log2)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-9]
    return float(-(p * np.log2(p)).sum())


def runner_up(probs, first):
    """The most probable option other than the first answer; NaN when tied with another option."""
    others = sorted((k for k in VALID if k != first), key=lambda k: probs[k - 1], reverse=True)
    return np.nan if probs[others[0] - 1] == probs[others[1] - 1] else float(others[0])


def first_answers(ref_dir, n, source):
    """Per item (question_index < n) of the reference run: question, its Phase 1 prompt, answer position, the first
    answer (1-4), the option probabilities it came with and their digit mass. source: "start_format" (no-trace
    readout of a Think run) or "phases" (Phase 1: no trace for Instruct, with trace for Think). NaN first answer: no
    readout, or the trace was cut."""
    ph = pd.read_csv(ref_dir / "phases.csv")
    p1 = ph[ph.phase == 1].set_index("question_index")
    ref = pd.DataFrame(index=pd.RangeIndex(n, name="question_index"))
    ref["question"] = p1.question
    ref["prompt"] = p1.prompt
    ref["gt_position"] = p1.gt_position
    if source == "start_format":
        sf = pd.read_csv(ref_dir / "start_format.csv")
        sf = sf[(sf.prefix == START_PREFIX) & (sf.phase == 1)].set_index("question_index")
        ref["first"] = pd.to_numeric(sf.forced_answer).reindex(ref.index)  # NaN: with-trace run truncated, no readout
        for k in VALID:
            ref[f"base_p{k}"] = sf[f"prob_{k}"].reindex(ref.index)
        ref["base_digit_mass"] = sf.digit_mass.reindex(ref.index)
    else:
        first = pd.to_numeric(p1.final_answer).reindex(ref.index)
        ref["first"] = first.where(p1.truncated.reindex(ref.index) == 0)  # a cut trace's answer is a logprob guess
        for k in VALID:
            ref[f"base_p{k}"] = p1[f"confidence_{k}"].reindex(ref.index)
        ref["base_digit_mass"] = p1.digit_mass.reindex(ref.index)
    return ref


def check_reference(cfg, ref_params, source):
    """The reference run is this checkpoint, in the settings the first answers were made with."""
    want = dict(model=cfg["model"], revision=cfg["revision"] or None, thinking=cfg["thinking"])
    got = {k: ref_params.get(k) or None for k in want}
    assert got == want, f"ref_tag {cfg['ref_tag']} is {got}, this run is {want}"
    if source != "start_format":  # the start_format readout reads one token: dtype is the run's own
        assert ref_params["dtype"] == cfg["dtype"], f"dtype {cfg['dtype']} vs the reference run's {ref_params['dtype']}"
    if cfg["think"]:
        for k in ("budget", "temperature", "seed"):
            assert ref_params[k] == cfg[k], f"{k} {cfg[k]} vs the reference run's {ref_params[k]}"
    assert ref_params.get("n", 1000) >= cfg["n"] and ref_params.get("questions") == cfg["questions"]


def start_readout(ctx, texts):
    """abstention_modal.start_probe's readout: one greedy token after the prefix, top-20 log-probs at position 0."""
    from vllm import SamplingParams
    outs = ctx.llm.generate(texts, SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    gens = []
    for o in outs:
        g = o.outputs[0]
        gens.append(dict(content=ctx.tok.decode(g.token_ids[:1]).strip(), reasoning="", reasoning_tokens=0,
                         truncated=0, generated_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                         top={lp.decoded_token: lp.logprob for lp in g.logprobs[0].values()}))
    return gens


class SecondChance:
    name = "second_chance"
    options = {"ref_tag": "", "think": 0}

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        assert f["thinking"] != "base", "second_chance needs a chat checkpoint (multi-turn messages)"
        assert opts["ref_tag"], "--task-opts ref_tag=<confidence_abstention run of this checkpoint> is required"
        assert not opts["think"] or f["thinking"] == "native", "think=1 runs Think checkpoints with --thinking native"
        assert f["questions"] in QUESTION_FILES, f"--questions must be one of {sorted(QUESTION_FILES)}"
        temperature = f["temperature"]
        if temperature < 0 and not opts["think"]:
            temperature = 0.0  # no trace: greedy, as the reference readouts; with trace the runner's 0.6
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=temperature, seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        full, arr = mc_items("/data/" + QUESTION_FILES[cfg["questions"]].format(1))
        df = full.head(cfg["n"])
        ref_dir = Path("/results") / cfg["ref_tag"]
        think = bool(cfg["think"])
        # Think checkpoints without trace: read after an empty think block, as the reference run's start_format
        source = "start_format" if ctx.thinking_active and not think else "phases"
        if source == "start_format":
            assert (ref_dir / "start_format.csv").exists(), f"{cfg['ref_tag']} has no start_format.csv"
        check_reference(cfg, json.loads((ref_dir / "params.json").read_text()), source)
        ref = first_answers(ref_dir, len(df), source)
        assert (ref.question.to_numpy() == df.question.to_numpy()).all(), "ref_tag run has different Phase 1 items"

        items = []
        for i, q in enumerate(df.question):
            c, gt, _ = arranged_choices(full, arr, i)
            assert gt == ref.gt_position[i], f"item {i}: option order differs from the reference run"
            p = ref.prompt[i]  # her stored Phase 1 prompt: the exact text the first answer was given to
            assert all(f"{k + 1}) {c[k]}\n" in p for k in range(4)) and f"Question: {q}\n" in p, f"item {i}"
            items.append((i, q, p))
        convs = {cond: [[{"role": "user", "content": p}, {"role": "assistant", "content": REDACTED},
                         {"role": "user", "content": head + p}] for _, _, p in items]
                 for cond, head in (("game", GAME), ("neutral", NEUTRAL))}
        if source == "start_format":
            r0 = ctx.render(convs["game"][0])
            prefix = probe_text(r0, "")[len(r0):] + START_PREFIX  # "\n</think>\n\nAnswer: ", as start_format
            max_tokens = 1
            gens = {cond: start_readout(ctx, [ctx.render(m, prefix) for m in convs[cond]]) for cond in convs}
        else:
            # with trace: her Phase 1 budget; Instruct Phase 1: 4 tokens after "Answer:"
            prefix, max_tokens = "", cfg["budget"] if think else 4
            gens = {cond: [g[0] for g in ctx.generate(convs[cond], max_tokens, prefix=prefix)] for cond in convs}
        # one-shot safety: the raw generations reach the volume before anything else can fail, and nothing after
        # this point raises (run() has Modal retries, which would redo the whole GPU job)
        raw = [dict(question_index=i, condition=cond, content=g["content"], reasoning=g["reasoning"],
                    top=json.dumps(g["top"]), truncated=g["truncated"], finish_reason=g["finish_reason"],
                    reasoning_tokens=g["reasoning_tokens"], generated_tokens=g["generated_tokens"])
               for cond in convs for (i, _, _), g in zip(items, gens[cond])]
        ctx.save("raw_generations.csv", raw)
        params = dict(prefix=prefix, outcome_source=source, max_tokens=max_tokens,
                      rendered_game=ctx.render(convs["game"][0], prefix),
                      rendered_neutral=ctx.render(convs["neutral"][0], prefix))
        try:
            return {"second_chance.csv": self.score(ctx, items, ref, convs, gens, prefix, think),
                    "raw_generations.csv": raw}, params
        except Exception:
            import traceback
            return {"raw_generations.csv": raw}, dict(params, score_error=traceback.format_exc())

    def score(self, ctx, items, ref, convs, gens, prefix, think):
        rows = []
        for n_row, (i, q, p) in enumerate(items):
            r = ref.loc[i]
            base = [r[f"base_p{k}"] for k in VALID]
            has_first = pd.notna(r["first"])
            base_ok = has_first and not np.isnan(base).any() and r.base_digit_mass >= MIN_DIGIT_MASS
            row = dict(question_index=i, question=q, gt_position=int(r.gt_position), first=r["first"],
                       first_correct=float(r["first"] == r.gt_position) if has_first else np.nan,
                       **{f"base_p{k}": b for k, b in zip(VALID, base)}, base_digit_mass=r.base_digit_mass,
                       base_entropy=entropy2(base) if base_ok else np.nan,
                       base_second=runner_up(base, r["first"]) if base_ok else np.nan)
            for cond in ("game", "neutral"):
                g = gens[cond][n_row]
                ans, lps = digit_readout(g, VALID, int(r.gt_position))
                conf = softmax(lps)
                mass = sum(math.exp(lps[k]) for k in VALID)
                cut = bool(g["truncated"])  # a trace cut at the budget has no answer
                prob_ok = not cut and mass >= MIN_DIGIT_MASS
                a = np.nan if cut else int(ans["final_answer"])
                row.update({f"{cond}_answer": a, f"{cond}_text": g["content"], f"{cond}_method": ans["parse_method"],
                            f"{cond}_correct": np.nan if cut else int(a == r.gt_position),
                            f"{cond}_changed": float(a != r["first"]) if has_first and not cut else np.nan,
                            f"{cond}_p_first": conf[int(r["first"])] if has_first and prob_ok else np.nan,
                            f"{cond}_entropy": entropy2(list(conf.values())) if prob_ok else np.nan,
                            f"{cond}_digit_mass": np.nan if cut else mass,
                            **{f"{cond}_p{k}": conf[k] for k in VALID},
                            f"{cond}_truncated": int(cut), f"{cond}_reasoning_tokens": g["reasoning_tokens"],
                            f"{cond}_mentions_prev": int(bool(PREV_RE.search(g["reasoning"]))) if think else np.nan,
                            f"{cond}_reasoning": g["reasoning"], f"{cond}_finish": g["finish_reason"]})
            row.update(game_prompt=ctx.render(convs["game"][n_row], prefix))
            rows.append(row)
        return rows

    def analyze(self, cfg, out):
        try:  # never raise in the container (see run); the analysis can be redone locally from the CSVs
            t = items(out)
            m = measures(t, n_boot=10000)
            return dict(run=out.name, n=len(t), **{k: round(float(v), 4) for k, v in m.items()})
        except Exception:
            import traceback
            return dict(run=out.name, analyze_error=traceback.format_exc())


TASK = SecondChance()


def rank_auroc(y, s):
    """Rank AUROC (ties averaged); as predict_before.rank_auroc."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (pd.Series(s).rank().to_numpy()[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binom_two_sided(k, n, p):
    """Exact two-sided binomial p-value (no scipy in the run container)."""
    if n == 0:
        return np.nan
    pk = [math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n + 1)]
    return float(min(1.0, sum(x for x in pk if x <= pk[k] * (1 + 1e-9))))


def mean_test(d):
    """Mean of paired differences, its normal 95% interval and two-sided p (Ackerman: mean +- 1.96 SE, t-test)."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan, np.nan
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se > 0 else np.nan
    return m, m - 1.96 * se, m + 1.96 * se, p


# ---- sweep hooks (analyze_sweep.py) ----------------------------------------------------------------

def items(d):
    return pd.read_csv(Path(d) / "second_chance.csv").set_index("question_index")


def lift(g, n):
    return (g.mean() - n.mean()) / (1 - n.mean()) if n.mean() < 1 else np.nan


def measures(t, n_boot=0, seed=0):
    """n_boot > 0 adds paired bootstrap intervals (analyze: 10000, as Ackerman; analyze_sweep bootstraps itself)."""
    trunc_game, trunc_neutral = t.game_truncated.mean(), t.neutral_truncated.mean()
    # sensitivity, as Ackerman scores them: a cut trace counts as an unchanged answer
    s = t[t["first"].notna()]
    lift_cut_unchanged = lift(s.game_changed.fillna(0).to_numpy(float), s.neutral_changed.fillna(0).to_numpy(float))
    t = t[t["first"].notna() & t.game_changed.notna() & t.neutral_changed.notna()]  # scored pairs
    g, nt = t.game_changed.to_numpy(float), t.neutral_changed.to_numpy(float)
    ci = {}
    if n_boot and len(t):
        ix = np.random.default_rng(seed).integers(0, len(t), (n_boot, len(t)))
        gm, nm = g[ix].mean(1), nt[ix].mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = np.where(nm < 1, (gm - nm) / (1 - nm), np.nan)
        ci = dict(lift_lo=np.nanpercentile(boot, 2.5), lift_hi=np.nanpercentile(boot, 97.5),
                  abs_lift_lo=np.percentile(gm - nm, 2.5), abs_lift_hi=np.percentile(gm - nm, 97.5))
    b, c = int(((g == 1) & (nt == 0)).sum()), int(((g == 0) & (nt == 1)).sum())  # McNemar discordant pairs
    wrong = t[t.first_correct == 0]
    changed_wrong = wrong[wrong.game_changed == 1]
    changed = t[(t.game_changed == 1) & t.base_second.notna()]
    sec = (changed.game_answer == changed.base_second).to_numpy(float)
    lp = t[t.game_p_first.notna() & t.neutral_p_first.notna()]
    ent_g = mean_test(t.game_entropy - t.base_entropy)
    ent_n = mean_test(t.neutral_entropy - t.base_entropy)
    has_ent = t.base_entropy.notna()
    return dict(
        n_scored=len(t), first_accuracy=t.first_correct.mean(),
        change_game=g.mean(), change_neutral=nt.mean(),
        lift=lift(g, nt), abs_lift=g.mean() - nt.mean(), **ci,
        mcnemar_p=binom_two_sided(b, b + c, 0.5), lift_cut_unchanged=lift_cut_unchanged,
        # the same contrast from the option probabilities: how much less probable the first answer becomes
        n_lp=len(lp), p_first_game=lp.game_p_first.mean(), p_first_neutral=lp.neutral_p_first.mean(),
        lift_lp=(lp.neutral_p_first.mean() - lp.game_p_first.mean()) / lp.neutral_p_first.mean(),
        # Ackerman's alternative-strategy tests (§3.2, Table 2)
        acc_incor=changed_wrong.game_correct.mean(), n_acc_incor=len(changed_wrong),   # Test 1, first wrong
        acc_incor_p=binom_two_sided(int(changed_wrong.game_correct.sum()), len(changed_wrong), 1 / 3),
        sec_choice=sec.mean() if len(sec) else np.nan, n_sec_choice=len(sec),          # Test 2, all changed
        sec_choice_p=binom_two_sided(int(sec.sum()), len(sec), 1 / 3),
        entropy_change_game=ent_g[0], entropy_change_game_lo=ent_g[1], entropy_change_game_hi=ent_g[2],
        entropy_change_game_p=ent_g[3],                                                  # Test 3
        entropy_change_neutral=ent_n[0], entropy_change_neutral_lo=ent_n[1], entropy_change_neutral_hi=ent_n[2],
        entropy_change_neutral_p=ent_n[3],                                               # his Fig. 20
        # does the first answer's uncertainty predict a change? (Ackerman: logit answer_changed ~ entropy)
        auroc_entropy_change=rank_auroc(g[has_ent.to_numpy()].astype(int), t.base_entropy[has_ent]),
        change_game_first_right=t[t.first_correct == 1].game_changed.mean(),
        change_game_first_wrong=wrong.game_changed.mean(),
        accuracy_game=t.game_correct.mean(), accuracy_neutral=t.neutral_correct.mean(),
        digit_mass_game=t.game_digit_mass.mean(), digit_mass_neutral=t.neutral_digit_mass.mean(),
        logprob_parsed=((t.game_method == "logprob") | (t.neutral_method == "logprob")).mean(),
        truncated_game=trunc_game, truncated_neutral=trunc_neutral,
        reasoning_tokens_game=t.game_reasoning_tokens.mean(), reasoning_tokens_neutral=t.neutral_reasoning_tokens.mean(),
        mentions_prev_game=t.game_mentions_prev.mean(), mentions_prev_neutral=t.neutral_mentions_prev.mean(),
    )


def gates(m, lo):
    """Both conditions must state an answer (not a log-prob fallback; a trace that ends); the neutral control must
    mostly repeat the first answer."""
    return dict(gate_format=bool(m["logprob_parsed"] < 0.05 and max(m["truncated_game"], m["truncated_neutral"]) < 0.05),
                gate_neutral=bool(m["change_neutral"] < 0.5))


show = ["first_accuracy", "change_game", "change_neutral", "lift", "lift_lp", "acc_incor", "sec_choice",
        "auroc_entropy_change"]
