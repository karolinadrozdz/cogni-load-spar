"""Second Chance (Ackerman 2025, "Evidence for Limited Metacognition in LLMs", arXiv 2509.21545, §2.5, Fig. 5;
code github.com/cma1114/self_awareness metacog/second_chance_game.py), on the PopQA Phase 1 items.

Each item is a three-turn conversation. The model's first answer is not shown ("[redacted]", as in the paper):
    user       the reference run's Phase 1 prompt (abstention_modal.answer_prompt, the prompt that gave the first answer)
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
  runs keep (TASKS.md). His "Output ONLY the letter of your choice" line is covered by her prompt's "single number".
- The neutral condition has no game line at all (his code sends the literal system text "None" there: a bug; the
  paper says the line is removed).
- The first answer and its option probabilities come from the reference run, and the second answer is read the
  same way, so both answers are in one mode:
  think=0 (default), no trace:
    Instruct (--thinking off): Phase 1 of phases.csv, 4 tokens, greedy.
    Think (--thinking native): the empty-think-block readout (start_format.csv, prefix "Answer: "), 1 token, greedy.
    (Ackerman: argmax of the first answer token's log-probs.)
  think=1, Think checkpoints with their trace (--thinking native --budget 8192 --max-model-len 12288, sampled at
    the runner's T 0.6, as her runs): her with-trace Phase 1 answer (phases.csv). The answer is parsed from the text
    after the trace (her parse_answer). Items whose trace hit the budget, in her run or here, are not scored.
    Ackerman ran thinking models (Gemini 2.5 Flash / Flash Lite "T") the same way: only the final answer is scored.
- Option entropy is over the four answer digits, renormalized (Ackerman: raw top-4 tokens, not renormalized).
- Ackerman's Test 1 (AccIncor) is the accuracy of the changed answers on items first answered wrong, against chance
  1/3; Test 2 (SecChoice) the share of changes to the first answer's runner-up, against 1/3; Test 3 game entropy
  minus first-answer entropy.

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

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import (QUESTION_FILES, answer_prompt, arranged_choices, default_tag, digit_readout, mc_items,
                              parse_task_opts, probe_text, softmax)

# Ackerman, second_chance_game.py L326-341 and Fig. 5 (verbatim; the game's system line + "\n\n")
GAME = ("Your previous answer was incorrect. Respond ONLY with your new answer\n\n"
        "Your answer was incorrect. Choose a different answer.\n\n")
NEUTRAL = "Your response was lost in transmission. Choose again.\n\n"
REDACTED = "[redacted]"
START_PREFIX = "Answer: "  # the start_format readout used as the no-trace answer of Think checkpoints
VALID = [1, 2, 3, 4]
# descriptive only (Ackerman did not read traces): does the trace reason about what the first answer was?
PREV_RE = re.compile(r"(previous|first|earlier|original|last) (answer|response|choice)|"
                     r"I (probably|likely|must have|might have) (chose|chosen|picked|answered|said|selected)", re.I)


def entropy2(p):
    """Entropy in bits of an option distribution (Ackerman: log2)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-9]
    return float(-(p * np.log2(p)).sum())


def first_answers(ref_dir, n, source):
    """Per item (question_index < n) of the reference run: question, answer position, the first answer (1-4) and
    the option probabilities it came with. source: "start_format" (no-trace readout of a Think run) or "phases"
    (Phase 1: no trace for Instruct, with trace for Think). NaN first answer: no readout, or the trace was cut."""
    ph = pd.read_csv(ref_dir / "phases.csv")
    p1 = ph[ph.phase == 1].set_index("question_index")
    ref = pd.DataFrame(index=pd.RangeIndex(n, name="question_index"))
    ref["question"] = p1.question
    ref["gt_position"] = p1.gt_position
    if source == "start_format":
        sf = pd.read_csv(ref_dir / "start_format.csv")
        sf = sf[(sf.prefix == START_PREFIX) & (sf.phase == 1)].set_index("question_index")
        ref["first"] = pd.to_numeric(sf.forced_answer).reindex(ref.index)  # NaN: with-trace run truncated, no readout
        for k in VALID:
            ref[f"base_p{k}"] = sf[f"prob_{k}"].reindex(ref.index)
    else:
        first = pd.to_numeric(p1.final_answer).reindex(ref.index)
        ref["first"] = first.where(p1.truncated.reindex(ref.index) == 0)  # a cut trace's answer is a logprob guess
        for k in VALID:
            ref[f"base_p{k}"] = p1[f"confidence_{k}"].reindex(ref.index)
    return ref


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
        ref = first_answers(ref_dir, len(df), source)
        assert (ref.question.to_numpy() == df.question.to_numpy()).all(), "ref_tag run has different Phase 1 items"
        if think:
            assert ref.base_p1.notna().mean() > 0.5, "think=1 needs a with-trace reference run"

        items = []
        for i, q in enumerate(df.question):
            c, gt, _ = arranged_choices(full, arr, i)
            assert gt == ref.gt_position[i], f"item {i}: option order differs from the reference run"
            items.append((i, q, answer_prompt(1, q, c)))
        convs = {cond: [[{"role": "user", "content": p}, {"role": "assistant", "content": REDACTED},
                         {"role": "user", "content": head + p}] for _, _, p in items]
                 for cond, head in (("game", GAME), ("neutral", NEUTRAL))}
        if source == "start_format":
            r0 = ctx.render(convs["game"][0])
            prefix = probe_text(r0, "")[len(r0):] + START_PREFIX  # "\n</think>\n\nAnswer: ", as start_format
            max_tokens = 1  # start_format reads one token
        elif think:
            prefix, max_tokens = "", cfg["budget"]  # her with-trace Phase 1: the whole budget
        else:
            prefix, max_tokens = "", 4  # Instruct Phase 1: 4 tokens after "Answer:"
        gens = {cond: [g[0] for g in ctx.generate(convs[cond], max_tokens, prefix=prefix)] for cond in convs}

        rows = []
        for n_row, (i, q, p) in enumerate(items):
            r = ref.loc[i]
            base = [r[f"base_p{k}"] for k in VALID]
            ok_base = not np.isnan(base).any()
            has_first = pd.notna(r["first"])
            row = dict(question_index=i, question=q, gt_position=int(r.gt_position), first=r["first"],
                       first_correct=float(r["first"] == r.gt_position) if has_first else np.nan,
                       **{f"base_p{k}": b for k, b in zip(VALID, base)},
                       base_entropy=entropy2(base) if ok_base else np.nan,
                       # runner-up: the most probable option other than the first answer (top ties occur in bf16)
                       base_second=float(max((k for k in VALID if k != r["first"]), key=lambda k: base[k - 1]))
                       if ok_base and has_first else np.nan)
            for cond in ("game", "neutral"):
                g = gens[cond][n_row]
                ans, lps = digit_readout(g, VALID, int(r.gt_position))
                conf = softmax(lps)
                cut = bool(g["truncated"])  # a trace cut at the budget has no answer
                a = np.nan if cut else int(ans["final_answer"])
                row.update({f"{cond}_answer": a, f"{cond}_text": g["content"], f"{cond}_method": ans["parse_method"],
                            f"{cond}_correct": np.nan if cut else int(a == r.gt_position),
                            f"{cond}_changed": float(a != r["first"]) if has_first and not cut else np.nan,
                            f"{cond}_p_first": conf[int(r["first"])] if has_first and not cut else np.nan,
                            f"{cond}_entropy": np.nan if cut else entropy2(list(conf.values())),
                            f"{cond}_digit_mass": np.nan if cut else sum(math.exp(lps[k]) for k in VALID),
                            **{f"{cond}_p{k}": conf[k] for k in VALID},
                            f"{cond}_truncated": int(cut), f"{cond}_reasoning_tokens": g["reasoning_tokens"],
                            f"{cond}_mentions_prev": int(bool(PREV_RE.search(g["reasoning"]))) if think else np.nan,
                            f"{cond}_reasoning": g["reasoning"], f"{cond}_finish": g["finish_reason"]})
            row.update(game_prompt=ctx.render(convs["game"][n_row], prefix))
            rows.append(row)
        return {"second_chance.csv": rows}, dict(
            prefix=prefix, outcome_source=source, max_tokens=max_tokens,
            rendered_game=ctx.render(convs["game"][0], prefix), rendered_neutral=ctx.render(convs["neutral"][0], prefix))

    def analyze(self, cfg, out):
        t = items(out)
        m = measures(t)
        return dict(run=out.name, n=len(t), **{k: round(float(v), 4) for k, v in m.items()})


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


# ---- sweep hooks (analyze_sweep.py) ----------------------------------------------------------------

def items(d):
    return pd.read_csv(Path(d) / "second_chance.csv").set_index("question_index")


def lift(g, n):
    return (g.mean() - n.mean()) / (1 - n.mean()) if n.mean() < 1 else np.nan


def measures(t, n_boot=2000, seed=0):
    trunc_game, trunc_neutral = t.game_truncated.mean(), t.neutral_truncated.mean()
    t = t[t["first"].notna() & t.game_changed.notna() & t.neutral_changed.notna()]  # scored pairs
    g, nt = t.game_changed.to_numpy(float), t.neutral_changed.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = [lift(g[ix], nt[ix]) for ix in (rng.integers(0, len(t), len(t)) for _ in range(n_boot))]
    b, c = int(((g == 1) & (nt == 0)).sum()), int(((g == 0) & (nt == 1)).sum())  # McNemar discordant pairs
    wrong = t[t.first_correct == 0]
    changed_wrong = wrong[wrong.game_changed == 1]
    changed = t[t.game_changed == 1]
    sec = (changed.game_answer == changed.base_second).to_numpy(float)
    return dict(
        n_scored=len(t), first_accuracy=t.first_correct.mean(),
        change_game=g.mean(), change_neutral=nt.mean(),
        lift=lift(g, nt), lift_lo=np.nanpercentile(boot, 2.5), lift_hi=np.nanpercentile(boot, 97.5),
        mcnemar_p=binom_two_sided(b, b + c, 0.5),
        # the same contrast from the option probabilities: how much less probable the first answer becomes
        p_first_game=t.game_p_first.mean(), p_first_neutral=t.neutral_p_first.mean(),
        lift_lp=(t.neutral_p_first.mean() - t.game_p_first.mean()) / t.neutral_p_first.mean(),
        # Ackerman's alternative-strategy tests (§3.2, Table 2)
        acc_incor=changed_wrong.game_correct.mean(),             # Test 1: accuracy of changed answers, first wrong
        acc_incor_p=binom_two_sided(int(changed_wrong.game_correct.sum()), len(changed_wrong), 1 / 3),
        sec_choice=sec.mean() if len(sec) else np.nan,           # Test 2: changed to the baseline runner-up
        sec_choice_p=binom_two_sided(int(sec.sum()), len(sec), 1 / 3),
        entropy_change=(t.game_entropy - t.base_entropy).mean(),  # Test 3: game entropy above baseline?
        auroc_entropy_change=rank_auroc(g.astype(int), t.base_entropy),  # does baseline entropy predict change?
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
