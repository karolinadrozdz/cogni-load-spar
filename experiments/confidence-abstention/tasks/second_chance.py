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
- The first answer and its option probabilities come from the reference run, in the same no-trace mode:
  Instruct (--thinking off): Phase 1 of phases.csv, 4 tokens, greedy.
  Think (--thinking native): the empty-think-block readout (start_format.csv, prefix "Answer: "), 1 token, greedy.
  The second answer is read the same way (Ackerman: argmax of the first token's log-probs).

- Option entropy is over the four answer digits, renormalized (Ackerman: raw top-4 tokens, not renormalized).
- The neutral condition has no game line at all (his code sends the literal system text "None" there: a bug; the
  paper says the line is removed).
- Ackerman's Test 1 (AccIncor) is the accuracy of the changed answers on items first answered wrong, against chance
  1/3; Test 2 (SecChoice) the share of changes to the first answer's runner-up, against 1/3; Test 3 game entropy
  minus first-answer entropy.

Launch lines (each checkpoint's mode and reference run as predict_before):
    modal run --detach abstention_modal.py::main --task second_chance --questions popqa --gpu H200         --model <model> [--revision <rev>] --thinking <mode> [--dtype bfloat16]         --task-opts ref_tag=<ref> --tag sc-<ref without olmo32b-popqa->
    Think (--thinking native): Olmo-3-32B-Think-SFT (think-sft-main), Olmo-3-32B-Think-DPO (think-dpo),
        Olmo-3.1-32B-Think (think-rl31-main)
    Instruct (--thinking off): Olmo-3.1-32B-Instruct-SFT --dtype bfloat16 (instruct-sft),
        Olmo-3.1-32B-Instruct-DPO --dtype bfloat16 (instruct-dpo), Olmo-3.1-32B-Instruct (instruct-rl)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from abstention_modal import (QUESTION_FILES, answer_prompt, arranged_choices, default_tag, digit_readout, mc_items,
                              parse_task_opts, probe_text, softmax)

# Ackerman, second_chance_game.py L326-341 and Fig. 5 (verbatim, the system line "+ \n\n")
GAME = "Your previous answer was incorrect. Respond ONLY with your new answer\n\nYour answer was incorrect. Choose a different answer.\n\n"
NEUTRAL = "Your response was lost in transmission. Choose again.\n\n"
REDACTED = "[redacted]"
START_PREFIX = "Answer: "  # the start_format readout used as the no-trace answer of Think checkpoints
VALID = [1, 2, 3, 4]


def entropy2(p):
    """Entropy in bits of an option distribution (Ackerman: log2)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-9]
    return float(-(p * np.log2(p)).sum())


def first_answers(ref_dir, n, notrace):
    """Per item (question_index < n) of the reference run: question, answer position, the first answer (1-4) and
    the option probabilities it came with, in the mode the second answer is read in."""
    ph = pd.read_csv(ref_dir / "phases.csv")
    p1 = ph[ph.phase == 1].set_index("question_index")
    ref = pd.DataFrame(index=pd.RangeIndex(n, name="question_index"))
    ref["question"] = p1.question
    ref["gt_position"] = p1.gt_position
    if notrace:
        sf = pd.read_csv(ref_dir / "start_format.csv")
        sf = sf[(sf.prefix == START_PREFIX) & (sf.phase == 1)].set_index("question_index")
        ref["first"] = pd.to_numeric(sf.forced_answer).reindex(ref.index)  # NaN: with-trace run truncated, no readout
        for k in VALID:
            ref[f"base_p{k}"] = sf[f"prob_{k}"].reindex(ref.index)
    else:
        ref["first"] = pd.to_numeric(p1.final_answer).reindex(ref.index)
        for k in VALID:
            ref[f"base_p{k}"] = p1[f"confidence_{k}"].reindex(ref.index)
    return ref


class SecondChance:
    name = "second_chance"
    options = {"ref_tag": "", "outcome": "auto"}

    def config(self, f):
        opts = parse_task_opts(f["task_opts"], self.options)
        assert f["thinking"] != "base", "second_chance needs a chat checkpoint (multi-turn messages)"
        assert opts["ref_tag"], "--task-opts ref_tag=<confidence_abstention run of this checkpoint> is required"
        assert opts["outcome"] in ("auto", "phases", "notrace"), "outcome is auto, phases or notrace"
        assert f["questions"] in QUESTION_FILES, f"--questions must be one of {sorted(QUESTION_FILES)}"
        temperature = 0.0 if f["temperature"] < 0 else f["temperature"]  # greedy, as the reference readouts
        return dict(task=self.name, model=f["model"], thinking=f["thinking"], n=f["n"], budget=f["budget"],
                    max_model_len=f["max_model_len"], temperature=temperature, seed=f["seed"], tp=f["tp"],
                    revision=f["revision"] or None, dtype=f["dtype"], questions=f["questions"], **opts,
                    tag=f["tag"] or default_tag(f, "_" + self.name))

    def run(self, ctx):
        cfg = ctx.cfg
        full, arr = mc_items("/data/" + QUESTION_FILES[cfg["questions"]].format(1))
        df = full.head(cfg["n"])
        ref_dir = Path("/results") / cfg["ref_tag"]
        # thinking checkpoints: no trace, read after an empty think block, as the reference run's start_format
        notrace = cfg["outcome"] == "notrace" or (cfg["outcome"] == "auto" and ctx.thinking_active)
        if notrace:
            assert (ref_dir / "start_format.csv").exists(), f"{cfg['ref_tag']} has no start_format.csv"
        ref = first_answers(ref_dir, len(df), notrace)
        assert (ref.question.to_numpy() == df.question.to_numpy()).all(), "ref_tag run has different Phase 1 items"

        items = []
        for i, q in enumerate(df.question):
            c, gt, _ = arranged_choices(full, arr, i)
            assert gt == ref.gt_position[i], f"item {i}: option order differs from the reference run"
            items.append((i, q, answer_prompt(1, q, c)))
        convs = {cond: [[{"role": "user", "content": p}, {"role": "assistant", "content": REDACTED},
                         {"role": "user", "content": head + p}] for _, _, p in items]
                 for cond, head in (("game", GAME), ("neutral", NEUTRAL))}
        if notrace:
            r0 = ctx.render(convs["game"][0])
            prefix = probe_text(r0, "")[len(r0):] + START_PREFIX  # "\n</think>\n\nAnswer: ", as start_format
            max_tokens = 1  # start_format reads one token
        else:
            prefix, max_tokens = "", 4  # Phase 1: 4 tokens after "Answer:"
        gens = {cond: [g[0] for g in ctx.generate(convs[cond], max_tokens, prefix=prefix)] for cond in convs}

        rows = []
        for n_row, (i, q, p) in enumerate(items):
            r = ref.loc[i]
            base = [r[f"base_p{k}"] for k in VALID]
            row = dict(question_index=i, question=q, gt_position=int(r.gt_position), first=r["first"],
                       first_correct=float(r["first"] == r.gt_position) if pd.notna(r["first"]) else np.nan,
                       **{f"base_p{k}": b for k, b in zip(VALID, base)},
                       base_entropy=entropy2(base) if not np.isnan(base).any() else np.nan,
                       base_second=float(sorted(VALID, key=lambda k: base[k - 1])[-2]) if not np.isnan(base).any() else np.nan)
            for cond in ("game", "neutral"):
                g = gens[cond][n_row]
                ans, lps = digit_readout(g, VALID, int(r.gt_position))
                conf = softmax(lps)
                a = int(ans["final_answer"])
                row.update({f"{cond}_answer": a, f"{cond}_text": g["content"], f"{cond}_method": ans["parse_method"],
                            f"{cond}_correct": int(a == r.gt_position),
                            f"{cond}_changed": float(a != r["first"]) if pd.notna(r["first"]) else np.nan,
                            f"{cond}_p_first": conf[int(r["first"])] if pd.notna(r["first"]) else np.nan,
                            f"{cond}_entropy": entropy2(list(conf.values())),
                            f"{cond}_digit_mass": sum(math.exp(lps[k]) for k in VALID),
                            **{f"{cond}_p{k}": conf[k] for k in VALID},
                            f"{cond}_finish": g["finish_reason"]})
            row.update(game_prompt=ctx.render(convs["game"][n_row], prefix))
            rows.append(row)
        return {"second_chance.csv": rows}, dict(
            prefix=prefix, outcome_source="start_format" if notrace else "phases",
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
    t = t[t["first"].notna()]
    g, nt = t.game_changed.to_numpy(float), t.neutral_changed.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = [lift(g[ix], nt[ix]) for ix in (rng.integers(0, len(t), len(t)) for _ in range(n_boot))]
    b, c = int(((g == 1) & (nt == 0)).sum()), int(((g == 0) & (nt == 1)).sum())  # McNemar discordant pairs
    wrong = t[t.first_correct == 0]
    changed_wrong = wrong[wrong.game_changed == 1]
    changed = t[t.game_changed == 1]
    sec = (changed.game_answer == changed.base_second).to_numpy(float)
    return dict(
        first_accuracy=t.first_correct.mean(),
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
        n_scored=len(t),
    )


def gates(m, lo):
    """Both conditions must answer with a digit; the neutral control must mostly repeat the first answer."""
    return dict(gate_format=bool(min(m["digit_mass_game"], m["digit_mass_neutral"]) >= 0.9),
                gate_neutral=bool(m["change_neutral"] < 0.5))


show = ["first_accuracy", "change_game", "change_neutral", "lift", "lift_lp", "acc_incor", "sec_choice",
        "auroc_entropy_change"]
