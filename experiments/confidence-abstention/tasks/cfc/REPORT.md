# Confidence forced choice (CFC): does post-training change how models read out their own correctness?

## The big picture

**Question.**
- Our earlier proxies (abstention, predict-before) suggested a pattern:
  - the model's internal signal about whether it is right is already there after SFT and stays flat;
  - what changes at DPO is how well its behaviour follows that signal.
- CFC tests this directly, with a task that has no threshold and no room for over- or under-confidence.

**Task.**
- The model sees two PopQA questions and picks the one it is more likely to get right. This is the confidence forced-choice paradigm from human metacognition research ([Mamassian & de Gardelle 2022](https://hal-cnrs.archives-ouvertes.fr/hal-02876786)).
- Pairs are matched on popularity and relation type, so "the famous one is easier" cannot work as a shortcut.
- A self-vs-other condition, following [Binder et al. 2024](https://arxiv.org/abs/2410.13787), asks whether a model judges **its own** answers better than a sibling model judges them.

**Findings.**
- **The models barely use their own signal.**
  - Stated accuracy is 0.51–0.59 (chance = 0.5).
  - Their internal signal, the answer probabilities, would reach ~0.75 on the same pairs.
- **It follows the same stage pattern as predict-before, only weaker.**
  - **Instruct:** improves at DPO. The share of the internal signal reaching the stated choice goes 0.18 → 0.33 → 0.38.
  - **Think:** drops at RL, 0.18 → 0.20 → 0.04; RL 3.1 ends at chance.
  - The pre-registered main DPO test narrowly misses significance (p = 0.056). The secondary versions of the same test are significant.
- **It is not self-specific.** Models judge a sibling's answers as well as their own (self-advantage ≈ 0). Whatever judging ability exists rests on knowledge any similar model has, not on privileged self-access.

**What we can and can't claim.**
- **Defensible, and behavioural:** post-training changes how well a model's explicit judgments track its own correctness. DPO helps on the Instruct line, RL hurts on the Think line, and the underlying knowledge signal stays flat. Three measures now agree on this.
- **Not supported:** metacognition in the strong sense, i.e. introspection or privileged self-knowledge.
  1. The self-vs-other test shows no self-advantage.
  2. That test only covered the conditions where the answers are shown. For the main "which would you get right?" condition, we never had another model predict this model's correctness.
  3. The comparison model is a near-twin: same base, same answer on 85% of items. Against it, even real self-access would be hard to detect. A fair test needs a judge from another model family, for all conditions.
- **Better framing:** "monitoring / readout of correctness", not "metacognition".

## What we ran

| | |
|---|---|
| Checkpoints | Instruct SFT / DPO / RL and Think SFT / DPO / RL 3.1 (OLMo-3 32B), no reasoning |
| Items | 1000 PopQA questions, with each checkpoint's own answers, correctness and answer probabilities from Artemis's runs |
| Pairs | 5,829 fixed pairs, the same for every checkpoint: 3,829 matched on relation and the same or a neighbouring popularity band, plus 2,000 random. Pairs that leak each other's answer are excluded. Each pair is shown in both orders, which cancels position preference. |
| Conditions | **P:** "Which of these two would you be more likely to answer correctly?" (no answers shown, the main condition). **S:** both questions with the model's own answers, "Which answer is more likely correct?". **X:** the same, with a sibling's answers (Instruct ↔ Think at the same stage). |
| Readout | One token, the probability of "A" vs "B", averaged over both orders. |
| Scoring | Pairs where exactly one answer is correct. Chance = 0.5. |
| Pilots | 3 checkpoints × 200 pairs. All passed the pre-set go rules: A/B probability ≥ 0.98, position bias within limits. |
| Cost | $1.55 (pilots) + $5.19 (main) = $6.74. |
| Code and plan | `tasks/cfc/` (plan, task, pair builder, analysis, local check, launch commands) |

## Results

**Accuracy per checkpoint.** Stated = the model's choice; internal = choosing by its own answer probabilities. 95% CI in brackets.

| Checkpoint | P stated (matched pairs) | P stated (all) | P internal | Share of signal used (P) | S stated | X stated |
|---|---|---|---|---|---|---|
| Instruct SFT | 0.54 [0.48, 0.59] | 0.54 | 0.74 | 0.18 | 0.62 | 0.63 |
| Instruct DPO | 0.56 [0.51, 0.62] | 0.58 | 0.74 | 0.33 | 0.69 | 0.68 |
| Instruct RL | 0.58 [0.52, 0.63] | 0.59 | 0.74 | 0.38 | 0.68 | 0.64 |
| Think SFT | 0.54 [0.49, 0.60] | 0.54 | 0.75 | 0.18 | 0.58 | 0.57 |
| Think DPO | 0.55 [0.49, 0.61] | 0.55 | 0.76 | 0.20 | 0.61 | 0.62 |
| Think RL 3.1 | 0.50 [0.45, 0.56] | 0.51 | 0.76 | 0.04 | 0.53 | 0.57 |

**Pre-registered tests.**

| Test | Result | Verdict |
|---|---|---|
| **H2 (primary):** Instruct SFT → DPO, P on matched pairs | +0.027 [−0.006, +0.061], one-sided p = 0.056 (Holm 0.11) | Right direction, not significant |
| H2, same step on random pairs / all pairs / graded measure | +0.047 / +0.034 / +0.035, all CIs exclude 0 | Consistent support |
| H2b: DPO → RL, Instruct | +0.014, not shown equivalent (margin 0.03) | Flat, but too uncertain to call equal |
| H2b: DPO → RL, Think | −0.047 (graded −0.082, CI excludes 0) | Gets worse at RL |
| **H4 (primary):** self-advantage beyond answer checking | −0.007 [−0.023, +0.009], p = 0.80 | No self-advantage |

All of these hold when pairs whose shown answer is the obviously-wrong distractor are left out.

**Other observations.**
- **Shortcuts are strong.** Popularity, relation type and question length all predict correctness on their own. The model's stated preference still adds signal beyond them in 5 of 6 checkpoints; Think RL 3.1 on matched pairs is the exception.
- **Seeing the answers helps** (S/X, 0.53–0.69, against 0.51–0.59 for P). The models are better at checking an answer than at predicting ahead.
- **Position.** The greedy A/B choice is mostly positional, and preferences sit close to 50/50. The signal is in small shifts between the two orders. That is why each pair is shown in both orders, and why the graded measure (which weights strong preferences more) was pre-registered after the pilots.

## Caveats

- **The main DPO test narrowly misses significance.** On its own, CFC is supporting evidence, not proof.
- **The self-vs-other comparison** covers S/X only, and uses a near-twin judge (see "What we can and can't claim").
- **Think RL 3.1 has a different built-in system prompt**, and a stronger position bias (0.33). Part of its drop may come from these.
- **Scope:** no reasoning; one dataset (PopQA); one model family.

## Files

- **Results:** `results/cfc-<checkpoint>/` (`cfc.csv` per pair and condition, `raw_generations.csv`, `analysis.json`, `params.json`), plus the full cross-run analysis in `results/cfc_analysis.txt` / `.json`. These are not in git, but they are on the Modal volume `abstention-results`.
- **Reproduce:** `python tasks/cfc/analyze_cfc.py --json results/cfc_analysis.json`.
