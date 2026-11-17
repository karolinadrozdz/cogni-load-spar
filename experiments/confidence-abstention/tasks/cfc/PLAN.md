# Plan: Confidence forced choice (CFC) across training stages

*Revised after two independent reviews (methods and engineering); the changes are listed at the end.*

## Idea in one paragraph

Our two proxies so far suggest a specific story.
- The model's **internal signal** about whether it is right (answer-confidence AUROC ~0.80) is already there after SFT and stays flat.
- What changes at **DPO** is how well the model's **reports follow** that signal.
- **RL** shifts the overall level (it abstains more) without making the behaviour track the signal better.

We test the readout part directly with the **confidence forced-choice** task from human metacognition research ([Mamassian & de Gardelle 2022, *Psych. Review*](https://hal-cnrs.archives-ouvertes.fr/hal-02876786); earlier [Barthelmé & Mamassian 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2730538/)). The model sees two PopQA items and picks the one it is **more likely to get right**. There is no threshold and no room for over- or under-confidence: only how well it ranks its own correctness counts.

As a secondary, exploratory question we ask whether the judgment is **self-specific** or just answer checking, using the self- vs cross-prediction logic of [Binder et al. 2024, *Looking Inward*](https://arxiv.org/abs/2410.13787).

## Hypotheses and pre-registered tests

| | Hypothesis | Measure | Prediction | Status |
|---|---|---|---|---|
| H2 | DPO improves how well the stated judgment follows the model's own signal | **Stated CFC accuracy on cue-matched pairs** (condition P) | Instruct SFT → DPO: increase | **Primary test 1** |
| H2b | ... and RL does not improve it further | the same, DPO → RL | Within ±0.03 (equivalence test, TOST) | Secondary |
| H4 | The judgment uses self-specific knowledge, beyond answer checking | **Stated self-advantage minus the internal self-advantage** (conditions S/X), pooled over stages | > 0 | **Primary test 2** (exploratory in spirit; see power) |
| — | Baseline, not a hypothesis | **Internal CFC accuracy** (the model's own answer probabilities). Known to be ~0.79–0.81 at every checkpoint | — | Reported as the reference point |

- **Holm correction** is applied over the two primary tests. Everything else is descriptive.
- **A null result is informative:** if stated CFC follows the internal signal equally well at every stage, DPO does not improve the readout here, which would contradict the readout story.
- The previous **H1** ("signal already there at SFT") is now a known baseline: internal CFC accuracy is just the answer-confidence AUROC, already measured. Claiming "already there" would need a pre-SFT point, which we don't have.
- The previous **H3** ("RL shifts level, not sensitivity") is not tested separately. It is H2b plus Artemis's existing abstention-rate result, and RL is confounded with the system-prompt change (see Controls).

## Items and pairs

- **Items:** the 1000 PopQA Phase 1 items. For each checkpoint, Artemis's runs give the first answer, its correctness and its internal probability, all in no-thinking mode:
  - Instruct: `phases.csv` Phase 1.
  - Think: `start_format.csv` (prefix "Answer: ", phase 1), with correctness = `forced_answer == gt_position`.
- **Usable items:** all 1000 for Instruct; 995 / 999 / 978 for Think SFT / DPO / RL 3.1, because items whose trace was cut have no readout.
- **Pair set: one fixed set** (`tasks/cfc/pairs_data.py`, built by `tasks/cfc/build_pairs.py` with seed 2026; a `.py` module because Modal ships only `.py` files; its SHA-256 is stored in params). **5829 pairs from 12 rounds:**
  - **8 cue-matched rounds (3829 pairs):** the two items share a **relation type** (`prop`), and their **popularity deciles** (`s_pop`, deciles over the 1000 items) are the **same or neighbouring** (1298 pairs in the same decile, 2531 in neighbouring deciles; the decile difference is stored per pair). Popularity and relation alone predict correctness with AUROC ~0.71, as well as we expect the model to, so the **primary H2 test uses only these pairs**. Within them the relation cannot help, and popularity differs by at most one decile; that **residual popularity difference is handled by the surface-cue regression**, which is also run on the matched pairs.
  - **4 random rounds (2000 pairs):** perfect matchings of all items, kept for comparison with the matched pairs, and for power.
  - Every item is in 4 random pairs and up to 8 matched pairs (873 items in 8). **4 items never enter a matched pair** (the only "color" item, and three items with no admissible partner left).
- **Constraints on every pair:**
  - no repeated pair;
  - no shared subject (**27 item pairs** share a subject; 25 subjects repeat in the item file);
  - **no exact answer leak:** **47 item pairs** (48 ordered cases) where one item's subject is among the other item's options, e.g. "What is Mexico City the capital of?" with "What is the capital of Mexico?";
  - **no fuzzy answer leak:** an option of one item appears as a whole word or phrase in the other item's question, or the other item's subject appears as a whole word inside one of this item's options (e.g. the option "screenwriter" next to "Who was the screenwriter for ...?", the subject "David" next to the option "David Trueba"). This excludes **2693 further item pairs**, mostly occupation options next to questions whose template names that occupation (12 pairs of the earlier 8-round set had such a leak);
  - subjects are parsed with 16 relation templates; note the trap "What is X the capital of?" vs "What is the capital of X?".
  - Matching: within each relation, a randomized greedy maximal matching, with free items placed by swapping partners with an existing pair; random rounds: random pairing, conflicts repaired by swapping partners. Rounds are never redrawn.
- **Each pair is shown in both orders.**

## Conditions

| | Condition | Shown | Judge | Target (whose correctness counts) |
|---|---|---|---|---|
| **P** | Prospective (**main**) | two questions, no answers | model M | M |
| **S** | Retrospective, self | two questions + **M's** answers | M | M |
| **X** | Retrospective, cross | two questions + **N's** answers | M | N |

**Pairs (M, N): the same stage across the two lines**, giving three crosses:
- Instruct SFT ↔ Think SFT
- Instruct DPO ↔ Think DPO
- Instruct RL ↔ Think RL 3.1

**Self-advantage** = ½ { [acc(M→M) − acc(N→M)] + [acc(N→N) − acc(M→N)] }
- The judge main effect cancels out.
- All four terms use the **same pairs**: those where both models have answers and the target has exactly one correct item.
- **The null is not 0. It is the same contrast computed from the internal probabilities**: the judge's own probability for the shown answer. Here that is +0.002 / 0.000 / −0.002 per stage, the empirical "pure knowledge-checking" baseline.
- The test is stated self-advantage minus internal self-advantage, **pooled over the three stages**. Per-stage differences ("does it appear at DPO?") are descriptive only.

**Power and honesty for H4:**
- The two lines agree on 85–87% of answers. On those pairs, S and X show identical prompts and cancel exactly, so only ~27% of pairs carry signal. Effects below ~0.02 will not be detectable.
- The cross-judge is a close **sibling** (same base, untrained on the target), not Binder's trained cross-predictor. A positive result means "better than a sibling at judging its own answers", which is weaker than privileged access.

## Prompts (single turn)

**P:**
```
You will be shown two multiple choice questions. Do not answer them.

Question A: {question}
1) {option 1}
2) {option 2}
3) {option 3}
4) {option 4}

Question B: {question}
1) ...

If you were asked both questions, which one would you be more likely to answer correctly? Respond only with A or B.
```

**S and X** (word-for-word identical; only whose answers are shown differs):
```
You will be shown two multiple choice questions, together with answers that were given to them.

Question A: {question}
1) ... 4) ...
Answer given: {k}) {option k}

Question B: ...
Answer given: {k}) {option k}

Which of the two answers is more likely to be correct? Respond only with A or B.
```

## Readout

- **One greedy token, top-20 log-probs at position 0.** Reuse `second_chance.start_readout`. Do not use `ctx.generate(max_tokens=1)`: its `split()` skips whitespace tokens and loses the log-probs.
- **Mode:**
  - Instruct: `--thinking off`, after `assistant\n`.
  - Think: closed think block + **`"Answer:"` (no trailing space)**.
  - The digit readout used a trailing space, but letters tokenize as " A", so a trailing space would force an unnatural token. **Check this in the pilot.**
- **Label log-prob:** a new `letter_logprobs(top, ["A","B"])` taking the max over the variants "A", " A", "A)", " A)", "A.", " A.". A label missing from the top-20 gets −50.
- **Preference per item X** (labels follow position): pref(X) = ½ [P(A | X first) + P(B | X second)], renormalized over A/B.
  - The choice is the item with pref > 0.5.
  - **pref = 0.5 exactly counts as half correct** (no silent B bias).
- **Diagnostics:** the A/B probability mass (should be ≥ 0.9), position bias (mean P(first label)), and the share of first tokens that are digits or "Question".
- **Internal ranking:** rank by the **raw log-prob** of the item's answer, not the probability. 18–39% of items have p > 0.99, so probabilities saturate. Items with a first-answer digit mass < 0.5 are excluded from the internal ranking.

## Metrics (bootstrap over items, 10k resamples)

1. **Stated CFC accuracy** (P, S, X): on pairs where the target has exactly one correct item. Chance = 0.5. Reported on **cue-matched** pairs (primary), random pairs, and all.
2. **Internal CFC accuracy:** the same, choosing by internal log-prob. This is the reference point, not a ceiling: stated choices can use other information.
3. **Agreement with the internal choice**, overall and **on pairs where the internal choice and the cue-based choice disagree** (cue-based choice = the item with the **higher log s_pop**). That shows whether the stated choice follows the model's own signal or the cues.
4. **Self-advantage** (S/X): stated minus internal, pooled and per stage.
5. **Surface-cue regression** (P, random pairs; also matched pairs, for their residual popularity difference): "chosen is correct" on the stated preference, Δlog s_pop, Δ relation base rate and Δ question length. The base rate leaves out **both** items of the pair (leave-one-out alone leaks the outcome on same-relation pairs), so it is 0 on same-relation pairs. *The other line's correctness is dropped as a covariate: it agrees with the target on 87–89% of items, nearly the outcome itself.*
6. **Diagnostics:** position bias, A/B mass, digit-first share, missing items. Results are also given **without pairs where a shown answer is the type-mismatched distractor** (1–2% of items).
7. **Stage differences:** paired bootstrap SFT → DPO and DPO → RL.

**Bootstrap scheme:** resample items; weight each pair by the product of its two items' resample counts; recompute the scored-pair set inside each resample. The naïve ±0.012 assumes independent pairs and is too optimistic.

## Controls

- **System prompt at RL: not run.** Artemis's runs with the DPO-era system prompt (`think-rl31-main-sys30`, `think-rl-main-sys31`) exist only **with** thinking (no start_format readout), and they use a harness option (`chat_template_from`) that is not in this branch. The RL system-prompt confound therefore stays a stated limitation.

## Implementation (all in `tasks/cfc/`, plus one registry line)

- **Options:** `ref_tag`, `cross_ref_tag` (partner, for S/X), `conditions` (default PSX), `n_pairs` (0 = all; pilots take the first n of a fixed shuffled order), `pairs_sha` (pins the pair set).
  - `check_reference` runs for both references. The partner's readout source comes from **its own** params (`thinking`). A fixed same-stage partner map makes a wrong pairing fail at launch.
- **Pair set:** `config()` checks locally that `pairs_data.py` is unedited (hash), indexes the current item file and meets all constraints. Pairs with a missing answer are skipped, never rendered as "nan".
- **One-shot safety, as in second_chance:**
  - all prompts are built and checked before generating;
  - the raw outputs are saved right after the single generate call;
  - scoring and analysis never raise.
- **Efficiency:** S and X prompts are deduplicated: ~73% are identical because the two lines gave the same answer.
- **Flags:**
  - `--gpu H200`;
  - `--dtype bfloat16` for Instruct SFT/DPO;
  - the default `--max-model-len` (prompts ≈ 170–350 tokens);
  - launch in groups of ~5, as TASKS.md advises.

## Runs, cost and order

| Step | What | Cost |
|---|---|---|
| 0 | Build the pair file (with constraint checks); write `tasks/cfc.py` | – |
| 1 | **Local check** with real reference files and real chat templates: every prompt, both orders, S/X identical apart from the answers, answer/option order equal to the reference, scoring checked on a scripted model, one-shot safety | – |
| 2 | **Pilots (n = 200 pairs):** Instruct DPO, **Instruct SFT** (failed the Second Chance format) and **Think DPO** (the new "Answer:" prefix). Check A/B mass ≥ 0.9, position bias, A/B in the top-20 | ~$2 |
| 3 | **Main:** the 6 checkpoints, conditions P + S + X | ~$8–10 |
| 5 | *(later)* other families: OLMo-2, Tülu 3 | ~$1.50–2 / checkpoint |

The main result costs about **$11–13** including the pilots.

**Pilot go/no-go (fixed before the pilots):**
- **GO** if the A/B probability mass is ≥ 0.9 and the position bias (mean P(first label)) is within 0.2–0.8.
- If **Instruct SFT** fails the format, it is reported as a format failure and the rest proceeds without redesign.
- Stated accuracy at chance in a pilot is a **result, not a reason to stop**.

## Known limitations

- **Cue-matched pairs control popularity and relation**, but not every possible cue (e.g. subject familiarity beyond `s_pop`).
- **H4 is low-powered, and its cross-judge is an untrained sibling;** a positive result is suggestive only.
- **The internal signal comes from Artemis's single-question prompt, not the pair prompt.** The comparison is between two prompts as well as between "stated" and "internal".
- **Comparisons across stages are within a line only;** the RL system-prompt confound remains unless the control is run.
- **PopQA label noise** lowers stated and internal accuracy alike.

## Changes after review

**Methods:**
- cue-matched pairs as the primary H2 test;
- H1 and H3 demoted, with equivalence testing for "flat";
- the H4 null taken from internal probabilities, pooled H4 test, same item set for all four terms, honest power statement;
- the item-level bootstrap scheme specified;
- two pre-registered primary tests with Holm correction;
- a regression without the near-outcome covariate;
- the type-mismatched-distractor check;
- the per-item preference definition.

**Second revision (after the implementation reviews):** matched pairs = same relation and same-or-neighbouring popularity decile, over 8 rounds (≈ twice as many matched pairs); the fuzzy answer-leak constraint; the base rate in the cue regression leaves out both items of a pair; pilot go/no-go rules fixed in advance.

**Engineering:**
- the "Answer:" prefix without a trailing space for A/B;
- the letter readout via start_readout;
- the cross-reference option and partner map;
- missing-item handling;
- answer-leak and subject constraints on the pairs, with swap repair;
- raw log-prob ranking for the saturated internal probabilities;
- S/X deduplication;
- pilots extended to Instruct SFT and Think DPO;
- flags and one-shot safety.
