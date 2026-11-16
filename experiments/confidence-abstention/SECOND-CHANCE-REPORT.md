# Meta-extension, stage 2: Second Chance — does the model know what it answered?

## The big picture

**The question.** Artemis's experiments found that OLMo-3 32B learns *selective abstention* (skipping the questions it would get wrong) at SFT and DPO, and that RL makes it abstain more without making it more selective. We are testing whether that pattern is specific to abstention or holds for metacognition in general, using different proxies on the same models and questions.

**The first two proxies.** Stage 1, *predict-before-answering*, also improved at **DPO** and then plateaued, the same shape as abstention.

**This proxy.** Stage 2 uses Ackerman's *Second Chance* game (2025). The model is told "Your answer was incorrect. Choose a different answer," but is not shown what it answered. A model that knows what it said will avoid that answer more often than when it is simply asked again.

Abstention and predict-before are about *how sure am I?* Second Chance is about *what did I say?*, which is self-modelling rather than confidence.

**What we found:**

- **Without reasoning, the models barely know their own earlier answer at any stage.** The lift is at most 0.03, against 0.1–0.34 for Ackerman's frontier models. Nothing appears at DPO the way abstention and predict-before do.
- **With reasoning (the Think line), the effect grows slowly from SFT to RL.**
  - The effect first becomes reliably above zero at DPO and is largest after the longest RL run (0.09).
  - The rise across the whole SFT → RL path is significant, but no single training stage is a clear jump.
  - RL keeps helping here, unlike abstention, where RL added no selectivity.
- **That growth looks like reasoning, not introspection.**
  - The later models talk themselves through it in the trace ("I might have picked X the first time…"). Traces that speculate about the earlier answer rise from 4% of the time at early SFT to 13% at DPO and **49%** after RL.
  - When the models do switch, they pick their second choice no more often than chance. That is the same pattern Ackerman saw in thinking models.
  - Much of the rising score also comes from later models simply answering more consistently when asked again; the switching rate itself stays flat.

**What this means for the study.** "Metacognition emerges at DPO" holds for **confidence-based** proxies: abstention and predict-before both show it. It does not extend to **self-modelling of one's own answer**. That ability is essentially absent without reasoning, and with reasoning it grows gradually, mostly through explicit reconstruction in the trace. So the DPO effect looks specific to knowing *how sure* you are, not a general switch for knowing yourself.

## All three proxies side by side

| Proxy (what it asks) | Instruct line, no reasoning (SFT → DPO → RL) | Think line, no reasoning | Think line, with reasoning |
|---|---|---|---|
| **Abstention selectivity** (Artemis): "skip what you'd get wrong" | 0.15 → **0.21** → 0.22 | 0.34 → 0.34 → 0.33 | 0.31 → **0.40** → 0.38 (midterm) |
| **Predict before answering** (stage 1, AUROC): "how likely am I to get this right?" | 0.57 → **0.65** → 0.63 | 0.62 → 0.63 → 0.58 | not run |
| **Second Chance** (stage 2, lift): "what did I answer before?" | 0.00\* → 0.01 → 0.02 | 0.01 → 0.03 → 0.02 | 0.02 → 0.06 → 0.06 / **0.09** |

**Bold** marks where the jump happens. \* Instruct-SFT mostly fails this format (see Caveats).

In the with-reasoning Second Chance column, the RL step has two numbers: OLMo 3 RL (step 750) and then OLMo 3.1 RL (step 2300). The two early SFT checkpoints (step 2000 and 6000) are in the detailed table below.

## What we ran

| | |
|---|---|
| Paradigm | Ackerman, *Evidence for Limited Metacognition in LLMs* (ICLR 2026). We copied his prompts and metrics from his code (github.com/cma1114/self_awareness), using two independent readers plus a verifier. |
| Game | Three turns. (1) Artemis's original question prompt; (2) the assistant turn shown as `[redacted]`; (3) "Your previous answer was incorrect. Respond ONLY with your new answer / Your answer was incorrect. Choose a different answer." followed by the same question. |
| Control | The same conversation, with turn 3 = "Your response was lost in transmission. Choose again." |
| First answer | Taken from Artemis's own run of the same checkpoint: her stored prompt, her answer, the same option order, the same mode. A check confirms that each reference run matches the checkpoint and settings of the run compared with it. |
| Items | 1000 PopQA multiple-choice questions, all played in both conditions, as Ackerman does. |
| Runs | 12 runs. **No reasoning** (greedy): Instruct SFT/DPO/RL and Think SFT/DPO/RL 3.1. **With reasoning** (Think checkpoints, T 0.6, 8192-token budget, Artemis's settings): SFT step 2000, step 6000 and final; DPO; RL 3.0; RL 3.1. |
| Main metric | **Normalized lift** = (game change rate − control change rate) / (1 − control change rate). 0 means the model doesn't avoid its earlier answer; 1 means it always does. |
| Cost | $14.65 for this stage; about $26 for the project so far. |
| Code | `tasks/second_chance.py` plus one registry line. Artemis's code is unchanged. |

## Results per checkpoint

Brackets give 95% bootstrap intervals (10,000 resamples). McNemar p is the paired test of game against control.

| Run | n | Change: game | Change: control | **Lift** [95% CI] | McNemar p |
|---|---|---|---|---|---|
| Instruct SFT (no reasoning) | 1000 | 0.10 | 0.10 | −0.00 [−0.02, 0.02] | 0.91 |
| Instruct DPO | 1000 | 0.12 | 0.11 | 0.01 [−0.01, 0.03] | 0.30 |
| Instruct RL | 1000 | 0.13 | 0.11 | **0.02** [0.01, 0.04] | 0.006 |
| Think SFT final (no reasoning) | 995 | 0.12 | 0.11 | 0.01 [−0.00, 0.03] | 0.12 |
| Think DPO | 999 | 0.13 | 0.10 | **0.03** [0.01, 0.04] | <0.001 |
| Think RL 3.1 | 978 | 0.11 | 0.09 | **0.02** [0.00, 0.03] | 0.04 |
| Think SFT step 2000 (reasoning) | 936 | 0.23 | 0.23 | 0.01 [−0.03, 0.04] | 0.70 |
| Think SFT step 6000 | 960 | 0.25 | 0.23 | 0.02 [−0.02, 0.05] | 0.46 |
| Think SFT final | 983 | 0.22 | 0.20 | 0.02 [−0.01, 0.06] | 0.19 |
| Think DPO | 998 | 0.22 | 0.17 | **0.06** [0.03, 0.09] | <0.001 |
| Think RL 3.0 | 995 | 0.22 | 0.17 | **0.06** [0.03, 0.10] | <0.001 |
| Think RL 3.1 | 966 | 0.23 | 0.15 | **0.09** [0.06, 0.12] | <0.001 |

### Change between stages

Paired bootstrap over the same items, 95% CI. \* marks a CI that excludes 0.

| Line | Step | Change in lift |
|---|---|---|
| Instruct, no reasoning | SFT → DPO | +0.01 [−0.01, +0.04] |
| | DPO → RL | +0.01 [−0.01, +0.04] |
| Think, no reasoning | SFT → DPO | +0.01 [+0.00, +0.03] \* |
| | DPO → RL 3.1 | −0.01 [−0.03, +0.01] |
| Think, with reasoning | SFT step 2000 → SFT final | +0.01 [−0.05, +0.06] |
| | SFT final → DPO | +0.03 [−0.01, +0.07] |
| | DPO → RL 3.0 | +0.01 [−0.04, +0.05] |
| | DPO → RL 3.1 | +0.04 [−0.01, +0.08] |
| | **SFT final → RL 3.1** | **+0.07 [+0.02, +0.11] \*** |

### Ackerman's three checks, plus the reasoning traces

These checks help tell whether a model really knows its answer or is using a shortcut. Chance is 1/3 for the first two.

| Run | Changed answers that are right, first answer wrong (n) | Changes that go to the 2nd choice | Does uncertainty predict a change? (AUROC) | Traces speculating about the earlier answer: game / control |
|---|---|---|---|---|
| Instruct SFT / DPO / RL | 0.63 (73) / 0.64 (91) / 0.66 (100) | 0.90 / 0.89 / 0.87 | 0.82 / 0.83 / 0.85 | — |
| Think SFT / DPO / RL 3.1, no reasoning | 0.64 (80) / 0.62 (84) / 0.53 (76) | 0.83 / 0.83 / 0.81 | 0.86 / 0.89 / 0.90 | — |
| Think SFT 2000 / 6000 / final, reasoning | 0.61 / 0.68 / 0.63 | 0.38 / 0.39 / 0.46 | 0.61 / 0.64 / 0.71 | 4% / 15% / 9% vs ≤1% |
| Think DPO / RL 3.0 / RL 3.1, reasoning | 0.64 / 0.67 / 0.66 | 0.42 / 0.44 / 0.40 | 0.75 / 0.68 / 0.60 | 13% / 21% / **49%** vs ≤1% |

**How to read these:**
- **When models change a wrong answer, they usually land on the right one** (0.53–0.68 vs 0.33). That is real knowledge, but it says nothing on its own about self-knowledge.
- **The second-choice check splits the two modes.**
  - Without reasoning, 81–90% of changes go to the second choice, so the models fall back on their own runner-up.
  - With reasoning, it is 38–46%, close to chance. This is Ackerman's finding for thinking models: a higher lift, but failing this check.
- **The traces.** "Speculating" means the trace explicitly reasons about what its earlier answer was. Spot-checked examples confirm it, and they often admit the model can't see it: "Wait, the first answer was [redacted], so maybe I initially guessed 1, 2, 3, or 4." This almost never happens in the control condition.
- **Entropy check (Test 3): mostly passed.** Being told "incorrect" doesn't simply make the models more uncertain.
  - The Think lines stay within ±0.02 bits of their first answer, except early SFT step 2000 with reasoning (+0.06).
  - Instruct DPO and RL become *more* certain in both conditions (−0.11 to −0.17 bits).

## Caveats

- **Instruct-SFT mostly fails the format.**
  - It answers `[redacted]`, copying the placeholder from its "previous" turn, in 66% of game trials and 34% of control trials. Its result above treats each of those as the digit the model rates most likely.
  - On the 334 items where it wrote a digit in both conditions, it repeats its first answer about 99% of the time in both, so the lift is −0.00 [−0.01, 0.00].
  - Either way it shows no lift. Instruct DPO and RL answer properly (≤0.5% fallback).
- **The rising lift with reasoning is partly a consistency effect.**
  - The game change rate is flat across stages (0.22–0.25). What falls is the control change rate (0.23 → 0.15): later models give the same answer more reliably when simply asked again.
  - Ackerman's metric counts that as more self-knowledge, but it can also be read as the model becoming more consistent.
- **The reasoning arm is one sample per item at T 0.6.** So some "changes" are just sampling noise, which the control condition absorbs. No single stage step is significant on its own (table above).
- **The system prompt changes at RL 3.1** ("You are Olmo… cutoff Dec 2024"). RL 3.0 keeps the DPO system prompt and shows the same lift as DPO (0.06 vs 0.06). So part of the RL 3.1 rise may come from the prompt or from the longer RL run; we can't separate the two.
- **Even "no reasoning" changes ~10% of answers in the control.** Answering inside a three-turn conversation differs from Artemis's single-turn prompt; the control condition accounts for this.
- **Scope:** one dataset (PopQA), one model family, and option numbers instead of Ackerman's letters.

## Files

- **Results:** `results/sc-*` (no reasoning) and `results/sct-*` (with reasoning). Each run has `second_chance.csv` (per item, with the traces), `raw_generations.csv`, `analysis.json` and `params.json`. They are also on the Modal volume `abstention-results` and are gitignored here.
- **Code:** `tasks/second_chance.py`. Its docstring lists every adaptation from Ackerman and the launch lines.
