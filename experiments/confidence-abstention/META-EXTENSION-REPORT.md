# Meta-extension: does "knowing what you know" emerge at the same training stage for other metacognitive proxies?

## The big picture

Artemis's experiments (midterm report, "Abstaining More, Not Better") found that in OLMo-3 32B, **selective abstention** emerges at SFT and DPO. Selective abstention means skipping more of the questions the model would get wrong. RL then makes the model abstain more overall without making it more selective. Our question is whether this pattern is **specific to abstention** or holds for metacognition more broadly.

To test it, we added a second, independent proxy on the same models and questions, with no other changes. The proxy is **predict-before-answering** (Barkan et al. 2025, Exp. 1): the model sees the full multiple-choice question and states, without answering, how likely it is to get it right (0–100%). We then check whether those predictions separate the questions it actually gets right from those it gets wrong.

**Main finding:**
- **Instruct line (no reasoning):** the ability jumps at **DPO** and stays flat at RL. That is the same shape as Artemis's abstention selectivity, so the pattern holds for a second metacognitive proxy.
- **Think line (thinking switched off):** nothing changes at DPO, and RL makes the predictions slightly worse.

In both lines, the model's internal answer confidence stays flat across stages. So what changes is how well the model *reports* what it knows, not how much it knows.

## What we ran

| | |
|---|---|
| Models | 6 checkpoints: Instruct SFT → DPO → RL, and Think SFT (final) → DPO → RL (3.1, step 2300) |
| Questions | Artemis's 1000 PopQA multiple-choice items, with options in her exact order |
| Prompt | Barkan et al. Exp. 1 wording, whole question shown, "respond only with a percentage" |
| Thinking | Off. This is the only setting with a matching Artemis readout for all six checkpoints. |
| Scored against | Artemis's own runs on the same checkpoints: same answers, same abstentions |
| Code | One new task file, `tasks/predict_before.py`, plus one registry line. Artemis's code is unchanged. |
| Cost | ~$4.10 for this run; ~$11.60 in total including the pilots |

**Main metric:** AUROC of the predicted %, i.e. how well the prediction separates right from wrong answers. 0.5 is chance and 1.0 is perfect. We also read the prediction from the model's token probabilities over the numbers ("LP"); this version is less noisy.

## Results

| Checkpoint | Accuracy | Mean prediction | Prediction AUROC [95% CI] | Prediction AUROC (LP) [95% CI] | Answer-confidence AUROC | Abstention selectivity Δ (Artemis) |
|---|---|---|---|---|---|---|
| Instruct SFT | 0.65 | 80% | 0.57 [0.55, 0.59] | 0.66 [0.62, 0.69] | 0.80 | 0.15 |
| Instruct DPO | 0.66 | 63% | **0.65** [0.62, 0.68] | 0.70 [0.67, 0.73] | 0.79 | 0.21 |
| Instruct RL | 0.64 | 71% | 0.63 [0.60, 0.67] | 0.70 [0.66, 0.73] | 0.80 | 0.22 |
| Think SFT | 0.68 | 42% | 0.62 [0.59, 0.66] | 0.65 [0.61, 0.68] | 0.80 | 0.34* |
| Think DPO | 0.68 | 55% | 0.63 [0.59, 0.66] | 0.66 [0.62, 0.69] | 0.81 | 0.34* |
| Think RL | 0.68 | 28% | **0.58** [0.55, 0.62] | 0.62 [0.59, 0.66] | 0.79 | 0.33* |

\* For the Think line this is Artemis's **no-thinking** abstention readout. Her midterm with-thinking numbers are different (0.31 → 0.40 → 0.38) and are not directly comparable to this run.

### Change between stages (paired bootstrap, 95% CI; **bold** = CI excludes 0)

| Line | SFT → DPO | DPO → RL |
|---|---|---|
| Instruct, prediction | **+0.08** [+0.05, +0.12] | −0.02 [−0.05, +0.01] |
| Instruct, prediction (LP) | **+0.04** [+0.03, +0.06] | −0.00 [−0.02, +0.02] |
| Think, prediction | +0.00 [−0.02, +0.02] | **−0.04** [−0.08, −0.01] |
| Think, prediction (LP) | +0.01 [−0.00, +0.02] | **−0.03** [−0.06, −0.01] |

## What it means

1. **Instruct: the same story as abstention.** Prediction improves at DPO, then plateaus at RL. This matches the abstention selectivity (0.15 → 0.21 → 0.22). Two different metacognitive proxies peak at the same training stage.
2. **The model's knowledge doesn't change; how it uses it does.** Answer-confidence AUROC is flat (~0.80) at every stage. DPO improves how well the model *states* what it knows, not the knowledge itself.
3. **Stated predictions are worse than the internal signal.** Prediction AUROC is 0.57–0.65, while the model's own answer confidence reaches ~0.80. This gap is consistent with what Barkan et al. found.
4. **Think line: no DPO jump without thinking, and a small drop at RL.** The RL model also becomes very pessimistic: it predicts 28% on average while getting 68% right.

## Caveats

- **The Think line is measured without thinking.** It shows what the model knows before it reasons, and should be labelled "no-thinking" on any plot next to the midterm numbers.
- **The RL checkpoints use a different built-in system prompt.** It changes from "You are a helpful AI assistant" to "You are Olmo…, cutoff Dec 2024". Part of the RL effect may come from this prompt rather than from RL itself.
- **Run-to-run noise is about ±0.01 AUROC.** Instruct-SFT, for example, hesitates between "80%" and "90%".
- **Parse rate is ≥ 98% everywhere.** The few unparsed Think replies started reasoning and ran past the 50-token limit.
- **Only one dataset (PopQA) and one model family.**

## Possible next steps (not started)

- Check Artemis's existing system-prompt-controlled runs (`-sys30` / `-sys31`) to separate the RL effect from the prompt change.
- Add a third proxy (e.g. "Second Chance") to see whether the DPO-emergence pattern holds again.
