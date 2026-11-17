# Summary note, 17/11/2026: what was done on branch `meta-extension`

A note for a future Claude session: what was run, where everything lives, what was found, and what to watch out for.

## Context

- **Repo:** SPAR project `cogni-load-spar`. Folder: `experiments/confidence-abstention/`.
- **Collaborator Artemis Veizi** built the harness (`abstention_modal.py`, a Modal + vLLM runner on H200, with a task registry in `tasks/`). Her branches are `exp/aveizi/confidence-abstention` and `exp/aveizi/paradigm-framework`. **Never modify her code or branches.** Our work adds task files plus registry lines only.
- **Her finding (midterm, "Abstaining More, Not Better"):** in OLMo-3 32B, selective abstention (Kumaran et al. protocol on PopQA) emerges at SFT/DPO. RL raises the abstention rate without raising selectivity.
- **Our question:** is that stage pattern specific to abstention, or general across metacognitive proxies?
- **Checkpoints used:**
  - Instruct line: 3.1 SFT / DPO / RL (SFT and DPO need `--dtype bfloat16`).
  - Think line, no reasoning: SFT-main / DPO / RL 3.1-main, scored against her `start_format.csv` readout (empty think block + "Answer: ").
  - Think line, with reasoning: SFT 1e-4-step2000 / step6000 / main, DPO, RL 3.0-main, RL 3.1-main.
  - Reference runs on the volume: `olmo32b-popqa-<key>`.
- **Modal:** results volume `abstention-results`, HF cache `abstention-hf-cache` (all checkpoints cached). Workspace `spar-f26`.

## The three experiments (all on `meta-extension`)

### 1. Predict before answering (Barkan et al. 2025, arXiv 2512.24661, Exp. 1)

- **Code:** `tasks/predict_before.py`. **Report:** `META-EXTENSION-REPORT.md`.
- **Setup:** the model sees the full 4-option question and states the percentage chance it would get it right, without answering. No reasoning, as in Barkan (budget 0; with a trace the model just answers in it). 50-token replies; the last "N%" is parsed. Run tags `pbo50-*`. Cost ~$4.
- **Result:** Instruct prediction AUROC 0.57 → **0.65** → 0.63 (jump at DPO; same shape as abstention 0.15 → 0.21 → 0.22). Think 0.62 → 0.63 → **0.58** (drop at RL). Answer-confidence AUROC is flat (~0.80) everywhere.
- **Note:** Barkan's Exp. 3 (SWE-bench) did run reasoning models with reasoning on, and they were no better. Exp. 1/2 used budget 0.

### 2. Second Chance (Ackerman 2025, arXiv 2509.21545)

- **Code:** `tasks/second_chance.py`. **Report:** `SECOND-CHANCE-REPORT.md`. Ackerman's code (github.com/cma1114/self_awareness) was read and verified by agents.
- **Setup:** three turns: her Phase 1 prompt → `[redacted]` → "Your answer was incorrect. Choose a different answer." Control: "Your response was lost in transmission. Choose again." Metric: normalized lift.
  - 12 runs: `sc-*` (no reasoning, 6 checkpoints) and `sct-*` (with reasoning, 6 Think checkpoints, T 0.6, budget 8192). Cost $14.65.
- **Result:**
  - Without reasoning, lift ≤ 0.03 at every stage.
  - With reasoning, 0.02 (SFT) → 0.06 (DPO) → 0.06 (RL 3.0) → 0.09 (RL 3.1). Only the whole SFT → RL 3.1 rise is significant.
  - Traces that speculate about the earlier answer go 4% → 13% → 49%.
  - Second-choice check: with reasoning near chance (0.38–0.46); without, 0.81–0.90.
  - Instruct-SFT copies `[redacted]` as its answer in 66% of game trials.
- **Interpretation, with corrections made during the session:**
  - The "85–90% of switches go to the second choice" figure is **not** evidence of self-knowledge: the control condition switches to the second choice at the same rate (near-ties flipping).
  - The paradigm mixes up self-knowledge with willingness to change, so the null without reasoning is ambiguous.
  - With reasoning, the rising lift is partly a consistency effect (the control's change rate falls).

### 3. Confidence forced choice (CFC), designed from scratch together with the user

- **Code, plan and report:** all in `tasks/cfc/`: `PLAN.md`, `REPORT.md`, `LAUNCH.md`, `__init__.py`, `build_pairs.py`, `pairs_data.py`, `analyze_cfc.py`, `check_local.py`.
- **Motivation:** a hypothesis built from experiments 1–2 and Artemis's data. The internal signal is flat; DPO improves the **readout** of it into behaviour; RL shifts the level.
- **Design:**
  - Mamassian & de Gardelle 2022 confidence forced choice: the model sees two PopQA questions; "which would you more likely answer correctly?" (P). Also S = its own answers shown, "which answer is more likely correct?", and X = the same with a sibling's answers.
  - Self-vs-other follows Binder et al. 2024 (arXiv 2410.13787): the same target, a different judge, both directions averaged, tested against the internal-probability baseline.
  - 5,829 fixed pairs (3,829 matched on relation + the same/neighbouring popularity decile; 2,000 random). Answer leaks and shared subjects are excluded. Each pair is shown in both orders. One-token A/B readout, debiased.
- **Pre-registration:** two primary tests with Holm (H2: Instruct SFT → DPO on matched P pairs; H4: pooled self-advantage). TOST for "flat at RL". A graded (strength-rank-weighted) secondary measure, added after the pilots and before the main runs.
- **Runs:** pilots `cfc-pilot-*` ($1.55), main `cfc-*` ($5.19).
- **Result:**
  - Stated accuracy 0.51–0.59 against internal ~0.75.
  - The model mostly answers by position: the same letter in both orders in ~77% of pairs (Instruct DPO), with preferences ~0.06 from 50/50. The real signal only leaks in as a small shift.
  - Share of the signal used: Instruct 0.18 → 0.33 → 0.38; Think 0.18 → 0.20 → 0.04.
  - H2 +0.027, p = 0.056 (narrowly not significant; the secondary variants are significant). H4 −0.007, so no self-advantage.
- **Framing agreed with the user:** this is **not** metacognition or introspection in the strong sense (a sibling judges as well). The defensible claim is "post-training changes the monitoring / readout of correctness; the knowledge signal stays flat". The sibling is a near-twin (85% same answers), and the cross test covered only S/X, so it is weak either way. A fair test needs a cross-family judge.

## Overall story

Three measures agree. On the Instruct line, the use of the internal correctness signal improves at DPO; on the Think line it drops at RL; the signal itself stays flat. The effects are small and nothing is self-specific. Possibly a workshop paper as a package; not publishable alone.

**Open next steps** (suggested, not started):
- hidden-state probes, to check the signal really is flat;
- a cross-family judge, e.g. Tülu 3 / OLMo-2 (which also have SFT/DPO/RL checkpoints);
- a "show the answer" control for Second Chance;
- fitting Kumaran's two-stage model per checkpoint.

**Total Modal spend ≈ $33.**

## Practical gotchas learned

- **Windows:** the Modal CLI needs `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`; subprocess readers need `encoding="utf-8"`.
- **Retries:** `run()` has Modal `retries=2`, so any exception after model load re-runs the GPU job up to 3×.
  - Save raw outputs right after generation (`ctx.save`).
  - Wrap scoring and analysis in try/except.
  - Check reference runs (`check_reference`) before generating.
- **One-token readouts:** use `second_chance.start_readout` (log-probs at position 0), not `ctx.generate(max_tokens=1)`. `split()` skips whitespace tokens.
- **Think prefixes:** for letter answers, use `"\n</think>\n\nAnswer:"` with NO trailing space (" A" is one token). Digits worked with "Answer: ".
- **Packaging:** `add_local_python_source` ships only `.py` files, so data inside `tasks/` must be a `.py` module (see `tasks/cfc/pairs_data.py`).
- **System messages:** they replace Olmo's default system prompt, so put instructions in the user turn. The RL 3.1 checkpoints have a different default system prompt ("You are Olmo…"), a standing confound. Artemis's `-sys30/-sys31` control runs exist only with reasoning, and need a harness option not in this branch.
- **Position bias:** greedy A/B choices are mostly positional, so always show both orders.
- **Background monitors** were once killed by Claude Code for low system memory. The Modal runs are unaffected; check status with a one-off `modal app list --json`.

## How this user likes to work

- Short, plain answers; a readable top first, numbers after; no long hedged walls of text.
- Before any Modal run: verify carefully, give a cost and time estimate, and wait for explicit approval. Monitor spend, but never auto-stop runs.
- **Delegation:** use agents (coder, reviewers, a plan↔code checker), but keep re-checks narrow (only the new parts) to save tokens and time.
- Be transparent about mistakes. The user caught that the Barkan setup at first hid the options, that 16-token replies cut answers off, and that the "second choice = self-knowledge" claim was wrong.
- Truncation and harness fixes are Artemis's job. Do not fix them.
- Keep each experiment's files together in one folder (as `tasks/cfc/`).
