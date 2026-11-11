# Spec: confidence steering with and without a reasoning trace

Question: Kumaran et al. (2026, NMI) show that adding a confidence direction to the residual stream shifts abstention
in a no-CoT model (Gemma 3 27B: 66.5% → 7.0% across the steering range, layer ~31 of 62). With a trace, abstention
could instead be the outcome of re-deriving an answer and comparing it with the options. This experiment tests whether
a confidence direction still governs abstention once the model deliberates.

Design in one line: Instruct carries the paper's pathway (positive control, same base); Think carries the trace; within
Think, the question is whether a pre-deliberation confidence direction changes what happens when the model then
deliberates.

Pre-registered outcomes (fill in after the run; do not revise the predictions):

| hypothesis | Instruct, no-trace gate | Think reasoning, manipulation check (§5.2) | Think reasoning, abstention by position | Think reasoning, by route |
|---|---|---|---|---|
| H-threshold: the paper's pathway survives reasoning | abstention monotone in −α, swing ≥ 20 pp | probe score moves with α | START and THINK: abstention monotone in −α, swing of the same order as Instruct's; POST: small | RESOLVE and JUDGE both move |
| H-mismatch: abstention is computed in the trace | same | probe score moves with α | START: small or none; THINK: a shift smaller than Instruct's; POST: none | JUDGE moves, RESOLVE does not |
| H-bypass: the confidence signal is not read in reasoning mode | same | probe score moves with α | no shift at any position | nothing moves |

Outcome mapping, per position mode:

- signal moves and behaviour moves → H-threshold or H-mismatch, separated by position and route as in the table;
- signal moves and behaviour does not → H-bypass;
- signal does not move → bad vector or scale; no inference about the hypotheses.

"Signal moves": the manipulation-check probe score at the answer position, (b) of §5.2, has a positive slope on α
(slope − 1.96 · item-clustered SE > 0). (a) is a scale check: it must move in POS-START and POS-THINK and stay flat in
POS-POST.
H-bypass additionally requires (i) the pre-trace vector to pass the validity bar (§5.1) and (ii) a positive-control
gate PASS (Instruct, or the prefixed Think arm) run in the same session.

Any of the three is a result. The Instruct gate is the positive control: if it does not reproduce the paper, stop and
debug before running reasoning mode.

## 1. Models and modes

Base family: Olmo-3.1-32B (dense, open; 64 layers, hidden size 5120, mixed sliding/full attention).

- **Primary: Olmo-3.1-32B-Think** (`allenai/Olmo-3.1-32B-Think`, main = RL step 2300), **reasoning mode** (native
  thinking). Harness run: `results/olmo32b-popqa-think-rl31-main` (T = 0.6, seed 42, budget
  8,192, k = 5 Phase 1 samples in `k_answers`, judge labels, in-trace probe). Olmo-3 has no think special tokens: the
  generation prompt ends in `<|im_start|>assistant\n<think>` as plain text, and `</think>` tokenizes as ordinary pieces
  that merge with neighbours (`</`=524, `think`=27963, `>`=29; `.</`=4005), so the hook detects the closing tag on the
  decoded text, as `split_think` does, not on token ids.
  Decoding is decided by the Qwen pilot (§6): greedy is preferred so that α is the only thing that changes between
  conditions, but Olmo-Think may loop under greedy. If the pilot's greedy truncation rate at budget 4,096 exceeds 10%,
  switch the main condition to T=0.6, seed 42, budget 8,192 (the harness settings) and add a second seed at
  α ∈ {−2, 0, +2}. Either way, the α = 0 condition is also run at the harness settings and compared with the harness run
  on the same 300 items; that, not the greedy α = 0, is the hook check.
- **Positive control: Olmo-3.1-32B-Instruct** (`allenai/Olmo-3.1-32B-Instruct`, `--thinking off`), **no-trace mode**:
  the chat template's prompt, single-token answer read at the last prompt position. The paper's setting on the same base.
- **Secondary arm on Think (robustness check), no-trace mode:** the native prompt followed by the force-closed empty
  block `<think>\n\n</think>\n\n` and an answer prefix (`--answer-prefix`, chosen by a format check that measures
  first-token digit mass; default none). Olmo-Think has no trained no-think mode, so this arm is a check on the Instruct
  gate, not the no-trace condition of the design.

Development pilot: **Qwen3-8B** (hybrid, `enable_thinking` on/off, 36 layers) on PopQA-MC. Build and debug the whole
pipeline there; its numbers are not part of the result.

## 2. Data

PopQA-MC (`data/popqa_phase0_questions.csv`, `popqa_phase1_questions.csv`), the paper's Phase 2 prompt (OSF), option
shuffle seed 1042 as in `abstention_modal.py`.

Instruct (and the Think secondary arm), the paper's construction:

- **Vector-building set:** the Phase 2 prompt (four options plus "5") on the 1,000 Phase 0 questions, no-trace mode.
- **Evaluation set:** 300 Phase 1 items, seed 0, stratified 150 correct / 150 incorrect by the model's own unsteered
  no-trace Phase 1 answer.

Think (reasoning mode), from the harness run `results/olmo32b-popqa-think-rl31-main`:

- **Labels:** k-sample consistency, the number of the item's 5 Phase 1 `k_answers` equal to `gt_position`. Sample 1 is
  the Phase 1 answer (the other four use seed + 1).
  high = 5/5; low = 1–4/5 and Phase 1 correct, i.e. sample 1 correct and 1–3 of the other four wrong. Both groups are
  Phase 1-correct, as in the paper's correct-only construction: the contrast is "knows it reliably" vs "knows it
  unreliably". mixed = 1–4/5 regardless of Phase 1 correctness (a secondary validity contrast); 0/5 is used in neither.
- **Evaluation set:** 300 Phase 1 items, seed 0, stratified 150 correct / 150 incorrect by the harness run's Phase 1
  correctness, stored in shuffled order.
- **Vector-building set:** the remaining 700 Phase 1 items. Separate from the evaluation items, as in the paper; it
  uses Phase 1 questions because the labels need the k samples.

## 3. Steering vectors

### 3.1 Instruct (and Think secondary arm): the paper's recipe

As in the paper's released notebook (OSF, Cells A and B), on the vector-building set. Residuals are read at the last
prompt token, the position whose residual produces the first generated token and where steering is applied (after the
chat template's assistant header; for the Think arm, after the force-closed block and answer prefix), layer L:

```
p_i        = softmax(logprob_i,1..5 / τ), τ fitted on the no-trace Phase 0 run (analyze.fit_tau; paper: 5.1 for Gemma)
margin_i   = max(p_i,1 … p_i,4) − p_i,5
candidates = among correct trials, the 75 with the largest margin and the 75 with the smallest
high       = walk the top candidates in descending margin, keeping a trial while fewer than int(25 · 0.28) = 7 of the
             group chose the same option, until 25 trials
low        = the same walk over the bottom candidates in ascending margin
v_L        = mean_i∈high h_L,i − mean_i∈low h_L,i          (residual stream after block L)
v̂_L       = v_L / ‖v_L‖
scale_L    = 0.03 · mean_i ‖h_L,i‖ over all vector-building trials   (one constant per layer)
steer(α)   : h_L ← h_L + α · scale_L · v̂_L                (α < 0: the paper's low-confidence vector, −v̂_L)
```

Record, per group, the mean calibrated P(best) = max(p_1 … p_4) and P(5), and the chosen-option counts.

### 3.2 Think: the pre-trace vector

Residuals at the last prompt token of the native reasoning-mode Phase 2 prompt (paper wording; the prompt ends in
`<|im_start|>assistant\n<think>`): the state just before deliberation, on-distribution, and exactly the POS-START
position. One forward pass per item (700 vector-building + 300 evaluation), no generation.

```
n          = 70, or the largest n ≥ 50 at which both draws below fill under the cap
high       = n vector-building items labelled 5/5, drawn at random (seed 0), keeping an item while fewer than
             int(0.28 · n) of the group share its gt_position (the nuisance here is answer position)
low        = n items labelled low (1–4/5, Phase 1 correct), the same draw
v_L, v̂_L, scale_L, steer(α) as in 3.1 (scale_L over the 700 vector-building items)
```

Groups are larger than the paper's 25 because the labels are readout-free and noisier than the paper's margins.

**L* (Think)**: 5-fold cross-validation within the vector-building set (folds over its 700 items, seed 0). In each
fold the direction is mean(high) − mean(low) over all training-fold high and low items, and the held-out fold's high
and low items are scored by their projection on it; L* = the sweep layer with the best mean CV AUROC. The 300
evaluation items are not used to choose L*, only to confirm it (§5.1).

### 3.3 Common to both

Vectors at every 4th layer (0, 4, …, 60), which covers the sweep layers 16, 20, …, 48 (25–75% of 64) and L* ± 8.
Saved with scales and selected items to `results/steer-<model>-<tag>/vectors.pt`.

Controls, at every vector layer, with the same scale_L:

- `random`: one unit vector drawn from a standard Gaussian of the hidden size (seed 0).
- `shuffled`: a balanced permutation of the selected items into two pseudo-groups, each half real high and half real
  low (seed 0); the difference of their means, with its component along v_L removed (exactly orthogonal to v_L),
  carries no real signal.

## 4. Conditions

### 4.1 Instruct no-trace gate (positive control)

Layer sweep: layers 16, 20, …, 48, α ∈ {−2, −1, 0, +1, +2}, 300 Phase 2 items, single-token answer. One forward pass per
item per condition. The paper's effect peaked at half depth (31 of 62); expect L* near 32.

Pass criterion: at the best layer, abstention is monotone in −α and the swing between α = −2 and α = +2 is at least
20 percentage points, with `random` and `shuffled` controls moving by less than 5 points. L* (Instruct) is the layer
with the largest swing.

Also Phase 1 (no abstain option) at every sweep layer, α ∈ {−2, +2}, to check steering does not change accuracy by more
than a few points. A confidence vector should not be a knowledge vector.

### 4.2 Think secondary arm (no-trace, force-closed block + answer prefix)

The same gate, criteria and Phase 1 check as 4.1, on Think with the 3.1 vector built in that rendering. Reported next to
the Instruct gate; it does not gate reasoning mode.

### 4.3 Think reasoning mode, pre-trace vector

L* (Think) as in 3.2; all layers reported. At L* (plus L* − 8 and L* + 8 at α = ±2 only), α ∈ {−2, −1, 0, +1, +2}, 300 evaluation items, three positions:

- **POS-START**: add at the last prompt token only, the position the vector was read at. Acts on the rest of the
  generation only through that position's KV entries, so its effect may be weak in a long trace.
- **POS-THINK**: add at that position and every generated position through the closing think tag. The main condition.
- **POS-POST**: add only at generated positions after the closing think tag. Tests whether the post-trace answer
  position still reads the signal (expected small).

Controls: `random` and `shuffled` at POS-THINK, α = ±2.

Sampled replicate: POS-THINK, L*, α ∈ {−2, 0, +2}, T = 0.6, seed 42.

Condition count: 5 α × 3 positions + 4 (layer offsets) + 4 (controls) + 3 (sampled) = 26 runs × 300 traces (α = 0 is one
shared unsteered run per temperature). Run α = 0 first and compare against the harness run on the same items
(abstention rate, mean length) as a correctness check on the hook.

### 4.4 Compute

| stage | work | estimate |
|---|---|---|
| Instruct `notrace` + `gate` | 3,000 + 29,700 single-token forward passes | ~$2.5 |
| Think `pretrace` | ~1,000 forward passes, no generation | ~$1 |
| Think `notrace` + `gate` (secondary arm) | as Instruct | ~$2.5 |
| Think `reason` | 26 runs × 300 traces, ~1,500 tokens each (~12M tokens), HF `generate`, bf16, 1×H200; plus one prefill pass per non-truncated trace for the manipulation check | several hours; recomputed from the α = 0 run's tokens/s and mean length before launching the rest |

## 5. Outcomes, per condition

From each Phase 2 generation, the harness's existing parsing (`final_answer`, `is_nota`, `reasoning_tokens`,
`wait_count`, `truncated`, `parse_method`):

| outcome | use |
|---|---|
| abstention rate | primary |
| P(abstain \| P1 right) and P(abstain \| P1 wrong): Instruct by its unsteered no-trace Phase 1 answer, Think by the harness run's Phase 1 answer | selectivity |
| route (RESOLVE / JUDGE / MIXED) of the item's α = 0 trace, from the judge pipeline (`trace_labels/`), passed via `--routes` | H-mismatch vs H-threshold |
| median `reasoning_tokens`, mean `wait_count` | does confidence steering change deliberation length |
| truncation count | sensitivity |
| Phase 1 accuracy at α = ±2 (4.1, 4.2) | capability check |

### 5.1 Pre-trace vector validity (offline, from the pretrace pass)

On the 300 evaluation items, per vector layer, of the projection h_L · v̂_L:

- held-out AUROC for high vs low, the vector's own contrast (primary), and for high vs all mixed (secondary);
- the CV AUROC of 3.2 alongside;
- Spearman correlation with log(1 + s_pop) (`data/popqa_phase1_questions.csv`): a difficulty-nuisance check, the
  direction should not be only popularity;
- mean projection of abstained minus answered items (harness Phase 2 `is_nota`), in SD units of the projection.

Validity bar (pre-registered): bootstrap 95% CI of the primary evaluation AUROC at L* (1,000 resamples of the labelled
evaluation items, seed 0). The vector is valid iff the lower bound exceeds 0.55.

### 5.2 Manipulation check (Think reasoning conditions)

At the steered layer the residual shifts by α · scale_L by construction, so the check reads a downstream layer.
Probe score = h_L′ · v̂_L′, the residual at L′ = L + 8, the steered layer plus 8, projected on that layer's own unit
pre-trace vector. L′ is capped at the last vector layer (60); if the cap leaves L′ ≤ L, the condition's probe is
uninformative (NaN, marked so in the table). The unsteered α = 0 run records the score at every vector layer, so each
condition compares against its own L′. Read in every reasoning condition at two positions:

- (a) the last prompt token, during prefill. Steered in POS-START and POS-THINK; in POS-POST the prefill is unsteered,
  so POS-POST's (a) is a no-effect reference.
- (b) the position that produces the first answer token: the last input position before the first content token
  after the closing think tag (the token Context.split reads the digit distribution from). One extra forward pass
  over prompt + generated ids up to that position, with steering added at exactly the positions the mode steered
  during generation. Truncated traces are skipped. The recompute must reproduce generation at that position: its top-1
  token equals the generation's recorded top-1 (and, under greedy decoding, the generated token), with log-prob
  agreement (fp32: every row, within 1e-3; bf16: ≥ 98% of rows, median within 0.05).

Reported per (position mode, layer, vector): mean probe score at each α for (a) and (b), OLS slope of the score on α
with item-clustered SE, and whether the means are monotone non-decreasing in α. Columns `probe_prompt_end`,
`probe_answer_pos` (plus `probe_layer`, `probe_top1_match`, `probe_lp_diff`) in the reason CSVs.

Analysis (`analyze_steering.py`):

- Logistic regression `abstain ~ α`, per position and layer, cluster-robust SE by item (same 300 items at every α).
  Report the slope, the abstention rate at each α, and the α = −2 vs +2 swing with a paired-bootstrap CI.
- Swing by route: the same regression restricted to items whose α = 0 trace is RESOLVE vs JUDGE (`--routes`). The
  H-mismatch prediction is a slope near zero for the first and a negative slope for the second.
- Swing by Phase 1 correctness.
- `log(reasoning_tokens) ~ α`, same structure.
- The validity table and bar of 5.1, and the manipulation-check table of 5.2, for pre-trace runs.
- One summary table: rows = conditions, columns = abstention at each α, swing [CI], swing by route, median length at
  α = ±2. Numbers only, `STEERING-RESULTS.md`.

## 6. Implementation notes

- HF `transformers`, bf16, `output_hidden_states` not needed: register a forward hook on the chosen decoder layer that
  adds α · scale_L · v̂_L (a fixed magnitude per layer, §3) at the selected positions. With KV caching the hook sees one
  new position per step; keep a step counter and track the closing think tag on the decoded text to implement the three
  position modes. Verify the hook fires exactly once per generated token (assert on the counter against
  `generated_tokens`). α = 0 is byte-identical to no hook.
- Reuse `abstention_modal.py`'s prompt construction, option shuffle, digit readout and parsing; do not reimplement
  them. Files: `steering_modal.py` (model load, hooks, generation; stages `notrace`, `gate`, `pretrace`, `reason`),
  `build_vector.py`, `analyze_steering.py`. Outputs to `results/steer-<model>-<tag>/` with the same `phases.csv`
  schema plus columns `alpha`, `layer`, `position_mode`, `vector`. The `pretrace` stage reads the harness run's
  `phases.csv` from the results volume and writes `pretrace_items.csv`, `pretrace_hiddens.pt`, `vectors.pt`,
  `eval_items.json`; `reason` then runs on that directory. `python build_vector.py <run dir>` rebuilds `vectors.pt`
  (selection, CV L*, validity, CI) from `pretrace_items.csv` and `pretrace_hiddens.pt` alone. The Think secondary arm uses its own tag so its `vectors.pt`
  does not replace the pre-trace one.
- Credit the paper in `NOTICE.md` (recipe and prompt).
- Pilot everything on Qwen3-8B with 50 items before touching the 32B.

## 7. What a result looks like

Instruct carries the paper's pathway (positive control, same base); Think carries the trace; within Think, the question
is whether a pre-deliberation confidence direction changes what happens when the model then deliberates.

Every Think reading starts from the manipulation check: if the probe score does not move with α, the vector or the
scale failed and nothing is inferred.

- H-threshold: a clear Instruct gate, the probe moving, and a POS-START / POS-THINK dose-response in Think of the same
  order. Confidence governs abstention through the trace as well as without it, and the trace-level labels were
  describing the surface.
- H-mismatch: a clear Instruct gate, the probe moving, and a Think swing concentrated in JUDGE items, RESOLVE items
  unmoved. Two abstention mechanisms, one of which confidence governs.
- H-bypass: a positive-control gate PASS in the same session, a valid pre-trace vector (§5.1 bar), the probe moving,
  and no abstention shift in Think reasoning mode at any position. The signal is there and is not read: the paper's
  causal pathway is specific to the no-CoT regime.

The Think secondary arm says whether the Think weights carry the paper's pathway when forced to answer without a trace;
it qualifies, but does not decide between, the hypotheses.

Also worth reporting whichever way it goes: whether steering changes trace length. If low-confidence steering
lengthens traces, that is the first causal link between internal confidence and deliberation length.
