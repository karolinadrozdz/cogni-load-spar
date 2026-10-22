# Confidence → abstention under reasoning (Kumaran et al. 2026, reasoning-on extension)

Runs the paper's Phase 0/1/2/4 + verbal-confidence pipeline on Qwen3 checkpoints with thinking
on, via vLLM on Modal. Prompts, option shuffle, logprob readout and CSV schema are the paper's
(OSF `resw4`, `Confidence_Drives_Abstention_Notebook.ipynb`); `data/README-osf.md` is their README.

## Files
- `abstention_modal.py` — the experiment (`main`) and the in-trace probe (`checkpoints`); see docstring for all flags
- `analyze.py` — the paper's two headline tests on one run directory
- `analyze_checkpoints.py` — analysis of the in-trace confidence checkpoints (below)
- `data/phase0_questions.csv`, `data/phase1_questions.csv` — the paper's calibration / main sets
  (SimpleQA 4-way MCQ with foils; `choice_1` is ground truth, shuffled at runtime with seed 1042)

## Run
```
modal profile activate spar-f26          # shared workspace
pip install modal                        # local only; everything else is in the image

# 1. quick early signal (~1 H100, phases 0/1/2 + verbal on 200 questions)
modal run abstention_modal.py::main --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native --n 200

# 2. paper-matched model, always thinks (80B bf16: --gpu H200:2 or H100:4)
modal run abstention_modal.py::main --model Qwen/Qwen3-Next-80B-A3B-Thinking --thinking native --gpu H200:2

# 3. hybrid: same weights, thinking off (paper's regime) then on
modal run abstention_modal.py::main --model Qwen/Qwen3-8B --thinking off --phases 0,1,2,4
modal run abstention_modal.py::main --model Qwen/Qwen3-8B --thinking on  --phases 0,1,2,4

# fetch + analyse (analysis.json is also computed inside the container at the end of the run)
modal volume get abstention-results <tag> results/
python analyze.py results/<tag>
```
The file has two entrypoints, so Modal needs the explicit `::main` / `::checkpoints` form.
`<tag>` is printed at the end of a run (`<model>_<thinking>_n<N>_<timestamp>`); `--tag` sets it.
Set `ABSTENTION_RESULTS_VOLUME=<name>` to write to your own volume in the shared workspace.

## Overnight run + hand-off
```
# launcher: --detach keeps the job running after this terminal/laptop goes away; fix the tag
modal run --detach abstention_modal.py::main --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native \
    --n 1000 --tag overnight-30b-thinking
modal app logs confidence-abstention      # progress, any time, from any machine in the workspace

# receiver (needs a Modal token for spar-f26 and `pip install -r requirements.txt`):
./handoff.sh overnight-30b-thinking       # waits for params.json, pulls results/<tag>/, runs analyze.py
```
`params.json` is written last, so its presence in the volume means the run is complete.
Everything the analysis needs is in `results/<tag>/` (`phases.csv`, `verbal.csv`, `analysis.json`,
`params.json`); the raw reasoning traces are in the `reasoning` column of both CSVs.

## In-trace checkpoints
After a thinking trace the answer-digit distribution is one-hot (prob > 0.999 on the chosen option
for every question): the decision was made in the text. This reads confidence *during* the trace instead.
```
modal run abstention_modal.py::checkpoints --tag 80b-thinking --gpu H200:2   # needs <tag>/phases.csv in the volume
modal volume get abstention-results 80b-thinking results/
python analyze_checkpoints.py results/80b-thinking
```
Options: `--phases 1,2`, `--every 256`, `--max-checkpoints 20`, `--n 0` (0 = all questions), `--model ""` (override the model in `params.json`).

**What the probe does.** For every complete (non-truncated) trace in `phases.csv`, it cuts the trace after the
first k tokens, appends `\n</think>\n\n` and reads the digit logprobs of the next token ("if you had to answer
now"), exactly as the main run does. Checkpoints (`kind`): `start` (empty trace), `fixed` (every `--every` tokens),
`marker` (just before a backtrack word: wait, actually, hmm, hold on, let me recheck/reconsider/rethink/double-check/think again,
on second thought, "no,") and `end` (full trace). Traces carry ~30 markers, mostly early, so of the `--max-checkpoints`
slots two go to `start`/`end`, half the rest to evenly spaced `fixed` points and the remainder to markers thinned
evenly along the trace (default 20 → about 9 fixed + 9 markers per trace).

`TRACE-CLASSIFICATION-SPEC.md` is the companion no-GPU analysis: label traces as re-solve-and-compare vs
judge-without-re-solving with an LLM judge, and cross-check the labels against the checkpoint trajectories.

**Caveat.** Closing `</think>` mid-trace is out of distribution: the model never saw a think block cut there, so a
forced answer need not equal what it would have answered. The `end` checkpoint is the check: the forced answer
with the whole trace should equal the run's `final_answer` (`sanity` in the analysis, flagged below 0.95). Only
trust the mid-trace readings if it passes.

**`checkpoints.csv`** (one row per checkpoint): `phase, question_index, checkpoint_idx, token_pos` (trace tokens before the
probe), `trace_tokens, frac` (`token_pos / trace_tokens`), `kind, marker, forced_answer` (argmax digit), `prob_forced`
(its softmax probability over the valid digits), `prob_1..prob_5` (5 is NaN in Phase 1), `prob_final` (probability of the
run's final answer), `matches_final`, plus `final_answer, is_correct, is_nota, gt_position` copied from `phases.csv`.
`checkpoints_params.json` records the settings and counts by kind.

**`checkpoints_analysis.json`**: `sanity`; `commitment` (earliest `frac` from which the forced answer stays equal to the final
one, by correctness in Phase 1 and by abstention in Phase 2); `before_backtrack` (`prob_forced` and its change at `marker`
vs `fixed` checkpoints, overall and by `frac` quartile); `trajectory_predicts_outcome` (AUROC of `min_prob, mean_prob,
prob_at_start, n_answer_changes, frac_committed` for `is_correct` / `is_nota`, with the post-trace `prob_final` as
baseline; raw AUROC, below 0.5 means the feature runs the other way); `abstention_onset` (Phase-2 abstentions: earliest `frac`
at which the forced answer is 5, and how many started on 1-4).

## Credit
Code and data adapted from Kumaran et al. (2026), OSF `resw4` (code MIT, data CC-BY-4.0); see `NOTICE.md`.

## What differs from the paper, and why
- **Decoding.** Paper: greedy, `max_tokens=3`. Thinking runs use Qwen's recommended sampling
  (T=0.6, top-p 0.95, top-k 20, fixed seed) and a 4096-token budget, because greedy decoding of
  long traces degenerates. `--thinking off` runs are greedy with a 4-token budget. `--temperature 0` forces greedy.
- **Where the logprobs come from.** Paper: the first output token after `Answer:`. Here: the first
  non-whitespace token after `</think>`. With thinking on, that distribution is expected to be
  near one-hot (the trace already commits to an answer), so `calibrated` in `analyze.py` will
  look uninformative in thinking runs. That is a finding, not a bug; the hybrid `off`/`on` pair
  is how to keep the paper's calibrated measure while varying thinking.
- **Verbal confidence.** Paper: 10-token budget, "NO REASONING". A thinking model reasons
  anyway; the class is parsed from the post-`</think>` content only, and the trace is saved.
- **Phase 4.** The prompt asks the model to rate its confidence "internally"; with thinking on
  that rating appears in the trace. Off by default (`--phases 0,1,2`); it is 11× the Phase-2 cost.
- **Unparseable answers** fall back to the argmax choice logprob exactly as in the paper;
  `parse_method` records `exact` / `regex` / `logprob`, and `truncated` flags traces that
  hit the budget without closing `</think>`. Exclude or report those rather than trusting them.

## Reading `analyze.py`
- `P1_auroc_correct`: `cal` (temperature-scaled, τ fit on Phase 0), `raw`, `verbal`, with a
  bootstrap CI on `cal − verbal`. Paper: cal > verbal. `frac_cal_above_0.99` shows the collapse.
  Thinking runs also report the trace's own signal (`neg_reasoning_tokens`, `neg_wait_count`).
- `P2_abstention`: Phase-2 abstention rate (paper: 27–82% across models; they dropped a model at 4%),
  abstention split by whether the Phase-1 answer was correct, and the nested logistic LR tests
  `verbal_beyond_cal` and `cal_beyond_verbal`. Paper: both significant, all p < 1e-6.
