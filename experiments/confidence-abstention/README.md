# Confidence → abstention under reasoning (Kumaran et al. 2026, reasoning-on extension)

Runs the paper's Phase 0/1/2/4 + verbal-confidence pipeline on Qwen3 and Olmo 3 checkpoints with thinking
on, via vLLM on Modal. Prompts, option shuffle, logprob readout and CSV schema are the paper's
(OSF `resw4`, `Confidence_Drives_Abstention_Notebook.ipynb`); `data/README-osf.md` is their README.

**Other experiments on the same checkpoints:** the runner is task-agnostic (`--task`, default `confidence_abstention`).
`TASKS.md` explains how to add a task (worked example: `--task short_answer`), what is reusable (the Olmo-3 32B checkpoint
list, the cached-weights volume, prefetch, batch launching, sweep statistics), the design rules and the costs.

## Files
- `abstention_modal.py` — the runner and tasks (`main`), the in-trace probe (`checkpoints`), the probe's
  start-checkpoint format check (`start_format`) and weight prefetch (`prefetch`); see docstring for all flags
- `TASKS.md` — adding a task; reusable infrastructure; checkpoint list and costs
- `data/short_answer_questions.csv` — items for the `short_answer` example task
- `analyze.py` — the paper's two headline tests on one run directory
- `analyze_checkpoints.py` — analysis of the in-trace confidence checkpoints (below)
- `analyze_sweep.py` — headline measures across checkpoints of one training run, with paired bootstrap CIs
  for adjacent checkpoints
- `sweep_batches.tsv`, `run_batch.sh` — the Olmo-32B PopQA sweep as launch lines; `./run_batch.sh <n>` launches batch `n`
  (client output in `logs/<tag>.log`)
- `analyze_length.py` — trace length vs correctness, abstention and difficulty across runs
  (writes `results/sweep/length_*.csv`)
- `reparse.py` — re-parses post-think answers of every `results/<tag>/phases.csv` with the current `parse_answer`
- `borrow_base.py` — gives a Phase-2-only run the Phase 0/1 rows and `verbal.csv` of its base run
- `trace_labels/` — LLM-judge labelling of traces (below; `TRACE-CLASSIFICATION-SPEC.md`)
- `steering_modal.py`, `build_vector.py`, `analyze_steering.py`, `test_steering_local.py` — confidence steering
  (below; `STEERING-SPEC.md`)
- `data/phase0_questions.csv`, `data/phase1_questions.csv` — the paper's calibration / main sets
  (SimpleQA 4-way MCQ with foils; `choice_1` is ground truth, shuffled at runtime with seed 1042)
- `data/popqa_phase{0,1}_questions.csv` — PopQA in the same format (`--questions popqa`), built by
  `build_popqa_mc.py`: 100 items per subject-popularity decile per file, two same-relation foils and one
  foil from another relation type; extra columns `s_pop`, `prop`

## Run
```
modal profile activate spar-f26          # shared workspace
pip install -r requirements.txt          # local scripts; GPU-side deps are in the Modal images

# 1. quick early signal (~1 H100, phases 0/1/2 + verbal on 200 questions)
modal run --detach abstention_modal.py::main --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native --n 200

# 2. paper-matched model, always thinks (80B bf16: --gpu H200:2 or H100:4)
modal run --detach abstention_modal.py::main --model Qwen/Qwen3-Next-80B-A3B-Thinking --thinking native --gpu H200:2

# 3. hybrid: same weights, thinking off (paper's regime) then on
modal run --detach abstention_modal.py::main --model Qwen/Qwen3-8B --thinking off --phases 0,1,2,4
modal run --detach abstention_modal.py::main --model Qwen/Qwen3-8B --thinking on  --phases 0,1,2,4

# fetch + analyse (analysis.json is also computed inside the container at the end of the run)
modal volume get abstention-results <tag> results/
python analyze.py results/<tag>
```
The file has several entrypoints, so Modal needs the explicit `::<name>` form. Every entrypoint spawns its GPU
function and returns; `--detach` keeps the app, and so the run, alive after the client exits.

Olmo 3 checkpoints (HF branches via `--revision`):
```
# base / pretraining checkpoint: no chat template, 4-shot completion, digit + cloze readouts, phases 0/1
modal run --detach abstention_modal.py::main --model allenai/Olmo-3-1125-32B --revision stage1-step62000 --thinking base --gpu H200
# Think checkpoint (the template opens <think>); Instruct checkpoints use --thinking off
modal run --detach abstention_modal.py::main --model allenai/Olmo-3.1-32B-Think --revision step_1100 --thinking native \
    --budget 8192 --max-model-len 12288 --gpu H200
```
`--thinking base` skips Phase 2 and the verbal pass unless `--base-phase2-verbal`; its exemplars are four Phase 0
items, one per answer position (none answers 5), and those items are left out of Phase 0 scoring.
`--k-samples N` adds N-1 sampled Phase 1 answers per question (`k_answers`, `consistency` = share of the modal answer).
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
modal run --detach abstention_modal.py::checkpoints --tag 80b-thinking --gpu H200:2   # needs <tag>/phases.csv in the volume
modal volume get abstention-results 80b-thinking results/
python analyze_checkpoints.py results/80b-thinking
```
Options: `--phases 1,2`, `--every 256`, `--max-checkpoints 20`, `--n 0` (0 = all questions), `--model ""` (override the model in `params.json`).
`modal run --detach abstention_modal.py::start_format --tag <tag> --gpu H200` (after `checkpoints`) writes
`<tag>/start_format.csv`: the digit mass of the `start` checkpoint and whether answer prefixes
(`--prefixes "Answer: |The answer is "`) restore a digit-first answer.

**What the probe does.** For every complete (non-truncated) trace in `phases.csv`, it cuts the trace after the
first k tokens, appends `\n</think>\n\n` and reads the digit logprobs of the next token ("if you had to answer
now"), exactly as the main run does. Checkpoints (`kind`): `start` (empty trace), `fixed` (every `--every` tokens),
`marker` (just before a backtrack word: wait, actually, hmm, hold on, let me recheck/reconsider/rethink/double-check/think again,
on second thought, "no,") and `end` (full trace). Traces carry ~30 markers, mostly early, so of the `--max-checkpoints`
slots two go to `start`/`end`, half the rest to evenly spaced `fixed` points and the remainder to markers thinned
evenly along the trace (default 20 → about 9 fixed + 9 markers per trace).

`TRACE-CLASSIFICATION-SPEC.md` is the companion no-GPU analysis: label traces as re-solve-and-compare vs
judge-without-re-solving, and cross-check the labels against the checkpoint trajectories (`trace_labels/` below).

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

## Trace labelling (`trace_labels/`, no GPU)
Labels each trace RESOLVE / JUDGE / MIXED / OTHER (`TRACE-CLASSIFICATION-SPEC.md`). The judges are Claude Code
subagents, run by `trace_labels/judge_workflow.js` through Claude Code's Workflow tool (not an API script): two Sonnet
judges per batch, one with `rubric_a.md` and one with `rubric_b.md`; batches of 5–10 traces, ids a judge drops are
re-done one trace per call; every item where the two kinds differ goes to a per-item Sonnet tiebreak with both judges'
labels shown. Pass `note` in the workflow args for runs that need an extra instruction (the reframed Phase-2 runs:
`judge_instructions_reframed.md`). `write_labels.py` turns the workflow's result into `labels_{a,b,tiebreak}_000.jsonl`.
```
python trace_labels/build_batches.py results/<tag> results/<tag>/trace_judging              # Phase 2 abstentions
python trace_labels/build_batches.py results/<tag> results/<tag>/trace_judging_answered \
    --answered 100,100 --batch-size 5                                                         # answered sample
# Workflow tool: trace_labels/judge_workflow.js with args {dir, tl, batches[, note]}; then
python trace_labels/write_labels.py <workflow output .json> results/<tag>/trace_judging
python trace_labels/summarize.py results/<tag>            # trace_judging/ -> trace_labels.csv, trace_labels_summary.json
python trace_labels/summarize_answered.py results/<tag>   # trace_judging_answered/, reweighted to the answered population
python trace_labels/crosscheck.py results/<tag>           # labels vs checkpoints.csv trajectories
```
`build_sample.py` draws sets A–D of the spec from `results/80b-thinking`; `build_reframed.py` / `summarize_reframed.py`
do the same for the reframed Phase-2 runs (`80b-thinking-p2skip`, `-p2neutral`).

## Confidence steering (`STEERING-SPEC.md`)
`steering_modal.py` runs the paper's steering recipe with HF transformers (own Modal image), each stage a separate
`modal run --detach steering_modal.py::<stage>`: `notrace` (no-trace runs, residuals, then `build_vector.py`), `gate`
(no-trace layer × alpha sweep with controls), `pretrace` (thinking checkpoints: pre-trace residuals labelled by a harness
run's k-sample consistency, then `build_vector.py`), `reason` (reasoning mode at one layer: alphas × positions). The
module docstring has the commands and the layout of `results/steer-<model>-<tag>/`.
```
modal volume get abstention-results steer-<model>-<tag> results/
python analyze_steering.py results/steer-<model>-<tag> [--harness results/<harness tag>]   # writes STEERING-RESULTS.md
python test_steering_local.py [e2e]     # CPU check on Qwen3-0.6B; e2e runs every stage end to end (~8 min)
```
The local steering scripts need `torch` and `transformers` (`requirements.txt`).

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
- `cloze` (base-mode runs): the same measures for the option-likelihood readout (`cloze_mean_k`, its own τ),
  plus `sum_raw` (unnormalised sum) and agreement with the digit answer. `phase<k>.mean_digit_mass` is the
  probability on the valid digits at the readout token; below ~0.95 the digit readout is not in MC format.
