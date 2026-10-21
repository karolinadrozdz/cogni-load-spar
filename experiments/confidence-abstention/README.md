# Confidence → abstention under reasoning (Kumaran et al. 2026, reasoning-on extension)

Runs the paper's Phase 0/1/2/4 + verbal-confidence pipeline on Qwen3 checkpoints with thinking
on, via vLLM on Modal. Prompts, option shuffle, logprob readout and CSV schema are the paper's
(OSF `resw4`, `Confidence_Drives_Abstention_Notebook.ipynb`); `data/README-osf.md` is their README.

## Files
- `abstention_modal.py` — the experiment (one Modal function; see docstring for all flags)
- `analyze.py` — the paper's two headline tests on one run directory
- `data/phase0_questions.csv`, `data/phase1_questions.csv` — the paper's calibration / main sets
  (SimpleQA 4-way MCQ with foils; `choice_1` is ground truth, shuffled at runtime with seed 1042)

## Run
```
modal profile activate spar-f26          # shared workspace
pip install modal                        # local only; everything else is in the image

# 1. quick early signal (~1 H100, phases 0/1/2 + verbal on 200 questions)
modal run abstention_modal.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native --n 200

# 2. paper-matched model, always thinks (80B bf16: --gpu H200:2 or H100:4)
modal run abstention_modal.py --model Qwen/Qwen3-Next-80B-A3B-Thinking --thinking native --gpu H200:2

# 3. hybrid: same weights, thinking off (paper's regime) then on
modal run abstention_modal.py --model Qwen/Qwen3-8B --thinking off --phases 0,1,2,4
modal run abstention_modal.py --model Qwen/Qwen3-8B --thinking on  --phases 0,1,2,4

# fetch + analyse (analysis.json is also computed inside the container at the end of the run)
modal volume get abstention-results <tag> results/
python analyze.py results/<tag>
```
`<tag>` is printed at the end of a run (`<model>_<thinking>_n<N>_<timestamp>`); `--tag` sets it.
Set `ABSTENTION_RESULTS_VOLUME=<name>` to write to your own volume in the shared workspace.

## Overnight run + hand-off
```
# launcher: --detach keeps the job running after this terminal/laptop goes away; fix the tag
modal run --detach abstention_modal.py --model Qwen/Qwen3-30B-A3B-Thinking-2507 --thinking native \
    --n 1000 --tag overnight-30b-thinking
modal app logs confidence-abstention      # progress, any time, from any machine in the workspace

# receiver (needs a Modal token for spar-f26 and `pip install -r requirements.txt`):
./handoff.sh overnight-30b-thinking       # waits for params.json, pulls results/<tag>/, runs analyze.py
```
`params.json` is written last, so its presence in the volume means the run is complete.
Everything the analysis needs is in `results/<tag>/` (`phases.csv`, `verbal.csv`, `analysis.json`,
`params.json`); the raw reasoning traces are in the `reasoning` column of both CSVs.

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
