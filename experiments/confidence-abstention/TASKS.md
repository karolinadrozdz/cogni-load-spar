# Running other experiments on the Olmo-3 32B checkpoints

`abstention_modal.py` is a model-agnostic runner, vLLM on Modal, plus pluggable tasks. `confidence_abstention`
(the Kumaran et al. paradigm, the default) is one task. `short_answer` (`tasks/short_answer.py`) is a small worked
example. A new experiment on emergent capabilities across training is a new task: one file in `tasks/` and one line in
the registry. It reuses everything below for free, and outputs that need heavier scoring get a grader (§ "Grading").

## What is reusable

- **Checkpoints** (§ "Checkpoint list"): 29 Olmo-3/3.1 32B checkpoints from pretraining step 1000 to the final RL
  step. Branch names and weight identities are checked.
- **Cached weights.** All of them are in the Modal volume `abstention-hf-cache` (≈ 2 TB, HF cache layout). Every
  function mounts it at `/cache`, so a run on a cached checkpoint starts without a download. **Do not delete it, or
  any repo inside it.** Re-downloading costs hours and GPU money, and other experiments use the same cache.
- **Prefetch.** This downloads new checkpoints on CPU containers into the same cache. A GPU run that downloads bills
  the GPU for the download:
  ```
  modal run --detach abstention_modal.py::prefetch --specs "allenai/Olmo-3-1125-32B@stage1-step19000,allenai/Olmo-3-32B-Think"
  ```
- **Runner** (`run` + `Context`). It handles:
  - loading the checkpoint at `--revision` / `--dtype`
  - the four `--thinking` modes
  - chat-template rendering
  - generation with a token budget
  - splitting off the think block, string-matched, because Olmo's `</think>` is three ordinary tokens
  - the truncation flag
  - top-20 logprobs of the first content token
  - prompt-logprob scoring
  - writing `<tag>/` with `params.json` last, so its presence means the run is complete
- **Launching.** `main` spawns rather than waiting, so a client crash does not cancel the run. `--detach` keeps the
  app alive. `run_batch.sh <n>` launches the batch-`n` lines of `sweep_batches.tsv` concurrently (nohup, logs in
  `logs/<tag>.log`). `./handoff.sh <tag>` waits for `params.json` and then fetches. Launch about 5 at a time: there
  is a GPU limit, and HF throttles parallel downloads.
- **Retries, and the "bad marshal data" flake.** Containers sometimes read a corrupt site-packages file at startup
  ("bad marshal data", "source code string cannot contain null bytes"). `run` retries twice, but a retry can land in
  the same container and fail the same way. If a run fails at startup like this, relaunch the same command. That
  makes a fresh app and a fresh container.
- **Do not touch the `image = (...)` block.** The cached image (vllm 0.32.0) is the only known-good one. Changing
  the image spec, even by one byte, triggers a rebuild. A rebuild pulls nvidia-cutlass-dsl 4.8.1, whose bytecode
  fails to load. `run_image` adds `tasks/` and `analyze.py` to it as mounts attached at container start
  (`add_local_python_source`, `copy=False`): same image id, no build. A task that needs a pip package the image
  lacks cannot get it here; score its outputs with a grader instead.
- **`analyze_sweep.py`.** It gives per-checkpoint measures with bootstrap CIs, paired bootstrap CIs for adjacent
  checkpoints (same items resampled in both, so item difficulty cancels), a check that both runs used the same
  items, and gate columns. A task plugs in its own measures (see "Sweep measures").
- **In-trace probe** (`::checkpoints`). It force-closes the think block at points along each trace and reads the
  answer distribution. It works only for digit-answer tasks (see "Making a task probe-able").

## The task interface

A task is a module in `tasks/` (`tasks/my_task.py`) plus one line in `tasks.MODULES` (`"my_task": "tasks.my_task"`).
`abstention_modal.py` does not change. `run` executes in `run_image`, which mounts `tasks/` and `analyze.py`; data
files go in `data/`, which the image mounts at `/data` at container start, so new CSVs and task files rebuild nothing.
`modal run abstention_modal.py::check_tasks` (CPU, seconds) imports every registered task in a run container.

`main()` imports the module locally to call `config`, so keep imports needed only in the container (vLLM, heavy
scorers) inside `run` / `analyze`. Shared helpers come from `abstention_modal` (`default_tag`, `parse_task_opts`, the
MC helpers below).

```python
# tasks/my_task.py
from abstention_modal import default_tag, parse_task_opts

class MyTask:
    name = "my_task"                 # --task my_task; also the "task" key in cfg / params.json
    options = {"shots": 4}           # extra settings, set with --task-opts "shots=2" (cast to the default's type)

    def config(self, f) -> dict:     # local, at launch: f = all main() flags + tp; validate, return the cfg dict
    def run(self, ctx) -> (files, params):
        # files: {"answers.csv": rows, ...} written to <tag>/; params: extra keys for params.json
    def analyze(self, cfg, out) -> dict | None:   # in the container; a dict is written to analysis.json

TASK = MyTask()
# sweep hooks for analyze_sweep.py (§ "Sweep measures"): items, measures, gates, show
```

`ctx` (`Context`) gives a task:

| helper | what it does |
|---|---|
| `ctx.cfg`, `ctx.tok`, `ctx.llm`, `ctx.out` | config, tokenizer, the vLLM engine, the output dir |
| `ctx.base`, `ctx.thinking_active` | `--thinking base`; whether the model thinks (native/on) |
| `ctx.render(prompt, prefix="")` | chat-template text for the `--thinking` mode. `prompt` is a user message, or a list of `{"role": "system"\|"user"\|"assistant", "content"}` messages ending in a user turn (multi-turn paradigms). `prefix` is appended after the generation prompt: a prefill of the assistant turn. Base mode: `prompt + prefix`, strings only |
| `ctx.generate(prompts, max_tokens, prefix="", **overrides)` | renders and generates. Per prompt, a list of samples (one unless `n=`), each a dict: `content` (after `</think>`, stripped), `reasoning`, `reasoning_tokens`, `truncated`, `generated_tokens`, `finish_reason`, `top` (decoded token → logprob, top-20 at the first non-blank content token), `wait_count`. `content` and `reasoning` exclude the prefix. Overrides go to `SamplingParams` (`stop`, `n`, `seed`, `temperature`, ...) |
| `ctx.max_tokens(k)` | `--budget` when the model thinks, else `k` (the task's own answer length) |
| `ctx.continuation_logprobs([(context, continuation), ...])` | per-token logprobs of each continuation given raw-text context, from prompt logprobs (what the cloze readout uses) |
| `ctx.save(name, rows)` | write `<tag>/<name>` now and commit, for partial results in long runs |
| `ctx.sampling` | default sampling: greedy unless thinking (T 0.6, top-p 0.95, top-k 20); `--temperature` overrides |

**Messages and prefill.**
- A **system message replaces the template's default system prompt.** Olmo's templates insert `You are a helpful AI
  assistant.` when there is none, and Olmo Think appends a no-functions sentence to a system message you pass. Prompts
  with a system message are therefore not comparable to runs without one; check `rendered_example`.
- Earlier assistant turns are rendered by the template: Qwen3 drops their `<think>` content, Olmo keeps the text as
  given. Put a redacted or fixed answer in `content`.
- **Think block and prefix.** A prefix containing `</think>` (e.g. `"<think>\n\n</think>\n\n"` or a prefilled trace
  ending in `</think>`) closes the block, so the whole generation is content: `reasoning` is empty, and a generation
  that hits the budget is not flagged `truncated` (`finish_reason` still says `length`). Any other prefix leaves the
  template's think handling unchanged: with a thinking template, a prefix of answer text lands *inside* the open
  think block. Native templates open `<think>` in the generation prompt, so the prefix that closes it is
  `"\n</think>\n\n"`-style; check `ctx.render(p, prefix)` once per model family.

**Multiple-choice helpers** (`abstention_modal.py`, what `confidence_abstention` itself uses):
- `mc_items(path)`: the question file (`question`, `choice_1` = answer, `choice_2..4`) and the fixed arrangement of
  every row (seed 1042, over the whole file, so `--n` does not change it).
- `arranged_choices(df, arr, i)`: options in arranged order, the answer's position (1–4), the order.
- `mc_item_fields(i, q, c, gt, order)`: the `phases.csv` item columns.
- `digit_readout(g, valid, gt)`: the `phases.csv` answer columns for one generation (`final_answer`, `is_correct`,
  `is_nota`, digit logprobs and softmax confidences, `parse_method`) and the digit logprobs.
- `digit_entropy_second(lps)`: entropy of the softmax over the valid digits, and the runner-up digit and its
  probability (`second_choice`, `confidence_second`), e.g. for a second-chance paradigm.

`main()` checks the task-independent flags: `--model --revision --thinking --n --budget --max-model-len
--temperature --seed --dtype --gpu --tag`. The confidence-only flags (`--phases --questions --phase2-prompt
--k-samples --base-phase2-verbal`) are ignored by other tasks. If `--tag` is not given, it defaults to
`<model>[_<rev>]_<thinking>_<task>_n<N>_<time>`.

**Checking a change to the shared code.** `python test_harness_local.py` (CPU, seconds, no vLLM) runs
`confidence_abstention` and `short_answer` in every `--thinking` mode on a fake engine, with this tree and with
`abstention_modal.py` at the reference revision `REF`, and asserts identical CSVs, params and analysis.
`python test_harness_local.py render` re-renders the `rendered_example` of every local `results/*/params.json` with
each run's tokenizer.

## Worked example: `short_answer`

The data is `data/short_answer_questions.csv`, with columns `split` (`shot` / `test`), `question` and `answer`
(aliases separated by `|`). The task is free-form short answers scored by exact match after normalisation
(lowercase, punctuation and articles removed). It differs from the MC task on every axis a new task has to decide:
- **Items:** the first `--n` test rows in file order. This is deterministic, so every checkpoint gets the same items.
  If a task samples or shuffles, seed it from a constant, not from `--seed`.
- **Base checkpoints:** `options["shots"]` exemplars (`Question: …\nAnswer: <first alias>\n\n`) before the item in
  completion format, `stop=["\n"]`, 16 tokens.
- **Chat checkpoints:** one instruction plus the item, rendered by `ctx.render`. Thinking modes get `--budget`.
- **Output:** `answers.csv`, one row per item: `question_index, question, answer, prompt, generated_text, is_correct,
  answer_words`, plus the runner fields `thinking, reasoning, reasoning_tokens, truncated, generated_tokens,
  finish_reason`. The analysis keys on `question_index` / `question` / `is_correct`.
- **`analysis.json`:** accuracy, the share of answers with ≤ 5 words (`short_format`), and the truncation count.
- **Sweep measures** (the hooks at the end of `tasks/short_answer.py`): `accuracy`, `short_format`,
  `auroc_neg_trace_len` and `truncated`. The gate is `gate_format` = `short_format ≥ 0.9` and truncation `< 5%`.
- **Grading:** `graders/exact_match.py` re-grades `answers.csv` after the run (§ "Grading").

Run it on one checkpoint, then across checkpoints:
```
modal profile activate spar-f26
modal run --detach abstention_modal.py::main --task short_answer --model allenai/Olmo-3-1125-32B \
    --revision stage1-step656000 --thinking base --gpu H200 --tag sa-olmo32b-base-s1-step656000
modal run --detach abstention_modal.py::main --task short_answer --model allenai/Olmo-3-32B-Think \
    --thinking native --budget 8192 --max-model-len 12288 --gpu H200 --tag sa-olmo32b-think-rl-main
./handoff.sh sa-olmo32b-base-s1-step656000          # fetches results/<tag>/ (analysis.json was made in the container)
python analyze_sweep.py results/sa-olmo32b-base-s1-step656000 results/sa-olmo32b-think-rl-main --out results/sweep/sa_olmo32b
modal run --detach grading_modal.py::grade --tag sa-olmo32b-think-rl-main --grader exact_match   # optional re-grade
```
Fewer exemplars: `--task-opts shots=2`.

## Sweep measures

`analyze_sweep.py` reads the task from each run's `params.json` (no `task` key = `confidence_abstention`). It refuses
to mix tasks. The hooks come from `SWEEP_TASKS[name]` (`confidence_abstention`) or, for every other task, the
module-level names of its `tasks/` module:
- `items(run_dir)`: one row per item, indexed by item id, with a `question` column. The paired diffs check that it
  matches across runs.
- `measures(items)`: a dict of floats. It is recomputed on each bootstrap resample, so keep it vectorised and
  NaN-safe (return NaN for a measure the run lacks).
- `gates(measures, ci_lo)`: a dict of `gate_*` booleans, or `None` where the tier was not measured.
- `show`: the columns printed.

The task module is also imported in the run container and by graders, which have no `analyze_sweep.py`: import
`analyze_sweep.auroc` inside `measures`, not at the top.

Outputs: `<out>.csv` (per run, each measure with `_lo` / `_hi`) and `<out>_diffs.csv` (later − earlier, paired CI).

## Grading

Scoring that needs packages the frozen image lacks, or that should be redone without the GPU (free-form answers
checked by a model, unit tests in a sandbox), runs after the run, on CPU, in the grader's own image:
```
modal run --detach grading_modal.py::grade --tag <tag> --grader exact_match [--opts "k=v,..."]
modal volume get abstention-results <tag> results/      # now with graded_<grader>.csv, grade_params_<grader>.json
```
A grader is a file in `graders/` plus one line in `graders.MODULES`. It defines:
- `image`: its own `modal.Image` (never `abstention_modal.image`). Add `.add_local_python_source("tasks",
  "abstention_modal")` to reuse a task's helpers; the `graders/` package itself is mounted automatically.
- `options`: defaults for `--opts`, cast like `--task-opts`.
- `grade(out, params, opts) -> (rows, extra)`: `out` is the run directory in the volume, `params` its `params.json`.
  `rows` become `<tag>/graded_<name>.csv`; `extra` (a dict of summary numbers) goes into
  `<tag>/grade_params_<name>.json` with the tag, grader, task, opts and row count. The JSON is written last.
- one Modal function, `grade_<name>(tag: str, opts: dict) -> str`, declared with `@grader_function(image)` and
  returning `grade_run("<name>", tag, grade, opts)`. `graders/exact_match.py` is the template.

`grading_modal.py` imports every grader module to register its function, so keep heavy imports inside `grade`.
`modal run` builds the image of every registered grader (each once, then cached), so a grader whose image fails to
build blocks every grading run until it is fixed. `grader_function(image, timeout=3600, **kw)` passes other options
(`cpu`, `memory`, `gpu`, `secrets`) to `app.function`. A grader keys its rows
on `question_index`, so `graded_<name>.csv` joins back to the task's CSV. Each grader writes its own files, so
several graders can grade the same tag.

## Making a task probe-able

`::checkpoints --tag <tag>` reads `<tag>/params.json` (`model`, `revision`, `thinking`, `max_model_len`, `budget`,
`seed`) and `<tag>/phases.csv`. It needs these columns:
- `phase`
- `question_index`
- `prompt`: the **unrendered** user message; the probe renders it with the chat template itself
- `reasoning`, `truncated`
- `final_answer`: a digit
- `is_correct`, `is_nota`, `gt_position`

It reads the next-token distribution over the digits 1–4 (plus 5 in phase 2) after a forced `</think>`. So only tasks
whose answer is a single digit, in a thinking mode (native/on), can be probed. A free-form task like `short_answer`
cannot: the probe stops with an assertion, because there is no `phases.csv`. To probe a new MC task, write
`phases.csv` with those columns, and those phase numbers if you want the phase-2 "5" option.

## Design rules

- **Same items at every checkpoint.** Paired statistics need it, and `analyze_sweep.py` asserts it. Item order and
  any shuffle must not depend on the model or the tag.
- **Base (pretraining) checkpoints need completion prompts with few-shot exemplars** (`--thinking base`): there is
  no chat template, and instructions are not followed. Leave exemplars out of scoring. Spread them over answer
  positions/types, so the shots do not teach an answer prior. Stop at `"\n"`.
- **Gate measures where the format is not learned.** Early checkpoints often do not answer in the task's format
  (digit mass < 0.95, rambling answers). Their accuracy and calibration are then undefined, not zero. Give each tier
  a gate, and report a measure only where its gate passes. A curve that "emerges" exactly where the gate turns on is
  a format artefact unless a format-free readout shows it too. For MC tasks, that is the cloze log-likelihood of each
  option, via `ctx.continuation_logprobs`.
- **Bridge runs for one format across training.** Post-trained checkpoints change the wrapper (chat template,
  thinking). Also run them in base mode (`--thinking base`, same items and prompts as pretraining). That gives one
  curve with a single format from step 1000 to the final RL step, and a paired wrapper-effect comparison at a fixed
  checkpoint.
- **Thinking runs:** check `truncated` < 5% per phase (raise `--budget` / `--max-model-len` if not). Check that
  `reasoning_tokens` > 0 (else `</think>` parsing failed). Check `params.json` → `rendered_example`; Olmo's template
  inserts a system prompt.
- **k-sample runs must store per-sample `reasoning_tokens` and `final_answer`.**

## Checkpoint list (Olmo-3 / 3.1 32B)

Branches were checked against the HF refs API on 2026-10-25.

| stage | repo @ revision | mode |
|---|---|---|
| pretraining ×12 | `allenai/Olmo-3-1125-32B@stage1-step{1000, 2000, 5000, 8000, 11000, 19000, 34000, 62000, 112000, 202000, 364000, 656000}` (8.4B … 5.5T tokens) | base |
| midtraining | `allenai/Olmo-3-1125-32B@stage2-ingredient1+2-step23842` (a merge) | base |
| final base | `allenai/Olmo-3-1125-32B@main` (long context; a merge, ≠ `stage3-step11921`) | base |
| Instruct SFT / DPO / RL | `allenai/Olmo-3.1-32B-Instruct-SFT`, `-DPO`, `allenai/Olmo-3.1-32B-Instruct` | off |
| Think SFT ×3 | `allenai/Olmo-3-32B-Think-SFT@1e-4-step2000`, `@1e-4-step6000`, `@main` | native |
| Think DPO | `allenai/Olmo-3-32B-Think-DPO@main` | native |
| Think RL 3.0 | `allenai/Olmo-3-32B-Think@step_050`, `@step_250`, `@step_500`, `@main` (= `step_750`) | native |
| Think RL 3.1 | `allenai/Olmo-3.1-32B-Think@step_1100`, `@step_1500`, `@step_1900`, `@main` (= `step_2300`) | native |

Notes:
- `Olmo-3-32B-Think-SFT@main` is a 1:1 linear merge of the `1e-4-step10790` and `5e-5-step10790` runs, not a step
  of either.
- **3.0 and 3.1 RL are one trajectory.** `Olmo-3.1-32B-Think@step_0050/0500/0750` are byte-identical to
  `Olmo-3-32B-Think@step_050/500/750`. Do not run them twice.
- **`Olmo-3.1-32B-Instruct-SFT` and `-DPO` are stored in float32** (129 GB). Run them with `--dtype bfloat16`;
  vLLM's `auto` would load them as float16.
- Stage 1–2 configs have `max_position_embeddings` 8192, so keep `--max-model-len 8192` for base runs. Think runs
  use `--budget 8192 --max-model-len 12288`.
- 32B in bf16 is 64.5 GB: use `--gpu H200`. 1×H100 leaves too little KV cache, and 2×H200 costs more per token.
- `sweep_batches.tsv` has the full PopQA pass as runnable lines (`./run_batch.sh <n>`). Copy it and edit `--task` /
  `--tag` for a new task.

## Cost (Modal H200 ≈ $4.54/h; startup ≈ 7 min ≈ $0.55 of every run)

| unit | ≈ $ per checkpoint |
|---|---|
| base or `--thinking off` mode, n = 1,000, short answers | 0.70 |
| reasoning mode (native), budget 8192 | 1.5–4 (scales with trace length) |
| one pass over the checkpoint list (≈ 28–29 runs, mixed modes) | 40–80 |

Before a production pass, run a cheap viability probe: n = 200 on 2–3 checkpoints, about $1–3 in total.
