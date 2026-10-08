# cogni-load-spar

Does the J-space hold the task state a model is maintaining, or only what is
recent or about to be output? If it does, can J-space content serve as a proxy
for cognitive load?

Every experiment builds the same seeded keep-track streams (a list of words
from eight categories, some of them tracked), reads them through the Jacobian
lens and a logit-lens control, and writes parquet. Rows join across
experiments on `(stream_id, c_t)`, where `c_t` is the number of tracked
categories.

Why each measurement is the way it is: [`DECISIONS.md`](DECISIONS.md). Results
so far: [`FINDINGS.md`](FINDINGS.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[local]"   # torch, pandas, modal, pytest, jinja2; no transformers or jlens
modal setup                 # once
pytest -q                   # no GPU needed
```

The GPU stack (`jlens`, `transformers`) is the `[gpu]` extra; `modal_app.py`
installs it into the image. On a CUDA box, `pip install -e ".[gpu]"` and run
the `python -m cogniload.cli` commands directly. macOS cannot run a stage.

## Change the model

`configs/models.yaml` is the only file that names a model. Add an alias block
(`hf_id`, `hf_revision` as a commit sha, `n_layers`, `d_model`, `lens_file`,
`enable_thinking: false`), then select it with `--model <alias>` or `model:` in
`configs/experiments.yaml`. Run `find_band` for the new alias first.

## Change the task

- **Streams:** the `stream:` block of `configs/experiments.yaml`, and the word
  pools in `exemplars.py`. All experiments share them.
- **Prompts:** `prompts.py` holds the shared stream instruction and the
  single question; each experiment's own prompts are constants at the top of
  its module. Every template used is written verbatim to the run manifest.
- **A new experiment:** a module with `run_stream(ctx, stream) -> (rows,
  summary)`, `run` and `report`, a config block, and its name in
  `cli.EXPERIMENTS`. The shared blocks are in `experiment.py`.

## Run

Every stage takes `--model dev|prod`, `--limit N` (streams per `c_t`, for a
smoke run; writes `_limit` files) and `--force` (redo shards that exist;
otherwise a stage resumes). On Modal the GPU follows the alias (`dev` L4,
`prod` H100, override with `--gpu`); `report` gets none. Run Modal commands
from the repo root.

### find_band (once per model)

```bash
python -m cogniload.cli find_band --model dev
modal run modal_app.py --stage find_band --model dev
```

Writes `band.json`: the layer band where the lens reads two-hop bridge
entities. Every experiment reads in this band.

### single_cue

Keep-track stream, then "What was the most recent {category}?". Read at the
comma inside the stream (`in_stream`), the final `.` (`stream_end`) and the
`Answer:` prefill (`answer`).

```bash
python -m cogniload.cli single_cue --model dev --limit 20
python -m cogniload.cli report single_cue --model dev
modal run modal_app.py --stage single_cue --model dev --limit 20
modal run modal_app.py --stage report --experiment single_cue --model dev
```

Read `FLOOR CHECK` first: if the queried target is present at the answer in
under 50% of streams on both rank columns, every number below is a null that
looks like a finding. Then `CAPACITY` (`mean` = non-queried targets present
per stream, at `readout_at = in_stream`, across `c_t`: a rise then a plateau is
a ceiling, flat at ~1 is the deflationary result), `SELECTIVITY`
(`target_non_queried` against `untracked_final` and `absent`), and `HEADLINE`
(every group, pooled over `c_t`; compare each to `absent`).

### forced_demand

The same stream, then "List the most recent word for each tracked category, in
alphabetical order": to emit the first word the model needs every target. Read
at the `Answer:` prefill.

```bash
python -m cogniload.cli forced_demand --model dev --limit 5
python -m cogniload.cli report forced_demand --model dev
modal run modal_app.py --stage forced_demand --model dev --limit 5
modal run modal_app.py --stage report --experiment forced_demand --model dev
```

Read the first table (`rank_wordlike`, J-lens): `targets_mean` (targets present
per stream), `t1`…`t6` (the target that is alphabetically 1st…6th, the order of
emission), `replaced`, `label_tracked`, `label_untracked`, against
`floor_word` and `floor_label`; behaviour in `q1_top1` (first word) and
`list_correct`. The single_cue table for the same streams follows it.

### retro_cue

Ask for tracked category A, put the model's one-token answer back as an
assistant turn, then ask for B. Read at both answers (`pos1`, `pos2`).

```bash
python -m cogniload.cli retro_cue --model dev --limit 5
python -m cogniload.cli report retro_cue --model dev
modal run modal_app.py --stage retro_cue --model dev --limit 5
modal run modal_app.py --stage report --experiment retro_cue --model dev
```

Read `target_A` and `target_B` at `position` 1 and 2, against `floor_word`:
does A's word leave when the cue moves to B?

### derived_state

The stream cut at a seeded point, then "How many {category} words have appeared
so far?". Arm `derived` shows the plain stream, so the counts exist only in the
model; arm `copyable` writes each tracked word's running count after it. Read at
every in-stream comma and at the answer, for the digits 0–9 and the words
zero–nine.

```bash
python -m cogniload.cli derived_state --arm derived --model dev --limit 5
python -m cogniload.cli derived_state --arm copyable --model dev --limit 5
python -m cogniload.cli report derived_state --model dev
modal run modal_app.py --stage derived_state --arm derived --model dev --limit 5
modal run modal_app.py --stage derived_state --arm copyable --model dev --limit 5
modal run modal_app.py --stage report --experiment derived_state --model dev
```

One table per arm. `ans_*` columns are at the answer position, `in_*` pooled
over in-stream commas: `ans_queried`, `ans_tracked`, `in_tracked` (split into
`in_tracked_just`, the word just read, and `in_tracked_other`), `*_untracked`,
`*_stale` (the superseded count). Compare them to `*_floor4` (4 is in range but
never a count) and `*_floor` (5–9). Behaviour is `top1`; `collision_rate` is
how often two tracked counts share a value.

### state_tracking

Not a keep-track stream. Three people each hold an object; each update names
its person by the object they hold now ("The person holding the key swaps it
for the pen."), so the answer to "What is {name} holding?" needs every update
in order. `k` updates hit the person asked about, `h` hit others; cells are
`k` 1-5 by `h` in {1, k, 2k}, 30 items each, the same items in both arms. Arm
`copyable` writes the full state after every update. Read at the answer and
at the period ending each update. `--limit N` is items per cell.

```bash
python -m cogniload.cli state_tracking --arm derived --model dev --limit 5
python -m cogniload.cli state_tracking --arm copyable --model dev --limit 5
python -m cogniload.cli report state_tracking --model dev
modal run modal_app.py --stage state_tracking --arm derived --model dev --limit 5
modal run modal_app.py --stage state_tracking --arm copyable --model dev --limit 5
modal run modal_app.py --stage report --experiment state_tracking --model dev
```

Read T1 first: `top1` per arm, `k` and `h` (derived should fall with `k`,
copyable stay at ceiling), `top1_hay_last` against `top1_needle_last` (if only
the latter is high, the model answers the last new object), and the error
shares (`poi_stale_j`, `other_*`, `out_of_prompt`). The `argmax` lines below
it must be objects, not ` the`. T2 is the answer position per arm and `k`:
`target`, `chain_1`...`chain_k` (the person's earlier objects, `j` steps
back), `other_current`, `other_stale`, `hay_matched` (recency control), all
against `floor`. In the copyable arm `target` must be well above `floor`, or
the readout fails its positive control. T3 (derived, `k` 3-4, correct items):
median first legible layer per chain object and the share in chain order, vs
`baseline`. T4 (derived): wrong items beside correct ones.

### hay_factorial

state_tracking's derived arm with the hays untied from the chain: `k` in {1,
3, 5} by `h` in {0, 2, 4, 8, 12}, 60 items per cell. Within each cell,
`tail_hays` (hays after the asked-about person's last needle) splits equally
over {0, 2, 4} (those ≤ `h`); `pre_hays` is the rest. Both are columns of the
summary. If the object pool is too small for `h = 12`, `h` is capped at 10 and
the manifest says so (`h_capped`). No `report` yet; the summaries carry
`correct`, `k`, `h`, `pre_hays`, `tail_hays`.

```bash
python -m cogniload.cli hay_factorial --model dev --limit 5
modal run modal_app.py --stage hay_factorial --model dev --limit 5
```

### hay_type

`k = 3`, `h = 6`, `tail_hays = 2`, 60 items, the same items in every arm; only
the hay sentences differ. `same`: "The person holding the lamp swaps it for
the map."; `named`: "Ben swaps the lamp for the map."; `reworded`: "Whoever has
the lamp trades it for the map.". Needles always use the first form. The
reference is hay_factorial's `k = 3, h = 0` cell. The smoke prints one prompt
per arm.

```bash
python -m cogniload.cli hay_type --arm same --model dev --limit 5
modal run modal_app.py --stage hay_type --arm same --model dev --limit 5
modal run modal_app.py --stage hay_type --arm named --model dev --limit 5
modal run modal_app.py --stage hay_type --arm reworded --model dev --limit 5
```

## Outputs

Under `results/<alias>/`, or the `cogniload-results` Modal volume at
`<alias>/` (fetch with `modal volume get cogniload-results prod ./results/prod`):

| file | contents |
|---|---|
| `band.json` | the band, and the per-layer medians that chose it |
| `<experiment>_ct{c}[_limit].parquet` | one row per word × layer × position × lens: `rank`, `rank_wordlike`, `role`, `readout_at`, `in_band` |
| `<experiment>_summary_ct{c}[_limit].parquet` | one row per stream: behaviour and read positions |
| `<experiment>_manifest.json` | model spec, prompt templates, exemplars, config |

derived_state names carry the arm: `derived_state_copyable_ct2.parquet`.
state_tracking shards are per cell, with the arm: `state_tracking_derived_k3_h6.parquet`.
hay_factorial shards are per cell (`hay_factorial_k3_h4.parquet`); hay_type's carry
the arm (`hay_type_named_k3_h6.parquet`).

Presence is band-min rank ≤ `readout.primary_k` (25). `rank` is over the full
vocabulary; `rank_wordlike` over word-like tokens only, which is what
Neuronpedia shows.

## Layout

```
configs/models.yaml       model + lens registry
configs/experiments.yaml  every experiment's settings; names an alias
modal_app.py              runs one CLI stage on Modal
src/cogniload/
  cli.py                  stage runner
  experiment.py           shared blocks: layers, streams, words read, two-lens readout,
                          scoring, the per-c_t shard loop, loading results
  find_band.py            band discovery
  single_cue.py           single_cue: prompt, run_stream, tables
  forced_demand.py        forced_demand: same shape
  retro_cue.py            retro_cue: same shape
  derived_state.py        derived_state: same shape, plus the arm
  state_tracking.py       state_tracking: its own items (no streams), the arm, tables
  hay_load.py             hay_factorial and hay_type: state_tracking items with the hays varied
  prompts.py              shared templates; chat rendering with the thinking assertion
  readout.py              lens ranks per word, layer and position
  stimuli.py              seeded stream generation (pure)
  exemplars.py            category word pools, filtered to single tokens
  registry.py, bands.py   alias -> spec; band.json
```
