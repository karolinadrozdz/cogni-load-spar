# cogni-load-spar — `metacog` branch

Does the J-space encode the model's confidence: the boundary between what it
knows and what it does not?

This branch holds only the shared infrastructure for reading a prompt through
the Jacobian lens and a logit-lens control. The keep-track experiments it was
cut from (`single_cue`, `forced_demand`, `retro_cue`, `derived_state`), their
specs and their findings are on `exp/aveizi/task-finding`.

Why each measurement is the way it is: [`DECISIONS.md`](DECISIONS.md).

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

## Run

### find_band (once per model)

```bash
python -m cogniload.cli find_band --model dev
modal run modal_app.py --stage find_band --model dev
```

Writes `results/<alias>/band.json`: the layer band where the lens reads two-hop
bridge entities. Every experiment reads in this band. On Modal the GPU follows
the alias (`dev` L4, `prod` H100, override with `--gpu`); run from the repo root.

### confidence

"What is the capital of {entity}? Answer in one word." for 50 real and 50
invented countries (`stimuli/capitals.csv`), prefilled with `Answer:` so the
next token is the answer. Read at that position for uncertainty words, words
for something not existing, and neutral control words.

```bash
python stimuli/capitals.py                                # only after editing the lists
python -m cogniload.cli confidence --model dev --limit 5  # 5 items per condition
python -m cogniload.cli confidence --model dev
python -m cogniload.cli report confidence --model dev
```

Read the first table (`rank_wordlike`, J-lens), one row per condition and
output type (`correct`, `guess`, `abstain`): `uncertain` and `nonexistent`
against `control`. The row that matters is `fictitious` / `guess`, where the
model names a capital and so no uncertainty word is about to be output.
`answer` is the capital itself, for real items. `out_*` is the share of items
where a word of that group is in the model's own top-k next tokens; where it
matches the lens columns, the lens is showing runners-up for the output.

The same tables follow for an earlier layer window
(`confidence.early_fraction`), where the answer is not yet readable, and then
a per-layer profile for each lens.

Outputs, under `results/<alias>/`: `confidence.parquet` (one row per word ×
layer × lens), `confidence_summary.parquet` (one row per item: the model's
output and its type) and `confidence_manifest.json`.

### Adding an experiment

A module in `src/cogniload/` with `run` and `report`, a block in
`configs/experiments.yaml`, and its name in `cli.EXPERIMENTS`. The shared
blocks are in `experiment.py`.

## Layout

```
configs/models.yaml       model + lens registry
configs/experiments.yaml  settings; names an alias
modal_app.py              runs one CLI stage on Modal
stimuli/capitals.py       the item lists; writes capitals.csv
src/cogniload/
  cli.py                  stage runner
  confidence.py           confidence: prompt, run_item, table
  experiment.py           shared blocks: layers, two-lens readout, scoring, generation
  find_band.py            band discovery
  prompts.py              chat rendering with the thinking assertion
  readout.py              lens ranks per word, layer and position
  registry.py, bands.py   alias -> spec; band.json
```

Presence is band-min rank ≤ `readout.primary_k` (25). `rank` is over the full
vocabulary; `rank_wordlike` over word-like tokens only, which is what
Neuronpedia shows.
