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

"What is the capital of {entity}? Answer in one word.", prefilled with
`Answer:` so the next token is the answer. Two stimuli sets, run separately:

| `--set` | file | condition | items |
|---|---|---|---|
| `capitals` | `stimuli/capitals.csv` | `real`, `fictitious` | 50 countries, 50 invented ones |
| `regions` | `stimuli/regions.csv` | `region` | 130 real regions, from famous to little known, from Wikidata |

Read at the prefill in two ways: a fixed list of uncertainty words, words for
something not existing and control words (`confidence.GROUPS`); and the top 25
word-like tokens at each layer, with nothing chosen in advance.

```bash
python -m cogniload.cli confidence --set regions --model dev --limit 5  # 5 items per condition
python -m cogniload.cli confidence --set regions --model dev
python -m cogniload.cli report confidence --model dev                   # every set that was run
```

The report, in order:

1. The fixed-list table, one row per condition and output type (`correct`,
   `guess` = a name that is not the capital, `abstain`): `uncertain` and
   `nonexistent` against `control`, in the band and then in an earlier window
   (`confidence.early_fraction`). `out_*` is the share of items where a word of
   that group is in the model's own top-k next tokens.
2. A per-layer profile for each lens.
3. Top tokens: the tokens in the top 25 of more items of one set than another,
   for `region` wrong names against correct ones, and `fictitious` against
   `real`. This part is exploratory.

The comparison that matters is `region` / `guess` against `region` /
`correct`: the output is a name in both, so a difference is not output staging.

Outputs, under `results/<alias>/`, per set: `confidence_<set>.parquet` (fixed
list: one row per word × layer × lens), `confidence_<set>_top.parquet` (top
tokens per item, layer and lens), `confidence_<set>_summary.parquet` (one row
per item: the model's output, its type and its probability) and
`confidence_<set>_manifest.json`.

To rebuild a stimuli file, run the script beside it (`stimuli/capitals.py`,
`stimuli/regions.py`). `regions.py` filters the committed Wikidata snapshot
`stimuli/regions_wikidata.csv` and needs the `dev` tokenizer from the Hub.

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
stimuli/regions.py        filters the Wikidata snapshot; writes regions.csv
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
