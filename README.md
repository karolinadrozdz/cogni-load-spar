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

## Outputs

Under `results/<alias>/`, or the `cogniload-results` Modal volume at
`<alias>/` (fetch with `modal volume get cogniload-results prod ./results/prod`):

| file | contents |
|---|---|
| `band.json` | the band, and the per-layer medians that chose it |
| `<experiment>_ct{c}[_limit].parquet` | one row per word × layer × position × lens: `rank`, `rank_wordlike`, `role`, `readout_at`, `in_band` |
| `<experiment>_summary_ct{c}[_limit].parquet` | one row per stream: behaviour and read positions |
| `<experiment>_manifest.json` | model spec, prompt templates, exemplars, config |

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
  prompts.py              shared templates; chat rendering with the thinking assertion
  readout.py              lens ranks per word, layer and position
  stimuli.py              seeded stream generation (pure)
  exemplars.py            category word pools, filtered to single tokens
  registry.py, bands.py   alias -> spec; band.json
```
