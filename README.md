# cogni-load-spar — Experiment 1

## The question

> Does the J-space hold the task state the model is maintaining, or just
> whatever is recent or about to be output? If it does, do we believe the
> J-space reasonably functions as a proxy for cognitive load, and J-space
> content can be used to indicate cognitive load (i.e. via content, eviction,
> etc.)?

The first clause is what the keep-track task is built to separate: a tracked
category's current word is task state the model must hold, a recency-matched
untracked word is not, and the two are made structurally identical so only the
instruction distinguishes them. The second clause is what capacity and eviction
speak to — whether what the J-space holds, and what it drops when something
replaces it, behaves like a load-bearing resource.

Spec: `experiments/experiment-1-spec.md` (in the shared notes repo).
Every open choice is recorded in [`DECISIONS.md`](DECISIONS.md) — **read it
before changing a measurement**, particularly D1 (what "present in the J-space"
means, and why the only subspace control is the logit lens) and D2 (what v1
does and does not claim).

## Running

```bash
pip install -e .
export HF_TOKEN=...

# 1. find the layer band (prints a J-lens vs logit-lens sanity check first,
#    then writes results/<alias>/band.json)
python -m cogniload.cli phase0 --config configs/exp1.yaml

# 2. FLOOR GATE — 20 streams, then look before committing
python -m cogniload.cli phase1 --config configs/exp1.yaml --limit 20
python -m cogniload.cli analyze --config configs/exp1.yaml

# 3. the run: 5 C_t levels x 30 streams = 150 streams
python -m cogniload.cli phase1 --config configs/exp1.yaml
python -m cogniload.cli analyze --config configs/exp1.yaml
```

Stages are idempotent; re-running skips completed work unless `--force`.

**Do not skip step 2.** If the queried target isn't present at the answer
position in ≥50% of streams, every measure below returns a null that looks like
a finding. Phase 0's R2 does not cover this — it tests single-concept prompts,
which is a far easier regime than a 24-word stream (DECISIONS.md D17).

## What the run produces

Three forward passes per stream (J-lens readout, logit-lens readout, Q2
generation) at two token positions — the **end-of-stream** delimiter and the
**answer** position. Q1 needs no generation of its own: `lens.apply` returns the
model's own logits at the readout positions, so the answer is consistent with
the measured ranks by construction.

`analyze` prints four things:

| | what it answers |
|---|---|
| **floor check** | is the readout working in the task at all? |
| **capacity** | non-queried targets present vs C_t — does it *rise* then flatten? |
| **selectivity / eviction** | target vs untracked-final vs superseded vs absent |
| **k sweep** | is the capacity shape an artefact of the threshold? |

Everything keys off one distinction: a **non-queried** target is task state the
model is *not* about to say, so it is the only item whose presence separates
"the J-space holds task state" from "the J-space holds what is about to be
output". The queried target is both at once and is reported separately.

A capacity curve that rises with C_t then flattens is a ceiling. A curve flat at
~1 is the deflationary result — and it would satisfy a naive "plateau"
criterion, so the rise is the part that matters.

## Three conventions that keep runs comparable

**Models are selected by alias, never by name.** `configs/models.yaml` is the
only file in the repo that names a model or a lens. `configs/exp1.yaml` says
`model: dev`; nothing downstream knows what that resolves to. To add a model,
add an alias block — weights, lens file, chat contract and the
(tokenizer-dependent) exemplar list all hang off it. Every ref is pinned to a
commit sha and `registry.resolve()` refuses an unpinned one.

```yaml
# configs/models.yaml
aliases:
  dev:   { hf_id: Qwen/Qwen3.5-4B,  hf_revision: 851bf6e8..., enable_thinking: false }
  prod:  { hf_id: Qwen/Qwen3.6-27B, hf_revision: 6a9e13bd..., enable_thinking: false }
```

`dev` and `prod` share a lens repo, revision, fitting corpus, model family,
tokenizer and chat template, so a result that holds on `dev` but not `prod`
implicates scale and nothing else.

**Declared config and discovered values live in different places.**
`configs/models.yaml` is hand-written and never machine-rewritten — a YAML
round-trip would silently drop every comment in it, including the one marking
`enable_thinking` mandatory. The workspace band is *discovered* by Phase 0 R1,
so it is written to `results/<alias>/band.json` and attached via
`registry.resolve(alias, results_dir=...)`. See DECISIONS.md D8.

**Prompts are frozen, content-addressed, and assembled in one place.** Every
prompt is a `Prompt` in `prompts.py` with an id, a version and a sha; the run
manifest records the sha of the whole set, so editing a template invalidates
cached results instead of silently contaminating them. Change a prompt by
bumping its `version`, never by editing in place.

How the stream and the queries *combine* is also fixed in code (`build_q1` /
`build_q2`), because that is exactly what two people would implement
differently in two scripts: Q1 is one user turn with the assistant prefilled,
Q2 is a fresh render with no Q1 in context. `build_q3` exists but v1 does not
use it. See DECISIONS.md D9.

`render_chat` **asserts** the thinking state it was asked for. This is not
defensive padding: the Qwen3 chat template opens a `<think>` block whenever
`enable_thinking` is unset, which would make the internal condition externalize
its state and silently become the externalized one. See DECISIONS.md D4.

## Layout

```
configs/models.yaml     model + lens registry — the only place a model is named
configs/exp1.yaml       experiment config — names an alias
src/cogniload/
  registry.py           alias -> ModelSpec; loads weights + lens, verifies shapes
  bands.py              workspace band persistence (results/<alias>/band.json)
  prompts.py            frozen hashed prompts; turn structure; chat assertions
  exemplars.py          category pools, filtered to single tokens per model
  stimuli.py            seeded keep-track stream generation (pure, no GPU)
  align.py              locating the end-of-stream token
  readout.py            lens ranks per layer/position (full-vocab + word-like)
  swap.py               the J-space swap intervention (the paper's formula)
  generate.py           Q2 only; Q1 comes free from the readout pass
  scoring.py            Q1/Q2 scoring; the committed R1 band criterion
  phase0.py             R1 band discovery + stack sanity check
  phase1.py             the keep-track run
  analyze.py            the three numbers + the floor check
  cli.py                stage runner
data/exemplars/         per-alias single-token exemplar lists (generated)
results/                parquet outputs, band.json, run manifests
```

## Measurement choices worth knowing before you read a number

- **Presence** is band-min lens rank ≤ k, with k swept — the paper's released
  protocol. There is no projection onto a subspace, so the only subspace
  control is the logit lens (D1).
- **Two rank columns.** `rank` is full-vocabulary and primary; `rank_wordlike`
  masks to word-like tokens and is what Neuronpedia displays. On Qwen the raw
  top-K is dominated by punctuation, so `k=1` on full vocab is stricter than it
  looks (D12).
- **The swap** is the paper's formula, `h' = h + V(σ(c) − c)` with `c = V⁺h` —
  not "replace one direction with another", which is ambiguous (D10).
- **The band criterion** is numeric and committed in `exp1.yaml` before R1
  runs, and uses R1 readouts only. Choosing it by what makes R2 pass would make
  the gate circular (D11).

## Upstream

- [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) —
  the lens. Read-only API; the swap intervention and the membership measure are
  ours (DECISIONS.md D1).
- `neuronpedia/jacobian-lens` @ `qwen-n1000` — pre-fitted lenses. Note the
  upstream per-model `config.yaml` is not reliable; we verify lenses by shape
  (DECISIONS.md D5).
- [`camilablank/workspace-bench`](https://github.com/camilablank/workspace-bench)
  — fallback task if the workspace gate fails (spec §6).

## Setup

The laptop only ever *launches* runs, so it does not need the GPU stack:

```bash
cd cogni-load-spar
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"      # modal, pytest, pandas — no transformers, no jlens
modal setup                    # once, if `modal profile current` is empty
```

Dependency groups:

| | contents | installed where |
|---|---|---|
| core | `torch`, `pyyaml`, `pandas`, `pyarrow`, `numpy`, `scipy` | everywhere |
| `[local]` | `modal`, `pytest`, `ruff`, `jinja2` | laptop |
| `[gpu]` | `jlens` (pinned sha), `transformers>=5.5`, `accelerate` | the GPU |

`modal_app.py` installs the `[gpu]` set into the image itself, so nothing
needs it locally. On a plain CUDA box instead of Modal, use
`pip install -e ".[gpu]"` and the `cogniload.cli` commands directly.

## Running on Modal

No local GPU needed — and macOS cannot run this at all (no CUDA).

```bash
source .venv/bin/activate
cd cogni-load-spar              # image paths are relative to the working directory

modal run modal_app.py --stage phase0 --model dev
modal run modal_app.py --stage phase1 --model dev --limit 20
modal run modal_app.py --stage analyze --model dev     # no GPU
```

Then the same three with `--model prod` once `dev` looks sane.

Two volumes persist between runs: `cogniload-hf-cache` (weights and lenses, so
the 54 GB prod download happens once) and `cogniload-results` (band.json,
parquet shards, manifests). GPU follows the alias — `dev` gets an `L4`, `prod`
an `H100`; override with `--gpu`.

Volumes are committed before an error is raised, so a crash part-way through
the C_t sweep keeps the shards already written and a re-run resumes.

To pull results down:

```bash
modal volume get cogniload-results / ./results
```
