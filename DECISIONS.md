# Decisions

Open choices, each with what was decided, why, and what would reopen it.

Only the entries that govern the shared infrastructure on this branch are kept,
with their original numbers and text. Some still mention the keep-track
experiments they were written for; those, and D2, D6, D7, D9, D15-D19 and
D21-D23, are on `exp/aveizi/task-finding`.

---

## D1 — "Present in the J-space" means band-min lens rank ≤ k, with k swept

**Decided:** presence of an item at a position is `min over band layers of
rank(item) ≤ k`, with `k ∈ {1, 5, 10, 25, 50}` swept. Not a sparse nonnegative
decomposition at `k_sparse ∈ {8, 16, 25}` as spec §3.2 states.

**Why:** the released reference implementation (`anthropics/jacobian-lens`)
has no sparse decomposition — its public API is read-only (`fit`, `from_hf`,
`JacobianLens.{transport, apply}`, `ActivationRecorder`). The paper's own
released protocol defines a hit as rank-1 in the band
(`data/experiments/README.md`), and `capacity.json` counts list words with
band-min rank ≤ k. Using it makes band discovery a replication rather than a
reconstruction, and keeps k a transparent post-hoc threshold instead of baking
it into a solver.

**Caveat, recorded deliberately:** this does *not* make capacity k-free. The
proposal's actual fix for the circularity risk (`jspace-workspace-proposal.md`
§6 F2) is a k-free probe-based capacity measure, and probes are out of scope (D2). So
the capacity claim is only **"a plateau stable across k and absent in the
logit-lens control"** — not "a k-free capacity ceiling." Do not write the
stronger claim without F2.

**Second caveat — what "same as the paper" means.** The paper's own capacity
number comes from a sparse nonnegative decomposition at `k ≤ 25`, which the
released code does not include. So we match the *released protocol*, not the
published number. That is the best reproducible target available and it is what
Neuronpedia implements; say it that way in any write-up.

**Consequence for the controls — this is what D1 changed downstream.** Spec
§3.4 asks for "a random subspace of matched dimension and a PCA subspace of
matched variance." That control belongs to the *projection* design in the
proposal's F2, where the residual is projected onto a low-dimensional subspace
and "matched dimension" is meaningful. D1's rank readout has no projection: it
transports through the full `d × d` matrix `J_ℓ` and ranks the word under the
model's unembedding. A random `d × d` matrix has no dimension to match — it
scrambles the residual and returns chance-level ranks, so the control passes
trivially and licenses nothing.

So:

- **Subspace control = the logit lens**, `lens.apply(..., use_jacobian=False)`:
  the same readout with `J_ℓ` replaced by the identity. This is the paper's own
  baseline and the comparison Neuronpedia shows side by side.
- **Within-stream control = untracked items matched to targets on recency**
  (§4.4), which is what actually separates "holds task state" from "holds
  whatever is recent."
- There are no random or PCA subspace controls; they return only if the
  probe design does (D2). The rank readout is the replication, and the
  projection measures are the proposal's extension beyond the paper.

**Reopens if:** the plateau moves with k (then there is no capacity finding and
the measure needs F2 before the indicator can be scored).

## D3 — Model selection goes through a registry alias, never a literal

**Decided:** `configs/models.yaml` is the only file naming a model or lens.
`configs/experiments.yaml` names an alias (`dev`, `prod`), and `--model`
overrides it. Every model-dependent constant hangs off the alias: weights, lens
file and chat contract; the exemplar list is filtered per tokenizer at run
time. Every ref, model and lens, is a commit sha, and `registry.resolve()`
refuses anything else (D20). The registry is read-only to the code (D8).

- `prod` = `Qwen/Qwen3.6-27B`.
- `dev` = `Qwen/Qwen3.5-4B`, **not** the spec's `google/gemma-3-12b-it`.

**Why dev is Qwen3.5-4B:** it shares the lens repo, revision and fitting
corpus with prod *and* the model family, tokenizer and chat template — so
dev→prod varies exactly one thing (scale). Lens availability is not the
deciding factor: a pre-fitted `gemma-3-12b-it` lens does exist in the same
Neuronpedia repo and revision. The deciding factor is that Gemma changes
family, tokenizer and chat template at once, which would make any dev→prod
discrepancy uninterpretable. (Gemma's export also lacks the `_n1000` fit the
two Qwen lenses share, so even the lens would not be matched.)

## D4 — Thinking mode is off and asserted, not defaulted

**Decided:** every experiment prompt renders through the chat template with
`enable_thinking=False`, pinned per alias in the registry and recorded in the
run manifest. `prompts.render_chat` *asserts* the rendered string ends in the
thinking-off marker and raises otherwise. Prefill (`Answer:`) is appended after
the template's empty think block; the readout position is the final token of
the rendered string.

**Why an assertion and not a default:** the Qwen3 template's branch is

```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- else %}
    {{- '<think>\n' }}      ← thinking is the DEFAULT
{%- endif %}
```

An unset flag opens a reasoning block, so C2 would silently externalize its
state and become C1 — a result-destroying failure with no error. Rendering the
real template with the flag omitted is byte-identical to `enable_thinking=True`.

`find_band` reads raw completion prompts, not chat renders: it measures the
lens, not the task.

## D5 — Lens integrity is checked by shape, not filename

**Decided:** `registry._assert_lens_matches` verifies `lens.d_model` and
`len(lens.source_layers)` against the registry before any readout.

**Why:** the upstream `qwen3.6-27b` lens directory ships a `config.yaml`
describing `openai/gpt-oss-20b` — a stale Neuronpedia export artifact. The `.pt`
itself is correct: 3.303 GB = `5120² × 63 × 2` bytes, matching Qwen3.6-27B
(a gpt-oss-20b lens would be 0.398 GB). Qwen3.5-4B checks out the same way
(`2560² × 31 × 2` = 0.406 GB). Upstream metadata is not trustworthy; shapes are.

Note the lens stores `n_layers − 1` matrices (layer ℓ → final; the final layer
needs no transport). Band indices must respect that.

Qwen3.6-27B is `Qwen3_5ForConditionalGeneration` (a multimodal wrapper with
hybrid linear/full attention); jlens resolves it via
`Layout("model.language_model")`, loaded with `AutoModelForCausalLM` as in the
jlens walkthrough.

## D8 — The registry is never machine-rewritten; the band lives with the results

**Decided:** `configs/models.yaml` is read-only to the code. `find_band` writes
the discovered band to `results/<alias>/band.json` (`bands.save`), and
`registry.resolve(alias, results_dir=...)` attaches it.

**Why:** writing the band into the registry with `yaml.safe_dump` round-trips
the file and silently drops every comment in it — including the block
explaining that `enable_thinking` is mandatory and why. Declared configuration
and discovered quantities should not share a file when one of them has to be
written programmatically.

## D10 — No swap intervention, no R2 report gate

**Decided:** neither is implemented. Every result here is a readout; nothing
intervenes on the residual stream. Re-add both together if a result needs the
lens validated under intervention.

## D11 — "Legible" is a committed numeric criterion, fixed before band discovery

**Decided:** the band is the **longest contiguous run of at least `min_width`
layers inside `candidate_fraction` whose median full-vocab rank over `find_band`'s items
is ≤ `k_band`**. Committed in `configs/experiments.yaml` as
`candidate_fraction: [0.25, 0.85]`, `k_band: 25`, `min_width: 3`.
`find_band.choose_band` raises rather than loosening the criterion on its own.

**Why it must be numeric and committed in advance:** spec §R1 says "record the
layer band where readouts are legible" without defining legible, and every
experiment then reads in that band. Choosing the band by which layers make an
experiment come out well would fit the band to its own answer, so the criterion
uses `find_band`'s readouts only. `k_band: 25` matches the paper's `k ≤ 25`
scale; any pre-committed number would do, choosing one afterwards would not.

## D12 — Two rank columns: full-vocab primary, word-like alongside

**Decided:** every readout row logs `rank` (full vocabulary, the paper's
protocol, primary in `single_cue`) and `rank_wordlike` (ranked among word-like
tokens only, via the library's own `jlens.vis._meaningful_token_mask`).
`single_cue` reports both; its capacity and `k` sweeps use `rank`. The
forced_demand, retro_cue and derived_state tables lead with `rank_wordlike`, because on Qwen3.6-27B
`<|im_end|>` and underscore runs flood the full-vocab top-25 (the queried target
reads 0.29 on `rank` against 0.74 on `rank_wordlike`).

**Why both:** the library's walkthrough notes that on Qwen "the interesting
word tokens trail punctuation and single-character tokens in the raw top-K",
and its visualiser masks the *display* to word-like tokens while keeping
full-vocab ranks for scoring. Consequences if we logged only one:

- only `rank` — `k = 1` and `k = 5` could be near-zero hit rates even where a
  concept is plainly present, making the low end of the k sweep uninformative
  for reasons that have nothing to do with the workspace;
- only `rank_wordlike` — we would no longer be running the paper's protocol.

Logging both also means a collaborator cross-checking Neuronpedia sees the
column that matches the site instead of concluding our numbers are wrong.

Ranks are **1-indexed** (`rank == 1` is top-1; `readout.rank_of` counts
strictly greater logits, plus one), so `rank <= k` reads as the paper states
it.

## D13 — One prefill convention, asserted

**Decided:** every prompt is rendered with `add_generation_prompt=True` and the
prefill appended to the resulting string, so the rendered tail is
`<think>\n\n</think>\n\n` + prefill.

**Why it needs deciding at all:** the paper reads at the token "immediately
before the name is produced", but under a chat template that token is whatever
prefill we choose; the paper's completion-style setup had no such choice.
`jlens`'s examples prefill by passing a final assistant message with
`continue_final_message=True`, which renders *without* the empty `<think>`
block. Both forms are legitimate, but they are different prompt surfaces, and
mixing them between experiments would compare two things. `render_chat`
only implements the appended form and asserts the thinking marker, so the other
form cannot reach a model through it.

## D14 — Dependencies are pinned on the same principle as models

**Decided:** `jlens` pinned to `581d398613e5602a5af361e1c34d3a92ea82ba8e`;
`transformers>=5.5`.

**Why:** `jlens` is an explicitly unmaintained reference implementation, so
following `main` would change the readout without changing our code or our
manifest — the registry's own rule, applied to code. `jlens` itself requires
`transformers>=5.5`, as do the Qwen3.5/3.6 model cards; a lower floor resolves
and then fails at load. The Modal image installs from `pyproject.toml`, so the
pin is written once.

## D20 — The lens is pinned to a commit

**Decided:** `lens_revision` in `models.yaml` is a commit sha, passed to
`JacobianLens.from_pretrained` as `revision=`; `registry.resolve` refuses a
non-sha. A branch such as `qwen-n1000` would follow its head, so the lens could
change under a fixed config (D3, D14).

## D24 — confidence: stimuli, word groups and output types

**Decided:** one template, "What is the capital of {entity}? Answer in one
word.", prefilled `Answer:`, over 50 real countries and 50 invented ones
(`stimuli/capitals.py`). Every real capital is one token after a space on the
Qwen tokenizer, checked against the pinned `dev` tokenizer. Countries with a
contested or multi-word capital, or named like their capital, are left out.

Read at the prefill, in three groups fixed before any run
(`confidence.GROUPS`): `uncertain`, `nonexistent`, and `control` as the floor.
Each concept is read in lower case and capitalised, whichever are single
tokens, and is present if either form is.

Each item's output is typed from eight greedy tokens: `correct`, `abstain`
(first word in `confidence.ABSTAIN`) or `guess`. The raw text is stored, so the
typing can be redone.

**Why:** an uncertainty word in the J-space is uninformative when the model is
about to say "Unknown"; it is then the next token. Only an invented country
answered with a name separates held uncertainty from output staging.

**Known limits:** invented names are 2-3 tokens where real ones are 1, so the
two conditions differ in form as well as in knowledge. The control words are
common nouns, not frequency-matched to the uncertainty words. The invented
names were not searched against real or fictional places.

## D25 — confidence: an early layer window beside the band

**Decided:** the report gives every table twice: in the band, and over
`confidence.early_fraction: [0.25, 0.60]` of depth (layers 8-18 on `dev`),
followed by a per-layer profile. Each table also reports, per word group, how
often one of its words is in the model's own top-k next tokens (`out_*`).

**Why:** the `dev` band is layers 23-26 of 32, where the logit lens reads
nearly what the J-lens does and a word can be present only as a runner-up for
the output. The early window is the range before the queried word became
readable in the keep-track runs (absent through L18). It was fixed after a
10-item smoke run seen in-band only, and before the full run.

## D26 — confidence: real regions, and an open top-token readout

**Decided:** a third condition, `region`: 130 real first-level regions with
their capitals, asked as "{region}, {country}" in the same template. They come
from a Wikidata query (`stimuli/regions.py`, snapshot committed), filtered by
rule: one capital, not a national capital, plain ASCII names, a one-word
capital that is one token and does not appear in the region's name, at least
20 Wikipedia language editions, and one region per capital. The number of
editions (`links`) is kept as a measure of how well known the region is.

Alongside the fixed word list, the top 25 word-like tokens at each layer are
stored for both lenses (`confidence_top.parquet`), and the report lists the
tokens found in more items of one set than of another.

**Why:** the first run had no confident wrong answers; the model never invents
a capital for an invented country, and it recognises those names as made up.
Real regions give right and wrong named answers under one condition, so the
two can be compared with the output type held fixed. The fixed list found its
effect in two of twelve words and could not have found a word that was not on
it; the top tokens show what the lens holds without that choice.

**Status of each:** the fixed list is the pre-committed test and is unchanged
from D24. The top-token comparison is exploratory.

The two stimuli sets run separately (`--set capitals`, `--set regions`) and
write separate files, so a new set does not require the earlier one to be run
again.

**Known limits:** the set is every region that passes the rules, not a sample
of obscure ones. Requiring a one-token capital removes most little-known
regions (162 of 1,035 otherwise usable pass it), so about a quarter of the set
is little known, and those capitals are often common words ("Salt", "Same", "Paradise"). An answer is scored against the Wikidata
label only, so another name for the same city counts as wrong. The Wikidata
entries were not checked one by one.

