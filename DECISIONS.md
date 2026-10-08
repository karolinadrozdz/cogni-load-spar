# Decisions

Open choices, each with what was decided, why, and what would reopen it.
Experiments: `find_band` (band discovery), `single_cue` (Experiment 1),
`forced_demand` and `retro_cue` (Experiment 2), `derived_state` (Experiment 3),
`state_tracking` (spec: `specs/state-tracking.md`).

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

## D2 — Scope: band discovery, then capacity, eviction, selectivity

**Decided:** `find_band`, then `single_cue` in condition C2 (internal) only, at
the `C_t` levels of D16. Measures: capacity, eviction, selectivity, plus the
list-all self-report scored descriptively. Controls: the logit lens, and
recency-matched untracked items within each stream (D1). The R2 report gate
and Q3 confidence are not implemented (D10).

**Deferred to v2, in order:** (1) held-out probes, (2) C1 externalized, (3) the
projection-based controls (random matched-dimension, PCA matched-variance) and
the k-free capacity measure, which travel with the probes, (4) R3
internal-reasoning replication.

**Why:** capacity, eviction and selectivity are the three questions the
keep-track task was chosen to answer; everything else is a covariate or a
control. Probes come first because analysis §4.5 — does self-report track
J-space presence or general decodability? — decides the project's direction
under spec §6, and it cannot run without a decodability measure to hold fixed.
Until then the self-report is descriptive only and **§6's branch cannot be
taken.**

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

## D6 — Every category present appears the same number of times; C_t varies only the instruction

**Signed off.**

**Decided:** `updates_per_tracked = 3`, `tail_guard = 3`, and *every* category
in the exemplar set appears exactly 3 times whether tracked or not. With 8
categories that is a 24-item stream at every `C_t`: 6/4/2 distractor categories
for `C_t` = 2/4/6, giving 18/12/6 distractor items. Stream length and
`n_distractor_categories` are both derived, not configured — to use fewer
categories, pass a subset of the exemplar dict.

**Why this deviates from spec §1 (`L = 18`, six categories present):** the spec
is internally inconsistent. With `L = 18` over six categories each tracked
category is updated ~3 times *regardless* of `C_t`, contradicting "3–9 times
depending on C_t"; and at `C_t = 6` all 18 slots are tracked, leaving no
distractors, which makes the selectivity analysis (§4.4) impossible at the
largest tracked-set size.

**Why equal frequency, and not just equal totals:** fixing the stream length
at 24 and letting two distractor categories absorb the remainder would give
those categories 9 items each at `C_t = 2` against 3 for each tracked
category. That makes untracked items differ from targets in category
*frequency* as well as in instruction — a confound sitting directly on the
selectivity contrast. Holding every category to 3 occurrences makes the stream
structurally identical across `C_t`, so the only thing that varies is the
"Track these categories" line, which is precisely the manipulation.

**Also an interpretation:** §1's rejection rule ("reject streams where any
tracked category's last exemplar is in the final 3 positions for all
categories") is ambiguous — read literally it can never fire for `C_t > 3`.
Implemented as *no target may fall in the last 3 positions*, so the answer is
never simply the most recent word overall, which is what the parenthetical
"keeps recency and category separable" asks for. Because any tracked item in
the tail is necessarily its category's last occurrence, this is exactly
equivalent to "the tail is all distractors", and is constructed directly rather
than reject-sampled (the admissible region is ~1% of orderings at `C_t = 6`).
Target recency spans 4–21.

**Reopens if:** C2 accuracy is at ceiling or floor (spec §5 anticipates
recalibrating L and C_t).

## D7 — Ambiguous exemplars are dropped, not assigned

**Decided:** a word may belong to exactly one of the eight categories.
`orange` and `olive` (fruit/colour) are excluded, and so are words whose second
category is simply absent from their own pool: `organ`, `bass`, `horn`
(instrument/body part, instrument/fish, instrument/body part), `file`, `level`,
`punch` (tool/other senses), `back` (body part/direction), `date` (fruit/time),
`bat` (animal/implement), `coral` (colour/animal), `kiwi` (fruit/bird).
`exemplars._assert_disjoint` enforces the in-pool case; the rest is judgement
recorded in the pool comment.

**Why, including the ones past position 12:** a word in two categories has no
defined role (target / replaced / untracked) when it appears in a stream. Spec
§8 bars exemplars equal to a category name for the same reason; this is the
same hazard one step out. Pruning words that filtering would usually never
reach still matters, because *which* words a pool reaches depends on the
tokenizer — a polysemous word can enter silently on one model and not another,
making a cross-model discrepancy look like a result.

## D8 — The registry is never machine-rewritten; the band lives with the results

**Decided:** `configs/models.yaml` is read-only to the code. `find_band` writes
the discovered band to `results/<alias>/band.json` (`bands.save`), and
`registry.resolve(alias, results_dir=...)` attaches it.

**Why:** writing the band into the registry with `yaml.safe_dump` round-trips
the file and silently drops every comment in it — including the block
explaining that `enable_thinking` is mandatory and why. Declared configuration
and discovered quantities should not share a file when one of them has to be
written programmatically.

## D9 — Turn structure is fixed in code, not left to the caller

**Decided:** each prompt is built by one function (`prompts.build_recent`,
`single_cue.build_list_all`, and each experiment's own builder).

- **Q1** (the single question) — one user turn containing the stream *and* the question; assistant
  prefilled `Answer:`; one token greedy. Two user turns would need an assistant
  reply in between, which would put model-generated text in context before the
  measurement. The readout position is the final token of the rendered prompt.
- **Q2** (list all) — a **fresh render from the same stream**, not a
  continuation of Q1. If Q2 followed Q1's answer, one tracked category would
  already be resolved in context and its self-report would not be comparable to
  the other categories'.

**Why in code:** the stream text and each question are separate templates, so
how they combine is exactly the kind of thing two collaborators would implement
differently in two scripts without noticing.

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

## D15 — Three readout positions, not per-position trajectories

**Decided:** `single_cue` reads every stream at three token positions:

- **`in_stream`** — the comma after the second-to-last word. Primary: every
  target has been seen, none is privileged, the query is not yet known, and the
  model is still disposed to continue the list.
- **`stream_end`** — the final `.`, where the model continues the instruction
  rather than the list. A contrast.
- **`answer`** — the `Answer:` prefill, a retrieval measure.

No per-position trajectories.

**Why `in_stream` is primary.** Q1 names one tracked category, so at the answer
position the queried target is simultaneously task state *and* the literal next
token — its presence cannot distinguish "the J-space holds task state" from
"the J-space holds what is about to be output". Inside the stream no target is
privileged and nothing is about to be emitted. That is what capacity and
selectivity want; the answer position is kept as the comparison. All three come
from one forward pass.

Reading the **delimiter** rather than the word follows the paper's own capacity
protocol ("at every comma position", `data/experiments/README.md`): a word is
trivially top-ranked at its own position because the model is processing it.
With `tail_guard` the final stream item is always untracked, so reading at the
word would plant a guaranteed false positive on the control side.

**Why no trajectories.** Three reasons, in order of weight:

1. *Confounded.* The spec's eviction analysis fits a step-vs-decay curve to an
   item's coefficient across positions. Every word spikes at its own position,
   so the putative "step at replacement" arrives at exactly the position where
   the replacement's own spike does. The single-position form — is a superseded
   item still present when its replacement is? — has no such artifact, with
   `untracked_superseded` as the "went stale without being task-relevant"
   baseline.
2. *Alignment.* Token positions and stream positions need a full per-item
   mapping; three positions need one lookup each.
3. *Cost.* ~120 positions per stream against 3.

**What is lost:** the shape of the decay. We keep the fact of it.

## D16 — Five C_t levels at 30 streams, not three at 100

**Decided:** `c_ts: [2, 3, 4, 5, 6]`, `n_streams: 30` — 150 streams.

**Why:** the capacity question is about the *shape* of the curve. Three points
leave no degrees of freedom to test a saturating curve against a line, so they
can only distinguish "flat" from "rising" — and if the knee is at a human-like
3–4, three levels straddle it invisibly. Precision per point matters less than
having points: the SE on a mean count is small even at n=30, while curve shape
is unidentifiable at any n with three levels. Sampling breadth over sampling
depth.

It also helps yield at the top end: the matched control (untracked
final, outside the tail) falls to ~0.1 per stream at `C_t = 6`, so the
intermediate levels carry the contrast.

## D17 — A floor gate before the full run

**Decided:** run `single_cue --limit 20` and check `report single_cue`'s floor
line before the full run. If the queried target is not present at the answer position in
≥50% of streams at primary k, stop and diagnose.

**Why `find_band` does not cover this:** it tests single-concept prompts. A
24-word stream is a far harder regime. If
presence is at floor in the task, all three measures return nulls that look
like findings, and we would only learn that after the full run. The absent-word
baselines make the check meaningful: they give "present" a chance floor, which
band-min rank over several layers otherwise lacks.

## D18 — forced_demand: the question, and the instruction before the readout

**Decided:** the question is "List the most recent word for each tracked
category, in alphabetical order. Answer with the words only, separated by
commas.", in the same user turn as the stream, prefilled `Answer:`, thinking
off. The readout sits at the `:` of `Answer:`, so the alphabetical instruction
comes before it: to emit the first word the model has to compare all `C_t`
targets. Behaviour is the rank of the alphabetically-first target (sorted
case-insensitively, since country names are capitalised) at that position;
`list_correct` is a `split(",")` + `strip()` comparison only. No fallback
question: first-word top-1 at `C_t = 2` was 33% on prod (40% on the dev smoke),
above the 20% at which the spec would switch to asking for a category name.

## D19 — retro_cue: cue selection and the three-message render

**Decided:** A = `queried_category`; B = the next tracked category after A in
`stream.tracked`, wrapping. `a1` is the argmax of the position-1 readout pass's
own logits (no separate generation). Position 2 is rendered as
`[user: stream + Q(A), assistant: "Answer: {a1}", user: Q(B)]` through
`render_chat(..., prefill="Answer:")` (`retro_cue.build_second_cue`). The Qwen3
template renders the historical assistant turn as plain text with no think
block; only the final generation prompt carries the empty think block.

## D20 — The lens is pinned to a commit

**Decided:** `lens_revision` in `models.yaml` is a commit sha, passed to
`JacobianLens.from_pretrained` as `revision=`; `registry.resolve` refuses a
non-sha. A branch such as `qwen-n1000` would follow its head, so the lens could
change under a fixed config (D3, D14).

## D21 — derived_state: digit answer prefill and scoring

**Decided:** prefill `"Answer: "` with the trailing space,
since `" 3"` is two tokens on this tokenizer (`" "` + `"3"`). The answer is
scored on the bare digit id at the last prompt token, from the readout pass's
own logits: rank, top-1, and rank among the ids for 0–3. `score_q1` is not
used: it encodes `" {expected}"`, which is two tokens for a digit. Two tokens
are greedy-generated and stored raw (`gen_tok1`, `gen_tok2`). On the dev smoke
the first generated token was a bare digit in 50/50 streams (both arms),
followed by `<|im_end|>`, so the spec's alternative (prefill `"Answer:"`, score
the second generated token) is not implemented. Readout tokens: bare digits
`0`–`9` and the space-prefixed number
words ` zero`–` nine`, each asserted to be one token; a value is present if its
digit or its word has band-min rank ≤ 25.

## D22 — derived_state: truncation and count collisions

**Decided:** streams are `experiment.streams` (the shared seeds), truncated to
`words[:p]` with
`p = random.Random(f"exp3-{stream_id}-{c_t}").randint(8, 21)` (inclusive;
`derived_state.trunc_range` in the config). Counts are over the truncated stream (0–3; the queried
answer can be 0). Groups are scored per category, not per token. An untracked
category's count is scored only where its value is not also some tracked
category's current count. A stale value (c − 1 for a tracked count c ≥ 1) is
skipped when it equals a tracked category's current count, and the number
skipped is reported. The table also reports the collision rate: positions
where two or more tracked categories share a count. Floor: values 5–9.

## D23 — derived_state: copyable-arm annotation

**Decided:** the copyable arm writes each **tracked** word as
`word (category n)`, with the category string as in `stream.tracked` and `n`
that category's running count including this word. Untracked words are not
annotated. The derived arm is the plain stream. Readout positions are the
comma token after items 1..p−1, located by character offsets. `),` may be a
single token, so the assertion is only that the read token contains `,`.

## D24 — state_tracking: item grammar, pools and grid

**Why:** each update names its person by the object they hold now, so the
final state cannot be read off the page without applying every update in
order; with thinking off, that chain is a load on one forward pass. Names and
objects must each be one token, so every readout and the emitted answer are
scored on a single id.

**Decided:** `n = 3` people. Initial sentence `"{name} holds the {obj}."`,
update `"The person holding the {cond} swaps it for the {new}."`, question
`"What is {poi} holding?"` in the same user turn, sentences joined by single
spaces, rendered through `render_chat` with thinking off. Pools are the
spec's candidates, kept if `readout.token_id` finds one token with a leading
space in the run's tokenizer, then dropping any object inside another;
asserted ≥ 6 names and ≥ 21 objects, and the surviving lists go in the
manifest (`templates.names`, `templates.objects`). Grid: `k ∈ {1..5}`,
`h ∈ {1, k, 2k}`. At `k = 1` that set is `{1, 2}`, so the duplicate cell is
dropped: 14 cells × 30 = 420 items per arm, not the spec's 450. Items are
seeded per cell (`random.Random(f"state_tracking-{k}-{h}")`), independent of
arm, so both arms see the same items and `--limit N` takes the first N of
each cell. One shard per cell: `state_tracking_<arm>_k{k}_h{h}[_limit].parquet`.

## D25 — state_tracking: no reused object, recency guard, floor

**Why:** a reused object would make a readout token ambiguous between roles.
If every item ended in a needle, "answer = the last new object" would score
without any tracking. A floor drawn from the same pool is matched to the
targets in frequency and form.

**Decided:** each item samples `3 + k + h` distinct objects: 3 for the initial
state, one fresh `new` per update. Items alternate: even-indexed items end in a
hay (`last_is_needle = False`), odd-indexed end in a needle, so each full cell
is 15/15 (with `--limit` the split is approximate). Needle positions are a
uniform sample of the slots allowed by that constraint; each hay's person is a
uniform non-PoI. Every item passes `state_tracking.validate`. The floor is
every pool object not used in the item (≥ 3), never in the prompt.

## D26 — state_tracking: prefill and first-token behaviour

**Why:** behaviour is scored on one token, so the token after the prefill has
to be the bare object. An article there would score every item wrong for a
format reason.

**Decided:** prefill is `state_tracking.prefill` in the config, default
`"Answer:"`; switching to `"Answer: the"` is a config change only. Behaviour is
the readout pass's own logits at the last prompt token: argmax, `score_q1`
against the target, restricted rank among the item's in-prompt objects, and an
error class from the decoded argmax (`target`, `poi_stale_j`,
`other_current`, `other_stale`, `out_of_prompt`). No generation. The decoded
argmax is printed per cell and in `report`. The question ends "Answer in one
word.", as single_cue's does: without it the argmax at `Answer:` is ` **` in
every item, with the target behind it. With it, the first token is a bare pool
object in every dev smoke item (70/70, derived arm), so the prefill stays
`"Answer:"`.

## D27 — state_tracking: copyable-arm annotation

**Why:** the copyable arm is the ceiling and the readout's positive control,
which needs every binding on the page at every step. Annotating only needles
would leave the hays' bindings to be derived.

**Decided:** every update, needle and hay, gets ` (Ann: pen, Ben: lamp, Tom:
cup)` before its period: the full state after that update, names in
initial-state order. Everything else is identical to the derived arm. The
readout at each update sits on the token containing the sentence's final
period, located by character offset (after `)` in this arm).

## D29 — hay_factorial and hay_type: terminal object, pre_hays, tail_hays

**Why:** in state_tracking, `h` grows with `k` and the hays after the PoI's
last needle are not controlled, so "hays interfere with the lookup" and "the
model answers a recent object" predict the same curve. Splitting the hays by
position against the PoI's last needle separates them.

**Decided:** a terminal object is one introduced in the prompt and never later
used as a condition; there are exactly three per item, one per person.
`tail_hays` is the number of hays after the PoI's last needle, `pre_hays =
h − tail_hays`. Both are stored per item (in the item and the summary row).
The A1 regression is the logistic `correct ~ k + pre_hays + tail_hays`,
reported with coefficients, Wald z and the likelihood-ratio p for dropping
each term.

## D30 — hay_factorial: grid, tail_hays strata, pool cap

**Why:** the regression in D29 is only clean if `k`, `pre_hays` and
`tail_hays` are independent by design, not by the luck of the interleaving.
`h = 0` is the reference with no hays at all.

**Decided:** derived arm only, `k ∈ {1, 3, 5}` × `h ∈ {0, 2, 4, 8, 12}`, 60
items per cell, 900 items. Item `i` of a cell has `tail_hays =
strata[i % len(strata)]`, where the strata are those of `{0, 2, 4}` that are
≤ `h`: `{0}` at `h = 0`, `{0, 2}` at `h = 2`, else `{0, 2, 4}`, so a full cell
splits equally (20/20/20 or 30/30; approximate under `--limit`). The PoI's last
needle sits just before the tail; its other `k − 1` needles are a uniform
sample of the slots before it, and each hay's person is a uniform non-PoI.
Items are seeded `random.Random(f"hay_factorial-{k}-{h}")`, apart from
state_tracking's cells, and every item passes `state_tracking.validate`
unchanged. Pool cap, checked at run time on the filtered pool: if
`3 + max k + max h + 3 floor` objects do not fit, every `h` above
`hay_factorial.h_cap` (10) becomes 10; the manifest records `hs` and
`h_capped`. One shard per cell: `hay_factorial_k{k}_h{h}[_limit].parquet`.

## D31 — hay_type: hay templates and shared items

**Why:** if the cost of a hay is the lookup among identical "holding the X"
sentences, a hay that names its person needs no lookup and a reworded hay does
not match the needle template; each should recover accuracy by a different
amount. Only the hay sentence may differ between arms, or a difference could
come from the items.

**Decided:** derived templates, `k = 3`, `h = 6`, `tail_hays = 2`, 60 items,
seeded `random.Random("hay_type-3-6")` independent of arm, so every arm sees
the same items. Needles always use the state_tracking update template. Hays:
`same` is that template, `named` is `"{who} swaps the {cond} for the {new}."`
(the hay's person by name), `reworded` is `"Whoever has the {cond} trades it
for the {new}."`. The arm's template rides on the item (`item["hay"]`), which
`state_tracking.build` reads for hays only, so `run_item`'s readout, period
offsets and scoring are reused unchanged; without the key `build` is
byte-identical to before. `same` renders exactly state_tracking's derived
prompt. The reference for every arm is hay_factorial's `k = 3, h = 0` cell.
Shards: `hay_type_<arm>_k3_h6[_limit].parquet`.
