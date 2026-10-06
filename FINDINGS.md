# Findings so far

Status: `dev` (Qwen3.5-4B) floor data collected and analysed. `prod`
(Qwen3.6-27B) full run pending. **This document is written before the `prod`
results arrive**, so that the interpretation table below commits us in advance
rather than letting us rationalise whatever comes back.

Everything numeric here is from the single_cue run (Experiment 1, the keep-track
task) at `n = 20` streams per
`C_t` (100 streams total), in-band, `k = 25`, unless stated. Design decisions
are recorded in `DECISIONS.md` (D1–D17); the measurement choices that matter
for reading these numbers are D1, D12 and D15.

---

## 1. The question, as it now stands

We started with: *does the J-space hold the task state the model is
maintaining, or just whatever is recent or about to be output?*

That has narrowed twice, both times because the original form turned out not to
be measurable as posed.

**First shift (D15): split the targets by whether they are being queried.** The
task asks the model to track `C_t` categories and then queries *one* of them.
At the answer position, the queried category's current word is simultaneously
task state *and* the literal next token, so its presence cannot distinguish the
two hypotheses — under either one it should be there. Only the other `C_t − 1`
targets discriminate. We were not recording which category had been queried, so
the field needed to compute the only informative contrast did not exist. It
does now (`Stream.queried_category`, drawn from the stream RNG).

**Second shift: item identity vs category structure.** The Neuronpedia evidence
in §3 shows that for these prompts the J-space readout is dominated by
*category* vocabulary, not by the remembered words. We had been measuring only
the words. The current question is therefore:

> At a position where the model must hold all `C_t` tracked items and does not
> yet know which will be queried, does the J-space contain (a) the item
> identities, (b) the tracked category labels specifically, or (c) the task's
> category vocabulary generically — and does any of it distinguish *tracked*
> from *untracked*?

This is narrower than the original and better posed, because each of (a), (b),
(c) has a distinct, measurable prediction against a floor we actually have.

---

## 2. What we measured

### Behaviour — the model can do the task

| | C_t=2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Q1 top-1 | 55% | 55% | 60% | 70% | 60% |
| Q2 per-category accuracy | 80% | 77% | 80% | 81% | 84% |

Q2 parse rate 100%. Pooled Q1: 60% top-1, **median rank of the expected word
= 1**, top-5 97% (n = 100). So where Q1 is "wrong" it is usually rank 2–3
behind a formatting token, not absent.

**Accuracy is flat, or very slightly rising, across `C_t`.** This matters a
lot and is discussed in §5.

### Item-level readout — pooled over `C_t`, in-band, k=25

| group | `in_stream` | `answer` | n |
|---|---|---|---|
| queried target | 0.060 | **0.700** | 100 |
| queried category's replaced items | 0.025 | **0.450** | 200 |
| non-queried targets | 0.077 | 0.087 | 300 |
| untracked stream words | 0.188 | 0.131 | 1200 |
| **absent words (floor)** | **0.091** | **0.065** | 800 |

Read every row against `absent`. The queried target is 70% against a 6.5%
floor — about 11×, unambiguous. **Non-queried targets are at the floor at both
positions**, and at `in_stream` they are numerically *below* it. Per-`C_t`,
`replaced_queried` at the answer position runs 0.50 / 0.50 / 0.40 / 0.50 / 0.35.

Supporting detail:

- `stream_end` (the `.` after the last stream word) is **exactly 0.000** for
  every group, every layer, every `C_t`. Explained in §4.
- Per-layer at the answer position, `target_queried` is 0.00 through L18, 0.13
  at L19, then 0.63 / 0.66 / 0.69 / 0.68 at L23–26 of 32. The signal is late
  and we verified there is no earlier peak by recording layers 8–26.
- The capacity count (non-queried targets present per stream) is
  0.20 / 0.35 / 0.20 / 0.05 / 0.50 at the answer position with **sem ≈ 0.05–0.16
  on n = 20**. That is noise, not a curve. No capacity plateau, no eviction
  signal, no selectivity at item level.
- Full-vocabulary and word-like ranks agree to within 2 points everywhere
  (0.700 vs 0.720; 0.087 vs 0.090; 0.091 vs 0.092), so punctuation crowding the
  top-K is **not** the explanation for the item-level null.

### Persistence probe (`dev`, every layer at every position)

| rung | own pos | after own | last pos |
|---|---|---|---|
| digits_bare | 1.00 | 0.83 | 0.00 |
| words_bare | 1.00 | 0.67 | 0.00 |
| words_long (chat) | 1.00 | 0.88 | 0.00 |
| words_long_then_text (chat) | 1.00 | 0.88 | 0.00 |
| keep_track_shape (chat) | 1.00 | 0.92 | 0.21 |

`own = 1.00` everywhere is the current-token diagonal and means nothing.
`after own` is presence at *any* position past the item, so it includes
distance 1 and conflates a short local tail with held state — which is why the
probe now buckets by distance (committed, not yet run). The informative column
today is `last pos ≈ 0.00`: items do not survive to the end of the list.

---

## 3. The Neuronpedia evidence that redirected us

Our prompts were run on **Qwen3.6-27B** through Neuronpedia's slice view. The
J-lens readout is dominated by category vocabulary, with the remembered items
far below it:

- `words_long_then_text`: `animals` 668, `colors` 450, `categories` 401,
  `animal` 400, `words` 392, `color` 341, `fruit` 304, `musical` 302.
- `keep_track_shape`: `animals` 277, `word` 260, `animal` 228, `color` 183,
  `fruit` 172, `categories` 170, and **`apple` at 200 in a prompt where apple
  never occurs.**
- The actual tracked items, searched directly: `fox` 112, `grape` 59.

The decisive rung is the digits. For the sequence **4, 7, 2, 9, 1, 3** the
readout lists:

| in the sequence | | not in the sequence | |
|---|---|---|---|
| 3 | 346 | **5** | **375** |
| 7 | 344 | **8** | 316 |
| 9 | 317 | **6** | 287 |
| 4 | 287 | | |
| 2 | 265 | | |

`5` outranks **every digit that was actually in the sequence**, and `8` and `6`
outrank `4` and `2`. So "the digits are in the J-space" is true but means
*digit-ness*, not the remembered values.

**Why this makes our item-level null credible rather than suspicious.** The
natural worry about a null is that the readout is broken. This evidence says it
isn't: the lens is reading something strong and legible at these positions —
just at a level of abstraction above the one we were probing. An identical
pattern appears at both ends: `apple` present at 200 without occurring, and
non-sequence digits outranking sequence digits. Our item-level null and the
Neuronpedia positive are the same observation.

One further detail, and it is the most predictive thing we have for the pending
run. In `keep_track_shape` the tracked categories were **animal** and **fruit**,
and animal was the queried one. The readout gives `animal`/`animals` 228/277
(tracked *and* queried), `fruit` 172 (tracked, not queried), and `color` 183
(**untracked**). Fruit does not beat colour. That parallels the item-level
result exactly — queried elevated, non-queried indistinguishable from
untracked — one level up. It is a single prompt, so treat it as weak evidence,
but it is directly on point.

---

## 4. Implementation history

Several apparent results were measurement artifacts. The sequence is recorded
because it is the reason to trust the current numbers more than the earlier
ones, not as a confessional.

| symptom | actual cause |
|---|---|
| Q1 = 0% | The model answers in markdown. Argmax at the prefill was `" **"`, with the target at **rank 2** in 4 of 6 sampled streams. Fixed by `"Answer in one word."` (the paper's own R2 phrasing) and by reporting `expected_rank`. |
| Q2 = 21%, 89% parse | 64 new tokens truncated the answer after 1–2 categories; it was *correct* on those. Raised to 320 → 80% at 100% parse. |
| `end_of_stream` = 0.00 at every layer | The readout sat on the `.` **after** the stream, where the prompt continues into instruction text, so the model is disposed to say `" Remember…"`. The paper's capacity protocol reads at commas **inside** the list. Replaced with `in_stream` (comma after the second-to-last word); `stream_end` retained as a contrast and still reads 0.00, as predicted. |
| band selection | `choose_band` grouped consecutive entries of a *list* rather than consecutive *layer indices*, so a strided R1 sweep could return a band covering layers R1 never measured — silently, since every layer is a valid lens input. Now refuses a gapped sweep. |
| `rank_wordlike` | Logged for a documented reason (D12) and then never read by any analysis function. Now threaded through all of them; it changed nothing, which is itself the useful result. |
| probe `present@anywhere` = 1.00 | Included each item's own position — the current-token diagonal. Now split into own / after-own / by-distance. |

Also, in passing: the `instrument` pool had only 8 single-token words under
Qwen's tokenizer (fixed by screening against the real `vocab.json`; `n` per
category lowered to 10), and `" 4"` is two tokens on this tokenizer, so digits
need the bare form. `dev` and `prod` share a byte-identical tokenizer, so one
exemplar list serves both.

---

## 5. Limitations, stated plainly

**There is no load effect to anchor against.** Accuracy is flat across
`C_t = 2…6`. The whole stream stays in the context window, so the model
re-reads rather than maintains. The keep-track task measures working memory in
*humans* because they cannot re-read. This weakens the item-level null
considerably: "task state the model is maintaining" may not be a thing that
exists here to be found, and it bears directly on the cognitive-load half of
the original question.

A direct consequence worth stating separately: given a behaviourally competent
model with no load effect, the *expected* result for any capacity measure is a
null. **Our capacity null therefore carries almost no evidential weight in
either direction.** It is not evidence against a capacity-limited J-space; it
is evidence that this task does not load one.

Also: `n = 20` per `C_t` with sem ≈ 0.1 on the capacity counts; the band is
late (layers 23–26, 72–81% depth) though we checked 8–26 for an earlier peak
and found none; the matched `untracked_final` control thins from 3.4
items/stream at `C_t = 2` to **0.1 at `C_t = 6`**, so the cleanest selectivity
contrast is only well-powered at `C_t = 2–4`; probes and the k-free capacity
measure are deferred (D2), so capacity claims are "stable across k and absent
in the logit-lens control", not "k-free"; **R2 was never built**, so we have no
replication of the paper's reportability result; and v1 cannot falsify the
*broad* output-staging hypothesis, because the Jacobian targets future
residuals by construction, so anything held because it might matter later is in
scope definitionally.

---

## 6. Pre-committed interpretation of the `prod` results

The row to read is `label_tracked` vs `label_untracked` vs `label_absent` at
`in_stream` in the `HEADLINE` table, with the item-level rows alongside.

| # | Outcome | What it implies | What we do |
|---|---|---|---|
| **a** | Tracked category labels clearly above untracked, both above `label_absent` | The J-space holds task state at the **category** level. The tracking assignment itself is represented. Direction is fruitful and tractable. | Proceed. Make the label contrast the primary measure, add probes for the k-free capacity question, build R2 for the replication line. |
| **b** | Tracked ≈ untracked, both clearly above `label_absent` | The J-space holds the task's **vocabulary** but not the assignment. Real but weaker: it is representing "this is a categorisation task about animals and colours", not "I am tracking animals". | Keep the paradigm, change the manipulation: the contrast has to be between two tasks over the *same* words, so vocabulary is held constant and only the instruction differs. |
| **c** | Item-level null replicates **and** labels show nothing above `label_absent` | For this paradigm the J-space is an output-staging buffer. The queried item and its competitors are there; nothing else is. | Redirect the paradigm. Need a task where held content must be *used* without being emitted — the proposal's "attended but unspoken" family (F4), e.g. multi-hop where the bridge entity is never said. |
| **d** | `prod` shows non-queried targets well above floor | The 4B null was scale-dependent. | Re-run everything at 27B, demote `dev` to a smoke tier only, and re-examine whether linear vs full attention also matters (`Qwen3-32B` is the dense control, lens available). |

**Most likely, in my judgement: (b), with (c) a close second.** The reason is
the `keep_track_shape` readout in §3 — `fruit` (tracked, 172) does not beat
`color` (untracked, 183), while the queried `animal` is clearly elevated at
228–277. And in `words_long_then_text`, where the prompt names *no* categories
at all, every category label is still present at 300–668. That is the signature
of task-vocabulary-without-assignment. I would be mildly surprised by (a) and
quite surprised by (d), since the Neuronpedia data *is* 27B and already shows
the category-over-item pattern.

Flagging honestly: this prediction rests on one prompt viewed through a UI, with
counts whose exact definition (cells in the layer × position grid, presumably
including own-position) we have not verified. It should not be treated as
quantitative. One piece of the §3 argument is robust to that worry, though:
`apple` at 200 and the non-sequence digits `5`/`8`/`6` **never occur in their
prompts**, so they have no own-position cells at all. Whatever `Count` measures,
those items cannot be scoring on the diagonal.

---

## 6b. What the `dev` full run actually showed

`n = 30` streams per `C_t`, 150 streams, 906,300 rows. Behaviour unchanged:
Q1 62% top-1, median expected-rank 1, top-5 96%; Q2 81% at 100% parse; still
flat across `C_t` (57 / 57 / 63 / 73 / 60).

**At the answer position the J-space is query-scoped, not task-scoped.**

| group | present | n |
|---|---|---|
| queried target | **0.707** | 150 |
| queried category's replaced items | **0.460** | 300 |
| **queried category label** | **0.105** | 247 |
| other tracked categories' labels | **0.000** | 717 |
| untracked labels | 0.005 | 986 |
| label floor (`flower`, `metal`, …) | 0.000 | 1200 |
| non-queried targets | 0.076 | 450 |
| word floor | 0.064 | 1200 |

`label_tracked = 0.000` on n = 717 is a hard zero — binomial 95% upper bound
≈ 0.4% — against `label_queried` at 10.5%. So the J-space holds the queried
category, its current word, and its superseded alternatives, and contains
nothing about the other tracked categories at either level of abstraction. The
item-level null replicated exactly (0.076 vs 0.064).

**Against the pre-committed table: this is (c), with a refinement.** Outcome
(b) is ruled out — the ordering is *inverted* from the prediction in §6, with
`label_untracked` (0.113) above `label_tracked` (0.063) at `in_stream` and
tracked labels at zero at the answer position. The refinement (c) did not
anticipate: the labels do carry signal, but only for the queried category. The
prediction in §6 was wrong; recording that here rather than reframing it.

**A confound in our own design, limiting the `in_stream` rows.** `tail_guard`
forces the last three stream items to be untracked distractors, so at
`in_stream` — the comma near the tail — the local continuation context is
untracked-heavy by construction. That inflates `label_untracked` and
`untracked_superseded` there; the latter rises 0.126 → 0.253 across `C_t` while
its n falls 348 → 87, which is the same recency effect. The answer-position
comparison sits far from the tail and is unaffected. Read `in_stream` as "local
continuation dominates, no task state visible" rather than as a quantitative
tracked-vs-untracked contrast.

**Revised reading of the question.** The J-space here looks like a
query-scoped retrieval buffer: on receiving a question it assembles the asked
category and its candidate members, and it carries no representation of the
remaining task state. That is a sharper claim than "output staging", because
the *competitor* items (0.460) are present without being emitted — so it is
not merely the next token — while the other tracked categories are absent
entirely.

---


Conditional on the above. If **(a)**: how many category labels can be held at
once, does the capacity plateau survive the k sweep, and does self-report track
label presence (`q2_item_correct ~ label_present + recency + C_t`, which needs
no probe and is already collectable)? If **(b)**: the key design is two tasks
over identical word lists differing only in instruction — that isolates
assignment from vocabulary, and it is the clean version of the selectivity
test. If **(c)**: move to attended-but-unspoken, where the deflationary and
workspace hypotheses actually diverge, and treat this run as the negative
control that justified the move.

In all three branches the same unresolved question sits underneath: whether a
transformer with the full stream in context has any "maintained state" to find,
or whether retrieval-on-demand makes the human working-memory framing the wrong
import. A paradigm where the content must be held *across a context boundary*
would settle it, and nothing we have built so far tests that.

---

## 7. Forced demand and retro cue (Experiment 2, `prod`, Qwen3.6-27B)

Spec: `specs/forced-demand-and-retro-cue.md`. Same streams as single_cue (joined on
`(stream_id, c_t)`), band (38, 54), `rank_wordlike`, J-lens, in-band band-min,
`k = 25`, answer position. forced_demand: 150 streams; retro_cue: 60 streams. Fallback question
not used (D18).

### 7a. forced_demand (Experiment 2a): "list all, alphabetical"

`t1…t6` = presence of the target that is alphabetically 1st…6th (the intended
emission order). Logit-lens column = targets present per stream under the
logit lens.

| C_t | n | targets / stream (sem) | t1 | t2 | t3 | t4 | t5 | t6 | replaced | tracked labels | untracked labels | untracked words | word floor | label floor | logit-lens targets / stream | Q1 top-1 | Q1 median rank | list correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 30 | 0.867 (0.115) | 0.467 | 0.400 | | | | | 0.417 | 0.263 | 0.041 | 0.002 | 0.096 | 0.000 | 0.667 | 33% | 2 | 37% |
| 3 | 30 | 0.967 (0.112) | 0.433 | 0.400 | 0.133 | | | | 0.261 | 0.194 | 0.028 | 0.009 | 0.054 | 0.000 | 0.367 | 43% | 2 | 20% |
| 4 | 30 | 0.700 (0.098) | 0.300 | 0.300 | 0.067 | 0.033 | | | 0.171 | 0.134 | 0.078 | 0.011 | 0.062 | 0.000 | 0.333 | 17% | 3 | 0% |
| 5 | 30 | 0.800 (0.088) | 0.300 | 0.167 | 0.167 | 0.133 | 0.033 | | 0.120 | 0.119 | 0.034 | 0.015 | 0.062 | 0.000 | 0.300 | 23% | 2.5 | 0% |
| 6 | 30 | 0.633 (0.131) | 0.300 | 0.100 | 0.067 | 0.100 | 0.033 | 0.033 | 0.094 | 0.129 | 0.011 | 0.006 | 0.050 | 0.000 | 0.067 | 30% | 3 | 0% |

single_cue, same streams, same position:

| C_t | n | queried target | non-queried targets | replaced (queried) | queried label | other tracked labels | untracked labels | word floor | label floor |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 30 | 0.767 | 0.000 | 0.400 | 0.653 | 0.000 | 0.000 | 0.050 | 0.000 |
| 3 | 30 | 0.733 | 0.000 | 0.317 | 0.543 | 0.000 | 0.000 | 0.029 | 0.000 |
| 4 | 30 | 0.767 | 0.000 | 0.367 | 0.460 | 0.000 | 0.000 | 0.058 | 0.000 |
| 5 | 30 | 0.667 | 0.000 | 0.500 | 0.608 | 0.000 | 0.000 | 0.067 | 0.000 |
| 6 | 30 | 0.767 | 0.000 | 0.317 | 0.431 | 0.000 | 0.000 | 0.042 | 0.000 |

Outcome: **B, partial** — about 1 target per stream, flat-to-falling in `C_t`, while list accuracy falls 37% → 0%; t1 is preferred only at `C_t ≥ 4` (t1 ≈ t2 at `C_t = 2–3`), so C is not excluded.

### 7b. retro_cue (Experiment 2b): the cue switches

| position | C_t | n | target A | target B | label A | label B | replaced A | replaced B | other targets | untracked words | word floor | label floor | a1 correct | Q(B) top-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 30 | 0.767 | 0.000 | 0.653 | 0.000 | 0.400 | 0.000 | | 0.006 | 0.050 | 0.000 | 90% | |
| 1 | 4 | 30 | 0.767 | 0.000 | 0.460 | 0.000 | 0.367 | 0.000 | 0.000 | 0.003 | 0.058 | 0.000 | 93% | |
| 2 | 2 | 30 | 0.000 | 0.833 | 0.000 | 0.630 | 0.000 | 0.633 | | 0.002 | 0.037 | 0.000 | | 100% |
| 2 | 4 | 30 | 0.000 | 0.867 | 0.000 | 0.600 | 0.000 | 0.450 | 0.000 | 0.008 | 0.029 | 0.000 | | 97% |

Outcome: **as predicted** — at position 2, B is present at about the single_cue queried rate and A falls to 0.000, below the word floor; no lingering content. Position 1 reproduces single_cue exactly.
