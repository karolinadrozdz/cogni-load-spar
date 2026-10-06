# Derived state (Experiment 3): derived vs copyable state (running counts)

Purpose: test whether the J-space holds task state that exists only in the residual stream (derived) while remaining blind to state the model can re-read from context (copyable). single_cue and forced_demand showed copyable state absent. This experiment puts the same task structure over state that is never written down, and the same state written down, on identical streams. Signal is the goal. One GPU-hour.

Audience: a Claude Code session in `cogni-load-spar`. Read `specs/forced-demand-and-retro-cue.md` ground rules; they all apply. Reuse `stimuli.generate`, `prompts.render_chat`, `readout.read`, `scoring.score_q1`, `analyze.headline`. New code: two prompt templates, stream truncation with count labelling, digit scoring, one analysis table. No refactors. Do not apply the audit cut list.

## Task

Same streams as single_cue and forced_demand (same seeds, `c_ts: [2, 3, 4, 5, 6]`, 30 streams each). Each stream is **truncated at a position p drawn from the stream RNG, uniform on [8, 21]**, so the per-category counts at p vary from 0 to 3 and the queried answer is not constant. The tracking instruction becomes:

> Track these categories: {tracked}. Keep a running count of how many words from each tracked category have appeared.

The question, in the same user turn after the stream:

> How many {category} words have appeared so far? Answer with a single digit.

Prefill `Answer: ` **with a trailing space**, because `" 3"` is two tokens on this tokenizer and the bare digit is what must be read and scored. Verify in the `dev` smoke that the first generated token is a bare digit; if it is not, prefill `Answer:` and score the second generated token instead. Record which in DECISIONS (D21).

Queried category: `queried_category` from the stream, as in single_cue. Thinking off, asserted.

## Two arms, identical streams

| arm | stream text | what state exists in context |
|---|---|---|
| **derived** | `cat, pear, dog, red, …` | nothing; counts must be carried in the residual |
| **copyable** | `cat (animal 1), pear (fruit 1), dog (animal 2), red, …` — the running count written after every **tracked** word; untracked words unannotated | every count is on the page |

Same instruction and question in both arms. The comma after each word is the in-stream readout position in both arms (in the copyable arm, the comma follows the closing parenthesis).

## Readout

At every in-stream comma and at the answer position. Band, `k = 25`, `rank_wordlike`, J-lens with logit-lens column, as before.

Target tokens for a count c: the bare digit `c` **and** the number word (`zero`…`three`); present if either is. Groups, scored per position:

| group | definition |
|---|---|
| queried count | the queried category's count at that position (answer position only; in-stream there is no query yet) |
| tracked counts | each tracked category's current count at that position (in-stream: all tracked categories; answer position: the non-queried ones) |
| untracked counts | each untracked category's current count, **only where that value is not also a tracked category's count** |
| count floor | digits 5–9 and their number words (never a count) |
| stale counts | for each tracked category, its count one step before the last increment (e.g. `1` when the current count is `2`) — the superseded value |

Collisions: two tracked categories with the same count share a token; count presence per category, not per token, and note the collision rate in the table.

Behaviour: `score_q1` with `expected` = the queried count's digit token; report top-1 and median rank per arm and `C_t`. Secondary: rank of the correct digit among digits 0–3 only.

## Table (the deliverable)

Per arm × `C_t`: top-1 accuracy; at the answer position — queried count, non-queried tracked counts, untracked counts, stale counts, floor; in-stream pooled over commas — tracked counts, untracked counts, stale counts, floor; logit-lens column for tracked counts. Two tables, same columns, one per arm.

## Pre-committed predictions

| outcome | derived arm | copyable arm | reading |
|---|---|---|---|
| A | tracked counts present in-stream and non-queried counts present at the answer position, both well above untracked and floor; accuracy falls with `C_t` | tracked counts at floor, as the words were in single_cue; accuracy at ceiling | J-space holds derived state, blind to copyable state. The capacity question is now well-posed and the count-vs-`C_t` curve is its first measurement |
| B | tracked counts at floor in both positions | at floor | J-space holds only what is being derived *for the next output*; carried state is invisible even when it is not in context. Stronger deflationary result than single_cue |
| C | floor digits present at the same rate as tracked counts | — | generic digit-ness; the count readout is not item-specific. Report and stop |
| D | derived arm accuracy at chance at `C_t ≥ 3` | — | the model cannot count in one pass; restrict the readout claims to `C_t = 2` and the positions where accuracy holds |

A and B are both results. D is likely at large `C_t` and does not invalidate the in-stream rows at positions where the running counts are small.

## Order and budget

1. Prompt templates, truncation + count labelling, digit scoring, analysis table. Smoke on `dev`: 5 streams at `c_t = 2`, both arms; print one rendered prompt per arm, the generated token, and the count-group table.
2. `prod`, `derived` arm, all `C_t`. Then `copyable` arm.
3. Append §8 to `FINDINGS.md`: the two tables, one line naming the outcome. No prose.

`prod`: 2 arms × 150 streams, one generation each, readout at ~20 positions per stream in one forward pass. About one GPU-hour.

## Do not

- Do not change seeds, band, `k`, or exemplar pools.
- Do not annotate untracked words in the copyable arm (the contrast is tracked state on the page vs not).
- Do not build a parser; score the first generated token.
- Do not run single_cue, forced_demand or retro_cue again.
- Do not interpret past the table.

## DECISIONS.md entries

- D21: prefill convention for the digit answer and the verified first-token behaviour.
- D22: truncation range [8, 21] and the count-collision rule for untracked counts.
- D23: copyable-arm annotation format and that it is applied to tracked words only.
