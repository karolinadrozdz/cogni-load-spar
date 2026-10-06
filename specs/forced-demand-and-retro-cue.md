# Forced demand and retro cue (Experiment 2): J-space content under forced simultaneous demand

Purpose: a fast follow-up to single_cue (Experiment 1) that puts the readout at a position where the two hypotheses disagree. single_cue read the J-space after a question had already cued one category; at that position both hypotheses predict the same thing (only the cued content matters for the future), and that is what we saw. This experiment reads it at a position where every tracked item is needed for the next token. Signal is the goal, not a publishable result. One GPU-afternoon.

Audience: a Claude Code session working in `cogni-load-spar` on branch `dev/aveizi/jspace-evaluation`. Read `README.md`, `DECISIONS.md` (D1–D17), `FINDINGS.md` §6c, and `AUDIT.md` first.

## Ground rules

- **Reuse, don't rebuild.** Everything needed exists: `stimuli.generate`, `prompts.render_chat`/`build_q1`, `readout.read`, `scoring.score_q1`, `analyze.headline`, the `prod` band on the Modal volume. Each experiment below is a new prompt builder, a new CLI stage, and one analysis table. If a change needs a new module or a new dataclass, it is probably wrong.
- **Minimal diff.** No refactors, no renames, no abstraction added "for later." Do not apply `AUDIT.md`'s cut list now; it is for after the collaborator meeting. The one audit item to apply: pin the lens to `repo_sha` (`registry.py` line that passes `revision=`; delete the branch name from `models.yaml`). Nothing else from the audit.
- **Result over edge cases.** If the model formats an answer oddly, measure the rank of the expected token instead of parsing. If a stream is unusable, drop it and count it. Do not build parsers, fallbacks, or retries for cases that have not occurred.
- **Same streams, same band, same readout.** Reuse the single_cue `prod` configuration exactly: `c_ts: [2, 3, 4, 5, 6]`, `n_streams: 30`, the same seeds, the band from `results/prod/band.json`, `k = 25`, `rank_wordlike` primary with full-vocab rank alongside, logit-lens control column. Rows must join to single_cue rows on `(stream_id, c_t)`. The single_cue numbers are the comparison arm; do not rerun them.
- **Reproducible.** One config block per experiment, containing only keys the code reads. Every prompt template goes into the run manifest verbatim. Pre-committed predictions are in this spec; append results to `FINDINGS.md` as §7 in the same table format as §6c, no narrative.
- **Smoke on `dev` first** (5 streams at `c_t = 2`), eyeball one rendered prompt and the generated answer, then run `prod`.

## forced_demand (2a, required)

**Question.** When the next output token depends on all `C_t` tracked items at once, does the J-space hold all of them, or one?

**Prompt.** Same stream turn as `build_q1`, with the question replaced by:

> List the most recent word for each tracked category, in alphabetical order. Answer with the words only, separated by commas.

Assistant prefill `Answer:` (same convention as Q1, D13). Thinking off, asserted. The alphabetical instruction must be in the user turn: at the readout position the model has to know that all `C_t` targets are needed to produce the first word.

**Generation.** Greedy, `max_new_tokens = 64`. Keep the raw text.

**Behaviour.** Primary: rank of the alphabetically-first target's token at the first generated position (`score_q1` with `expected = min(targets)`); report top-1 and median rank per `C_t`. Secondary, only if it falls out of the raw text with a split on commas: fraction of streams where the emitted list equals the sorted targets. Do not write a parser beyond `split(",")` and `strip()`.

**Readout.** Last prompt token (the `:` of `Answer:`), band-min rank ≤ 25, exactly as the single_cue answer position. Groups, each against its floor:

| group | definition |
|---|---|
| target, by emission order | each tracked category's final word, labelled 1…`C_t` by its alphabetical position |
| replaced items | superseded words of every tracked category |
| tracked labels | the `C_t` category names |
| untracked labels | the other category names in the stream |
| untracked words | all untracked stream words |
| word floor / label floor | as in single_cue |

**Table (the deliverable).** Per `C_t`: mean number of targets present per stream (with sem), presence rate of target 1 / 2 / … / `C_t` by emission order, replaced-item rate, tracked-label rate, untracked-label rate, floors; plus the same for the logit-lens column. One row per `C_t`, one table. Alongside it, the single_cue numbers for the same streams.

**Pre-committed predictions.**

| outcome | targets present per stream | behaviour vs `C_t` | reading |
|---|---|---|---|
| A | ≈ `C_t` at all `C_t` (all targets present) | flat or gently falling | J-space tracks output influence; no capacity property at this position |
| B | ≈ 1–2, flat in `C_t`; the present one is target 1 (next to be emitted) | falls with `C_t` | J-space is a serial focus of attention; the comparison over the rest happens outside it |
| C | ≈ 1–2 but *not* preferentially target 1 | any | J-space holds a sample, not the emission queue; worth a second look before interpreting |
| D | ≈ 0 for all targets, category labels present | any | the readout at this position is category-level only; report as such |

Either A or B is the result we want. C and D are reported, not rescued.

**Fallback.** If the `dev` smoke shows the model cannot do the alphabetical task at `c_t = 2` (top-1 < 20%), switch the question to "Which tracked category's most recent word comes first alphabetically? Answer with the category name." and set `expected` to that category's label token. The readout groups do not change. Record the switch as a decision.

## retro_cue (2b, required if forced_demand is running; otherwise optional)

**Question.** When the cue changes, does the J-space content change with it?

**Design.** `c_t ∈ {2, 4}`, 30 streams each, same seeds. Two tracked categories A ≠ B drawn from the stream RNG (A is `queried_category`; B is the next tracked category in the stream's order). Two user turns:

1. Stream turn + "What was the most recent {A}?" — prefill `Answer:`, generate 1 token, keep it as `a1`.
2. Assistant turn `Answer: {a1}`; user turn "What was the most recent {B}?" — prefill `Answer:`.

Thinking off at both generation prompts, asserted. Render the second prompt as a three-message chat exactly as `build_q3` did (the pattern is in `render_chat`; the earlier assistant turn is history, so the template renders it without a think block).

**Readout.** Both answer positions. Groups: target A, target B, label A, label B, replaced items of A, replaced items of B, non-cued targets, untracked words, floors.

**Table.** Two rows (position 1, position 2) × those groups, per `c_t`.

**Predictions.** At position 2: B present at the single_cue queried-target rate (~0.7), A at or near floor. If A persists at position 2 well above floor, that is a lingering-content result and also interesting; report it either way.

## 2c. Causal check (optional; only after forced_demand and retro_cue have produced tables)

Swap the queried target with one of its replaced items in the J-space at the single_cue answer position (every band layer, that position only), using the paper's formula already implemented in `swap.py` (`V = [v_s, v_t]`, `c = V⁺h`, `h′ = h + V(σ(c) − c)`). Measure answer-change rate over 30 streams at `c_t = 2`. This turns the "output buffer" reading into a causal one. Skip it if time is short; the two tables above are what the meeting needs.

## Order and budget

1. Pin the lens sha. Add the forced_demand prompt builder, CLI stage, analysis table. Smoke on `dev`. Run `prod` forced_demand.
2. While `prod` forced_demand runs: add retro_cue. Smoke on `dev`. Run `prod` retro_cue.
3. Append §7 to `FINDINGS.md`: the forced_demand table with the single_cue column beside it, then the retro_cue table. Same format as §6c. No prose beyond one line naming which pre-committed outcome each table matches.
4. 2c only if 1–3 are done.

`prod` budget: 150 streams × 1 generation + readout for forced_demand, 60 streams × 2 for retro_cue. Under two GPU-hours including model load.

## Do not

- Do not rerun single_cue or touch its results.
- Do not add per-position readouts, probes, PCA/random controls, or R2.
- Do not change `tail_guard`, exemplar pools, seeds, band, or `k`.
- Do not write new tests beyond one per new prompt builder asserting the rendered string ends in the thinking-off marker plus `Answer:`.
- Do not refactor, rename, or apply the audit cut list.
- Do not interpret beyond the prediction table. The writing happens after the meeting.

## DECISIONS.md entries to add

- D18: forced_demand question wording and the requirement that the alphabetical instruction precede the readout position; the fallback wording if used.
- D19: retro_cue cue selection rule (A = `queried_category`, B = next tracked in stream order) and the three-message render.
- D20: lens pinned to `repo_sha`; branch name removed.
