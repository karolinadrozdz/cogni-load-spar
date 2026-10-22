# Spec: does the model ever judge its answer without re-solving the question?

## Why
On Qwen3-Next-80B-A3B-Thinking, every Phase-2 abstention trace we have read re-derives an answer
from scratch and abstains because that answer is "not among the options"; every "No chance" in the
verbal pass re-derives an answer and rates the earlier one against it. That is recompute-and-compare,
not a reading of confidence in the first answer. The question is whether the other kind exists at
all — traces where the model judges its answer or the options *without* producing its own candidate —
and, if it does, whether that kind tracks correctness differently. If the second kind is absent, the
paper's two-stage architecture (confidence representation → threshold) has no second stage under
thinking; it has a second first stage.

## Inputs
`results/80b-thinking/phases.csv` and `results/80b-thinking/verbal.csv` (columns: `question`,
`choice_1..4_arranged`, `gt_position`, `final_answer`, `is_correct`, `is_nota`, `reasoning`,
`reasoning_tokens`, `truncated`; verbal.csv adds `confidence_class`, `confidence_output`).
Exclude `truncated == 1`.

Traces to label (≈ 650):
- A. Phase 2, `is_nota == 1`: all (≈ 207).
- B. Phase 2, `is_nota == 0`: a random sample of 200 (seed 0), stratified half correct / half incorrect.
- C. Verbal pass, `confidence_class == "No chance"`: all (≈ 98).
- D. Verbal pass, `confidence_class` in {"Highly likely", "Almost certain"}: random 100, stratified by `is_correct`.

## The labels (one per trace)
Read the whole trace. Assign exactly one `kind`:

- **RESOLVE** — the trace works out its own answer to the question (recalls or derives a specific
  fact: "Oka was a mathematician", "founded in 1951", "he wrote none") and then compares that
  candidate to the options or to the previous answer. The decision cites the candidate.
- **JUDGE** — the trace evaluates the listed options or the previous answer without asserting its own
  candidate: "I don't recall this", "2 and 4 both seem plausible and I can't separate them",
  "this looks right but I'm not certain", "the instructions say pick 5 if unsure". The decision cites
  uncertainty or plausibility, not a competing fact.
- **MIXED** — a candidate is asserted but the decision is driven by stated uncertainty about it
  ("I think it's 1951 but I'm really not sure, so 5"), or the trace alternates without settling.
- **OTHER** — none of the above (formatting confusion, misread question, degenerate loop). Rare.

And these fields:
- `candidate`: the model's own asserted answer in the trace, verbatim short form, or empty for JUDGE.
- `candidate_in_options`: 1–4 if the candidate matches a listed option (string or obvious paraphrase), else 0.
- `candidate_correct`: 1 if the candidate matches the ground-truth option text, else 0, empty if no candidate.
- `decision_reason`: one of `not_listed` (candidate isn't among the options), `question_wrong`
  (trace argues the question itself is flawed), `uncertain_recall`, `tie_between_options`,
  `instruction_license` (abstains by citing the prompt's "choose 5 if unsure" without other grounds),
  `confident_pick`, `other`.
- `uncertainty_phrases`: count of first-person uncertainty expressions ("not sure", "don't know",
  "can't recall", "I think", "maybe") in the trace. Cheap proxy to sanity-check the judge.
- `rationale`: one sentence quoting the decisive span of the trace.

## How to label
An LLM judge (Claude via the API) with the rubric above as the system prompt and one trace per call,
returning JSON with exactly those fields. Run it twice with two different models (or two different
prompt phrasings) and report agreement on `kind` (Cohen's κ; target ≥ 0.75). Where they disagree, a
third call with both rationales shown breaks the tie, and the row is flagged `disagreed = 1`.
Then a human reads 40 rows (10 per kind, random) and records agreement; report it. Do not tune the
rubric to the traces after reading the results — if the rubric needs changing, change it, relabel
everything, and say so.

Output: `results/80b-thinking/trace_labels.csv` (all fields above plus `set` A–D, `question_index`,
`phase`, `is_correct`, `is_nota`, `confidence_class`), and `trace_labels_summary.json`.

## What we learn, and how we would know
1. **Does JUDGE exist?** Report the share of each kind per set. If JUDGE + MIXED is under ~5% of
   abstentions, the model essentially never abstains without recomputing; the "second stage" is
   absent under thinking. If it is substantial, the next two checks matter.
2. **Which kind is the better error detector?** For sets A+B: P(Phase-1 answer wrong | abstained, kind)
   and P(abstained | Phase-1 wrong, kind). If JUDGE abstentions are more often on wrong answers than
   RESOLVE abstentions, something other than recomputation is carrying information about correctness —
   the closest thing to a confidence readout we could find. If RESOLVE is better, error detection here
   is just solving the problem twice.
3. **Anatomy of RESOLVE failures.** Among RESOLVE abstentions and "No chance" ratings, split by
   (`candidate_correct`, `candidate_in_options`):
   - candidate correct and listed, yet abstained or rated "no chance" → a *matching* failure
     (the model had the answer and failed to align it with the option text or its own earlier pick);
   - candidate incorrect → a *knowledge* error in the second solve, overriding a correct first solve
     when `is_correct == 1`;
   - `decision_reason == question_wrong` → the model disputes the item.
   We already know 47% of "No chance" ratings sit on correct answers; this says which failure they are.
4. **Instruction licensing.** Share of abstentions with `decision_reason == instruction_license`. If
   high, the Phase-2 framing ("some questions have no correct answer") is doing the work, and the
   reframed-prompt run is the control.
5. **Cross-check with the checkpoint probe** (`checkpoints.csv`, when available). RESOLVE traces should
   show the forced early-exit answer committing to the candidate before the decision; JUDGE traces
   should show no stable commitment and lower forced-answer probability throughout. If the labels and
   the probe disagree systematically, one of them is wrong — report which traces.

## Deliverable
`trace_labels.csv`, `trace_labels_summary.json`, and a half-page note: kind shares per set, the two
error-detection rates by kind with counts, the RESOLVE failure split, agreement statistics, and five
quoted examples (two RESOLVE, two JUDGE if any exist, one MIXED). Results only; no narrative of the
process.
