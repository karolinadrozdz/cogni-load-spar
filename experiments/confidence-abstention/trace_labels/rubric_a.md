# Rubric A

Each item is a reasoning trace from a reasoning model. Two task types:
- phase 2: the model answered a 4-option multiple-choice question under the instruction "some questions have NO correct answer among the options. If you don't see a clearly correct answer, choose '5'". `final_answer` is what it chose (5 = abstained).
- verbal: the model was shown the question, options, and its own earlier answer (`previous_answer`) and asked to rate how likely that answer is correct ("No chance" ... "Almost certain").

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
- `candidate`: the model's own asserted answer in the trace, verbatim short form, or "" for JUDGE.
  If the trace asserts several, give the one the final decision rests on.
- `candidate_in_options`: 1–4 if the candidate matches a listed option (string or obvious paraphrase), else 0. Use 0 when candidate is "".
- `decision_reason`: one of `not_listed` (candidate isn't among the options), `question_wrong`
  (trace argues the question itself is flawed), `uncertain_recall`, `tie_between_options`,
  `instruction_license` (abstains by citing the prompt's "choose 5 if unsure" without other grounds),
  `confident_pick`, `other`.
  For verbal items, use `confident_pick` when the rating follows from a confidently held candidate
  (whether it agrees or disagrees with the previous answer) and the candidate is a listed option.
- `rationale`: one sentence quoting the decisive span of the trace.
