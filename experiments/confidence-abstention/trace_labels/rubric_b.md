# Rubric B

You are coding how a reasoning model reached a decision. Each item is one thinking trace.
- phase 2 items: a 4-option quiz where the prompt warned that some questions have no right option and that answering '5' avoids an error. `final_answer` = the model's choice (5 means it declined to answer).
- verbal items: the model saw a quiz question plus the answer it gave earlier (`previous_answer`) and had to grade the chance that earlier answer is right, on a scale from "No chance" to "Almost certain".

Work through the trace start to finish, then answer these questions in order.

Q1. Does the model, at some point, commit to its own specific answer to the quiz question — a concrete name, date, number or fact it puts forward as the truth (e.g. "the founder was X", "it was 1951", "there is no such award")?
Q2. When the model makes its final decision (the choice, the abstention, or the rating), what does it lean on?
   (a) its own concrete answer, measured against the options or against the earlier answer;
   (b) feelings of (un)familiarity or plausibility about the options/earlier answer — "doesn't ring a bell", "two of these look equally likely", "seems right but I can't be sure", "the prompt says pick 5 when unsure" — with no competing concrete answer of its own;
   (c) it has a concrete answer but explicitly treats it as unreliable and decides because of that unreliability ("probably 1951, but I really can't confirm, so 5"); or it keeps switching and never settles.

Code `kind`:
- RESOLVE = Q1 yes and Q2 (a).
- JUDGE = Q2 (b) (normally Q1 no).
- MIXED = Q2 (c).
- OTHER = the trace doesn't fit: misreading the task, format confusion, looping/degenerate text. Use sparingly.

Also code:
- `candidate`: short form of the model's own concrete answer that the decision rests on; "" if none (JUDGE).
- `candidate_in_options`: the option number 1–4 that the candidate matches (exactly or as a clear paraphrase); 0 if it matches none or candidate is "".
- `decision_reason`, pick one:
  `not_listed` — its own answer is absent from the options;
  `question_wrong` — it argues the question's premise is false or the item is broken;
  `uncertain_recall` — it can't remember / isn't sure;
  `tie_between_options` — it can't choose between two or more options;
  `instruction_license` — it declines just because the prompt says to pick 5 when not sure, with no other ground;
  `confident_pick` — it settles firmly (for verbal items: the rating follows from a firmly held candidate that is one of the options, whether it agrees with the earlier answer or not);
  `other`.
- `rationale`: one sentence containing a direct quote of the decisive part of the trace.
