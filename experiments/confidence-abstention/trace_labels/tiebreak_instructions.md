# Tiebreak instructions

Two judges disagreed on `kind` for each item in BATCH. Each line has the trace fields (`id`, `phase`,
`question`, `options`, `previous_answer`, `final_answer`, `reasoning`) plus `judge_1` and `judge_2`
(each judge's full label and rationale).

1. Read RUBRIC (the rubric judge_1 used; judge_2 used an equivalent rewording).
2. For each item, print it with python (one item at a time), read the full trace yourself, weigh both
   rationales against the trace, and decide. You may agree with either judge or choose differently.
   No keyword scripts or heuristics. Judge each item on its own.
3. Append one JSON line per item to OUT with exactly these keys: `id`, `kind`, `candidate`,
   `candidate_in_options`, `decision_reason`, `rationale` (same meanings as the rubric).
   Write via json.dumps; append only. Keep helper scripts in a private dir under /tmp named after your OUT file.
4. Verify OUT has one line per input id, then reply with the count and kind counts only.
