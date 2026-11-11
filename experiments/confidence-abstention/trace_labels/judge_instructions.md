# Judge instructions

You will label reasoning traces according to a rubric file. Inputs:
- RUBRIC: path given in your task.
- BATCH: a JSONL file; each line has `id`, `phase` (2 or "verbal"), `question`, `options` (1-4),
  `previous_answer` (verbal only), `final_answer` (phase 2 only), `reasoning` (the trace).
- OUT: JSONL path to write.

Procedure:
1. Read the rubric.
2. Process items one at a time. Print one item with e.g.
   `python3 -I -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])][int(sys.argv[2])]; print(json.dumps({k:v for k,v in r.items() if k!='reasoning'})); print(r['reasoning'])" BATCH i`
   Read the entire trace yourself and judge it. Do NOT classify with keyword scripts, regexes, or heuristics — every label must come from your reading of that trace.
   Judge each trace on its own; do not let earlier items shape later labels.
3. After each item (or every few items), append one JSON object per item to OUT with exactly these keys:
   `id`, `kind` (RESOLVE|JUDGE|MIXED|OTHER), `candidate` (string), `candidate_in_options` (int 0-4),
   `decision_reason` (one of the rubric's values), `rationale` (string).
   Write via python json.dumps so the file stays valid JSONL. Append; never overwrite earlier lines.
4. When done, verify OUT has one line per input item with unique ids (re-do any missing), then reply with just: the count labeled and the count of each kind.
Do not write anything else anywhere; do not modify the batch files.
