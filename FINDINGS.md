# Findings: confidence experiment, first run

Run of 8 Oct 2026 on `dev` (Qwen3.5-4B, `bfloat16`, Colab T4). 100 items:
"What is the capital of {entity}? Answer in one word." with the `Answer:`
prefill, 50 real and 50 invented countries. Read at the prefill token, layers
8–26; band (23, 27) from `find_band`. Presence is minimum rank ≤ 25 over the
stated layers, on the word-like rank unless noted. Design: `DECISIONS.md` D24–D25.
Data: stimuli set `capitals` (`results/dev/confidence_capitals*.parquet`).

## Summary

The model marks invented countries internally: "unknown" and "nonexistent" are
present under the J-lens on all 50 invented items and on almost no real ones,
with control words at zero. This is not yet evidence of a confidence signal
held apart from the output. "Unknown" is among the model's top next-token
candidates on every invented item, and the nonexistence words are read equally
well by the logit-lens control. The clearest J-lens-specific effect is timing:
it shows "Unknown" about four layers before the logit lens does.

## Behaviour

| condition | n | output |
|---|---|---|
| real | 50 | 50 correct; the capital is the top next token on every item |
| invented | 50 | 22 "None"/"Unknown", 9 a sentence ("There is no such place…"), 18 the country name repeated (three slightly altered), 1 real guess (Drenmark → Copenhagen) |

The model does not invent capitals. The "guess" type is therefore 18 echoes
and one guess, and the design's critical cell (a confident wrong answer) is
empty. The top candidates on invented items ("None", "Unknown", the first token
of the name) are within about one logit of each other, so which one wins is
close to arbitrary.

## Presence by condition

Share of items with the word present, J-lens. "Output top-25" is the same word
group in the model's own next-token distribution.

| | invented | real |
|---|---|---|
| "unknown", band | 1.00 | 0.06 |
| "unknown", early (L8–18) | 1.00 | 0.46 |
| "nonexistent", band | 1.00 | 0.06 |
| "nonexistent", early | 1.00 | 0.12 |
| "fictional" / "fake", early | 0.68 / 0.74 | 0.00 / 0.00 |
| any control word, band or early | 0.00 | 0.00 |
| an uncertainty word in output top-25 | 1.00 | 0.12 |
| a nonexistence word in output top-25 | 0.22 | 0.00 |

Two words carry almost everything: "unknown" (mostly as `Unknown`) and
"nonexistent". "Unsure" appears on 32% of invented items in the band. The
other six uncertainty words and "imaginary" never appear.

Echoes and abstentions do not differ: "unknown" and "nonexistent" are at 1.00
in both, in both windows. The readout follows the condition, not the output.

## Timing

Share of items with the word present at each layer.

| layer | "unknown", invented, J-lens | same, logit lens | "nonexistent", invented, J-lens | same, logit lens | "unknown", real, J-lens | capital, real, J-lens | capital, real, logit lens |
|---|---|---|---|---|---|---|---|
| 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 0.78 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 1.00 | 0.00 | 0.34 | 0.10 | 0.00 | 0.00 | 0.00 |
| 17 | 1.00 | 0.00 | 0.88 | 0.00 | 0.00 | 0.00 | 0.00 |
| 18 | 1.00 | 0.00 | 1.00 | 0.84 | 0.46 | 0.00 | 0.00 |
| 19 | 1.00 | 0.28 | 1.00 | 0.66 | 0.90 | 0.08 | 0.00 |
| 20 | 1.00 | 0.32 | 1.00 | 0.78 | 0.10 | 0.78 | 0.00 |
| 23 | 1.00 | 0.20 | 1.00 | 1.00 | 0.04 | 0.98 | 0.42 |
| 26 | 1.00 | 0.42 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |

- On invented items "unknown" appears under the J-lens at layer 15–16 and stays
  at rank 1–2 to the end. The logit lens first shows it at layer 19 and never
  on more than 60% of items.
- The J-lens leads the logit lens by a similar margin for the capital on real
  items (layer 20 against 23–24).
- "Nonexistent" appears at layers 16–18 under both lenses.
- On real items "unknown" is present on 46% at layer 18 and 90% at layer 19,
  then falls to 10% at layer 20, where the capital appears. This is weak: the
  word-like rank is about 16–20, the full-vocabulary rank about 70–230, and it
  is absent on the full-vocabulary column.

## Reading

1. **Uncertainty words are not separable from output staging here.** An
   uncertainty word has a median rank of 3 in the model's own next-token
   distribution on invented items, including the echoes. The J-lens shows it
   earlier than the logit lens, which is what a lens built to read future
   output would do.
2. **Nonexistence words are present without being output candidates** (median
   output rank 55; in the top-25 on none of the echo items). The logit lens
   reads them just as well, so this is not specific to the J-lens.
3. **The transient "unknown" on real items** is consistent with a default
   "unknown" state that retrieval of the answer overrides. It rests on one
   word, one rank column and two layers, and was found after looking at the
   data.

## Limitations

- One run, one model, one template; no statistics beyond proportions of 50.
- Invented names differ from real ones in form (2–3 tokens against 1) and the
  model recognises them as made up. The contrast may be "detects an invented
  name", not "does not know a fact".
- The word list fixes what can be found, and two of twelve words carry the
  result. "None", the most frequent abstention, was not read.
- The early window was fixed after a 10-item smoke run seen in-band only. The
  per-word and per-layer breakdowns are exploratory.
- Layer 19, where the real-item transient peaks, lies in neither window.

## Next

1. Real but obscure entities as a third condition, to obtain confident wrong
   answers and to separate an invented name from an unknown fact.
2. Add "none" to the words read, and report the per-word and per-layer tables
   from the code so this analysis is reproducible.
3. Repeat on `prod` (Qwen3.6-27B) once a larger GPU is available.
