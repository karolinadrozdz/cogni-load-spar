# Findings: confidence experiment

# 1. First run: real and invented countries

Run of 8 Oct 2026 on `dev` (Qwen3.5-4B, `bfloat16`, Colab T4). 100 items:
"What is the capital of {entity}? Answer in one word." with the `Answer:`
prefill, 50 real and 50 invented countries. Read at the prefill token, layers
8–26; band (23, 27) from `find_band`. Presence is minimum rank ≤ 25 over the
stated layers, on the word-like rank unless noted. Design: `DECISIONS.md` D24–D25.
Data: stimuli set `capitals` (`results/dev/confidence_capitals*.parquet`).

## Summary

The model marks invented countries internally: "unknown" and "nonexistent" are
present under the J-lens on all 50 invented items and, in the band, on almost
no real ones (6%), with control words at zero. This is not yet evidence of a
confidence signal held apart from the output. An uncertainty word is among the
model's top 25 next-token candidates on every invented item, and the
nonexistence words are read equally well by the logit-lens control. The clearest J-lens-specific effect is timing:
it shows "Unknown" about four layers before the logit lens does.

## Behaviour

| condition | n | output |
|---|---|---|
| real | 50 | 50 correct; the capital is the top next token on every item |
| invented | 50 | 22 "None"/"Unknown", 9 a sentence ("There is no such place…"), 18 the country name repeated (three slightly altered), 1 real guess (Drenmark → Copenhagen) |

The model does not invent capitals. The "guess" type is therefore 18 echoes
and one guess, and the design's critical cell (a confident wrong answer) is
empty. The report's `echo` type catches 17 of the 18 echoes ("Sorvonia" is not
contained in "Sorvania"). The top candidates on invented items ("None", "Unknown", the first token
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

In the band two words carry almost everything: "unknown" (mostly as `Unknown`)
and "nonexistent", with "unsure" on 32% of invented items. In the early window
"fictional" and "fake" join them. The other six uncertainty words and
"imaginary" never appear.

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

- On invented items "unknown" appears under the J-lens at layer 15–16; from
  layer 17 its median rank is 1–2 to the end. The logit lens first shows it at layer 19 and never
  on more than 60% of items.
- The J-lens leads the logit lens by a similar margin for the capital on real
  items (layer 20 against 23–24).
- "Nonexistent" appears at layers 16–18 under both lenses.
- On real items "unknown" is present on 46% at layer 18 and 90% at layer 19,
  then falls to 10% at layer 20, where the capital appears. This is weak: where
  present, its median word-like rank is 16–20 and its full-vocabulary rank is
  between 60 and 340, so it is absent on the full-vocabulary column.

## Reading

1. **Uncertainty words are not separable from output staging here.** An
   uncertainty word has a median rank of 3 in the model's own next-token
   distribution on invented items, including the echoes. The J-lens shows it
   earlier than the logit lens, which is what a lens built to read future
   output would do.
2. **Nonexistence words are present without being output candidates** (median
   output rank 56; in the top-25 on none of the echo items). The logit lens
   reads them just as well, so this is not specific to the J-lens.
3. **The transient "unknown" on real items** is consistent with a default
   "unknown" state that retrieval of the answer overrides. It rests on one
   word, one rank column and two layers, and was found after looking at the
   data.

## Limitations

- One run, one model, one template; no statistics beyond proportions of 50.
- Invented names differ from real ones in form (2–3 tokens against mostly 1)
  and the
  model recognises them as made up. The contrast may be "detects an invented
  name", not "does not know a fact".
- The word list fixes what can be found, and a handful of its twelve words
  carry the result. "None", the most frequent abstention, was not read.
- The early window was fixed after a 10-item smoke run seen in-band only. The
  per-word and per-layer breakdowns are exploratory.
- Layer 19, where the real-item transient peaks, lies in neither window.

## Next

1. Real but obscure entities as a third condition, to obtain confident wrong
   answers and to separate an invented name from an unknown fact.
2. Add "none" to the words read, and report the per-word and per-layer tables
   from the code so this analysis is reproducible.
3. Repeat on `prod` (Qwen3.6-27B) once a larger GPU is available.

# 2. Second run: real regions

Run of 8 Oct 2026, same model and setup. Stimuli set `regions`: 130 real
regions with single-token capitals, from Wikidata (`DECISIONS.md` D26). Data:
`results/dev/confidence_regions*.parquet`. New in this run: the top 25
word-like tokens per layer are stored, alongside the fixed word list.

## Summary

The pre-committed test is negative: the fixed uncertainty and nonexistence
words are no more present when the model names a wrong capital than when it
names the right one. The open top-25 view suggests where a difference does
lie. On wrong answers the J-lens shows words for "nothing" (nada, nil, null) at
layers 19-22, where on correct answers it shows city names. That contrast was
found by looking and has to be tested on new items before it counts.

## Behaviour

| output | n | median familiarity (`links`) | median probability of first token |
|---|---|---|---|
| correct | 90 | 157 | 0.79 |
| wrong name | 28 | 41 | 0.20 |
| region name repeated | 6 | 45 | 0.22 |
| abstains | 6 | 39 | 0.20 |

Output types are worked out from the stored text by the current rule in
`confidence.output_type`. At run time the code counted "N'Djamena" and "N'Zi"
as abstentions and did not separate repeats of the region name from other
wrong names; both are fixed, and the report applies the rule again on reading.
In the report "wrong name" is the type `guess` and "region name repeated" is
`echo`.

The wrong answers are not confident errors. The probability of the model's
first token is low on them (0.20 against 0.79; only 4 of 28 are above 0.5), and
they are the little-known regions. On 8 of the 28 that first token is a line
break and the name follows. Being wrong, being unfamiliar and being
unsure at the output are nearly the same items in this set.

## Fixed word list (pre-committed)

Share of items with at least one word of the group present, J-lens.

| | wrong name (28) | correct (90) | Fisher p |
|---|---|---|---|
| uncertainty word, band | 0.21 | 0.13 | 0.37 |
| uncertainty word, early (L8-18) | 0.32 | 0.36 | 0.82 |
| nonexistence word, band | 0.11 | 0.12 | 1.00 |
| nonexistence word, early | 0.14 | 0.29 | 0.14 |

Of the twelve listed words only "unknown" and "nonexistent" ever appear in
this run. Neither separates wrong from correct. The control word "bridge" is
present on 5 of the 130 items. The six abstentions do show them (uncertainty
word in the band on 4 of 6), which is the output-staging case again.

## The correct capital on wrong answers

On 11 of 28 wrong answers (39%) the correct capital is present under the J-lens
in the band while the model names another city (21% under the logit lens). On
9 of those 11 it is also in the model's own top 25 next tokens, so it is mostly
a runner-up for the output, not something held apart from it.

## Open top-25 tokens (exploratory)

Share of items with the token among the J-lens top 25 in the stated layers.

| tokens, layers | wrong name | correct | correct, not US (61) | correct, `links` < 110 (19) |
|---|---|---|---|---|
| "none", L16-19 | 0.96 | 1.00 | 1.00 | 1.00 |
| nope / nothing / neither / nobody, L15-18 | 0.39 | 0.96 | 0.93 | 0.79 |
| "incorrect", L18 | 0.00 | 0.63 | 0.51 | 0.21 |
| nada / nil / null, L19-22 | 0.68 | 0.04 | 0.07 | 0.21 |
| town / local / municip / rural / village, L17-18 | 0.93 | 0.19 | 0.28 | 0.74 |
| "or", L15 | 1.00 | 0.24 | 0.36 | 0.84 |

1. **A "no answer" vocabulary appears at layers 15-19 on nearly every item**,
   before the capital does: "none" (98% of items) and, in Chinese, "does not
   exist" (100%) and "none yet" (88%), right or wrong. This is the transient seen in the
   first run, now without a word list. It is a stage every answer passes
   through, not a sign of not knowing; several of these words are in fact more
   frequent on correct answers.
2. **At layers 19-22 the two diverge.** Correct answers show specific cities.
   Wrong answers keep words for "nothing" (68% against 4%). The difference
   holds against correct answers outside the US (7%) and against little-known
   correct ones (21%; 19 of 26 against 4 of 19 among items with `links` < 110,
   Fisher p = 0.001).
3. **Place-type words and "or" follow familiarity, not correctness**: they are
   nearly as frequent on little-known regions the model gets right.

The "nothing" words are not what the model is about to say: by layers 25-26
they are on 7% of wrong answers and 1% of correct ones. The logit lens also
shows them at layers 19-22, but less selectively (93% of wrong answers and 30%
of correct ones, against 68% and 4% under the J-lens).

Whether point 2 adds anything to the model's output probability is open. The
"nothing" words correlate with it (Spearman -0.61) and with familiarity
(-0.64) over the 118 wrong and correct answers. Among the 29 of those with a
probability below 0.4 they are on 18 of 24 wrong and 1 of 5 correct (p = 0.04),
which is too few items to carry weight.

## Is this metacognition?

Not on this evidence. The J-space differs between items the model knows and
items it does not, which is a correlate of the knowledge boundary. It could be
no more than an empty retrieval. Self-monitoring in the sense of the
commentary this project started from requires that the model use the signal,
and here it still answers with a name and no hedge. Showing that the signal
predicts hedging, declining or self-correction, or that changing it changes
the model's confidence, would be the evidence.

## Limitations

- The tokens in point 2 were chosen after inspecting thousands of candidates.
  The p-values there do not correct for that and are descriptive only.
- No confident wrong answers: correctness, familiarity and output probability
  are confounded, so this run cannot show a signal that departs from what the
  model says.
- Wrong and correct items differ in topic (wrong ones are mostly African and
  Latin American regions; a third of correct ones are US states). Topic words
  dominate the raw token differences.
- "Wrong" rests on Wikidata labels, and 5 of the 28 should not count: one where
  the model is right and Wikidata is not ("Georgetown" for Demerara-Mahaica,
  confirmed on Wikipedia), one spelling variant ("Kolonia" for Colonia), one
  region with no clear capital (Saint Andrew Parish, Dominica) and two garbled
  outputs ("N'Zi", "Awdaghou"). On the remaining 23 the two main comparisons
  are unchanged: "nothing" words at L19-22 on 0.61 against 0.04, and an
  uncertainty word in the band on 0.22 against 0.13 (p = 0.33). Only the
  Georgetown case was checked against a second source.
- Further reference answers are open to doubt, on a reviewer's knowledge and
  unchecked: West Kazakhstan's "Oral" is better known as Uralsk; Orellana's
  "Coca" is a colloquial name; Bougainville's "Buka" is an interim seat;
  "Nazinon" is a recent name for a region the model probably knows under its
  old one. Several capitals are also everyday words (Salt, Same, Coca, Oral,
  Paradise), which may affect a rank readout.
- On 8 of 28 wrong answers the first generated token is a line break, so the
  position read is one token before the name.
- One model, one template, one run. The first run's capitals set has no
  top-token data, so point 1 has not been checked on invented countries.

## Next

1. Test point 2 as a pre-committed hypothesis on new items: fix the token set
   (nada, nil, null, none) and the layers (19-22) now, and draw fresh regions.
2. Find confident wrong answers, the cell both runs lack. Candidates: lift the
   one-token rule to reach more obscure regions, or use questions with a
   plausible but wrong default, such as a region whose best-known city is not
   its capital (New York here; the model answered "New York City").
3. Re-run the capitals set to get its top tokens.
4. Move the per-word, per-layer and re-typing analyses into the code.
5. Read the J-space after the answer, for words of failure following a wrong
   answer; and plant wrong answers on countries the model knows, so the same
   question is compared with a right and a wrong answer.
6. Check every reference answer against a second source before a
   confirmatory run.

