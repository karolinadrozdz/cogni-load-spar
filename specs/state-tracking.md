# state_tracking: sequential state tracking (mini-CogniLoad)

Purpose: load the forward pass. single_cue, forced_demand, retro_cue and derived_state produced no behavioural variance (accuracy flat in `C_t`) because every task could be resolved at the question by attending back: the words were in the prompt, and counting is parallel. This experiment uses conditional state updates in the style of CogniLoad (Kaiser et al., ICLR 2026): each update's condition refers to the *current* state, so the final state cannot be computed without computing the intermediates in order. With thinking off, that chain is a load on the single pass. The intermediates are derived, never written as bindings, and used without being emitted. Signal is the goal; the first deliverable is the accuracy-vs-chain-length curve, which tells us whether we have a load at all. One GPU-hour.

Audience: a Claude Code session in `cogni-load-spar`. Read `specs/forced-demand-and-retro-cue.md` ground rules; they all apply. Reuse `prompts.render_chat`, `readout.read`, `readout.resolve_token_id`, `scoring.score_q1`, `exp3.score_digit`'s pattern for restricted-set ranks, `analyze.headline`. New code: one stimulus generator (`state_tracking.py`), two prompt templates, one analysis module with the tables below. `stimuli.generate` is not reused (different item structure). No refactors. Do not apply the audit cut list.

## Task

Three named people each hold one object. A sequence of update statements changes who holds what. Each update names its person by the object they **currently** hold, not by name, so applying update t requires the state after update t−1. At the end, one person (the PoI) is asked about.

Rendered item (derived arm, `k = 3`, `h = 2`; needles marked here for the reader only, not in the prompt):

> Ann holds the key. Ben holds the lamp. Tom holds the cup. The person holding the key swaps it for the pen. [needle] The person holding the lamp swaps it for the map. [hay] The person holding the pen swaps it for the coin. [needle] The person holding the coin swaps it for the bell. [needle] The person holding the cup swaps it for the ring. [hay] What is Ann holding? Answer:

Target: `bell`. Ann's chain: key → pen → coin → bell. Thinking off, asserted (`_assert_thinking`). Greedy, one token generated; keep logits at the first generated position.

CogniLoad mapping: `d = 1` (one attribute), `N = k + h`, `ρ = k / (k + h)`. Their design cannot vary needle count and hay count independently; ours does. Natural-language templates are ours; the logical form (condition on `S_{t−1}`, update `S_t`, needle = affects PoI, hay = affects a non-PoI, no two people ever share a value) is theirs. Their generator is not used.

### Parameters

| symbol | meaning | values |
|---|---|---|
| `n` | people | 3 (fixed) |
| `k` | needles = PoI chain length | 1, 2, 3, 4, 5 |
| `h` | hay updates | 1, `k`, `2k` |
| items per cell | | 30, seeded |

15 cells × 30 = 450 items per arm.

### Pools (filter programmatically; record the surviving lists in the run manifest)

- **Names:** single-token with leading space in the Qwen3.6 tokenizer; need ≥ 6. Candidates: Ann, Ben, Tom, Sam, Kim, Max, Dan, Joe, Amy, Eve.
- **Objects:** single-token with leading space, concrete nouns, none a substring of another; need ≥ 20. Candidates: key, lamp, cup, pen, map, coin, bell, ring, box, bag, fan, jar, rope, drum, comb, nail, clip, book, hat, sock, fork, bowl, vase, flag, kite, rock, shell, stick, brush, chain.
- No object reused within an item: every object in an item appears exactly once as a "new" value or once in the initial state. This keeps every readout token unambiguous (each object is exactly one of: PoI chain, other-person chain, or absent). Requires `n + k + h ≤ 18` distinct objects per item; the pool must have ≥ 21 so at least 3 floor objects remain in every item.
- Floor objects are drawn from the **same filtered pool** — never in the prompt. This frequency-matches the floor to the targets by construction (the gap between in-prompt zeros and a 5% random-word floor in single_cue is not interpretable; this is).

### Generation rules

1. Initial state: `n` distinct names, `n` distinct objects.
2. Choose the PoI uniformly. Build the update sequence of length `k + h` by choosing a random interleaving of `k` needles and `h` hays, subject to the **recency guard**: in exactly half of the items in every cell, at least one hay follows the last needle (the other half end with a needle). Record `last_is_needle` per item. Without this, "answer = the last new object" is a shortcut that needs no tracking.
3. Each needle: condition = PoI's current object; new object = fresh from the pool. Each hay: condition = a uniformly chosen non-PoI's current object; new object = fresh.
4. Validation (CogniLoad's): after every update no two people hold the same object (guaranteed by freshness); the PoI's chain has exactly `k` steps; every hay affects exactly one non-PoI.
5. Store per item: names, PoI, full chain for every person (`chain[p] = [initial, …, final]`), the update list with `(step, is_needle, cond_obj, new_obj)`, `last_is_needle`, the floor objects, and the character offset of the period ending each update sentence.

### Prefill

`Answer:` then generate one token. The expected first token is the bare object (` bell`). Verify in the `dev` smoke that the first generated token is an object and not ` the` / ` a`; if it is an article, change the prefill to `Answer: the` and re-verify. Record in D26.

## Two arms, identical items

| arm | update sentence | what is on the page |
|---|---|---|
| **derived** | `The person holding the key swaps it for the pen.` | conditions and new values only; bindings must be derived |
| **copyable** | `The person holding the key swaps it for the pen (Ann: pen, Ben: lamp, Tom: cup).` | the full state after **every** update, needle and hay |

Same initial state, same updates, same question. The copyable arm is both the behavioural ceiling (if it is not at ceiling the task is failing for reasons other than tracking) and the positive control for the readout (the target object must be present at the answer position in this arm; if it is not, the readout cannot see objects at this position and nothing in the derived arm is interpretable).

## Readout

At the answer position (the `:` of `Answer:`), and, recorded but secondary, at the period ending each update sentence. Band from `results/prod/band.json`, `k = 25`, `rank_wordlike`, J-lens with logit-lens column, band-min rank ≤ 25 = present, as in single_cue, forced_demand, retro_cue and derived_state.

Groups at the answer position, each object scored by `resolve_token_id`:

| group | definition |
|---|---|
| target | PoI's final object |
| chain `−j` | PoI's object `j` steps before the final one, `j = 1 … k` (`−k` is the initial object); report by `j` |
| other current | each non-PoI's final object |
| other stale | each non-PoI's superseded objects |
| hay new, recency-matched | the hay new-object introduced closest before each needle new-object (control for position) |
| floor | ≥ 3 pool objects never in the prompt |
| PoI name / other names | binding readout; name tokens |
| answered token | whatever token the model emitted, if it is a pool object |

Also record the per-layer rank of every chain object (`−k … 0`) across the band — not just band-min — for Table 3.

Secondary, at the period after each update: presence of that update's new object and of the condition object; presence of the PoI's current object after hays (is the PoI binding carried across unrelated updates). Exploratory: in-stream readouts failed their positive control in derived_state and are not in the deliverable.

## Behaviour

Primary: `score_q1` with `expected = target`; top-1 and median rank per cell and arm. Restricted rank: rank of the target among the item's in-prompt objects only (`exp3.score_digit` pattern).

Error class of the emitted token (first generated token, no parser): `target` / `PoI stale −j` (which `j`) / `other current` / `other stale` / `out of prompt`. Report the distribution per `k`. Split accuracy by `last_is_needle`; if accuracy is much higher when the last update is a needle, the recency shortcut is in use and only the guard-satisfying half is interpretable — say so in the table.

## Tables (the deliverable)

**T1 — Load.** Per arm × `k` × `h`: top-1, median rank, restricted rank, `n`; accuracy split by `last_is_needle`; error-class distribution. This table alone answers "do we have a load." Target shape: derived falls with `k` and with `h`; copyable at ceiling throughout.

**T2 — Answer-position contents.** Per `k` (pool `h`), per arm: presence of target, chain `−1 … −k` as separate columns, other current, other stale, recency-matched hay, floor, PoI name, other names, logit-lens target and logit-lens chain `−1`. Same layout as the single_cue/derived_state tables.

**T3 — Layer order of the chain.** For `k ∈ {3, 4}`, derived arm, correct items only: for each chain object, the first band layer at which its rank ≤ 25 (or the layer of its minimum rank). Report (a) the median first-legible layer per chain position and (b) the fraction of items where first-legible layers are weakly monotone in chain order (initial earliest, target latest), against the permutation baseline `1/k!` for strict order. This is the paper's spider→legs ordering extended to `k` hops.

**T4 — Wrong trials.** Derived arm, incorrect items only, per `k`: presence of the target (correct but not emitted), presence of the emitted token, number of in-prompt objects present (fog), each beside the same quantities on correct items at the same `k`.

Append as §9 of `FINDINGS.md`: T1–T4, one line naming the outcome. No prose.

## Pre-committed outcomes

Read T1 first; the rest is conditional on having a load.

| # | T1 | T2–T4 | reading |
|---|---|---|---|
| 0 | derived accuracy ≥ 90% at `k = 5, h = 2k`, or ≤ 40% at `k = 1` | — | no usable dynamic range. Extend `k` to 7 with `h ∈ {1, k}` (needs pool ≥ 21) or drop to `n = 2`, respectively; re-run T1 before reading anything else |
| 0′ | copyable arm target presence at the answer position near floor | — | readout fails its positive control at this position; stop and report |
| A | derived falls with `k`; copyable at ceiling | chain `−1 … −k` present above floor **and** above recency-matched hay, presence falling with `j`; T3 ordering above baseline | J-space exposes the serial computation: intermediates are held while unspoken, in chain order. The capacity question becomes the `j` at which presence hits floor vs the `k` at which accuracy breaks |
| B | derived falls with `k`; copyable at ceiling | target present; chain, other, hay all at floor | J-space is an output buffer even for derived, serial state. Strongest deflationary result so far; combined with single_cue, forced_demand, retro_cue and derived_state this is the paper |
| C | derived falls with `k` | chain ≈ other stale ≈ hay, all well above floor | the readout here is "objects in the prompt", not chain-selective — the opposite of single_cue's in-prompt zeros. Report; check whether presence tracks recency; do not interpret as workspace content |
| D | any load | on wrong trials the target is present well above floor (signal present, report fails), **or** the emitted wrong token dominates and the target is at floor (confident wrong) | either is a result. The first is the dissociation the scoping document asks for; the second says the J-space reflects the model's commitment, not the truth |

A, B, C and D are not exclusive with each other beyond A/B/C being alternatives for T2. Pre-register that the headline is the pair (T1 shape, T2 outcome).

## Order and budget

1. Pool filtering, `state_tracking.py` generator with validation and the recency guard, two prompt templates, scoring, tables. Unit test: every generated item passes validation; exactly half of each cell satisfies `last_is_needle = False`.
2. Smoke on `dev`: 5 items at `k = 2, h = 1`, both arms. Print one rendered prompt per arm, the generated token, and T1 for the 5 items. Confirm the first generated token is a pool object (D26).
3. `prod`, derived arm, all cells. Read T1 immediately; if outcome 0, adjust and re-run before the copyable arm.
4. `prod`, copyable arm.
5. Append §9.

`prod`: 2 arms × 450 items, one generation each, readout at ≤ 12 positions per item in the same forward pass. Under one GPU-hour.

## Do not

- Do not enable thinking. (Thinking-on with pre-write readouts is the follow-up; it is a different experiment.)
- Do not reuse an object within an item, and do not let names appear as conditions.
- Do not let every item end with a needle; the recency guard is mandatory.
- Do not annotate only needles in the copyable arm; the full state after every update is the contrast.
- Do not build a parser; score the first generated token.
- Do not change band, `k`, or the readout definition.
- Do not use CogniLoad's generator or dataset; the item structure here is deliberately smaller and single-attribute.
- Do not interpret past the outcome table. The writing happens after the meeting.

## DECISIONS.md entries to add

- D24: state_tracking item grammar, name and object pools after tokenizer filtering, `n = 3`, the `k × h` grid.
- D25: no-reuse rule for objects; recency guard (half of each cell ends in hay); floor drawn from the same pool.
- D26: prefill convention and the verified first-token behaviour.
- D27: copyable-arm annotation format (full state after every update).

## References

Kaiser, Frigessi, Ramezani-Kebrya, Ricaud 2026, CogniLoad (ICLR 2026; github.com/kaiserdan/cogniload). Gurnee et al. 2026, arXiv 2607.15495 (spider→legs ordering). Kim & Schuster 2023, Entity Tracking in Language Models. Prakash et al. 2024, binding-ID mechanism for entity tracking in Llama. Dziri et al. 2023, Faith and Fate (compositional depth limits).
