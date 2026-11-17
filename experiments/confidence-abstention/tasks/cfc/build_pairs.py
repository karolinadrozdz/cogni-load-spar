"""Build the CFC pair set over the PopQA Phase 1 items (tasks/cfc/PLAN.md, "Items and pairs").

    python tasks/cfc/build_pairs.py           # writes tasks/cfc/pairs_data.py (+ pairs.csv, local only) and a summary
    python tasks/cfc/build_pairs.py --check   # rebuilds in memory and asserts pairs_data.py is identical

The pair set lives in a .py module (pairs_data.py), not a CSV: Modal's add_local_python_source ships only *.py files.

12 rounds, fixed seed:
- rounds 1-8 "matched": both items share the relation (prop), and their s_pop deciles (over the 1000 items) differ by at
  most 1. Each relation is matched separately: a randomized greedy maximal matching (the item with the fewest free
  partners first, a random partner among them), then free items are placed by swapping partners with an existing pair.
  Items that cannot be placed stay unpaired that round.
- rounds 9-12 "random": perfect matchings of all 1000 items (random pairing, conflicts repaired by swapping partners).
Every pair: no repeated pair across rounds; no shared subject; no answer leak:
- exact: the subject of one item equals one of the other item's options (case-insensitive, stripped);
- fuzzy: an option of one item appears as a whole word/phrase (case-insensitive) in the other item's question, or the
  other item's subject appears as a whole word/phrase inside one of this item's options ("screenwriter" as an option
  next to "Who was the screenwriter for ...?"; subject "David" next to the option "David Trueba").
Each pair stores ddec = |decile_i - decile_j|.

The helpers (subjects, forbidden pairs, check_pairs, pairs_sha256) are imported by the task (tasks/cfc/__init__.py).
"""

import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 2026
N_ROUNDS_MATCHED, N_ROUNDS_RANDOM = 8, 4
MAX_DECILE_DIFF = 1
HERE = Path(__file__).resolve().parent
QUESTIONS = HERE.parents[1] / "data" / "popqa_phase1_questions.csv"

# one template per relation (prop). Note "What is X the capital of?" (capital of) vs "What is the capital of X?" (capital)
TEMPLATES = {
    "author": r"Who is the author of (.+)\?",
    "capital": r"What is the capital of (.+)\?",
    "capital of": r"What is (.+) the capital of\?",
    "color": r"What color is (.+)\?",
    "composer": r"Who was the composer of (.+)\?",
    "country": r"In what country is (.+)\?",
    "director": r"Who was the director of (.+)\?",
    "father": r"Who is the father of (.+)\?",
    "genre": r"What genre is (.+)\?",
    "mother": r"Who is the mother of (.+)\?",
    "occupation": r"What is (.+)'s occupation\?",
    "place of birth": r"In what city was (.+) born\?",
    "producer": r"Who was the producer of (.+)\?",
    "religion": r"What is the religion of (.+)\?",
    "screenwriter": r"Who was the screenwriter for (.+)\?",
    "sport": r"What sport does (.+) play\?",
}


def norm(s):
    return str(s).strip().lower()


def subjects(df):
    """The subject of every question, parsed with its own relation's template; asserts each question parses and
    matches no other relation's template."""
    out = []
    for q, prop in zip(df.question, df.prop):
        m = re.fullmatch(TEMPLATES[prop], q)
        assert m, f"question does not parse with its template ({prop}): {q!r}"
        others = [p for p, t in TEMPLATES.items() if p != prop and re.fullmatch(t, q)]
        assert not others, f"{q!r} ({prop}) also matches {others}"
        out.append(m.group(1))
    return out


def deciles(df):
    """s_pop decile (0-9) of every item, over the whole file; equal s_pop values share a decile."""
    import pandas as pd
    d = pd.qcut(df.s_pop, 10, labels=False)
    assert d.nunique() == 10
    return d.to_numpy()


def whole(needle, hay):
    """needle occurs in hay as a whole word/phrase (not inside a longer word)."""
    return needle in hay and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay) is not None


def forbidden(df):
    """Unordered pairs that may never be shown together: same subject, an exact answer leak (one item's subject is
    among the other's options) or a fuzzy one (module docstring). Returns (set of frozensets, counts by reason; the
    fuzzy count is of pairs not already excluded by the exact rules)."""
    subj = [norm(s) for s in subjects(df)]
    opts = [[norm(df[f"choice_{k}"].iat[i]) for k in range(1, 5)] for i in range(len(df))]
    ques = [q.lower() for q in df.question]
    by_subj = {}
    for i, s in enumerate(subj):
        by_subj.setdefault(s, []).append(i)
    same = {frozenset((a, b)) for g in by_subj.values() for a in g for b in g if a < b}
    leak = {frozenset((i, j)) for j in range(len(df)) for o in set(opts[j]) for i in by_subj.get(o, []) if i != j}
    fuzzy = {frozenset((x, y)) for x in range(len(df)) for y in range(len(df)) if x != y
             and (any(whole(o, ques[y]) for o in opts[x]) or any(whole(subj[y], o) for o in opts[x]))}
    return same | leak | fuzzy, dict(shared_subject=len(same), answer_leak=len(leak),
                                     fuzzy_leak_extra=len(fuzzy - same - leak))


def greedy_round(items, rng, ok):
    """Randomized greedy maximal matching of `items` under ok (fewest free partners first, a random one of them), then
    two free items a, b are placed through a pair (c, d): (a, c) + (b, d) or (a, d) + (b, c)."""
    items = list(items)
    rng.shuffle(items)
    adj = {a: [b for b in items if b != a and ok(a, b)] for a in items}
    free, pairs = set(items), []
    while True:
        cand = {a: [b for b in adj[a] if b in free] for a in items if a in free}
        live = [a for a in items if a in free and cand[a]]
        if not live:
            break
        a = min(live, key=lambda x: len(cand[x]))
        b = rng.choice(cand[a])
        pairs.append([a, b])
        free -= {a, b}
    changed = True
    while changed:
        changed = False
        fl = [a for a in items if a in free]
        for x, a in enumerate(fl):
            for b in fl[x + 1:]:
                for k, (c, d) in enumerate(pairs):
                    if ok(a, c) and ok(b, d):
                        pairs[k:k + 1] = [[a, c], [b, d]]
                    elif ok(a, d) and ok(b, c):
                        pairs[k:k + 1] = [[a, d], [b, c]]
                    else:
                        continue
                    free -= {a, b}
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return [tuple(sorted(p)) for p in pairs]


def match_round(items, rng, ok):
    """Random pairing of `items` (consecutive after a seeded shuffle), then conflicts repaired by swapping partners
    with another pair of the round (or with the spare item of an odd group). Unrepairable pairs are dropped."""
    order = list(items)
    rng.shuffle(order)
    pairs = [[order[k], order[k + 1]] for k in range(0, len(order) - 1, 2)]
    spare = [order[-1]] if len(order) % 2 else []
    changed = True
    while changed:
        changed = False
        for k in range(len(pairs)):
            a, b = pairs[k]
            if ok(a, b):
                continue
            others = [m for m in range(len(pairs)) if m != k]
            rng.shuffle(others)
            for m in others:
                c, d = pairs[m]
                if ok(a, d) and ok(c, b):
                    pairs[k], pairs[m] = [a, d], [c, b]
                elif ok(a, c) and ok(b, d):
                    pairs[k], pairs[m] = [a, c], [b, d]
                else:
                    continue
                changed = True
                break
            else:
                if spare and ok(a, spare[0]):
                    pairs[k], spare = [a, spare[0]], [b]
                    changed = True
                elif spare and ok(spare[0], b):
                    pairs[k], spare = [spare[0], b], [a]
                    changed = True
    return [tuple(sorted(p)) for p in pairs if ok(*p)]


def build(df, seed=SEED):
    """The canonical pair list: (pair_id, i, j, kind, round, ddec), i < j, sorted by (round, i, j)."""
    bad, _ = forbidden(df)
    used = set()
    ok = lambda a, b: frozenset((a, b)) not in bad and frozenset((a, b)) not in used
    rng = random.Random(seed)
    dec = deciles(df)
    near = lambda a, b: ok(a, b) and abs(int(dec[a]) - int(dec[b])) <= MAX_DECILE_DIFF
    props = {}
    for i, p in enumerate(df.prop):
        props.setdefault(p, []).append(i)
    out = []
    for rnd in range(1, N_ROUNDS_MATCHED + N_ROUNDS_RANDOM + 1):
        if rnd <= N_ROUNDS_MATCHED:
            kind, got = "matched", []
            for p in sorted(props):
                got += greedy_round(props[p], rng, near)
        else:
            kind, got = "random", match_round(range(len(df)), rng, ok)
        for p in got:
            used.add(frozenset(p))
        out += [(i, j, kind, rnd) for i, j in sorted(got)]
    return [(pid, i, j, kind, rnd, abs(int(dec[i]) - int(dec[j]))) for pid, (i, j, kind, rnd) in enumerate(out)]


def pairs_sha256(pairs):
    """SHA-256 of the canonical text of the pair list (one "pair_id,i,j,kind,round,ddec" line per pair)."""
    return hashlib.sha256("\n".join(",".join(map(str, p)) for p in pairs).encode()).hexdigest()


def items_sha256(df):
    """SHA-256 of the items the pairs index (questions, options, prop, s_pop), independent of CSV line endings."""
    cols = ["question", "choice_1", "choice_2", "choice_3", "choice_4", "prop", "s_pop"]
    return hashlib.sha256("\n".join("\t".join(map(str, r)) for r in df[cols].itertuples(index=False)).encode()).hexdigest()


def check_pairs(df, pairs):
    """Every constraint of the pair set; raises AssertionError on the first violation."""
    bad, _ = forbidden(df)
    dec = deciles(df)
    seen, per_round = set(), {}
    assert [p[0] for p in pairs] == list(range(len(pairs))), "pair ids must be 0..n-1 in order"
    for pid, i, j, kind, rnd, ddec in pairs:
        assert 0 <= i < j < len(df), (pid, i, j)
        assert ddec == abs(int(dec[i]) - int(dec[j])), (pid, ddec)
        key = frozenset((i, j))
        assert key not in seen, f"pair {pid} ({i}, {j}) repeats an earlier pair"
        seen.add(key)
        assert key not in bad, f"pair {pid} ({i}, {j}) shares a subject or leaks an answer"
        r = per_round.setdefault(rnd, Counter())
        r[i] += 1
        r[j] += 1
        if kind == "matched":
            assert 1 <= rnd <= N_ROUNDS_MATCHED, (pid, rnd)
            assert df.prop.iat[i] == df.prop.iat[j] and ddec <= MAX_DECILE_DIFF, f"matched pair {pid} is not cue-matched"
        else:
            assert kind == "random" and N_ROUNDS_MATCHED < rnd <= N_ROUNDS_MATCHED + N_ROUNDS_RANDOM, (pid, kind, rnd)
    for rnd, c in per_round.items():
        assert max(c.values()) == 1, f"round {rnd}: an item is in two pairs"
        if rnd > N_ROUNDS_MATCHED:
            assert len(c) == len(df), f"random round {rnd} is not a perfect matching"
    return True


def summary(df, pairs):
    import pandas as pd
    t = pd.DataFrame(pairs, columns=["pair_id", "i", "j", "kind", "round", "ddec"])
    _, reasons = forbidden(df)
    lines = [f"{len(t)} pairs; forbidden unordered pairs: {reasons}",
             "pairs per round: " + ", ".join(f"{r}:{n}" for r, n in t.groupby("round").size().items()),
             "pairs per kind: " + ", ".join(f"{k}:{n}" for k, n in t.groupby("kind").size().items()),
             "matched pairs by decile difference: " + ", ".join(
                 f"{d}:{n}" for d, n in t[t.kind == "matched"].groupby("ddec").size().items())]
    for kind in ("matched", "random", None):
        s = t if kind is None else t[t.kind == kind]
        app = pd.Series(list(s.i) + list(s.j)).value_counts().reindex(range(len(df)), fill_value=0)
        lines.append(f"appearances per item ({kind or 'all'}): " +
                     ", ".join(f"{k}x:{v}" for k, v in app.value_counts().sort_index().items()))
    return "\n".join(lines)


def render_module(df, pairs):
    rows = "\n".join(f"    {p!r}," for p in pairs)
    return (f'"""Generated by tasks/cfc/build_pairs.py (seed {SEED}); do not edit. The CFC pair set over\n'
            f"data/popqa_phase1_questions.csv: (pair_id, i, j, kind, round, ddec), i < j = question_index (row of the\n"
            f"file), ddec = |s_pop decile difference|.\n"
            f'kind "matched" (rounds 1-{N_ROUNDS_MATCHED}: same prop, ddec <= {MAX_DECILE_DIFF}) or "random" '
            f'(rounds {N_ROUNDS_MATCHED + 1}-{N_ROUNDS_MATCHED + N_ROUNDS_RANDOM}).\n"""\n\n'
            f"SEED = {SEED}\nITEMS_SHA256 = {items_sha256(df)!r}\nPAIRS_SHA256 = {pairs_sha256(pairs)!r}\n"
            f"PAIRS = [\n{rows}\n]\n")


def main():
    import pandas as pd
    df = pd.read_csv(QUESTIONS)
    assert len(df) == 1000 and not df.question.duplicated().any()
    pairs = build(df)
    check_pairs(df, pairs)
    text = render_module(df, pairs)
    target = HERE / "pairs_data.py"
    if "--check" in sys.argv:
        assert target.read_text(encoding="utf-8") == text, "pairs_data.py differs from a fresh build"
        print("pairs_data.py matches a fresh build")
    else:
        target.write_text(text, encoding="utf-8", newline="\n")
        subj = subjects(df)
        dec = deciles(df)
        rows = [dict(pair_id=pid, i=i, j=j, kind=kind, round=rnd, decile_diff=ddec, prop_i=df.prop[i], prop_j=df.prop[j],
                     decile_i=int(dec[i]), decile_j=int(dec[j]), s_pop_i=df.s_pop[i], s_pop_j=df.s_pop[j],
                     subject_i=subj[i], subject_j=subj[j], question_i=df.question[i], question_j=df.question[j])
                for pid, i, j, kind, rnd, ddec in pairs]
        pd.DataFrame(rows).to_csv(HERE / "pairs.csv", index=False)
        print(f"wrote {target} and {HERE / 'pairs.csv'}")
    print(summary(df, pairs))
    print(f"PAIRS_SHA256 {pairs_sha256(pairs)}")


if __name__ == "__main__":
    main()
