"""Give a Phase-2-only run (e.g. --phase2-prompt skip) the Phase 0/1 rows and verbal.csv of its base run, then analyse.

    python borrow_base.py results/80b-thinking results/80b-thinking-p2skip
Same model, n and seeds give the same questions and option order, so only Phase 2 differs.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

base, new = Path(sys.argv[1]), Path(sys.argv[2])
b, n = pd.read_csv(base / "phases.csv"), pd.read_csv(new / "phases.csv")
assert set(n.phase) == {2}, "expected a Phase-2-only run"
b2 = b[b.phase == 2].set_index("question_index")
n2 = n.set_index("question_index")
assert (b2.question == n2.question).all() and (b2.gt_position == n2.gt_position).all(), "questions/arrangement differ"
pd.concat([b[b.phase.isin([0, 1])], n]).to_csv(new / "phases.csv", index=False)
shutil.copy(base / "verbal.csv", new / "verbal.csv")
subprocess.run([sys.executable, str(Path(__file__).parent / "analyze.py"), str(new)], check=True)
