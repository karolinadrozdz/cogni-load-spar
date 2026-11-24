#!/usr/bin/env python3
"""Build a compact inventory from every completed result directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "manifest.csv"
OUT_MD = RESULTS / "SUMMARY.md"


def nested(obj: dict, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return ""
        obj = obj.get(key, "")
    return obj


rows = []
for analysis_path in sorted(RESULTS.rglob("analysis.json")):
    run_dir = analysis_path.parent
    params_path = run_dir / "params.json"
    if not params_path.exists():
        continue
    analysis = json.loads(analysis_path.read_text())
    params = json.loads(params_path.read_text())
    rel = run_dir.relative_to(RESULTS)
    group = rel.parts[0]
    rows.append(
        {
            "group": group,
            "run": run_dir.name,
            "model": params.get("model", ""),
            "revision": params.get("revision", params.get("model_commit", "")),
            "thinking": params.get("thinking", ""),
            "n_requested": params.get("n", ""),
            "n_phase0": nested(analysis, "n", "0"),
            "n_phase1": nested(analysis, "n", "1"),
            "n_phase2": nested(analysis, "n", "2"),
            "phase1_accuracy": nested(analysis, "phase1", "accuracy"),
            "auroc_correct_cal": nested(analysis, "P1_auroc_correct", "cal"),
            "auroc_correct_verbal": nested(analysis, "P1_auroc_correct", "verbal"),
            "phase2_abstention": nested(analysis, "phase2", "abstention"),
            "abstain_given_wrong": nested(analysis, "P2_abstention", "abstention_given_p1_incorrect"),
            "abstain_given_right": nested(analysis, "P2_abstention", "abstention_given_p1_correct"),
            "verbal_beyond_cal_p": nested(analysis, "P2_abstention", "verbal_beyond_cal", "p"),
            "gpu": params.get("gpu", ""),
            "vllm": params.get("vllm", ""),
            "path": rel.as_posix(),
        }
    )

# Stage A also has a consolidated, bootstrap-annotated table. Use it as the
# canonical fallback if an item-level directory was archived without its
# analysis JSON.
stage_a_table = RESULTS / "stage-a" / "k2h32b-popqa-stage-a-n200.csv"
known_runs = {row["run"] for row in rows}
if stage_a_table.exists():
    with stage_a_table.open(newline="") as handle:
        for item in csv.DictReader(handle):
            if item["run"] in known_runs:
                continue
            rows.append(
                {
                    "group": "stage-a",
                    "run": item["run"],
                    "model": item["model"],
                    "revision": item["revision"],
                    "thinking": item["thinking"],
                    "n_requested": item["n_items"],
                    "n_phase0": item["n_items"],
                    "n_phase1": item["n_items"],
                    "n_phase2": item["n_items"] if item["thinking"] == "native" else "",
                    "phase1_accuracy": item["accuracy"],
                    "auroc_correct_cal": item["auroc_cal"],
                    "auroc_correct_verbal": item["auroc_verbal"],
                    "phase2_abstention": item["abstention"],
                    "abstain_given_wrong": "",
                    "abstain_given_right": "",
                    "verbal_beyond_cal_p": "",
                    "gpu": "",
                    "vllm": "",
                    "path": f"stage-a/{item['run']}",
                }
            )

rows.sort(key=lambda row: (row["group"], row["run"]))

fields = list(rows[0]) if rows else []
with OUT_CSV.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

group_counts: dict[str, int] = {}
for row in rows:
    group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1

stage_b = [row for row in rows if row["group"] == "runs"]
native = [row for row in stage_b if row["thinking"] == "native"]
base = [row for row in stage_b if row["thinking"] == "base"]
stage_a = [row for row in rows if row["group"] == "stage-a"]

lines = [
    "# Result snapshot",
    "",
    "Generated from the checked-in `analysis.json` and `params.json` files.",
    "",
    "## Inventory",
    "",
    "| Group | Completed runs |",
    "|---|---:|",
]
for group, count in sorted(group_counts.items()):
    lines.append(f"| `{group}` | {count} |")

lines += [
    "",
    "## Stage A probes",
    "",
    "| Run | Mode | Accuracy | AUROC cal. | AUROC verbal | Abstention |",
    "|---|---|---:|---:|---:|---:|",
]
for row in stage_a:
    lines.append(
        f"| `{row['run']}` | {row['thinking']} | {row['phase1_accuracy']} | "
        f"{row['auroc_correct_cal']} | {row['auroc_correct_verbal']} | "
        f"{row['phase2_abstention']} |"
    )

lines += [
    "",
    "## Stage B: base checkpoints",
    "",
    "| Run | Accuracy | AUROC(calibrated → correctness) |",
    "|---|---:|---:|",
]
for row in base:
    lines.append(
        f"| `{row['run']}` | {row['phase1_accuracy']} | {row['auroc_correct_cal']} |"
    )

lines += [
    "",
    "## Stage B: native-thinking checkpoints",
    "",
    "| Run | Accuracy | AUROC cal. | AUROC verbal | Abstention | P(abstain\u2223wrong) | P(abstain\u2223right) | verbal beyond cal. p |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in native:
    lines.append(
        f"| `{row['run']}` | {row['phase1_accuracy']} | "
        f"{row['auroc_correct_cal']} | {row['auroc_correct_verbal']} | "
        f"{row['phase2_abstention']} | {row['abstain_given_wrong']} | "
        f"{row['abstain_given_right']} | {row['verbal_beyond_cal_p']} |"
    )

lines += [
    "",
    "The machine-readable version, including revisions and runtime metadata, is in `manifest.csv`.",
    "",
]
OUT_MD.write_text("\n".join(lines))
print(f"Wrote {OUT_CSV} and {OUT_MD} for {len(rows)} completed runs")
