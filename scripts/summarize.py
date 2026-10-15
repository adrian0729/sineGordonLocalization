"""Summarize results/runs.jsonl for one ablation table: mean ± std over seeds, as markdown.

    python scripts/summarize.py phase2     # prints the tables and writes results/phase2_tables.md
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402

METRICS = [("mae", "Mean distance", 2, 1), ("acc_0", "Exact", 1, 100), ("acc_1", "±1", 1, 100),
           ("acc_3", "±3", 1, 100), ("over_10", ">10 off", 1, 100)]


def load_records():
    records = {}
    with open(experiment.RESULTS) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("config_id"):
                    records[rec["config_id"]] = rec  # the latest record wins
    return records


def fmt(values, digits, scale):
    v = np.asarray(values) * scale
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def summarize(phase):
    records = load_records()
    rows = defaultdict(list)  # (group, name) -> records
    order = []
    for cfg in ablations.PHASES[phase]():
        key = (cfg["group"], cfg["name"])
        if key not in order:
            order.append(key)
        rec = records.get(experiment.config_id(cfg))
        if rec:
            rows[key].append(rec)
    tables = {}
    for group in dict.fromkeys(g for g, _ in order):
        lines = ["| Config | Seeds | Params | Val mean distance | " + " | ".join(m[1] for m in METRICS) + " | Train time |",
                 "|---|---|---|---|" + "---|" * len(METRICS) + "---|"]
        for g, name in order:
            if g != group:
                continue
            recs = rows.get((g, name), [])
            if not recs:
                lines.append(f"| {name} | 0 | | | " + " | ".join("" for _ in METRICS) + " | pending |")
                continue
            cells = [fmt([r["test"][m] for r in recs], d, s) for m, _, d, s in METRICS]
            val = fmt([r["val"]["mae"] for r in recs], 2, 1)
            minutes = np.mean([r["train_seconds"] for r in recs]) / 60
            lines.append(f"| {name} | {len(recs)} | {recs[0]['params']:,} | {val} | " + " | ".join(cells) + f" | {minutes:.0f} min |")
        tables[group] = "\n".join(lines)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(ablations.PHASES))
    args = parser.parse_args()
    tables = summarize(args.phase)
    text = "\n\n".join(f"### {group}\n\nValidation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).\n\n{table}"
                       for group, table in tables.items())
    print(text)
    (ROOT / "results" / f"{args.phase}_tables.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
