"""Why is the hump label easier to predict than the swing-averaged label? Writes results/hump_label.json.

    python scripts/hump_label.py

Both labels average the symmetric energy over a swing, so neither jitters within one. They differ in two ways: the
hump label sums 3 neighbouring pendulums, and it fires earlier (median tau 54.5 against 60), when the hump has just
formed. The hump_top label separates the two: it fires with the hump label but names the top pendulum of the
swing-averaged energy, as the averaged label does.
- "3 pendulums or 1": hump_top against the hump label, at the same moment
- "drift": hump_top against the averaged label, which is read about 5 time units later

Then the final models' saved test predictions (pred_test.npy): each model scored on every label, and where each
model's misses land.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from top2 import FINAL  # noqa: E402
from sgloc import data, metrics  # noqa: E402


def share(d, n):
    return {"exact": float(np.mean(d == 0)), "within1": float(np.mean(d <= 1)), f"over{n}": float(np.mean(d > n))}


def main():
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    labels = {name: test[f"site_{name}"].astype(int) for name in ("right", "avg", "hump", "hump_top")}
    hump, avg, right, top = (labels[n] for n in ("hump", "avg", "right", "hump_top"))
    ok = avg >= 0  # 51 runs never pass the averaged label's threshold by tau = 150
    gap = test["time_avg"][ok] - test["time_hump"][ok]
    drift = metrics.ring_distance(top[ok], avg[ok])
    out = {"runs": int(len(hump)), "avg_localized": int(ok.sum()),
           "time_gap_avg_minus_hump": {"median": float(np.median(gap)), "p10": float(np.percentile(gap, 10)),
                                       "p90": float(np.percentile(gap, 90))},
           "one_vs_three_pendulums": share(metrics.ring_distance(top, hump), 1),
           "drift_to_avg_time": share(drift, 1),
           "hump_vs_avg_label": share(metrics.ring_distance(hump[ok], avg[ok]), 1),
           "hump_vs_v1_label": share(metrics.ring_distance(hump, right), 1)}
    print(json.dumps(out, indent=1), flush=True)

    out["models"] = {}
    for trained, pattern in FINAL.items():
        preds = [np.load(d / "pred_test.npy").astype(int) for d in sorted(ROOT.glob(pattern))]
        if not preds:
            continue
        scored = {}
        for name, truth in labels.items():
            keep = truth >= 0
            dist = [metrics.ring_distance(pred[keep], truth[keep]) for pred in preds]
            scored[name] = {"mae": float(np.mean([d.mean() for d in dist])),
                            "exact": float(np.mean([np.mean(d == 0) for d in dist])),
                            "within1": float(np.mean([np.mean(d <= 1) for d in dist]))}
        # misses on the model's own label (2 or more sites off): how many sit within 1 of another label instead?
        own = labels[trained]
        keep = ok  # runs with every label
        misses = [metrics.ring_distance(pred[keep], own[keep]) >= 2 for pred in preds]
        near = {name: float(np.mean([np.mean(metrics.ring_distance(pred[keep][m], truth[keep][m]) <= 1)
                                     for pred, m in zip(preds, misses)]))
                for name, truth in labels.items() if name not in (trained, "right")}
        drifted = drift >= 2 if trained == "avg" else None
        entry = {"seeds": len(preds), "scored_on": scored, "miss_share": float(np.mean([m.mean() for m in misses])),
                 "misses_within1_of": near}
        if drifted is not None:  # the averaged label's misses on runs where its top pendulum moved after the hump formed
            entry["share_drifted"] = float(drifted.mean())
            entry["within1_drifted"] = float(np.mean([np.mean(~m[drifted]) for m in misses]))
            entry["within1_not_drifted"] = float(np.mean([np.mean(~m[~drifted]) for m in misses]))
        out["models"][trained] = entry
        print(trained, json.dumps(entry), flush=True)
    (ROOT / "results" / "hump_label.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
