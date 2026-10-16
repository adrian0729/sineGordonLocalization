"""Why is the hump label easier to predict than the swing-averaged label? (Phase 9.) Writes results/hump_label.json.

    python scripts/hump_label.py

Both labels average the symmetric energy over a swing, so neither jitters within one. They differ in two ways: the
hump label sums 3 neighbouring pendulums, and it fires earlier (median tau 54.5 against 60), when the hump has just
formed. To separate the two, read the single-pendulum swing-averaged profile at the moment the hump label fires:
- "3 pendulums or 1": its top pendulum against the hump label, at the same moment
- "drift": its top pendulum against the averaged label, which is read about 5 time units later

Then the final models' saved test predictions (pred_test.npy): each model scored on every label, and where the
averaged-label model's misses land.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from top2 import FINAL, swing_profiles  # noqa: E402
from sgloc import data, metrics, sim  # noqa: E402


def share(d, n):
    return {"exact": float(np.mean(d == 0)), "within1": float(np.mean(d <= 1)), f"over{n}": float(np.mean(d > n))}


def main():
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    p, chunk = sim.Params(), 1000
    jobs = [(test["theta0"][i:i + chunk], {"hump": test["time_hump"][i:i + chunk].astype(float)}, p)
            for i in range(0, len(test["theta0"]), chunk)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        top = np.concatenate([part["hump"].argmax(axis=1) for part in pool.map(swing_profiles, jobs)])
    hump, avg, right = (test[f"site_{n}"].astype(int) for n in ("hump", "avg", "right"))
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

    labels = {"right": right, "avg": avg, "hump": hump, "top_at_hump_time": top}
    out["models"] = {}
    for trained, pattern in FINAL.items():
        preds = [np.load(d / "pred_test.npy").astype(int) for d in sorted(ROOT.glob(pattern))]
        if not preds:
            continue
        scored = {}
        for name, truth in labels.items():
            keep = truth >= 0
            dist = [metrics.ring_distance(pred[keep], truth[keep]) for pred in preds]
            scored[name] = {"mae": float(np.mean([d.mean() for d in dist])), "within1": float(np.mean([np.mean(d <= 1) for d in dist]))}
        # misses on the model's own label (2 or more sites off): how many sit within 1 of another label instead?
        own = labels[trained]
        keep = ok  # runs with all three labels
        misses = [metrics.ring_distance(pred[keep], own[keep]) >= 2 for pred in preds]
        near = {name: float(np.mean([np.mean(metrics.ring_distance(pred[keep][m], truth[keep][m]) <= 1)
                                     for pred, m in zip(preds, misses)]))
                for name, truth in labels.items() if name != trained and name != "right"}
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
