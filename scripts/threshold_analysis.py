"""Threshold study (REPORT.md Sec. 10, "Which threshold?"): label diagnostics, physics baselines and trained models for every
localization label of scripts/threshold_labels.py, and the two threshold choices. Writes results/threshold.json.

    python scripts/threshold_analysis.py

1. Label diagnostics on the validation and test pools (no training): share localized by tau = 150, median time,
   lag after the first pendulum passes the top (ott), agreement with the hump centre (Sec. 10) and with the ott
   site, and how often splitting each spring evenly moves the site.
2. Physics baselines per label, each tuned on validation: the box-car on the site energy (frame and width) and the
   Floquet filter (number of periods); and the hump-centre oracle with the best constant offset.
3. Trained models from results/runs.jsonl (phase "threshold"): the v2 final recipe (3 seeds per label) and the
   rebuilt v1 (trained seeds; stalled tries are counted, not scored).
4. Two choices of threshold among the energy shares. The physics choice is fixed in advance from the validation
   diagnostics alone: the smallest share whose label fires no earlier than the first pendulum past the top in at
   least 95% of runs, agrees with the hump centre within +-1 in at least 98% of runs, and moves under the even
   spring split least. The accuracy choice is the share with the highest mean v2 accuracy within +-1 on test.
   Spearman correlations across the shares relate model accuracy to the label diagnostics and the baselines.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, floquet, metrics, sim  # noqa: E402
from threshold_labels import E_SEP, SEPARATRIX, SHARES  # noqa: E402

SHARE_LABELS = [f"thr{s:02d}" for s in SHARES]
PHYSICAL = ["sep2", "sep3", "ott"]
REFERENCE = ["avg", "hump"]


def boxcar(x, K):
    return sum(np.roll(x, j, axis=-1) for j in range(-(K // 2), K // 2 + 1)) / K


def ring_acc(pred, true, k=1):
    keep = true >= 0
    return float((metrics.ring_distance(pred[keep], true[keep]) <= k).mean())


def diagnostics(pool, label):
    site, t = pool[f"site_{label}"], pool[f"time_{label}"]
    both = (site >= 0) & (pool["site_ott"] >= 0)
    out = {"localized": float((site >= 0).mean()), "median_time": float(np.nanmedian(t)),
           "lag_after_ott_median": float(np.median(t[both] - pool["time_ott"][both])),
           "fires_at_or_after_ott": float((t[both] >= pool["time_ott"][both]).mean()),
           "hump_pm1": ring_acc(pool["site_hump"], site), "hump_exact": ring_acc(pool["site_hump"], site, 0),
           "ott_pm1": ring_acc(pool["site_ott"], site)}
    if label.startswith("thr"):
        sym = pool[f"site_sym{label[3:]}"]
        keep = (site >= 0) & (sym >= 0)
        out["flip_even_split"] = float((site[keep] != sym[keep]).mean())
    return out


def baselines(pools, labels):
    """Box-car and Floquet filter, tuned on validation for each label, scored on test."""
    p = sim.Params()
    E = {s: sim.site_energy(*sim.evolve(pools[s]["theta0"], 10, p), p.g, p.h, "right") for s in ("val", "test")}
    a_hat = {s: pools[s]["theta0"].mean(1) for s in ("val", "test")}
    widths = np.arange(1, 40, 2)
    box = {s: {(k, K): boxcar(E[s][:, k], K).argmax(1) for k in range(10) for K in widths} for s in ("val", "test")}
    flo = {s: {n: floquet.floquet_filter(pools[s]["theta0"], a_hat[s], periods=n) for n in range(3, 10)} for s in ("val", "test")}
    out = {}
    for label in labels:
        yv, yt = pools["val"][f"site_{label}"], pools["test"][f"site_{label}"]
        kb = min(box["val"], key=lambda key: metrics.ring_distance(box["val"][key][yv >= 0], yv[yv >= 0]).mean())
        kf = min(flo["val"], key=lambda n: metrics.ring_distance(flo["val"][n][yv >= 0], yv[yv >= 0]).mean())
        offsets = range(-3, 4)
        hump = {o: (pools["val"]["site_hump"] + o) % 100 for o in offsets}
        ko = max(offsets, key=lambda o: ring_acc(hump[o], yv))
        out[label] = {"boxcar": {"tau": kb[0] * 0.5, "K": int(kb[1]), **metrics.site_metrics(box["test"][kb], yt)},
                      "floquet": {"periods": int(kf), **metrics.site_metrics(flo["test"][kf], yt)},
                      "hump_oracle": {"offset": int(ko), **metrics.site_metrics((pools["test"]["site_hump"] + ko) % 100, yt)}}
    return out


def models_by_label():
    v2, v1 = {}, {}
    with open(ROOT / "results" / "runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") != "threshold":
                continue
            if r.get("group") == "v1":
                v1.setdefault(f"{r['label']}|{r['pool_tag']}", []).append(r)
            else:
                v2.setdefault(r["config"]["label"], []).append(r)

    def summary(rows):
        t = [r["test"] for r in rows]
        out = {"seeds": [r["seed"] for r in rows], "n_test": t[0]["n"]}
        for k in ("acc_0", "acc_1", "acc_2", "acc_3", "mae", "over_10"):
            v = np.array([x[k] for x in t])
            out[k] = float(v.mean())
            out[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
        return out

    v2s = {k: summary(rows) for k, rows in v2.items()}
    v1s = {}
    for k, rows in v1.items():
        trained = [r for r in rows if not r["stalled"]]
        v1s[k] = {"tries": [r["seed"] for r in rows], "stalled": [r["seed"] for r in rows if r["stalled"]],
                  **(summary(trained) if trained else {})}
    return v2s, v1s


def main():
    pools = {s: data.load_pool(ROOT / "data" / f"thrpool_full_{s}.npz") for s in ("val", "test")}
    labels = SHARE_LABELS + [f"sep{k:g}".replace(".", "p") for k in SEPARATRIX] + ["ott"] + REFERENCE
    diag = {s: {l: diagnostics(pools[s], l) for l in labels} for s in pools}
    E0 = sim.site_energy(pools["test"]["theta0"], np.zeros_like(pools["test"]["theta0"]), 0.75, 0.5, "right").sum(1)
    scale = {"E_sep": E_SEP, "E_tot_median": float(np.median(E0)), "E_tot_range": [float(E0.min()), float(E0.max())],
             "E_sep_share_median": float(np.median(E_SEP / E0)),
             "share_in_E_sep": {l: float(np.median(int(l[3:]) / 100 * E0 / E_SEP)) for l in SHARE_LABELS}}
    base = baselines(pools, labels)
    v2, v1 = models_by_label()

    # the physics choice, from validation diagnostics only (the rule was fixed before any model was trained)
    dv = diag["val"]
    passes = {l: dv[l]["fires_at_or_after_ott"] >= 0.95 and dv[l]["hump_pm1"] >= 0.98 for l in labels if l not in REFERENCE}
    ok = [l for l in SHARE_LABELS if passes[l]]
    physics = min(ok, key=lambda l: (dv[l]["flip_even_split"], int(l[3:]))) if ok else None
    ok_any = [l for l, p in passes.items() if p]  # the rule over every candidate, shares or not
    scored = [l for l in SHARE_LABELS if l in v2]
    accuracy = max(scored, key=lambda l: v2[l]["acc_1"]) if scored else None
    corr = {}
    if len(scored) >= 4:
        acc = [v2[l]["acc_1"] for l in scored]
        for name, xs in {"hump_pm1": [diag["test"][l]["hump_pm1"] for l in scored],
                         "flip_even_split": [diag["test"][l]["flip_even_split"] for l in scored],
                         "lag_after_ott": [diag["test"][l]["lag_after_ott_median"] for l in scored],
                         "floquet_pm1": [base[l]["floquet"]["acc_1"] for l in scored],
                         "boxcar_pm1": [base[l]["boxcar"]["acc_1"] for l in scored],
                         "hump_oracle_pm1": [base[l]["hump_oracle"]["acc_1"] for l in scored]}.items():
            rho, pval = spearmanr(acc, xs)
            corr[f"v2_acc1_vs_{name}"] = {"rho": float(rho), "p": float(pval), "n": len(scored)}
    out = {"labels": labels, "scale": scale, "diagnostics": diag, "baselines": base, "v2": v2, "v1": v1,
           "choice": {"physics_rule": "smallest share with fires_at_or_after_ott >= 0.95 and hump_pm1 >= 0.98 on "
                                      "validation, fewest even-split flips", "physics": physics,
                      "physics_candidates": ok, "rule_passes_any_label": ok_any,
                      "rule_margins": {l: {"fires_at_or_after_ott": dv[l]["fires_at_or_after_ott"], "hump_pm1": dv[l]["hump_pm1"]}
                                       for l in ("thr04", "thr05", "thr06", "sep3", "ott")},
                      "accuracy": accuracy}, "spearman": corr}
    (ROOT / "results" / "threshold.json").write_text(json.dumps(out, indent=1))
    print(f"physics choice among shares {physics} (candidates {ok}); rule passes {ok_any}; accuracy choice {accuracy}")
    for l in labels:
        d, b = diag["test"][l], base[l]
        m2 = v2.get(l, {})
        print(f"{l:6s} loc {d['localized']:.4f} t {d['median_time']:5.1f} lag {d['lag_after_ott_median']:+5.1f} "
              f"hump±1 {d['hump_pm1']:.3f} flip {d.get('flip_even_split', float('nan')):.3f} | box {b['boxcar']['acc_1']:.3f} "
              f"flq {b['floquet']['acc_1']:.3f} oracle {b['hump_oracle']['acc_1']:.3f} | v2 {m2.get('acc_1', float('nan')):.3f}")


if __name__ == "__main__":
    main()
