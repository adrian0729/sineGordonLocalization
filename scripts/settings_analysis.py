"""Settings study: where v1's settings come from (the hump label's knobs, amplitude, ring size N, offset size).
Writes results/settings.json and results/settings_tables.md.

    python scripts/settings_analysis.py [--records PATH ...] [--parts A B C D] [--skip-rescore]

Inputs: results/runs.jsonl (plus any --records files; deduped by config_id, seed, created), phases settings_hump,
settings_amplitude, settings_ring, settings_offsets; pools data/setpool_<tag>_<split>.npz (scripts/setting_labels.py);
saved runs under runs/ (pred_test.npy for cross-scoring, model.pt etc. for re-scoring with experiment.evaluate_saved).
Anything whose pool, run record or run directory is missing is skipped and listed under "notes".

A  the hump label's knobs on the main pools: label diagnostics (val, test), trained v2 per label, cross-scoring of the
   settings_hump runs against hump / eh_k3 / ott, the decision rule (validation seed means) and the grid's spread.
B  amplitude: Floquet growth, label timing in e-folding times, the Floquet filter, specialists vs the wide model.
C  ring size N: growing modes, label diagnostics, ring_n<N> models and cross-N transfer.
D  offset size: timing shift against -ln(s)/sigma1, how often the site moves, models and transfer of the scale-1 model.

The target label L is sgloc.ablations.SETTINGS_LABEL. While it is None, parts B-D report only the label-only
diagnostics (with label "hump") and no models. Distances are around the ring.
"""
import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, data, floquet, metrics, sim  # noqa: E402

PHASES = ("settings_hump", "settings_amplitude", "settings_ring", "settings_offsets")
HUMP_LABELS = [f"hump_k{k}_s{s}" for k in (3, 5, 7) for s in (10, 15, 20)]
EH_LABELS = [f"eh_k{k}" for k in (1, 3, 5, 7)]
A_LABELS = HUMP_LABELS + EH_LABELS + ["thr04", "ott", "right"]
BASE = "hump_k3_s15"
GROW = 1e-6
A_RANGE = (1.65, 1.85)
ETA = 0.025          # half-width of the offsets at scale 1 (data.NOISE / 2)
HUMP_RUNS = 600      # test runs integrated for the hump width
NAN = float("nan")
NOTES = []


def note(msg):
    if msg not in NOTES:
        NOTES.append(msg)
        print("NOTE:", msg, flush=True)


# ---------------------------------------------------------------------------------------------------- inputs
def load_records(extra):
    seen, rows = set(), []
    for path in [ROOT / "results" / "runs.jsonl"] + [Path(p) for p in extra]:
        if not path.exists():
            note(f"records file {path} not found")
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("phase") not in PHASES:
                    continue
                key = (r.get("config_id"), r.get("seed"), r.get("created"))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def run_ok(r, *files):
    d = ROOT / r["run_dir"]
    return all((d / f).exists() for f in files)


_POOLS = {}


def pool_of(tag, split):
    key = (tag, split)
    if key not in _POOLS:
        path = ROOT / "data" / f"setpool_{tag}_{split}.npz"
        _POOLS[key] = data.load_pool(path) if path.exists() else None
        if _POOLS[key] is None:
            note(f"pool setpool_{tag}_{split} missing")
    return _POOLS[key]


def pool_has(pool, *labels):
    return pool is not None and all(f"site_{l}" in pool and f"time_{l}" in pool for l in labels)


# ---------------------------------------------------------------------------------------------------- stats helpers
def med(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else NAN


def ring_acc(pred, true, k=1, n_sites=100):
    """Share within ring distance k, over runs where both sites exist (>= 0)."""
    pred, true = np.asarray(pred), np.asarray(true)
    keep = (true >= 0) & (pred >= 0)
    return float((metrics.ring_distance(pred[keep], true[keep], n_sites) <= k).mean()) if keep.any() else NAN


def n_of(pool):
    return pool["theta0"].shape[1]


def rival_share(pool, key="rival_eh3"):
    if key not in pool:
        return NAN
    r = pool[key]
    fin = np.isfinite(r)
    return float((r[fin] >= 0.5).mean()) if fin.any() else NAN


def label_diag(pool, label):
    n = n_of(pool)
    site, t = pool[f"site_{label}"], pool[f"time_{label}"]
    ott_site, ott_t = pool["site_ott"], pool["time_ott"]
    both = (site >= 0) & (ott_site >= 0)
    return {"localized": float((site >= 0).mean()), "median_time": med(t),
            "lag_after_ott_median": med(t[both] - ott_t[both]),
            "hump_pm1": ring_acc(pool["site_hump"], site, 1, n), "hump_exact": ring_acc(pool["site_hump"], site, 0, n),
            "ott_pm1": ring_acc(ott_site, site, 1, n), "eh_k3_pm1": ring_acc(pool["site_eh_k3"], site, 1, n)}


def condition_stats(pool, label):
    """Localized share, median time of the label and of ott, rival share on a test pool."""
    return {"n_runs": int(len(pool["A"])), "localized": float((pool[f"site_{label}"] >= 0).mean()),
            "median_time_L": med(pool[f"time_{label}"]), "median_time_ott": med(pool["time_ott"]),
            "ott_localized": float((pool["site_ott"] >= 0).mean()), "rival_share": rival_share(pool)}


@lru_cache(maxsize=None)
def rates(A, n_sites):
    return floquet.growth_rates(A, range(1, n_sites // 2 + 1), n_sites=n_sites)


def sigma1(A, n_sites=100):
    return float(rates(round(float(A), 6), n_sites).max())


def n_growing(A, n_sites=100):
    return int((rates(round(float(A), 6), n_sites) > GROW).sum())


def per_run_growing(A, n_sites, points=21):
    """Median over runs of the number of growing modes at each run's A (nearest of a grid of amplitudes)."""
    A = np.asarray(A, dtype=float)
    grid = np.unique(np.round(np.linspace(A.min(), A.max(), points), 6)) if np.ptp(A) > 0 else A[:1]
    counts = np.array([n_growing(a, n_sites) for a in grid])
    return float(np.median(counts[np.abs(A[:, None] - grid[None]).argmin(1)])), counts


_FLOQ = {}   # (label, tag) -> (periods, predicted sites on the test pool), filled by floquet_baseline


def floquet_baseline(label, tag):
    """Floquet filter on label: number of periods (3..9) chosen on val by mean ring distance, scored on test."""
    val, test = pool_of(tag, "val"), pool_of(tag, "test")
    if not (pool_has(val, label) and pool_has(test, label)):
        return None
    A_all = np.concatenate([val["A"], test["A"]])
    grid = np.unique(np.round(np.linspace(A_all.min(), A_all.max(), 41), 6)) if np.ptp(A_all) > 1e-9 else A_all[:1]
    n = n_of(val)
    yv, yt = val[f"site_{label}"], test[f"site_{label}"]

    def predict(pool, periods):
        return floquet.floquet_filter(pool["theta0"], pool["A"], periods=periods, a_grid=grid)

    dist = {p: float(metrics.ring_distance(predict(val, p)[yv >= 0], yv[yv >= 0], n).mean()) for p in range(3, 10)}
    best = min(dist, key=dist.get)
    pred = predict(test, best)
    _FLOQ[(label, tag)] = (int(best), pred)
    return {"periods": int(best), "val_mean_distance": dist[best], **metrics.site_metrics(pred, yt, n)}


# ---------------------------------------------------------------------------------------------------- models
def seed_stats(rows, keys=("acc_0", "acc_1", "acc_3", "mae", "over_10"), splits=("test", "val")):
    if not rows:
        return None
    out = {"n_seeds": len(rows), "seeds": sorted(r["seed"] for r in rows)}
    for s in splits:
        out[s] = {"n": rows[0][s]["n"]}
        for k in keys:
            v = np.array([r[s][k] for r in rows], dtype=float)
            out[s][k] = float(v.mean())
            out[s][f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
    return out


def by_name(records, phase, key="name"):
    out = {}
    for r in records:
        if r["phase"] == phase:
            k = r["config"]["label"] if key == "label" else r["name"]
            out.setdefault(k, []).append(r)
    return out


class Rescorer:
    def __init__(self, skip):
        self.skip, self.cache, self.ev = skip, {}, None

    def one(self, r, pool_name):
        key = (r["run_dir"], pool_name)
        if key in self.cache:
            return self.cache[key]
        res = None
        if self.skip:
            note("re-scoring skipped (--skip-rescore)")
        elif not (ROOT / "data" / f"{pool_name}.npz").exists():
            note(f"pool {pool_name} missing: nothing re-scored on it")
        elif not run_ok(r, "model.pt", "model_spec.json", "scaler.npz"):
            note(f"run dir missing for some runs ({r['name']} seed {r['seed']}): not re-scored")
        else:
            try:
                if self.ev is None:
                    from sgloc import experiment
                    self.ev = experiment.evaluate_saved
                res = self.ev(r["run_dir"], pool_name)
            except Exception as e:  # keep going; the failure is listed
                note(f"re-scoring {r['run_dir']} on {pool_name} failed: {type(e).__name__}: {e}")
        self.cache[key] = res
        return res

    def seeds(self, rows, pool_name, keys=("acc_0", "acc_1", "mae", "over_10")):
        """Seed mean of the re-scored metrics over the rows that could be scored."""
        res = [m for m in (self.one(r, pool_name) for r in rows) if m and m.get("n")]
        if not res:
            return None
        out = {"n_seeds": len(res), "n": res[0]["n"]}
        for k in keys:
            v = np.array([m[k] for m in res], dtype=float)
            out[k] = float(v.mean())
            out[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else None
        return out


def std_rows(hump_rows):
    """The settings_hump runs at the standard architecture for the study label L."""
    L = ablations.SETTINGS_LABEL
    if not L:
        return [], None
    name = BASE if L == "hump" else L
    return hump_rows.get(name, []), name


def spearman(x, y):
    ok = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(ok) < 5:
        return {"n": len(ok), "note": "fewer than 5 conditions with models"}
    xs, ys = zip(*ok)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"n": len(ok), "note": "constant input"}
    rho, p = spearmanr(xs, ys)
    return {"rho": float(rho), "p": float(p), "n": len(ok)}


# ---------------------------------------------------------------------------------------------------- Part A
def part_a(records, rescorer):
    out = {"diagnostics": {}, "v2": {}, "cross": {}}
    for split in ("val", "test"):
        pool = pool_of("main", split)
        if not pool_has(pool, *A_LABELS, "hump", "eh_k3"):
            note(f"part A: setpool_main_{split} missing or lacks labels")
            continue
        d = {l: label_diag(pool, l) for l in A_LABELS}
        d["_pool"] = {"rival_eh3": rival_share(pool), "rival_hump": rival_share(pool, "rival_hump"),
                      "n_runs": int(len(pool["A"]))}
        out["diagnostics"][split] = d
    hump_rows = by_name(records, "settings_hump", "label")
    if not hump_rows:
        note("part A: no settings_hump run records yet")
    out["v2"] = {l: seed_stats(hump_rows[l]) for l in A_LABELS if l in hump_rows}
    out["v2"].update({l: seed_stats(rows) for l, rows in hump_rows.items() if l not in out["v2"]})

    test = pool_of("main", "test")
    if pool_has(test, "hump", "eh_k3", "ott"):
        n = n_of(test)
        for l, rows in hump_rows.items():
            res = []
            for r in rows:
                f = ROOT / r["run_dir"] / "pred_test.npy"
                if not f.exists():
                    note("part A: pred_test.npy missing for some settings_hump runs")
                    continue
                pred = np.load(f)
                if len(pred) != len(test["A"]):
                    note(f"part A: pred_test.npy of {r['run_dir']} has the wrong length")
                    continue
                res.append({f"{t}_{k}": ring_acc(pred, test[f"site_{t}"], k, n)
                            for t in ("hump", "eh_k3", "ott") for k in (1, 0)})
            if res:
                out["cross"][l] = {"n_seeds": len(res), **{k: float(np.mean([x[k] for x in res])) for k in res[0]}}

    # decision rule, fixed before the runs: validation seed means
    v = out["v2"]
    if v.get("eh_k3") and v.get(BASE):
        d_acc = v["eh_k3"]["val"]["acc_1"] - v[BASE]["val"]["acc_1"]
        d_mae = v["eh_k3"]["val"]["mae"] - v[BASE]["val"]["mae"]
        ok = v["eh_k3"]["val"]["acc_1"] >= v[BASE]["val"]["acc_1"] - 0.003 and \
            v["eh_k3"]["val"]["mae"] <= v[BASE]["val"]["mae"] + 0.03
        out["decision"] = {"rule": "val seed means: eh_k3 if acc_1(eh_k3) >= acc_1(hump_k3_s15) - 0.003 and "
                                   "mae(eh_k3) <= mae(hump_k3_s15) + 0.03, else hump_k3_s15",
                           "val_acc1_delta": float(d_acc), "val_acc1_delta_points": float(100 * d_acc),
                           "val_mae_delta": float(d_mae), "decision": "eh_k3" if ok else BASE}
    else:
        out["decision"] = None
        note("part A: decision rule needs settings_hump runs for eh_k3 and hump_k3_s15 (none yet)")
    spread = {}
    for name, labels in (("threshold_timed_9", HUMP_LABELS), ("eh_k3_k5_k7", ["eh_k3", "eh_k5", "eh_k7"])):
        a = [v[l]["test"]["acc_1"] for l in labels if v.get(l)]
        spread[name] = {"n_labels": len(a), "max_minus_min_test_acc_1": float(max(a) - min(a)) if len(a) == len(labels) else None}
    out["spread"] = spread
    return out


# ---------------------------------------------------------------------------------------------------- Parts B, C, D
def Lname():
    return ablations.SETTINGS_LABEL or "hump"


def part_b(records, rescorer):
    L = Lname()
    amp = by_name(records, "settings_amplitude")
    hump_rows = by_name(records, "settings_hump", "label")
    std, std_name = std_rows(hump_rows)
    wide = [r for r in amp.get("ampwide", [])]
    out = {"label": L, "standard_label": std_name, "per_amplitude": {}, "ampwide": None}
    for a in ablations.AMPLITUDES:
        tag = f"amp{ablations.tag_of(a)}"
        s1 = sigma1(a)
        entry = {"A": a, "sigma1": s1, "n_growing": n_growing(a), "efold_time": 1 / s1, "in_v1_range": A_RANGE[0] <= a <= A_RANGE[1]}
        test = pool_of(tag, "test")
        if pool_has(test, L, "ott"):
            st = condition_stats(test, L)
            st.update(median_time_L_x_sigma1=st["median_time_L"] * s1, median_time_ott_x_sigma1=st["median_time_ott"] * s1)
            entry["pool"] = st
            entry["floquet"] = floquet_baseline(L, tag)
        entry["specialist"] = seed_stats(amp.get(tag, []))
        pool_name = f"setpool_{tag}_test"
        entry["wide_rescored"] = rescorer.seeds(wide, pool_name) if wide else None
        entry["standard_rescored"] = rescorer.seeds(std, pool_name) if std else None
        entry["standard_in_range"] = entry["in_v1_range"]
        sp, wd = entry["specialist"], entry["wide_rescored"]
        if sp and wd:
            entry["specialist_minus_wide"] = {"acc_1_points": 100 * (sp["test"]["acc_1"] - wd["acc_1"]), "mae": sp["test"]["mae"] - wd["mae"]}
        out["per_amplitude"][tag] = entry
    if not amp:
        note("part B: no settings_amplitude run records yet")

    # the wide pool: t * sigma1(A) per run, sigma1 interpolated from a 36-point grid, 7 amplitude bins
    wp = pool_of("ampwide", "test")
    if pool_has(wp, L, "ott"):
        grid = np.linspace(1.60, 1.95, 36)
        sg = np.array([sigma1(a) for a in grid])
        s_run = np.interp(wp["A"], grid, sg)
        edges = np.linspace(1.60, 1.95, 8)
        b = np.clip(np.digitize(wp["A"], edges) - 1, 0, 6)
        w = {"bins": [], "wide_models": seed_stats(wide)}
        for j in range(7):
            m = b == j
            e = {"A_lo": float(edges[j]), "A_hi": float(edges[j + 1]), "n_runs": int(m.sum())}
            for name, lab in (("L", L), ("ott", "ott")):
                t = wp[f"time_{lab}"][m] * s_run[m]
                e[f"median_t_sigma1_{name}"] = med(t)
            w["bins"].append(e)
        for name, lab in (("L", L), ("ott", "ott")):
            t = wp[f"time_{lab}"] * s_run
            bm = [e[f"median_t_sigma1_{name}"] for e in w["bins"]]
            w[f"overall_{name}"] = {"median": med(t), "bin_median_range": float(np.nanmax(bm) - np.nanmin(bm)),
                                    "bin_median_sd": float(np.nanstd(bm, ddof=1)),
                                    "q25": float(np.nanpercentile(t, 25)), "q75": float(np.nanpercentile(t, 75))}
        w["stats"] = condition_stats(wp, L)
        w["floquet"] = floquet_baseline(L, "ampwide")
        out["ampwide"] = w
    return out


def part_c(records, rescorer):
    L = Lname()
    ring = by_name(records, "settings_ring")
    hump_rows = by_name(records, "settings_hump", "label")
    std, std_name = std_rows(hump_rows)
    out = {"label": L, "standard_label": std_name, "per_n": {}, "cross": {}}
    for N in (50, 100, 200, 400):
        tag = "main" if N == 100 else f"n{N}"
        entry = {"N": N, "pool_tag": tag, "n_growing_A1p75": n_growing(1.75, N), "sigma1_A1p75": sigma1(1.75, N)}
        test = pool_of(tag, "test")
        if pool_has(test, L, "ott"):
            entry["n_growing_median_per_run"], _ = per_run_growing(test["A"], N)
            entry["pool"] = condition_stats(test, L)
            entry["floquet"] = floquet_baseline(L, tag)
        entry["model"] = seed_stats(ring.get(f"ring_n{N}", []), keys=("acc_0", "acc_1", "mae", "over_10"))
        out["per_n"][f"n{N}"] = entry
    if not ring:
        note("part C: no settings_ring run records yet")
    cross = out["cross"]
    for src, targets in ((100, (50, 200, 400)), (200, (100,)), (400, (100,))):
        rows = ring.get(f"ring_n{src}", [])
        for t in targets:
            tag = "main" if t == 100 else f"n{t}"
            cross[f"ring_n{src}_on_n{t}"] = rescorer.seeds(rows, f"setpool_{tag}_test") if rows else None
    for t in (200, 400):
        cross[f"standard_on_n{t}"] = rescorer.seeds(std, f"setpool_n{t}_test") if std else None
    return out


def part_d(records, rescorer):
    L = Lname()
    off = by_name(records, "settings_offsets")
    hump_rows = by_name(records, "settings_hump", "label")
    std, std_name = std_rows(hump_rows)
    scales = sorted(set(ablations.OFFSET_SCALES) | {1})
    out = {"label": L, "standard_label": std_name, "per_scale": {}, "bins": None}
    main = pool_of("main", "test")
    if not pool_has(main, L, "ott"):
        note("part D: setpool_main_test missing or lacks the label")
        return out
    n = n_of(main)
    edges = np.linspace(*A_RANGE, 6)
    b = np.digitize(main["A"], edges) - 1
    b[main["A"] == edges[-1]] = 4
    medA = [float(np.median(main["A"][b == j])) if (b == j).any() else NAN for j in range(5)]
    sg = [sigma1(a) if np.isfinite(a) else NAN for a in medA]
    out["bins"] = [{"A_lo": float(edges[j]), "A_hi": float(edges[j + 1]), "median_A": medA[j], "sigma1": sg[j],
                    "n_runs": int((b == j).sum())} for j in range(5)]
    for s in scales:
        tag = "main" if s == 1 else f"off{ablations.tag_of(s)}"
        entry = {"scale": s, "pool_tag": tag}
        test = pool_of(tag, "test")
        if pool_has(test, L, "ott"):
            entry["pool"] = condition_stats(test, L)
            entry["floquet"] = floquet_baseline(L, tag)
            if len(test["A"]) == len(main["A"]) and np.allclose(test["A"], main["A"]):
                if s != 1:
                    dbins = [{"A_lo": float(edges[j]), "A_hi": float(edges[j + 1]),
                              "predicted": float(-math.log(s) / sg[j]) if np.isfinite(sg[j]) else None} for j in range(5)]
                    entry["delay"] = {"bins": dbins, "prediction_sign": "positive for s < 1"}
                    for name, lab in (("L", L), ("ott", "ott")):
                        dt = test[f"time_{lab}"].astype(float) - main[f"time_{lab}"].astype(float)
                        for j in range(5):
                            dbins[j][f"median_delay_{name}"] = med(dt[(b == j) & np.isfinite(dt)])
                    entry["site_change"] = {}
                    for name, lab in (("L", L), ("ott", "ott")):
                        a, c = test[f"site_{lab}"], main[f"site_{lab}"]
                        keep = (a >= 0) & (c >= 0)
                        d = metrics.ring_distance(a[keep], c[keep], n) if keep.any() else np.array([])
                        entry["site_change"][name] = {"n": int(keep.sum()), "identical": float((d == 0).mean()),
                                                      "within_1": float((d <= 1).mean()), "over_10": float((d > 10).mean())}
            else:
                note(f"part D: setpool_{tag}_test does not share the main pool's runs; delays/site changes skipped")
        entry["model"] = seed_stats(off.get(f"off{ablations.tag_of(s)}", [])) if s != 1 else None
        if s == 1:
            entry["standard_model"] = seed_stats(std) if std else None
            entry["standard_rescored"] = entry["standard_model"]["test"] if std else None
        else:
            entry["standard_rescored"] = rescorer.seeds(std, f"setpool_{tag}_test") if std else None
        out["per_scale"][f"s{ablations.tag_of(s)}"] = entry
    if not off:
        note("part D: no settings_offsets run records yet")
    return out


# ---------------------------------------------------------------------------------------------------- new 1a, 1b
def spec_pred(r, n_runs):
    """pred_test.npy of a run if it exists and matches the pool length."""
    f = ROOT / r["run_dir"] / "pred_test.npy"
    if not f.exists():
        return None
    pred = np.load(f)
    return pred if len(pred) == n_runs else None


def floquet_vs_ratio(records, out):
    """Floquet filter and v2 within +-1 against r = eta_max / (A - A_c): the fixed-amplitude pools (part B) and the
    five amplitude bins of every offset pool (part D)."""
    L = Lname()
    A_c = float(floquet.threshold_amplitude())
    pts = []
    for tag, e in out.get("B", {}).get("per_amplitude", {}).items():
        f, sp = e.get("floquet"), e.get("specialist")
        if not f:
            continue
        pts.append({"source": f"fixed A={e['A']:g}", "kind": "fixed", "A": e["A"], "scale": 1, "eta_max": ETA,
                    "r": ETA / (e["A"] - A_c), "floquet_acc_1": f["acc_1"], "v2_acc_1": g(sp, "test", "acc_1"),
                    "n_runs": g(e, "pool", "n_runs")})
    d = out.get("D")
    main = pool_of("main", "test")
    if d and d.get("bins") and pool_has(main, L):
        n = n_of(main)
        edges = np.linspace(*A_RANGE, 6)
        b = np.digitize(main["A"], edges) - 1
        b[main["A"] == edges[-1]] = 4
        hump_rows = by_name(records, "settings_hump", "label")
        std, _ = std_rows(hump_rows)
        off = by_name(records, "settings_offsets")
        for key, e in d["per_scale"].items():
            s, tag = e["scale"], e["pool_tag"]
            test = pool_of(tag, "test")
            if not pool_has(test, L):
                continue
            if (L, tag) not in _FLOQ:
                floquet_baseline(L, tag)
            if (L, tag) not in _FLOQ or len(test["A"]) != len(main["A"]):
                continue
            pred = _FLOQ[(L, tag)][1]
            rows = std if s == 1 else off.get(f"off{ablations.tag_of(s)}", [])
            preds = [p for p in (spec_pred(r, len(test["A"])) for r in rows) if p is not None]
            for j in range(5):
                m = b == j
                if not m.any():
                    continue
                medA = float(np.median(test["A"][m]))
                v2 = [ring_acc(p[m], test[f"site_{L}"][m], 1, n) for p in preds]
                pts.append({"source": f"scale {s:g}, bin {j + 1}", "kind": "offset", "A": medA, "scale": s, "bin": j,
                            "eta_max": ETA * s, "r": ETA * s / (medA - A_c),
                            "floquet_acc_1": ring_acc(pred[m], test[f"site_{L}"][m], 1, n),
                            "v2_acc_1": float(np.mean(v2)) if v2 else None, "v2_n_seeds": len(v2), "n_runs": int(m.sum())})
    pts.sort(key=lambda q: q["r"])
    return {"A_c": A_c, "points": pts}


def md_ratio(fr):
    rows = [[q["source"], num(q["A"], 3), num(q["eta_max"], 4), num(q["r"], 3), q["n_runs"], pct(q["floquet_acc_1"]),
             pct(q.get("v2_acc_1"))] for q in fr["points"]]
    return (f"## Floquet filter against the offsets' size (A_c = {fr['A_c']:.4f}; r = eta_max / (A - A_c))\n\n"
            + table(["Point", "A (median)", "eta_max", "r", "Runs", "Floquet ±1 %", "v2 ±1 %"], rows))


def hump_width_pool(tag, L_unused=None, n_runs=HUMP_RUNS, chunk=50):
    """FWHM of the site-energy hump at the event: first n_runs test runs, 'sym' energy averaged over samples
    k0-8..k0+7 with k0 = round(time_ott / sample), width counted contiguously around the ring from the top site."""
    test = pool_of(tag, "test")
    if test is None or "time_ott" not in test:
        return None
    p = sim.Params()
    idx = np.arange(min(n_runs, len(test["A"])))
    k0 = np.round(test["time_ott"][idx] / p.sample)
    idx = idx[np.isfinite(k0) & (test["site_ott"][idx] >= 0)]
    k0 = np.round(test["time_ott"][idx] / p.sample).astype(int)
    order = np.argsort(k0)
    idx, k0 = idx[order], k0[order]
    N, widths = n_of(test), []
    for s in range(0, len(idx), chunk):
        sub, ks = idx[s:s + chunk], k0[s:s + chunk]
        th, om = sim.evolve(test["theta0"][sub], int(ks.max()) + 8, p)
        e = sim.site_energy(th, om, p.g, p.h, bond="sym")
        for j, k in enumerate(ks):
            prof = e[j, max(k - 8, 0):k + 8].mean(axis=0)
            top = int(prof.argmax())
            above, width = prof >= prof[top] / 2, 1
            for step in (1, -1):
                n = top
                while above[(n + step) % N] and width < N:
                    n, width = (n + step) % N, width + 1
            widths.append(width)
        del th, om, e
    w = np.asarray(widths, dtype=float)
    return {"n_runs": int(len(w)), "median": float(np.median(w)), "p10": float(np.percentile(w, 10)),
            "p90": float(np.percentile(w, 90)), "mean": float(w.mean())}


def hump_widths(out):
    res = {}
    todo = []   # (group, key, tag)
    for a in ablations.AMPLITUDES:
        todo.append(("fixed_amplitude", f"A={a:g}", f"amp{ablations.tag_of(a)}"))
    for s in sorted(set(ablations.OFFSET_SCALES) | {1}):
        todo.append(("offsets", f"scale {s:g}", "main" if s == 1 else f"off{ablations.tag_of(s)}"))
    for N in (50, 100, 200, 400):
        todo.append(("ring", f"N={N}", "main" if N == 100 else f"n{N}"))
    cache = {}
    for group, key, tag in todo:
        if tag not in cache:
            if pool_of(tag, "test") is None:
                cache[tag] = None
            else:
                print(f"hump width: {tag}", flush=True)
                cache[tag] = hump_width_pool(tag)
        if cache[tag]:
            res.setdefault(group, {})[key] = {"pool_tag": tag, **cache[tag]}
    return res


def md_hump(h):
    s = [f"## Hump width at the event (FWHM in pendulums, first {HUMP_RUNS} test runs; site energy averaged over samples k0-8..k0+7)\n"]
    for group, d in h.items():
        s.append(f"### {group}\n")
        s.append(table(["Condition", "Pool", "Runs", "Median", "10th pct", "90th pct", "Mean"],
                       [[k, v["pool_tag"], v["n_runs"], num(v["median"], 1), num(v["p10"], 1), num(v["p90"], 1), num(v["mean"], 2)]
                        for k, v in d.items()]))
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------------------------------------------- output
def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def num(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def pm(m, sd, scale=100, d=1):
    if m is None:
        return "–"
    s = f"{scale * m:.{d}f}"
    return s + (f" ± {scale * sd:.{d}f}" if sd is not None else "")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def g(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def md_a(a):
    s = ["## Part A: the hump label's knobs (main pools)\n"]
    for split in ("val", "test"):
        d = a["diagnostics"].get(split)
        if not d:
            continue
        p = d["_pool"]
        s.append(f"### Label diagnostics, {split} ({p['n_runs']:,} runs; rival_eh3 >= 0.5 in {pct(p['rival_eh3'])}% of runs)\n")
        s.append(table(["Label", "Localized %", "Median time", "Lag after ott", "Hump ±1 %", "Hump exact %", "ott ±1 %", "eh_k3 ±1 %"],
                       [[l, pct(x["localized"]), num(x["median_time"], 1), num(x["lag_after_ott_median"], 1), pct(x["hump_pm1"]),
                         pct(x["hump_exact"]), pct(x["ott_pm1"]), pct(x["eh_k3_pm1"])] for l, x in d.items() if l != "_pool"]))
    s.append("### Trained v2 (seed mean ± sd; accuracies in %, mean distance in pendulums)\n")
    rows = [[l, v["n_seeds"], pm(g(v, "test", "acc_0"), g(v, "test", "acc_0_sd")), pm(g(v, "test", "acc_1"), g(v, "test", "acc_1_sd")),
             pm(g(v, "test", "acc_3"), g(v, "test", "acc_3_sd")), pm(g(v, "test", "mae"), g(v, "test", "mae_sd"), 1, 2),
             pm(g(v, "test", "over_10"), g(v, "test", "over_10_sd")), pm(g(v, "val", "acc_1"), g(v, "val", "acc_1_sd")),
             pm(g(v, "val", "mae"), g(v, "val", "mae_sd"), 1, 2)] for l, v in a["v2"].items() if v]
    s.append(table(["Label", "Seeds", "Exact", "±1", "±3", "Mean dist", ">10 off", "Val ±1", "Val mean dist"], rows)
             if rows else "No settings_hump runs yet.\n")
    s.append("### Cross-scoring of the settings_hump runs (test, seed means)\n")
    rows = [[l, c["n_seeds"]] + [pct(c[f"{t}_{k}"]) for t in ("hump", "eh_k3", "ott") for k in (1, 0)] for l, c in a["cross"].items()]
    s.append(table(["Trained on", "Seeds", "vs hump ±1", "vs hump exact", "vs eh_k3 ±1", "vs eh_k3 exact", "vs ott ±1", "vs ott exact"], rows)
             if rows else "No settings_hump predictions available yet.\n")
    dec = a.get("decision")
    s.append("### Decision and spread\n")
    if dec:
        s.append(f"- Validation eh_k3 − hump_k3_s15: ±1 {100 * dec['val_acc1_delta']:+.2f} points (rule: >= −0.30), "
                 f"mean distance {dec['val_mae_delta']:+.3f} (rule: <= +0.03). **Decision: {dec['decision']}**")
    else:
        s.append("- Decision: not available (settings_hump runs for eh_k3 and hump_k3_s15 needed).")
    for k, v in a["spread"].items():
        s.append(f"- Test ±1 spread (max − min) across {k}: " + (f"{100 * v['max_minus_min_test_acc_1']:.2f} points" if v["max_minus_min_test_acc_1"] is not None else "–"))
    return "\n".join(s) + "\n"


def md_b(b):
    s = [f"## Part B: amplitude (label {b['label']}; standard model: {b['standard_label'] or 'n/a'})\n", "### Physics and label timing (test pools)\n"]
    rows = []
    for tag, e in b["per_amplitude"].items():
        p, f = e.get("pool"), e.get("floquet")
        rows.append([e["A"], num(e["sigma1"], 4), e["n_growing"], num(e["efold_time"], 1),
                     pct(g(p, "localized")), num(g(p, "median_time_L"), 1), num(g(p, "median_time_ott"), 1),
                     num(g(p, "median_time_L_x_sigma1")), num(g(p, "median_time_ott_x_sigma1")), pct(g(p, "rival_share")),
                     f"{g(f, 'periods')}: {pct(g(f, 'acc_1'))} / {num(g(f, 'mae'))}" if f else "–"])
    s.append(table(["A", "sigma1", "Growing modes", "e-fold time", "Localized %", "t(L)", "t(ott)", "t(L)·sigma1", "t(ott)·sigma1",
                    "Rival %", "Floquet periods: ±1 % / dist"], rows))
    s.append("### Models (test; accuracies in %, mean distance in pendulums)\n")
    rows = []
    for tag, e in b["per_amplitude"].items():
        sp, wd, sd_ = e.get("specialist"), e.get("wide_rescored"), e.get("standard_rescored")
        diff = e.get("specialist_minus_wide")
        rows.append([e["A"], sp["n_seeds"] if sp else "–", pm(g(sp, "test", "acc_1"), g(sp, "test", "acc_1_sd")),
                     pm(g(sp, "test", "mae"), g(sp, "test", "mae_sd"), 1, 2), pm(g(sp, "test", "acc_0"), g(sp, "test", "acc_0_sd")),
                     pm(g(sp, "test", "over_10"), g(sp, "test", "over_10_sd")),
                     pct(g(wd, "acc_1")), num(g(wd, "mae")), f"{diff['acc_1_points']:+.2f}" if diff else "–", f"{diff['mae']:+.3f}" if diff else "–",
                     pct(g(sd_, "acc_1")), num(g(sd_, "mae")), "in range" if e["standard_in_range"] else "extrapolated"])
    s.append(table(["A", "Seeds", "Specialist ±1", "Specialist dist", "Specialist exact", "Specialist >10", "Wide ±1", "Wide dist",
                    "Spec − wide ±1 (pts)", "Spec − wide dist", "Standard ±1", "Standard dist", "Standard model"], rows))
    w = b.get("ampwide")
    if w:
        s.append("### Wide pool: median t·sigma1(A) per amplitude bin (test)\n")
        rows = [[f"{e['A_lo']:.3f}–{e['A_hi']:.3f}", e["n_runs"], num(e["median_t_sigma1_L"]), num(e["median_t_sigma1_ott"])] for e in w["bins"]]
        for name in ("L", "ott"):
            o = w[f"overall_{name}"]
            rows.append([f"overall {name}", "", f"median {num(o['median'])}, bin range {num(o['bin_median_range'])}, sd {num(o['bin_median_sd'])}, IQR {num(o['q25'])}–{num(o['q75'])}", ""])
        s.append(table(["Bin", "Runs", "t(L)·sigma1", "t(ott)·sigma1"], rows))
    return "\n".join(s) + "\n"


def md_c(c):
    s = [f"## Part C: ring size (label {c['label']}; standard model: {c['standard_label'] or 'n/a'})\n", "### Physics and label timing (test pools)\n"]
    rows = []
    for k, e in c["per_n"].items():
        p, f = e.get("pool"), e.get("floquet")
        rows.append([e["N"], e["n_growing_A1p75"], num(e.get("n_growing_median_per_run"), 1), pct(g(p, "localized")),
                     num(g(p, "median_time_L"), 1), num(g(p, "median_time_ott"), 1), pct(g(p, "rival_share")),
                     f"{g(f, 'periods')}: {pct(g(f, 'acc_1'))} / {num(g(f, 'mae'))}" if f else "–"])
    s.append(table(["N", "Growing modes (A=1.75)", "Growing modes (median per run)", "Localized %", "t(L)", "t(ott)", "Rival %",
                    "Floquet periods: ±1 % / dist"], rows))
    s.append("### ring_n<N> models (test)\n")
    rows = []
    for k, e in c["per_n"].items():
        m = e.get("model")
        rows.append([e["N"], m["n_seeds"] if m else "–"] + [pm(g(m, "test", q), g(m, "test", q + "_sd"), 1, 2) if q == "mae" else
                                                          pm(g(m, "test", q), g(m, "test", q + "_sd")) for q in ("acc_0", "acc_1", "mae", "over_10")])
    s.append(table(["N", "Seeds", "Exact", "±1", "Mean dist", ">10 off"], rows))
    s.append("### Cross-N transfer (re-scored on the test pool of the target N; seed means)\n")
    rows = [[k, g(v, "n_seeds") or "–", pct(g(v, "acc_0")), pct(g(v, "acc_1")), num(g(v, "mae")), pct(g(v, "over_10"))] for k, v in c["cross"].items()]
    s.append(table(["Model on pool", "Seeds", "Exact", "±1", "Mean dist", ">10 off"], rows))
    return "\n".join(s) + "\n"


def md_d(d):
    s = [f"## Part D: offset size (label {d['label']}; standard model: {d['standard_label'] or 'n/a'})\n"]
    if d["bins"] is None:
        return s[0] + "\nMain test pool missing.\n"
    s.append("### Test pools\n")
    rows = []
    for k, e in d["per_scale"].items():
        p, f, c = e.get("pool"), e.get("floquet"), e.get("site_change", {})
        sc = lambda n, q: pct(g(c, n, q))  # noqa: E731
        rows.append([e["scale"], pct(g(p, "localized")), num(g(p, "median_time_L"), 1), num(g(p, "median_time_ott"), 1),
                     sc("L", "identical"), sc("L", "within_1"), sc("L", "over_10"), sc("ott", "identical"), sc("ott", "within_1"), sc("ott", "over_10"),
                     f"{g(f, 'periods')}: {pct(g(f, 'acc_1'))} / {num(g(f, 'mae'))}" if f else "–"])
    s.append(table(["Scale", "Localized %", "t(L)", "t(ott)", "L same %", "L ±1 %", "L >10 %", "ott same %", "ott ±1 %", "ott >10 %",
                    "Floquet periods: ±1 % / dist"], rows))
    s.append("### Median delay vs scale 1 per amplitude bin (time at scale s − time at scale 1; measured L / ott, predicted −ln(s)/sigma1)\n")
    header = ["Scale"] + [f"{b_['A_lo']:.2f}–{b_['A_hi']:.2f} (A={num(b_['median_A'], 3)})" for b_ in d["bins"]]
    rows = []
    for k, e in d["per_scale"].items():
        dl = e.get("delay")
        if dl:
            rows.append([e["scale"]] + [f"{num(b_.get('median_delay_L'), 1)} / {num(b_.get('median_delay_ott'), 1)} / {num(b_.get('predicted'), 1)}" for b_ in dl["bins"]])
    s.append(table(header, rows))
    s.append("### Models (test)\n")
    rows = []
    for k, e in d["per_scale"].items():
        m, st = e.get("model"), e.get("standard_rescored")
        rows.append([e["scale"], m["n_seeds"] if m else "–", pm(g(m, "test", "acc_1"), g(m, "test", "acc_1_sd")),
                     pm(g(m, "test", "mae"), g(m, "test", "mae_sd"), 1, 2), pm(g(m, "test", "acc_0"), g(m, "test", "acc_0_sd")),
                     pm(g(m, "test", "over_10"), g(m, "test", "over_10_sd")),
                     pct(g(st, "acc_1")), num(g(st, "mae"))])
    s.append(table(["Scale", "Seeds", "Specialist ±1", "Specialist dist", "Specialist exact", "Specialist >10", "Scale-1 model ±1", "Scale-1 model dist"], rows))
    return "\n".join(s) + "\n"


def md_spearman(sp):
    rows = [[k, num(v.get("rho"), 3), num(v.get("p"), 4), v["n"], v.get("note", "")] for k, v in sp.items()]
    return "## Spearman (v2 test ±1 against condition properties)\n\n" + table(["Pair", "rho", "p", "n", "Note"], rows)


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="*", default=[], help="extra JSON-lines run files")
    ap.add_argument("--parts", nargs="+", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--skip-rescore", action="store_true", help="skip experiment.evaluate_saved calls (slow)")
    args = ap.parse_args()

    records = load_records(args.records)
    rescorer = Rescorer(args.skip_rescore)
    if not ablations.SETTINGS_LABEL:
        note("SETTINGS_LABEL is None: parts B-D use label 'hump' for the label-only diagnostics and report no models")
    out = {"label": ablations.SETTINGS_LABEL, "label_for_diagnostics": Lname(),
           "records": {p: sum(r["phase"] == p for r in records) for p in PHASES}}
    md = ["# Settings study\n"]
    fns = {"A": (part_a, md_a), "B": (part_b, md_b), "C": (part_c, md_c), "D": (part_d, md_d)}
    for part in args.parts:
        fn, mdfn = fns[part]
        out[part] = fn(records, rescorer)
        md.append(mdfn(out[part]))

    # Spearman across conditions with models
    sp = {}
    if "B" in out:
        pe = out["B"]["per_amplitude"]
        tags = [t for t, e in pe.items() if e.get("specialist") and e.get("pool")]
        acc = [pe[t]["specialist"]["test"]["acc_1"] for t in tags]
        sp["amplitude: acc_1 vs growing modes"] = spearman(acc, [pe[t]["n_growing"] for t in tags])
        sp["amplitude: acc_1 vs rival share"] = spearman(acc, [pe[t]["pool"]["rival_share"] for t in tags])
    if "C" in out:
        pn = out["C"]["per_n"]
        ks = [k for k, e in pn.items() if e.get("model") and e.get("pool")]
        acc = [pn[k]["model"]["test"]["acc_1"] for k in ks]
        sp["N: acc_1 vs growing modes (median per run)"] = spearman(acc, [pn[k]["n_growing_median_per_run"] for k in ks])
        sp["N: acc_1 vs rival share"] = spearman(acc, [pn[k]["pool"]["rival_share"] for k in ks])
    if sp:
        out["spearman"] = sp
        md.append(md_spearman(sp))
    if "B" in out or "D" in out:
        out["floquet_vs_ratio"] = floquet_vs_ratio(records, out)
        md.append(md_ratio(out["floquet_vs_ratio"]))
    out["hump_width"] = hump_widths(out)
    md.append(md_hump(out["hump_width"]))
    out["notes"] = NOTES
    if NOTES:
        md.append("## Notes\n\n" + "\n".join(f"- {n}" for n in NOTES) + "\n")

    (ROOT / "results" / "settings.json").write_text(json.dumps(clean(out), indent=1))
    (ROOT / "results" / "settings_tables.md").write_text("\n".join(md))

    # short summary
    print(f"label {ablations.SETTINGS_LABEL}; records {out['records']}")
    if "A" in out:
        for split, d in out["A"]["diagnostics"].items():
            print(f"[A {split}] rival_eh3>=0.5 in {pct(d['_pool']['rival_eh3'])}% of runs")
            for l, x in d.items():
                if l != "_pool":
                    print(f"{l:12s} loc {x['localized']:.4f} t {x['median_time']:5.1f} lag {x['lag_after_ott_median']:+5.1f} "
                          f"hump±1 {x['hump_pm1']:.3f} hump0 {x['hump_exact']:.3f} ott±1 {x['ott_pm1']:.3f} eh3±1 {x['eh_k3_pm1']:.3f}")
        print("[A] decision:", out["A"]["decision"] and (out["A"]["decision"]["decision"], round(out["A"]["decision"]["val_acc1_delta_points"], 2),
                                                          round(out["A"]["decision"]["val_mae_delta"], 3)))
        print("[A] spread:", out["A"]["spread"])
    if "B" in out:
        for tag, e in out["B"]["per_amplitude"].items():
            p = e.get("pool")
            print(f"[B {tag}] sigma1 {e['sigma1']:.4f} modes {e['n_growing']}" + (f" t(L)·s1 {p['median_time_L_x_sigma1']:.2f} rival {p['rival_share']:.3f}" if p else " (no pool)"))
    if "C" in out:
        for k, e in out["C"]["per_n"].items():
            p = e.get("pool")
            print(f"[C {k}] modes@1.75 {e['n_growing_A1p75']}" + (f" median/run {e['n_growing_median_per_run']:.1f} rival {p['rival_share']:.3f}" if p else " (no pool)"))
    if "D" in out:
        for k, e in out["D"]["per_scale"].items():
            p = e.get("pool")
            print(f"[D {k}] " + (f"t(L) {p['median_time_L']:.1f} t(ott) {p['median_time_ott']:.1f}" if p else "(no pool)"))
    for q in out.get("floquet_vs_ratio", {}).get("points", []):
        print(f"[ratio] {q['source']:22s} r {q['r']:.3f} floquet {pct(q['floquet_acc_1'])} v2 {pct(q.get('v2_acc_1'))}")
    for grp, d in out["hump_width"].items():
        for k, v in d.items():
            print(f"[hump width {grp}] {k}: median {v['median']:.0f} (10th {v['p10']:.0f}, 90th {v['p90']:.0f})")
    for k, v in sp.items():
        print("[spearman]", k, v)
    print("wrote results/settings.json and results/settings_tables.md")


if __name__ == "__main__":
    main()
