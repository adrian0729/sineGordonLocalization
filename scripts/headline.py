"""The v1 vs v2 headline tables, assembled from the saved results into results/headline.md.

    python scripts/headline.py

Reads results/runs.jsonl (training runs), results/robustness.jsonl (re-evaluation of trained runs on
shifted test conditions, including damped chains) and results/phase1.json (physics baselines). Anything
not run yet shows as pending.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402

COLUMNS = [("mae", "Mean distance", 2, 1), ("acc_0", "Exact", 1, 100), ("acc_1", "±1", 1, 100),
           ("acc_3", "±3", 1, 100), ("over_10", ">10 off", 1, 100)]
CONDITIONS = [("pool_test", "nominal test pool"), ("robust_g-10", "coupling g 10% low"), ("robust_g+10", "coupling g 10% high"),
              ("robust_h-10", "global coupling h 10% low"), ("robust_h+10", "global coupling h 10% high"),
              ("robust_noise2x", "twice the initial spread"), ("robust_N80", "N = 80 pendulums"), ("robust_N150", "N = 150 pendulums"),
              ("robust_impurity1", "one pendulum 1% weaker (2 mm longer)"), ("test_noise0.0001", "sensor noise 1e-4 rad (untrained)"),
              ("test_noise0.0003", "sensor noise 3e-4 rad (untrained)"), ("test_noise0.001", "sensor noise 1e-3 rad (untrained)"),
              ("test_noise0.003", "sensor noise 3e-3 rad (untrained)")]


def stat(values, digits=2, scale=1):
    v = np.asarray(values, float) * scale
    if len(v) == 0:
        return "pending"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def cells(tests):
    return [stat([t[key] for t in tests], d, s) for key, _, d, s in COLUMNS]


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def load():
    by_id, v1 = {}, []
    with open(experiment.RESULTS) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("config_id"):
                    by_id[rec["config_id"]] = rec
                elif rec.get("model") == "v1_cnn_bilstm" and rec.get("label", "right") == "right":
                    v1.append(rec)
    phase6 = defaultdict(list)  # name -> records
    dirs = defaultdict(list)    # name -> run dirs
    notes = {}
    for cfg in ablations.phase6():
        cid = experiment.config_id(cfg)
        notes[cfg["name"]] = (cfg["group"], cfg["note"])
        if cid in by_id:
            phase6[cfg["name"]].append(by_id[cid])
            dirs[cfg["name"]].append(f"runs/{cfg['name']}_s{cfg['seed']}_{cid}")
    robust = {}
    path = ROOT / "results" / "robustness.jsonl"
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    robust[(rec["run_dir"], rec["condition"])] = rec["test"]  # the latest evaluation wins
    return v1, phase6, dirs, notes, robust


GAMMAS = [("robust_gamma0", 0.0), ("robust_gamma1e-4", 1e-4), ("robust_gamma3e-4", 3e-4), ("robust_gamma1e-3", 1e-3),
          ("robust_gamma3e-3", 3e-3), ("robust_gamma1e-2", 1e-2)]
DAMPED = [("damped_trained", "v2 recipe, trained on damped chains", "recipe"),
          ("damped_trained_localizes_head", "... plus the 'does it localize' head", "+ head"),
          ("damped_trained_avg_label", "... plus the head, on the swing-averaged label", "+ head, averaged label"),
          ("damped_deploy_camera_every2_noise1e-3", "... plus the head, camera on every 2nd pendulum, 1e-3 rad",
           "+ head, camera")]


def damping_section(v1_dirs, p6, dirs, robust):
    """Damped chains: models trained undamped and re-scored on damped pools, and the recipe retrained on them."""
    def scored(run_dirs, cond):
        return [robust[(d, cond)] for d in run_dirs if (d, cond) in robust]

    def site(tests, key="mae", digits=2, scale=1):
        tests = [t for t in tests if t.get("n")]
        return stat([t[key] for t in tests], digits, scale) if tests else "n/a"

    def head(tests, key="accuracy"):
        tests = [t["localizes"] for t in tests if "localizes" in t]
        return stat([t[key] for t in tests if key in t], 1, 100) if tests else "n/a"

    out = ["## 9. Damped chains (linear damping gamma per time unit; quality factor Q = 1 / gamma)", "",
           "Six pools of 5,000 runs share their initial conditions and differ only in gamma. Sites are scored over the runs "
           "that localize by τ = 150, against the label each model was trained on. \"Undamped frames\" feeds a model the "
           "first frames of the same runs without damping and scores it against the damped outcome, which separates a "
           "changed outcome from changed inputs.", ""]
    rows = []
    v2, avg = dirs.get("v2_final", []), dirs.get("final_avg_label", [])
    for cond, gamma in GAMMAS:
        any_test = next(iter(scored(v2, cond)), None)
        if any_test is None:
            continue
        b, c = scored(v2, cond), scored(avg, cond)
        b0, c0 = scored(v2, f"{cond}_labels_only"), scored(avg, f"{cond}_labels_only")
        rows.append([f"{gamma:g}", f"{1 / gamma:,.0f}" if gamma else "∞", f"{100 * any_test['n'] / 5000:.1f}%",
                     site(scored(v1_dirs, cond)), site(b), site(b, "acc_1", 1, 100), site(b0), site(c), site(c, "acc_1", 1, 100),
                     site(c0)])
    out += ["### Trained on undamped chains (mean distance; ±1 in %)", "",
            table(["gamma", "Q", "Localize by τ = 150", "v1 rebuilt", "v2 final", "v2 final ±1", "v2 final, undamped frames",
                   "v2 swing-averaged label", "its ±1", "its undamped frames"], rows), ""]
    rows = []
    for cond, gamma in GAMMAS:
        cols = []
        for name, _, _ in DAMPED:
            tests = scored(dirs.get(name, []), cond)
            cols += [site(tests), site(tests, "acc_1", 1, 100)]
        if any(c not in ("n/a", "pending") for c in cols) or cond == "robust_gamma1e-2":
            heads = head(scored(dirs.get("damped_trained_localizes_head", []), cond))
            rows.append([f"{gamma:g}"] + cols + [heads])
    if rows:
        out += ["### Trained on damped chains (gamma log-uniform in 1e-4 .. 1e-2), scored at each gamma", "",
                table(["gamma"] + [f"{short}: {m}" for _, _, short in DAMPED for m in ("mean distance", "±1")] +
                      ["Localizes head: accuracy"], rows), ""]
    rows = []
    for label, who in (("v1 rebuilt, trained undamped", v1_dirs), ("v2 final, trained undamped", dirs.get("v2_final", [])),
                       ("v2 swing-averaged label, trained undamped", dirs.get("final_avg_label", []))):
        tests = scored(who, "robust_damped_test")
        rows.append([label, len(tests)] + cells(tests) + ["n/a"] * 3)
    for name, label, _ in DAMPED:
        tests = [r["test"] for r in p6.get(name, [])]
        rows.append([label, len(tests)] + cells(tests) + [head(tests), head(tests, "auc"), head(tests, "false_alarm")])
    out += ["### The damped test pool (10,000 runs, gamma log-uniform in 1e-4 .. 1e-2, 80% localize)", "",
            table(["Model", "Seeds", *[c[1] for c in COLUMNS], "Localizes: accuracy", "ROC area", "False alarms"], rows), ""]
    return out


def minutes(recs):
    return f"{np.mean([r['train_seconds'] for r in recs]) / 60:.0f} min" if recs else ""


def main():
    v1, p6, dirs, notes, robust = load()
    final = p6.get("v2_final", [])
    out = ["# v1 vs v2: headline results", "",
           "Generated by `scripts/headline.py`. Test pool of 20,000 runs unless noted; mean ± standard deviation over seeds.", ""]

    # 1. headline
    base = json.loads((ROOT / "results" / "phase1.json").read_text())["baselines"]
    boxcar = min(base["boxcar"], key=lambda b: b["mae"])
    rows = [["v1 as published (paper, not reproduced)", "26.5M", "", "1.44", "", "94.4", "", ""],
            [f"v1 rebuilt ({len(v1)} seeds)", f"{v1[0]['params'] / 1e6:.1f}M" if v1 else "", minutes(v1)] + cells([r["test"] for r in v1]),
            [f"Box-car width {boxcar['K']}, best frame (physics)", "0", ""] + cells([boxcar]),
            ["Floquet filter (physics)", "0", ""] + cells([base["floquet"]["test"]]),
            ["Learned linear ring filter", "200", ""] + cells([base["linear"]["linear_dtheta0_amp"]["test"]]),
            [f"**v2 final ({len(final)} seeds)**", f"{final[0]['params']:,}" if final else "", minutes(final)] +
            [f"**{c}**" for c in cells([r["test"] for r in final])]]
    out += ["## 1. Accuracy, size and cost", "",
            table(["Model", "Parameters", "Train time", *[c[1] for c in COLUMNS]], rows), ""]

    # 2. leave-one-out
    f_mae = np.array([r["test"]["mae"] for r in final])
    rows = []
    for name, (group, note) in notes.items():
        if group != "leave_one_out":
            continue
        recs = p6.get(name, [])
        mae = np.array([r["test"]["mae"] for r in recs])
        if len(mae) >= 2 and len(f_mae) >= 2:
            se = np.sqrt(mae.var(ddof=1) / len(mae) + f_mae.var(ddof=1) / len(f_mae))
            delta = f"{mae.mean() - f_mae.mean():+.2f} ± {se:.2f}"
        else:
            delta = "pending"
        rows.append([note.replace("Q5 leave-one-out: final model ", ""), len(recs), stat(mae), delta,
                     stat([r["test"]["acc_1"] for r in recs], 1, 100), stat([r["test"]["acc_3"] for r in recs], 1, 100)])
    out += ["## 2. Leave one choice out", "",
            "Δ is the change in mean distance against the final model (positive = worse), ± its standard error.", "",
            table(["Final model ...", "Seeds", "Mean distance", "Δ vs final", "±1", "±3"], rows), ""]

    # 3. data efficiency
    rows = []
    v1_mae = np.mean([r["test"]["mae"] for r in v1]) if v1 else None
    for n in (1000, 3000, 10_000, 30_000, 60_000, 100_000):
        recs = final if n == 60_000 else p6.get(f"final_n{n}", [])
        rows.append([f"{n:,}", len(recs), stat([r["test"]["mae"] for r in recs]),
                     stat([r["test"]["acc_1"] for r in recs], 1, 100), stat([r["test"]["acc_3"] for r in recs], 1, 100)])
    rows.append(["60,000 (v1 rebuilt)", len(v1), stat([r["test"]["mae"] for r in v1]),
                 stat([r["test"]["acc_1"] for r in v1], 1, 100), stat([r["test"]["acc_3"] for r in v1], 1, 100)])
    out += ["## 3. Data efficiency (v2 final recipe, fixed number of steps)", "",
            table(["Training runs", "Seeds", "Mean distance", "±1", "±3"], rows), ""]
    if v1_mae is not None:
        beat = [n for n in (1000, 3000, 10_000, 30_000) if p6.get(f"final_n{n}") and
                np.mean([r["test"]["mae"] for r in p6[f"final_n{n}"]]) < v1_mae]
        if beat:
            out += [f"v2 beats the rebuilt v1 (mean distance {v1_mae:.2f}) from {min(beat):,} training runs on.", ""]

    # 4. robustness without retraining
    v1_dirs = [f"runs/{r['run_id']}" for r in v1]
    rows = []
    for cond, label in CONDITIONS:
        a = [robust[(d, cond)] for d in v1_dirs if (d, cond) in robust]
        b = [robust[(d, cond)] for d in dirs.get("v2_final", []) if (d, cond) in robust]
        if not a and not b:
            continue
        rows.append([label, stat([t["mae"] for t in a]) if a else "n/a", stat([t["acc_3"] for t in a], 1, 100) if a else "n/a",
                     stat([t["mae"] for t in b]), stat([t["acc_3"] for t in b], 1, 100)])
    out += ["## 4. Robustness: trained on the nominal chain, tested on shifted ones", "",
            "Shifted pools have 5,000 runs (10,000 for the impurity pool). v1 needs exactly 100 pendulums, so N = 80 and 150 are n/a.", "",
            table(["Test condition", "v1 mean distance", "v1 ±3", "v2 mean distance", "v2 ±3"], rows), ""]

    # 5. noise and impurity, with training
    rows = []
    for cond, who, label in (("test_noise0.001", v1_dirs, "v1, trained noise-free"),
                             ("test_noise0.001", dirs.get("v2_final", []), "v2 final, trained noise-free")):
        tests = [robust[(d, cond)] for d in who if (d, cond) in robust]
        rows.append([label, len(tests)] + cells(tests))
    recs = p6.get("final_trained_noise1e-3", [])
    rows.append(["v2 final recipe, trained with 1e-3 rad noise", len(recs)] + cells([r["test"] for r in recs]))
    out += ["## 5. Sensor noise of 1e-3 rad at test time", "", table(["Model", "Seeds", *[c[1] for c in COLUMNS]], rows), ""]

    rows = []
    tests = [robust[(d, "robust_impurity_test")] for d in dirs.get("v2_final", []) if (d, "robust_impurity_test") in robust]
    rows.append(["v2 final, trained on uniform chains", len(tests)] + cells(tests))
    tests = [robust[(d, "robust_impurity_test")] for d in v1_dirs if (d, "robust_impurity_test") in robust]
    rows.insert(0, ["v1, trained on uniform chains", len(tests)] + cells(tests))
    for name, label in (("impurity_trained", "v2 recipe, trained with impurities, no map"),
                        ("impurity_trained_with_map", "v2 recipe, trained with impurities, given a measured map of every pendulum")):
        recs = p6.get(name, [])
        rows.append([label, len(recs)] + cells([r["test"] for r in recs]))
    out += ["## 6. Chains with one impurity (one random pendulum's restoring force 0–2% weaker; 10,000 test runs)", "",
            table(["Model", "Seeds", *[c[1] for c in COLUMNS]], rows), ""]

    # 7. deployable sensing setups
    rows = []
    for name, (group, note) in notes.items():
        if group == "deploy":
            recs = p6.get(name, [])
            rows.append([note.replace("Q1/Q5: final recipe, ", ""), len(recs)] + cells([r["test"] for r in recs]))
    out += ["## 7. The final recipe on real-world sensing setups", "", table(["Setup", "Seeds", *[c[1] for c in COLUMNS]], rows), ""]

    # 8. the period-averaged label
    recs = p6.get("final_avg_label", [])
    sym = p6.get("final_sym_label", [])
    v1_avg = []
    with open(experiment.RESULTS) as f:
        v1_avg = [r for r in (json.loads(l) for l in f if l.strip()) if r.get("model") == "v1_cnn_bilstm" and r.get("label") == "avg"]
    rows = [["v2 final: trained and scored on v1's label (first site over 10%)", len(final)] + cells([r["test"] for r in final]),
            ["v2 recipe: trained and scored on the symmetric label (springs split, not averaged)", len(sym)] + cells([r["test"] for r in sym]),
            ["v2 recipe: trained and scored on the period-averaged label", len(recs)] + cells([r["test"] for r in recs]),
            ["the same model, scored on v1's label", len(recs)] +
            cells([r["test"]["right_label"] for r in recs if "right_label" in r["test"]]),
            ["v1 rebuilt: trained and scored on v1's label", len(v1)] + cells([r["test"] for r in v1]),
            ["v1 rebuilt: trained and scored on the period-averaged label", len(v1_avg)] + cells([r["test"] for r in v1_avg])]
    out += ["## 8. A smoother target: where the energy sits on average over a swing", "",
            table(["Model", "Seeds", *[c[1] for c in COLUMNS]], rows), ""]
    out += damping_section(v1_dirs, p6, dirs, robust)

    text = "\n".join(out)
    (ROOT / "results" / "headline.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
