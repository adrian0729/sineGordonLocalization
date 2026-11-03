"""Ring-size transfer beyond the settings study (paper Sec. 3.7; REPORT.md "Where the settings come from").
Writes results/ring_transfer.json.

    python scripts/setting_labels.py --tag n80 --n-sites 80 --sizes 0 0 10000 --seed 5080 --splits test --workers 8
    python scripts/setting_labels.py --tag n150 --n-sites 150 --sizes 0 0 10000 --seed 5150 --splits test --workers 8
    python scripts/ring_transfer.py

Part C of scripts/settings_analysis.py scores the wider network trained on rings of 50, 100, 200 and 400 pendulums
(phase settings_ring, runs ring_n<N>) only on rings whose size divides its training ring's or is a multiple of it.
On a ring whose size divides the training ring's, a circular CNN computes exactly what it computes on its own ring
when the state is repeated around it, so those transfers cannot tell whether a network carries over to shorter
rings in general. Here:

1. transfers: the whole matrix. Each of these networks, and the standard network (v2) trained at N = 100 on the same
   label (eh_k3), scored on the test pools of 50, 80, 100, 150, 200 and 400 pendulums, its own ring included. The
   pools of 80 and 150 pendulums are new (same amplitudes, offsets and couplings as the main pools; 80 divides only
   400, 150 none of the training rings). Pairs that part C also scores repeat settings.json C.cross, and the own
   rings repeat the training runs' test scores (C.per_n). Scored as in part C (settings_analysis.Rescorer:
   experiment.evaluate_saved on the run's own view and scaler); seed means, spreads and each seed's accuracy.
2. identity: for the transfers to rings whose size divides the training ring's, the saved networks' site scores on
   64 random states of the shorter ring against their scores on the same states repeated around the training ring.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import settings_analysis as sa  # noqa: E402
from sgloc import ablations, models  # noqa: E402

TRAIN_N = (50, 100, 200, 400)
TEST_N = (50, 80, 100, 150, 200, 400)
NEW_N = (80, 150)
# (training ring, shorter ring that divides it)
IDENTITY = ((100, 50), (200, 50), (200, 100), (400, 50), (400, 80), (400, 100), (400, 200))


def pool_name(n):
    return "setpool_main_test" if n == 100 else f"setpool_n{n}_test"


def relation(train_n, test_n):
    if test_n == train_n:
        return "own ring"
    if test_n > train_n:
        return "longer"
    return "divides" if train_n % test_n == 0 else "does not divide"


def scored(rescorer, rows, pool_name):
    """Seed stats as in settings.json C.cross, plus each seed's accuracy within +-1."""
    out = rescorer.seeds(rows, pool_name)
    if out is not None:
        per = sorted((r["seed"], rescorer.one(r, pool_name)) for r in rows)
        out["acc_1_by_seed"] = {str(s): m["acc_1"] for s, m in per if m}
        out["over_10_by_seed"] = {str(s): m["over_10"] for s, m in per if m}
    return out


def identity(rows, short_n, n_states=64):
    """Largest difference between the site scores on the shorter ring and on the training ring with the state
    repeated around it, over the seeds, and how often the predicted sites agree around the shorter ring."""
    torch.manual_seed(0)
    worst, worst_scale, agree = 0.0, 0.0, []
    for r in rows:
        run = ROOT / r["run_dir"]
        spec = json.loads((run / "model_spec.json").read_text())
        n_train = spec["input_shape"][1]
        channels = spec["input_shape"][0]
        x = torch.randn(n_states, channels, short_n)
        k = n_train // short_n
        scores = {}
        for n, xin in ((short_n, x), (n_train, x.repeat(1, 1, k))):
            m = models.build(spec["spec"], (channels, n))
            m.load_state_dict(torch.load(run / "model.pt"))
            m.eval()
            with torch.no_grad():
                scores[n] = m(xin)["site"]
        short, long_ = scores[short_n], scores[n_train]
        worst = max(worst, float((long_[:, :short_n] - short).abs().max()))
        worst_scale = max(worst_scale, float(short.abs().max()))
        agree.append(float((long_.argmax(-1) % short_n == short.argmax(-1)).float().mean()))
    return {"seeds": len(rows), "states": n_states, "max_abs_score_difference": worst, "max_abs_score": worst_scale,
            "argmax_agrees_around_shorter_ring": float(np.mean(agree))}


def main():
    records = sa.load_records([])
    rescorer = sa.Rescorer(False)
    ring = sa.by_name(records, "settings_ring")
    std, std_name = sa.std_rows(sa.by_name(records, "settings_hump", "label"))
    out = {"label": ablations.SETTINGS_LABEL, "standard_label": std_name, "pools": {}, "transfers": {}, "identity": {}}
    for t in NEW_N:
        path = ROOT / "data" / f"setpool_n{t}_test.npz"
        if path.exists():
            with np.load(path, allow_pickle=True) as f:
                meta = json.loads(str(f["meta"]))
                out["pools"][f"n{t}"] = {"pool": path.stem, "n_runs": meta["n_runs"], "n_sites": meta["n_sites"],
                                         "seed": meta["seed"], "a_range": meta["a_range"], "offset_width": meta["offset_width"],
                                         "eh_k3_fired": float((f["site_eh_k3"] >= 0).mean())}
        else:
            sa.note(f"pool {path.name} missing: run scripts/setting_labels.py --tag n{t} first")
    for src in TRAIN_N:
        rows = ring.get(f"ring_n{src}", [])
        for t in TEST_N:
            res = scored(rescorer, rows, pool_name(t)) if rows else None
            out["transfers"][f"ring_n{src}_on_n{t}"] = res and {"train_N": src, "test_N": t, "relation": relation(src, t), **res}
    for t in TEST_N:
        res = scored(rescorer, std, pool_name(t)) if std else None
        out["transfers"][f"standard_on_n{t}"] = res and {"train_N": 100, "test_N": t, "relation": relation(100, t), **res}
    for src, t in IDENTITY:
        rows = ring.get(f"ring_n{src}", [])
        if rows:
            out["identity"][f"ring_n{src}_on_n{t}"] = identity(rows, t)
    out["notes"] = sa.NOTES
    path = ROOT / "results" / "ring_transfer.json"
    path.write_text(json.dumps(sa.clean(out), indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))

    print("\nwithin +-1 on test (%), trained on (rows) and scored on (columns)")
    print("  " + " " * 10 + "".join(f"{t:>8d}" for t in TEST_N))
    for name, key in [(f"wide {s}", f"ring_n{s}") for s in TRAIN_N] + [("v2 100", "standard")]:
        cells = [out["transfers"].get(f"{key}_on_n{t}") for t in TEST_N]
        print("  " + f"{name:10s}" + "".join(f"{100 * c['acc_1']:8.1f}" if c else f"{'-':>8s}" for c in cells))
    print("\nwithin +-1 on test (seed mean, sd; each seed) and far misses")
    for key, v in out["transfers"].items():
        if v:
            seeds = ", ".join(f"{100 * a:.1f}" for a in v["acc_1_by_seed"].values())
            print(f"  {key:20s} {v['relation']:16s} {100 * v['acc_1']:5.1f}% (sd {100 * (v['acc_1_sd'] or 0):4.1f}; {seeds}); "
                  f"over 10 sites {100 * v['over_10']:4.1f}%")
    print("\nshorter ring vs training ring with the state repeated (saved networks, random states)")
    for key, v in out["identity"].items():
        print(f"  {key:20s} max |score difference| {v['max_abs_score_difference']:.2e} (scores up to {v['max_abs_score']:.0f}); "
              f"argmax agrees {100 * v['argmax_agrees_around_shorter_ring']:.1f}%")


if __name__ == "__main__":
    main()
