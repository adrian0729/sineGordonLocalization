"""Extra read-outs of trained final models on the test pool: the time head and prediction sets.

    python scripts/eval_final.py --runs runs/v2_final_s*_*

Time head: it was trained only as a helper for the site, on log(time). Its relative error is
compared with two baselines that need no network: the training pool's median localization time,
and the median time for the run's amplitude A (the growth rate depends on A, so A alone already
predicts some of the time). Written to results/time_head.json.

Prediction sets: instead of one site, name the smallest set of sites holding a given share of the
predicted probability. Reports how often the true site is in the set (coverage) and the set's size.
Written to results/prediction_sets.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, experiment, models, views  # noqa: E402

LEVELS = (0.5, 0.8, 0.9, 0.95)


def outputs(run_dir, pool):
    """Site probabilities and predicted localization times of a saved run on a pool's test split."""
    cfg = json.loads((run_dir / "config.json").read_text())
    spec = json.loads((run_dir / "model_spec.json").read_text())
    scaler = np.load(run_dir / "scaler.npz")
    view = views.View(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["view"].items()})
    x = experiment.build_inputs(view, "test", pool, workers=4)
    x = torch.from_numpy(np.nan_to_num((x - scaler["mean"]) / scaler["std"], nan=0.0).astype(np.float32))
    model = models.build(spec["spec"], tuple(x.shape[1:]))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    probs, times = [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out = model(x[i:i + 512])
            probs.append(out["site"].softmax(dim=-1))
            times.append(out["time"].reshape(-1))
    times = np.exp(torch.cat(times).numpy() * experiment.TIME_STD + experiment.TIME_MEAN)
    return torch.cat(probs).numpy(), times


def time_summary(pred, true):
    rel = np.abs(pred - true) / true
    return {"median_rel_error": float(np.median(rel)), "p90_rel_error": float(np.percentile(rel, 90)),
            "median_abs_error": float(np.median(np.abs(pred - true))), "within_10pct": float(np.mean(rel <= 0.1))}


def set_summary(probs, true):
    order = np.argsort(-probs, axis=1)
    cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    rank = np.argmax(order == true[:, None], axis=1)  # position of the true site in the sorted list
    out = {}
    for level in LEVELS:
        size = np.argmax(cum >= level, axis=1) + 1
        out[f"{level:g}"] = {"coverage": float(np.mean(rank < size)), "mean_size": float(size.mean()),
                             "median_size": float(np.median(size))}
    out["top3"] = float(np.mean(rank < 3))
    out["top5"] = float(np.mean(rank < 5))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train = data.load_pool(ROOT / "data" / "pool_train.npz")
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    true_t, true_site = test["time_right"], test["site_right"].astype(int)
    edges = np.quantile(train["A"], np.linspace(0, 1, 21))
    bins = np.clip(np.searchsorted(edges, train["A"]) - 1, 0, 19)
    by_bin = np.array([np.median(train["time_right"][bins == b]) for b in range(20)])
    from_a = by_bin[np.clip(np.searchsorted(edges, test["A"]) - 1, 0, 19)]
    times = {"baseline_constant": time_summary(np.full_like(true_t, np.median(train["time_right"])), true_t),
             "baseline_amplitude": time_summary(from_a, true_t), "runs": {}}
    sets = {"levels": list(LEVELS), "runs": {}}
    for run in args.runs:
        run_dir = (ROOT / run).resolve()
        name = str(run_dir.relative_to(ROOT))
        probs, pred_t = outputs(run_dir, test)
        times["runs"][name] = time_summary(pred_t, true_t)
        sets["runs"][name] = set_summary(probs, true_site)
    for name, s in [("time: constant (training median)", times["baseline_constant"]),
                    ("time: from the amplitude alone", times["baseline_amplitude"])] + list(times["runs"].items()):
        print(f"{name:44s} median error {100 * s['median_rel_error']:.1f}%  90th pct {100 * s['p90_rel_error']:.1f}%  "
              f"within 10% {100 * s['within_10pct']:.1f}%")
    for name, s in sets["runs"].items():
        print(f"{name:44s} " + "  ".join(f"{k}: covers {100 * v['coverage']:.1f}% with {v['mean_size']:.1f} sites"
                                          for k, v in s.items() if isinstance(v, dict)))
    (ROOT / "results" / "time_head.json").write_text(json.dumps(times, indent=1))
    (ROOT / "results" / "prediction_sets.json").write_text(json.dumps(sets, indent=1))


if __name__ == "__main__":
    main()
