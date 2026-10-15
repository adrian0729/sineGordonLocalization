"""How well a trained model's auxiliary time head predicts *when* localization happens (test pool).

    python scripts/eval_time.py --runs runs/v2_final_s*_*

The time head was trained only as a helper for the site, on log(time). This reports its relative error
against two baselines that need no network: the median localization time of the training pool, and the
median time for the run's amplitude A (the growth rate depends on A, so A alone already predicts some
of the time). Writes results/time_head.json.
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


def predicted_times(run_dir, pool):
    cfg = json.loads((run_dir / "config.json").read_text())
    spec = json.loads((run_dir / "model_spec.json").read_text())
    scaler = np.load(run_dir / "scaler.npz")
    view = views.View(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["view"].items()})
    x = experiment.build_inputs(view, "test", pool, workers=4)
    x = torch.from_numpy(np.nan_to_num((x - scaler["mean"]) / scaler["std"], nan=0.0).astype(np.float32))
    model = models.build(spec["spec"], tuple(x.shape[1:]))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out.append(model(x[i:i + 512])["time"].reshape(-1))
    return np.exp(torch.cat(out).numpy() * experiment.TIME_STD + experiment.TIME_MEAN)


def summary(pred, true):
    rel = np.abs(pred - true) / true
    return {"median_rel_error": float(np.median(rel)), "p90_rel_error": float(np.percentile(rel, 90)),
            "median_abs_error": float(np.median(np.abs(pred - true))), "within_10pct": float(np.mean(rel <= 0.1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train = data.load_pool(ROOT / "data" / "pool_train.npz")
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    true = test["time_right"]
    edges = np.quantile(train["A"], np.linspace(0, 1, 21))
    bins = np.clip(np.searchsorted(edges, train["A"]) - 1, 0, 19)
    by_bin = np.array([np.median(train["time_right"][bins == b]) for b in range(20)])
    from_a = by_bin[np.clip(np.searchsorted(edges, test["A"]) - 1, 0, 19)]
    result = {"baseline_constant": summary(np.full_like(true, np.median(train["time_right"])), true),
              "baseline_amplitude": summary(from_a, true), "runs": {}}
    for run in args.runs:
        run_dir = (ROOT / run).resolve()
        result["runs"][str(run_dir.relative_to(ROOT))] = summary(predicted_times(run_dir, test), true)
    for name, s in [("constant (training median)", result["baseline_constant"]),
                    ("from the amplitude alone", result["baseline_amplitude"])] + list(result["runs"].items()):
        print(f"{name:44s} median error {100 * s['median_rel_error']:.1f}%  90th pct {100 * s['p90_rel_error']:.1f}%  "
              f"within 10% {100 * s['within_10pct']:.1f}%")
    (ROOT / "results" / "time_head.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
