"""Phase 8 analysis: how each regularizer changes overfitting with little data.

    python scripts/regularization.py      # tables: results/phase8_regularization.md; data: results/phase8_curves.json

For every Phase 8 run (including the Phase 6 controls it reuses) it reads the validation curve the
run logged (runs/<run>/history.json; mean distance on the first 5,000 validation runs), and scores
the saved model on its own training runs (cached in results/phase8_train.json). Needs runs/.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, data, experiment, losses, metrics, models, train  # noqa: E402

TRAIN_CACHE = ROOT / "results" / "phase8_train.json"


def train_metrics(rec, workers=4):
    """Site metrics of a saved run on the training runs it was fit to (the originals, not mirror images)."""
    cfg, run_dir = rec["config"], ROOT / rec["run_dir"]
    spec = json.loads((run_dir / "model_spec.json").read_text())
    scaler = np.load(run_dir / "scaler.npz")
    pool = data.load_pool(ROOT / "data" / f"{(cfg.get('pools') or {}).get('train', 'pool_train')}.npz", cfg["n_train"])
    x = experiment.build_inputs(experiment.views_for(cfg)["train"], "train", pool, workers)
    x = torch.from_numpy(np.nan_to_num((x - scaler["mean"]) / scaler["std"], nan=0.0).astype(np.float32))
    model = models.build(spec["spec"], tuple(x.shape[1:]))
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    pred = train.predict_sites(model, losses.SiteLoss(**cfg.get("loss", {})), x)
    return metrics.site_metrics(pred, pool[f"site_{cfg.get('label', 'right')}"])


def collect():
    """(group, name) -> list of per-seed dicts, in table order."""
    records = {}
    with open(experiment.RESULTS) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("config_id"):
                    records[rec["config_id"]] = rec
    cached = json.loads(TRAIN_CACHE.read_text()) if TRAIN_CACHE.exists() else {}
    rows = defaultdict(list)
    for cfg in ablations.phase8():
        rec = records.get(experiment.config_id(cfg))
        if rec is None:
            continue
        cid = rec["config_id"]
        if cid not in cached:
            cached[cid] = train_metrics(rec)
            TRAIN_CACHE.write_text(json.dumps(cached, indent=1, sort_keys=True))
        history = json.loads((ROOT / rec["run_dir"] / "history.json").read_text())
        rows[(cfg["group"], cfg["name"])].append({
            "seed": cfg["seed"], "n_train": cfg["n_train"], "test": rec["test"], "train": cached[cid],
            "best_step": rec["best_step"], "steps": [h["step"] for h in history], "val": [h["val_mae"] for h in history]})
    return rows


def mean_sd(values, digits=2, scale=1):
    v = np.asarray(values, dtype=float) * scale
    return f"{v.mean():.{digits}f}" if len(v) == 1 else f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def tables(rows):
    out = []
    for group in dict.fromkeys(g for g, _ in rows):
        out += [f"### {group}", "",
                "| Config | Seeds | Best step | Val: best → last step | Train | Test | Gap | Test ±1 |",
                "|---|---|---|---|---|---|---|---|"]
        for (g, name), runs in rows.items():
            if g != group:
                continue
            best = [min(r["val"]) for r in runs]
            last = [r["val"][-1] for r in runs]
            tr = [r["train"]["mae"] for r in runs]
            te = [r["test"]["mae"] for r in runs]
            out.append(f"| {name} | {len(runs)} | {', '.join(str(r['best_step']) for r in runs)} | "
                       f"{np.mean(best):.2f} → {np.mean(last):.2f} (+{100 * (np.mean(last) / np.mean(best) - 1):.0f}%) | "
                       f"{mean_sd(tr)} | {mean_sd(te)} | {np.mean(te) - np.mean(tr):+.2f} | "
                       f"{mean_sd([r['test']['acc_1'] for r in runs], 1, 100)}% |")
        out.append("")
    return "\n".join(out)


def curves(rows):
    """Seed-mean validation curves, for the report figure (only configs whose seeds share one step grid)."""
    out = {}
    for (group, name), runs in rows.items():
        if len({tuple(r["steps"]) for r in runs}) == 1:
            out[name] = {"n_train": runs[0]["n_train"], "steps": runs[0]["steps"],
                         "val": np.mean([r["val"] for r in runs], axis=0).round(4).tolist(),
                         "test": [r["test"]["mae"] for r in runs], "train": [r["train"]["mae"] for r in runs],
                         "test_acc_1": [r["test"]["acc_1"] for r in runs], "best_step": [r["best_step"] for r in runs]}
    return out


def main():
    rows = collect()
    text = ("Phase 8: final recipe, 7,050 steps at every size. Val = mean distance on the first 5,000 validation runs "
            "(checkpoint selection), at the best checkpoint and at the last step. Train = the saved (best) model on its "
            "own training runs; test = the 20,000-run test pool; gap = test - train.\n\n" + tables(rows))
    print(text)
    (ROOT / "results" / "phase8_regularization.md").write_text(text)
    (ROOT / "results" / "phase8_curves.json").write_text(json.dumps(curves(rows)))


if __name__ == "__main__":
    main()
