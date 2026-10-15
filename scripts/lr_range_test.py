"""Learning-rate range test (Smith, 2017) for each optimizer of Phase 5.

    python scripts/lr_range_test.py

The learning rate grows geometrically from 1e-5 to 1 over 300 steps while the smoothed training
loss is recorded. Suggested rates: the rate at the loss minimum divided by 3 (peak rate for
one-cycle) and by 10 (starting rate for cosine). Writes results/lr_range.json.
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment, losses, models, train  # noqa: E402


def main():
    torch.set_num_threads(2)
    cfg = ablations.merge(ablations.BASE, {"model": ablations.ARCH, "view": ablations.VIEW, "loss": ablations.LOSS})
    pools = experiment.load_pools(cfg["n_train"])
    splits, _ = experiment.prepare(cfg, pools, workers=4)
    out = {}
    for name, extra in (("adamw", {"weight_decay": 1e-4}), ("adam", {}), ("sgd", {"weight_decay": 1e-4}),
                        ("lion", {"weight_decay": 1e-3})):
        torch.manual_seed(0)
        model = models.build(ablations.ARCH, tuple(splits["train"]["x"].shape[1:]))
        curve = train.lr_range_test(model, losses.SiteLoss(**ablations.LOSS), splits["train"],
                                    optim={"name": name, **extra}, lr_min=1e-5, lr_max=1.0, steps=300)
        lr_best = min(curve, key=lambda c: c[1])[0]
        out[name] = {"curve": curve, "lr_at_min_loss": lr_best, "suggest_peak": lr_best / 3, "suggest_start": lr_best / 10}
        print(f"{name:6s}: loss minimum at lr {lr_best:.2e} -> peak {lr_best / 3:.2e}, start {lr_best / 10:.2e}", flush=True)
    (ROOT / "results" / "lr_range.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
