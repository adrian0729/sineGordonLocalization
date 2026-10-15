"""Train the paper's CNN-BiLSTM (v1) on the data pools and log the result.

    python scripts/train_v1.py --seed 0
    python scripts/train_v1.py --n-train 2000 --max-epochs 1 --tag _smoke --no-record   # quick check

Follows Sec. 2.3 and the Appendix: inputs are the site energy at tau = 0, 0.5, ..., 4.5 plus
theta(0) repeated per frame, standardized per channel; Adam (lr 1e-3), batch 64, up to 100
epochs, early stopping on validation loss with patience 10, best weights restored.
One deliberate change: the scaler is fit on the training split only (the paper fit it
before splitting). Results are appended to results/runs.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, losses, models, sim, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=60_000)  # the paper's 60% of 100k
    parser.add_argument("--label", default="right", choices=sim.LABELS)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-record", action="store_true", help="don't append to results/runs.jsonl")
    parser.add_argument("--note", default="Phase 0: the paper's CNN-BiLSTM re-implemented from Sec. 2.3 "
                                          "and the Appendix, as the v1 baseline")
    args = parser.parse_args()

    device = train.pick_device(args.device)
    run_id = f"v1_seed{args.seed}{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True)
    log_file = open(run_dir / "log.txt", "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    params = sim.Params()
    sizes = {"train": args.n_train, "val": None, "test": None}
    pools = {split: data.load_pool(ROOT / "data" / f"pool_{split}.npz", size) for split, size in sizes.items()}
    start = time.time()
    inputs = {split: data.v1_inputs(pool["theta0"], params) for split, pool in pools.items()}
    flat = inputs["train"].reshape(-1, 2)
    mean, std = flat.mean(axis=0, dtype=np.float64), flat.std(axis=0, dtype=np.float64)
    log(f"{run_id} on {device}: inputs built in {time.time() - start:.0f}s; "
        f"train {len(inputs['train'])}, val {len(inputs['val'])}, test {len(inputs['test'])}; "
        f"channel mean {mean.round(4)}, std {std.round(4)}")

    tensors = {}
    for split, x in inputs.items():
        y_site = pools[split][f"site_{args.label}"].astype(np.int64)
        x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        tensors[split] = {"x": torch.from_numpy(x).to(device),
                          "targets": (torch.from_numpy((y_site >= 0).astype(np.float32)).to(device),
                                      torch.from_numpy(y_site).to(device))}
    del inputs

    torch.manual_seed(args.seed)
    model = models.V1CnnBiLstm().to(device)
    start = time.time()
    history, best_epoch = train.fit(model, losses.v1_loss, tensors["train"], tensors["val"],
                                    max_epochs=args.max_epochs, patience=args.patience,
                                    batch_size=args.batch_size, seed=args.seed, log=log)
    train_seconds = time.time() - start

    results = {}
    for split in ("val", "test"):
        results[split], pred = train.evaluate(model, tensors[split]["x"], pools[split][f"site_{args.label}"])
        np.save(run_dir / f"pred_{split}.npy", pred.astype(np.int16))
        m = results[split]
        log(f"{split}: mean distance {m['mae']:.3f} (linear {m['mae_linear']:.3f})  ±1 {m['acc_1']:.2%} "
            f"(linear {m['acc_1_linear']:.2%})  ±3 {m['acc_3']:.2%}  >10 {m['over_10']:.2%}  "
            f"localizes {m['acc_localizes']:.2%}  nll {m['nll']:.3f}")

    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    record = {"run_id": run_id, "phase": 0, "model": "v1_cnn_bilstm", "note": args.note, "seed": args.seed,
              "label": args.label, "inputs": "E(tau=0..4.5, spring to the right) + theta(0)",
              "n_train": len(tensors["train"]["x"]), "n_val": len(tensors["val"]["x"]),
              "n_test": len(tensors["test"]["x"]), "params": models.count_parameters(model),
              "optimizer": "Adam lr 1e-3 eps 1e-7", "batch_size": args.batch_size,
              "epochs": len(history), "best_epoch": best_epoch, "train_seconds": round(train_seconds),
              "device": str(device), "torch": torch.__version__, "scaler": {"mean": mean.tolist(), "std": std.tolist()},
              "val": results["val"], "test": results["test"], "created": time.strftime("%Y-%m-%d %H:%M")}
    (run_dir / "record.json").write_text(json.dumps(record, indent=1))
    if not args.no_record:
        (ROOT / "results").mkdir(exist_ok=True)
        with open(ROOT / "results" / "runs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    log(f"saved {run_dir.relative_to(ROOT)}" + ("" if args.no_record else " and appended results/runs.jsonl"))


if __name__ == "__main__":
    main()
