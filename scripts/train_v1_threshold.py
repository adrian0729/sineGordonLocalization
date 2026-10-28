"""Train the rebuilt v1 CNN-BiLSTM on a threshold-study label (scripts/threshold_labels.py) and log the result.

    python scripts/train_v1_threshold.py --label thr04 --seed 0
    python scripts/train_v1_threshold.py --label thr04_tf100 --pool-tag half --seed 30

Same network, inputs, loss, optimizer and early stopping as scripts/train_v1.py; only the label (and, with
--pool-tag half, the offsets) change. At low initial noise most seeds never leave the ln(100) plateau of a
uniform guess, so a run whose validation loss is still at or above --stall-loss after --stall-epoch epochs is
stopped and recorded as stalled; scripts/threshold_analysis.py counts those tries but scores only trained runs.
Writes runs/threshold_v1_<label>_<tag>_s<seed>/ and appends a record (phase "threshold") to results/runs.jsonl.
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
from sgloc import data, losses, metrics, models, sim, train  # noqa: E402


def profile(pred, y):
    """Where the misses fall: off by exactly 2 vs exactly 1, and the far misses (more than 4 sites)."""
    keep = y >= 0
    d = metrics.ring_distance(pred[keep], y[keep])
    p1, p2 = (d == 1).mean(), (d == 2).mean()
    far = d[d > 4]
    return {"P2_over_P1": float(p2 / p1) if p1 else None, "far_frac": float(far.size / d.size),
            "far_mean": float(far.mean()) if far.size else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pool-tag", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stall-epoch", type=int, default=8)
    parser.add_argument("--stall-loss", type=float, default=4.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    device = train.pick_device(args.device)
    run_id = f"threshold_v1_{args.label}_{args.pool_tag}_s{args.seed}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "log.txt", "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    params = sim.Params()
    sizes = {"train": args.n_train, "val": None, "test": None}
    pools = {s: data.load_pool(ROOT / "data" / f"thrpool_{args.pool_tag}_{s}.npz", n) for s, n in sizes.items()}
    inputs = {s: data.v1_inputs(pool["theta0"], params) for s, pool in pools.items()}
    flat = inputs["train"].reshape(-1, 2)
    mean, std = flat.mean(axis=0, dtype=np.float64), flat.std(axis=0, dtype=np.float64)
    y = {s: pools[s][f"site_{args.label}"].astype(np.int64) for s in pools}
    log(f"{run_id} on {device}: train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}; unlocalized "
        f"{', '.join(f'{s} {(y[s] < 0).sum()}' for s in y)}; channel mean {mean.round(4)}, std {std.round(4)}")
    tensors = {}
    for s, x in inputs.items():
        x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        tensors[s] = {"x": torch.from_numpy(x).to(device),
                      "targets": (torch.from_numpy((y[s] >= 0).astype(np.float32)).to(device),
                                  torch.from_numpy(y[s]).to(device))}
    del inputs

    torch.manual_seed(args.seed)
    model = models.V1CnnBiLstm().to(device)
    # train.fit's loop (Adam 1e-3, eps 1e-7, early stopping on validation loss, best weights restored), plus the stall stop
    shuffle = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    n = len(tensors["train"]["x"])
    best_loss, best_state, best_epoch, waited, stalled, history = float("inf"), None, 0, 0, False, []
    start = time.time()
    for epoch in range(1, args.max_epochs + 1):
        tick = time.time()
        model.train()
        order = torch.randperm(n, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            loss, _ = losses.v1_loss(model(tensors["train"]["x"][idx]), *(t[idx] for t in tensors["train"]["targets"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.detach() * len(idx)
        train_loss, val_loss = running.item() / n, train.mean_loss(model, losses.v1_loss, tensors["val"])
        if val_loss < best_loss:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            waited += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": round(time.time() - tick, 1)})
        log(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  best {best_loss:.4f} (epoch {best_epoch})  "
            f"{time.time() - tick:.0f}s")
        if epoch >= args.stall_epoch and best_loss >= args.stall_loss:
            stalled = True
            log(f"STALLED: validation loss still >= {args.stall_loss} after epoch {epoch} (ln 100 = 4.605)")
            break
        if waited >= args.patience:
            break
    train_seconds = time.time() - start
    model.load_state_dict(best_state)

    results = {}
    for s in ("val", "test"):
        results[s], pred = train.evaluate(model, tensors[s]["x"], y[s])
        results[s].update(profile(pred, y[s]))
        np.save(run_dir / f"pred_{s}.npy", pred.astype(np.int16))
        m = results[s]
        log(f"{s}: n {m['n']}  mean {m['mae']:.3f}  exact {m['acc_0']:.2%}  ±1 {m['acc_1']:.2%}  ±2 {m['acc_2']:.2%}  "
            f"±4 {m['acc_4']:.2%}  P2/P1 {m['P2_over_P1'] or 0:.3f}  far {m['far_frac']:.2%}")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    record = {"run_id": run_id, "phase": "threshold", "group": "v1", "model": "v1_cnn_bilstm", "label": args.label,
              "pool_tag": args.pool_tag, "seed": args.seed, "stalled": stalled,
              "note": "threshold study: the rebuilt v1 on a threshold-study label",
              "n_train": n, "params": models.count_parameters(model), "batch_size": args.batch_size,
              "epochs": len(history), "best_epoch": best_epoch, "train_seconds": round(train_seconds),
              "device": str(device), "torch": torch.__version__, "val": results["val"], "test": results["test"],
              "created": time.strftime("%Y-%m-%d %H:%M")}
    (run_dir / "record.json").write_text(json.dumps(record, indent=1))
    if not args.no_record:
        with open(ROOT / "results" / "runs.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    log("done" + (" (stalled)" if stalled else ""))


if __name__ == "__main__":
    main()
