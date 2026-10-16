"""Phase 6 robustness: re-evaluate trained runs on shifted test conditions (no retraining).

    python scripts/robustness.py --runs <run_dir> [<run_dir> ...] [--v1]

Conditions: the nominal test pool; g or h off by 10%; twice the initial noise; N = 80 and 150;
one 1% impurity per chain, or one 0-2% impurity as in the Phase 6 impurity test pool; linear
damping gamma = 0 .. 1e-2 (six pools sharing their initial conditions) and gamma log-uniform in
[1e-4, 1e-2] (pools from scripts/make_robustness_pools.py, always observed through the nominal
model); and sensor noise of 1e-4 .. 3e-3 rad added at test time only. "<damped pool>_labels_only"
scores predictions from the undamped chain's inputs (robust_gamma0, the same initial conditions)
against the damped chain's labels: how much of the damage is a changed outcome, and how much
changed inputs. With --v1 the rebuilt v1 runs from Phase 0 are evaluated on the conditions they
support (N = 100 only).

Sites are scored over the runs that localize by tau = 150, against the label each run was trained
on. Appends one line per (run, condition) to results/robustness.jsonl.
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
from sgloc import data, experiment, metrics, models, views  # noqa: E402

POOLS = ["pool_test", "robust_g-10", "robust_g+10", "robust_h-10", "robust_h+10", "robust_noise2x",
         "robust_N80", "robust_N150", "robust_impurity1", "robust_impurity_test",
         "robust_gamma0", "robust_gamma1e-4", "robust_gamma3e-4", "robust_gamma1e-3", "robust_gamma3e-3",
         "robust_gamma1e-2", "robust_damped_test"]
NOISES = [1e-4, 3e-4, 1e-3, 3e-3]
LABELS_ONLY = ["robust_gamma1e-4", "robust_gamma3e-4", "robust_gamma1e-3", "robust_gamma3e-3"]


def conditions(only=None):
    """(name, input pool, view override, label pool)"""
    for pool in POOLS:
        if not only or pool in only:
            yield pool, pool, None, None
    for pool in LABELS_ONLY:
        if not only or f"{pool}_labels_only" in only:
            yield f"{pool}_labels_only", "robust_gamma0", None, pool
    for noise in NOISES:
        if not only or f"test_noise{noise:g}" in only:
            yield f"test_noise{noise:g}", "pool_test", {"noise": noise}, None


def summary(m):
    if not m["n"]:
        text = "no run localizes"
    else:
        text = f"mean {m['mae']:.3f}  ±1 {m['acc_1']:.2%}  ±3 {m['acc_3']:.2%}  (n {m['n']})"
    if "localizes" in m:
        text += f"  localizes: accuracy {m['localizes']['accuracy']:.2%}"
    return text


def evaluate_v1(record, pool_name, override, device_name="auto", label_pool=None):
    pool = data.load_pool(ROOT / "data" / f"{pool_name}.npz")
    if pool["theta0"].shape[1] != 100:
        return None
    view = views.View(channels=("E",), frames=tuple(range(10)), theta0=True, layout="frames",
                      noise=(override or {}).get("noise", 0.0))
    x = experiment.build_inputs(view, "test", pool, workers=4)  # (runs, 10, 2, N)
    mean, std = np.array(record["scaler"]["mean"]), np.array(record["scaler"]["std"])
    x = ((x.transpose(0, 1, 3, 2) - mean) / std).astype(np.float32)  # v1 layout (runs, 10, N, 2)
    model = models.V1CnnBiLstm()
    model.load_state_dict(torch.load(ROOT / "runs" / record["run_id"] / "model.pt", map_location="cpu"))
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            preds.append(model(torch.from_numpy(x[i:i + 1024]).to(device))[1].argmax(1).cpu())
    truth = pool if label_pool is None else data.load_pool(ROOT / "data" / f"{label_pool}.npz")
    return metrics.site_metrics(torch.cat(preds).numpy(), truth["site_right"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--device", default="auto", help="device for the v1 evaluation (cpu keeps the GPU free)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None, help="evaluate only these conditions (names as logged)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = open(ROOT / "results" / "robustness.jsonl", "a")
    for run_dir in args.runs:
        run_dir = str(Path(run_dir).resolve().relative_to(ROOT))
        for name, pool, override, label_pool in conditions(args.only):
            m = experiment.evaluate_saved(run_dir, pool, override, label_pool=label_pool)
            out.write(json.dumps({"run_dir": run_dir, "condition": name, "test": m,
                                  "created": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
            out.flush()
            print(f"{run_dir:48s} {name:20s} {summary(m)}", flush=True)
    if args.v1:
        with open(experiment.RESULTS) as f:
            v1_runs = [r for r in (json.loads(l) for l in f if '"v1_cnn_bilstm"' in l) if r.get("label", "right") == "right"]
        for rec in v1_runs:
            for name, pool, override, label_pool in conditions(args.only):
                m = evaluate_v1(rec, pool, override, args.device, label_pool)
                if m is None:
                    continue
                out.write(json.dumps({"run_dir": f"runs/{rec['run_id']}", "model": "v1", "condition": name, "test": m,
                                      "created": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
                out.flush()
                print(f"v1 {rec['run_id']:40s} {name:20s} {summary(m)}", flush=True)


if __name__ == "__main__":
    main()
