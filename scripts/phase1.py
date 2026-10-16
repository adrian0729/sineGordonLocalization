"""Phase 1: physics baselines, the learned-linear model, and the physics checks behind the plan.

    python scripts/phase1.py

1. Growth spectrum sigma_m(A) on a grid (for the report figure).
2. Best box-car width vs amplitude: measured on fixed-A runs vs the prediction K* = N / m_c(A).
3. Mode knockouts and perturbation sensitivity on test runs (which scales decide the site; how
   precisely the initial state must be known).
4. Baselines on the official pools (v1 label): box-car (width chosen on validation), Floquet
   filter (periods chosen on validation), and learned-linear models fit by L-BFGS.
Writes results/phase1.json and appends the baselines to results/runs.jsonl.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, floquet, metrics, models, sim, train  # noqa: E402

N = 100


def boxcar(x, K):
    return sum(np.roll(x, j, axis=-1) for j in range(-(K // 2), K // 2 + 1)) / K


def unstable_modes(A, h=0.5):
    sigma = floquet.growth_rates(A, np.arange(1, N // 2 + 1), h=h)
    return np.flatnonzero(sigma > 1e-6) + 1


def growth_grid():
    amps = np.round(np.arange(1.55, 1.951, 0.01), 2)
    sigma = np.array([floquet.growth_rates(a, np.arange(1, 13)) for a in amps])
    return {"A": amps.tolist(), "modes": list(range(1, 13)), "sigma": np.round(sigma, 6).tolist(),
            "threshold": floquet.threshold_amplitude()}


def best_width_vs_amplitude(workers, runs=2000, boot=300):
    rng = np.random.default_rng(11)
    widths = np.arange(3, 50, 2)
    out = []
    for i, A in enumerate((1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90)):
        _, theta0 = data.initial_conditions(runs, seed=200 + i, a_range=(A, A))
        lab = data.localize_parallel(theta0, sim.Params(t_final=300.0), chunk=250, workers=workers, labels=("right",))
        keep = lab["site_right"] >= 0
        E0 = sim.site_energy(theta0, np.zeros_like(theta0), 0.75, 0.5, "right")[keep]
        dist = np.array([metrics.ring_distance(boxcar(E0, K).argmax(1), lab["site_right"][keep]) for K in widths])
        curve = dist.mean(1)
        picks = [widths[dist[:, rng.integers(0, keep.sum(), keep.sum())].mean(1).argmin()] for _ in range(boot)]
        m_c = unstable_modes(A).max() + 1
        out.append({"A": A, "localized": float(keep.mean()), "unstable_modes": unstable_modes(A).tolist(),
                    "predicted_K": N / m_c, "measured_K": int(widths[curve.argmin()]),
                    "measured_K_16_84": [float(np.percentile(picks, 16)), float(np.percentile(picks, 84))],
                    "widths": widths.tolist(), "mean_distance": np.round(curve, 3).tolist()})
        print(f"  A={A:.2f}: localized {keep.mean():.1%}, unstable m={out[-1]['unstable_modes']}, "
              f"predicted K*={N / m_c:.1f}, measured K*={out[-1]['measured_K']} "
              f"(16-84%: {out[-1]['measured_K_16_84']})", flush=True)
    return out


def sensitivity(test, workers, runs=2000):
    rng = np.random.default_rng(12)
    theta0, base = test["theta0"][:runs], test["site_right"][:runs]
    F = np.fft.rfft(theta0, axis=1)
    ms = np.arange(F.shape[1])

    def shift(new_theta0):
        lab = data.localize_parallel(new_theta0, chunk=250, workers=workers, labels=("right",))
        d = metrics.ring_distance(lab["site_right"], base)
        return {"mean_shift": float(d.mean()), "moved_over_1": float((d > 1).mean()),
                "moved_over_3": float((d > 3).mean()), "moved_over_10": float((d > 10).mean())}

    knockouts = {}
    for name, keep in [(f"keep m<={M}", ms <= M) for M in (2, 3, 5, 6, 8, 12, 25)] + [("remove m<=5", ms > 5)]:
        keep = keep.copy()
        keep[0] = True  # the mean swing amplitude is never touched
        G = F.copy()
        G[:, ~keep] = 0
        knockouts[name] = shift(np.fft.irfft(G, n=N, axis=1))
        print(f"  knockout {name:12s} {knockouts[name]}", flush=True)

    perturb = {}
    for rms in (1e-9, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        perturb[f"white {rms:g}"] = shift(theta0 + rms * rng.standard_normal(theta0.shape))
        print(f"  white noise rms {rms:g}: {perturb[f'white {rms:g}']}", flush=True)
    for name, band in (("m<=5", (ms >= 1) & (ms <= 5)), ("m>=6", ms >= 6)):
        P = np.fft.rfft(rng.standard_normal(theta0.shape), axis=1)
        P[:, ~band] = 0
        p = np.fft.irfft(P, n=N, axis=1)
        p *= 1e-3 / p.std(axis=1, keepdims=True)
        perturb[f"band {name} 0.001"] = shift(theta0 + p)
        print(f"  band {name} rms 1e-3: {perturb[f'band {name} 0.001']}", flush=True)
    return {"runs": runs, "knockouts": knockouts, "perturbations": perturb}


def fit_linear(X_tr, y_tr, a_tr, X_va, y_va, a_va, amplitude, l2_grid=(0.0, 1e-4, 1e-3)):
    """Fit RingLinear for each L2 strength, keep the one with the lowest validation loss."""
    best = None
    for l2 in l2_grid:
        model = models.RingLinear(X_tr.shape[1], amplitude=amplitude)
        train_loss = train.fit_lbfgs(lambda: model.scores_from_spectrum(X_tr, a_tr), model.parameters(),
                                     y_tr, l2=l2, max_iter=500)
        with torch.no_grad():
            val_loss = torch.nn.functional.cross_entropy(model.scores_from_spectrum(X_va, a_va), y_va).item()
        if best is None or val_loss < best[0]:
            best = (val_loss, l2, train_loss, model)
    val_loss, l2, train_loss, model = best
    return model, l2, train_loss, val_loss


def baselines(pools, record):
    out = {}
    params = sim.Params()
    y = {s: pools[s]["site_right"] for s in pools}
    frames = {s: sim.evolve(pools[s]["theta0"], 10, params) for s in ("val", "test")}
    E = {s: sim.site_energy(*frames[s], params.g, params.h, "right") for s in frames}

    # box-car: width chosen on validation for every frame
    widths = np.arange(1, 40, 2)
    rows = []
    for k in range(10):
        val_mae = [metrics.ring_distance(boxcar(E["val"][:, k], K).argmax(1), y["val"]).mean() for K in widths]
        K = int(widths[int(np.argmin(val_mae))])
        m = metrics.site_metrics(boxcar(E["test"][:, k], K).argmax(1), y["test"])
        rows.append({"tau": k * 0.5, "K": K, **m})
    out["boxcar"] = rows
    best = min(rows, key=lambda r: r["mae"])
    record("boxcar", f"box-car on E, width chosen on validation (best frame tau={best['tau']}, K={best['K']})", 0, best)
    print(f"  box-car best: tau {best['tau']} K {best['K']}: mean {best['mae']:.2f} ±1 {best['acc_1']:.1%}", flush=True)

    # Floquet filter: number of periods chosen on validation, amplitude estimated as the chain mean
    a_hat = {s: pools[s]["theta0"].mean(1) for s in ("val", "test")}
    val_mae = {p: metrics.ring_distance(floquet.floquet_filter(pools["val"]["theta0"], a_hat["val"], periods=p),
                                        y["val"]).mean() for p in range(3, 10)}
    p_best = min(val_mae, key=val_mae.get)
    m = metrics.site_metrics(floquet.floquet_filter(pools["test"]["theta0"], a_hat["test"], periods=p_best), y["test"])
    out["floquet"] = {"val_mae_by_periods": val_mae, "periods": p_best, "test": m}
    record("floquet_filter", f"training-free Floquet filter, {p_best} periods (chosen on validation)", 0, m)
    print(f"  Floquet filter ({p_best} periods): mean {m['mae']:.2f} ±1 {m['acc_1']:.1%}", flush=True)

    # learned linear models, fit on the same 60k training runs as v1
    torch.set_num_threads(8)
    th = {s: pools[s]["theta0"] for s in pools}
    d_std = (th["train"] - th["train"].mean(1, keepdims=True)).std()
    amp_mean, amp_std = th["train"].mean(1).mean(), th["train"].mean(1).std()

    def spec(x):
        return torch.fft.rfft(torch.from_numpy(x), dim=-1)

    def dtheta(s):
        return ((th[s] - th[s].mean(1, keepdims=True)) / d_std)[:, None, :]

    a = {s: torch.from_numpy((th[s].mean(1) - amp_mean) / amp_std) for s in pools}
    yt = {s: torch.from_numpy(y[s].astype(np.int64)) for s in pools}
    linear_rows = {}
    for name, amplitude in (("linear_dtheta0", False), ("linear_dtheta0_amp", True)):
        X = {s: spec(dtheta(s)) for s in pools}
        start = time.time()
        model, l2, tr_loss, va_loss = fit_linear(X["train"], yt["train"], a["train"], X["val"], yt["val"], a["val"], amplitude)
        with torch.no_grad():
            pred = model.scores_from_spectrum(X["test"], a["test"]).argmax(1).numpy()
        m = metrics.site_metrics(pred, y["test"])
        linear_rows[name] = {"l2": l2, "train_loss": tr_loss, "val_loss": va_loss, "seconds": round(time.time() - start), "test": m,
                             "filter": model.w.detach().numpy()[0].tolist(),
                             "amp_filter": None if model.u is None else model.u.detach().numpy()[0].tolist()}
        record(name, f"convex learned-linear ring filter on delta theta(0){' with amplitude-dependent filter' if amplitude else ''}, "
                     f"L-BFGS, L2={l2}", models.count_parameters(model), m)
        print(f"  {name}: mean {m['mae']:.2f} ±1 {m['acc_1']:.1%} ±3 {m['acc_3']:.1%} (l2 {l2}, {time.time() - start:.0f}s)", flush=True)

    # linear model on v1's inputs: energy frames tau = 0..4.5 plus theta(0), standardized like v1
    v1x = {s: data.v1_inputs(th[s], params) for s in pools}
    flat = v1x["train"].reshape(-1, 2)
    mu, sd = flat.mean(0, dtype=np.float64), flat.std(0, dtype=np.float64)

    def v1_channels(s):
        x = (v1x[s] - mu) / sd  # (runs, 10, N, 2)
        return np.concatenate([x[..., 0], x[:, :1, :, 1]], axis=1)  # 10 energy frames + theta(0)

    X = {s: spec(v1_channels(s)) for s in pools}
    del v1x
    start = time.time()
    model, l2, tr_loss, va_loss = fit_linear(X["train"], yt["train"], None, X["val"], yt["val"], None, False)
    with torch.no_grad():
        pred = model.scores_from_spectrum(X["test"]).argmax(1).numpy()
    m = metrics.site_metrics(pred, y["test"])
    linear_rows["linear_v1_inputs"] = {"l2": l2, "train_loss": tr_loss, "val_loss": va_loss, "seconds": round(time.time() - start), "test": m}
    record("linear_v1_inputs", f"convex learned-linear ring filters on v1's inputs (11 channels), L-BFGS, L2={l2}",
           models.count_parameters(model), m)
    print(f"  linear_v1_inputs: mean {m['mae']:.2f} ±1 {m['acc_1']:.1%} ±3 {m['acc_3']:.1%} ({time.time() - start:.0f}s)", flush=True)
    out["linear"] = linear_rows
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = {"created": time.strftime("%Y-%m-%d %H:%M")}

    def record(model, note, n_params, test_metrics):
        line = {"run_id": f"phase1_{model}", "phase": 1, "model": model, "note": note, "seed": None,
                "label": "right", "n_train": 60_000, "params": n_params, "test": test_metrics,
                "created": time.strftime("%Y-%m-%d %H:%M")}
        with open(ROOT / "results" / "runs.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")

    print("growth spectrum grid", flush=True)
    out["growth"] = growth_grid()
    print("best box-car width vs amplitude", flush=True)
    out["best_width"] = best_width_vs_amplitude(args.workers)
    pools = {"train": data.load_pool(ROOT / "data" / "pool_train.npz", 60_000),
             "val": data.load_pool(ROOT / "data" / "pool_val.npz"),
             "test": data.load_pool(ROOT / "data" / "pool_test.npz")}
    print("sensitivity of the label to the initial state", flush=True)
    out["sensitivity"] = sensitivity(pools["test"], args.workers)
    print("baselines on the test pool", flush=True)
    out["baselines"] = baselines(pools, record)
    (ROOT / "results" / "phase1.json").write_text(json.dumps(out, indent=1))
    print("wrote results/phase1.json", flush=True)


if __name__ == "__main__":
    main()
