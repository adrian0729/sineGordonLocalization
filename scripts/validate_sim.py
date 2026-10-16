"""Check the vectorized simulator against the paper's own code path and known results.

    python scripts/validate_sim.py

1. The paper's pipeline (its odeint right-hand side, energy formula and first-crossing loop)
   vs our RK4 labels on the same initial conditions.
2. Step-size convergence of all four label conventions.
3. Box-car replication: Fig. 5 of the paper reports 5.41 for kernel 17 at "t = 5" (sample 5).
4. Floquet threshold and unstable band, plus the box-car and Floquet-filter baselines on the
   test pool.
Writes results/validation.json.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, floquet, metrics, sim  # noqa: E402


def paper_rhs(x, t, g, h):
    # f() from the paper's notebooks
    x1, x2 = x[0:100], x[100:200]
    dx2 = -np.sin(x1) + g * (np.roll(x1, 1) + np.roll(x1, -1) - 2 * x1) + h * (np.average(x1) * np.ones(100) - x1)
    return np.hstack([x2, dx2])


def paper_label(theta0, g=0.75, h=0.5):
    """Site and time from the paper's pipeline: odeint, the notebooks' energy formula (doubled,
    spring to the right neighbour) and the first-crossing loop of energyLoc()/locTime()."""
    t = np.arange(0, 150, 0.5)
    x = odeint(paper_rhs, np.hstack([theta0, np.zeros(100)]), t, args=(g, h))
    theta, dtheta = x[:, :100], x[:, 100:]
    energy = (dtheta**2 - 2 * np.cos(theta) + 2 + g * (theta - np.roll(theta, -1, axis=1)) ** 2
              + h * (theta - theta.mean(axis=1, keepdims=True)) ** 2)
    entotal = energy[0].sum()
    for n in range(300):
        if energy[n].max() > 0.1 * entotal:
            return int(energy[n].argmax()), n * 0.5
    return -1, np.nan


def boxcar(x, K):
    """Centered circular moving average over an odd window of K sites."""
    return sum(np.roll(x, j, axis=-1) for j in range(-(K // 2), K // 2 + 1)) / K


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odeint-runs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    out = {"created": time.strftime("%Y-%m-%d %H:%M")}

    n = args.odeint_runs
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(paper_label, test["theta0"][:n], chunksize=8))
    site_ode = np.array([r[0] for r in res])
    time_ode = np.array([r[1] for r in res])
    d = metrics.ring_distance(site_ode, test["site_right"][:n])
    out["paper_odeint_vs_rk4"] = {
        "runs": n, "same_site": float((d == 0).mean()), "within_1": float((d <= 1).mean()),
        "mean_distance": float(d.mean()), "same_time": float((time_ode == test["time_right"][:n]).mean()),
        "max_time_difference": float(np.nanmax(np.abs(time_ode - test["time_right"][:n]))),
        "seconds": round(time.time() - start, 1)}
    print("paper odeint vs RK4:", out["paper_odeint_vs_rk4"], flush=True)

    fine = data.localize_parallel(test["theta0"][:1000], sim.Params(dt=0.025), chunk=100, workers=args.workers)
    out["dt_halved_same_site"] = {name: float((fine[f"site_{name}"] == test[f"site_{name}"][:1000]).mean())
                                  for name in sim.LABELS}
    print("dt 0.05 vs 0.025, fraction with identical site:", out["dt_halved_same_site"], flush=True)

    _, theta0 = data.initial_conditions(2000, seed=7, a_range=(1.75, 1.75))
    labels = data.localize_parallel(theta0, chunk=200, workers=args.workers)
    theta, omega = sim.evolve(theta0, 11)
    E = sim.site_energy(theta, omega, 0.75, 0.5, "right")
    out["boxcar_A175_mean_distance"] = {
        f"sample{k}_K{K}": float(metrics.ring_distance(boxcar(E[:, k], K).argmax(1), labels["site_right"]).mean())
        for k in (5, 10) for K in (1, 15, 17, 19)}
    print("box-car at A = 1.75 (paper Fig. 5: K17 at sample 5 -> 5.41):", out["boxcar_A175_mean_distance"], flush=True)

    sigma = floquet.growth_rates(1.75, np.arange(1, 51))
    out["floquet"] = {"threshold_amplitude": floquet.threshold_amplitude(),
                      "unstable_modes_A175": (np.flatnonzero(sigma > 1e-6) + 1).tolist(),
                      "growth_rates_A175": np.round(sigma[:6], 5).tolist()}
    print("Floquet:", out["floquet"], flush=True)

    theta, omega = sim.evolve(test["theta0"], 10)
    E = sim.site_energy(theta, omega, 0.75, 0.5, "right")
    baselines = {f"boxcar_K17_tau{k * 0.5:g}": metrics.site_metrics(boxcar(E[:, k], 17).argmax(1), test["site_right"])
                 for k in (0, 9)}
    baselines["floquet_filter_6_periods"] = metrics.site_metrics(
        floquet.floquet_filter(test["theta0"], test["A"], periods=6), test["site_right"])
    out["test_pool_baselines"] = baselines
    for name, m in baselines.items():
        print(f"{name:28s} mean {m['mae']:.2f}  ±1 {m['acc_1']:.1%}  ±3 {m['acc_3']:.1%}  >10 {m['over_10']:.1%}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
