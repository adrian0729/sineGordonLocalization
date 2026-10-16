"""The Floquet trade-off between sensor noise and waiting, fitted to the noise-and-timing grid on both labels.

    python scripts/noise_tradeoff.py     # prints the tables; writes results/noise_tradeoff.json (needs runs/)

The grid trains one model per sensing setup (S1-S3), start time tau and sensor noise sigma_s: Phase 3 on v1's
label, and its addendum `phase3_avg` on the swing-averaged label (same configs plus label="avg"). If waiting
helps only because the growing modes rise above a fixed sensor noise, the error depends on sigma_s and tau only
through sigma_s * exp(-k * tau) with k = sigma_1 = 0.118, the fastest Floquet growth rate at A = 1.75, and a 10x
noisier sensor costs ln(10) / k = 19.5 time units of waiting.

Fit, per setup and label, to the 24 cells in log space (one seed each):
    mean distance = floor(tau) + A * (sigma_s * exp(-k * tau))^p
with a free noise-free floor per start time, so that waiting without noise is not mistaken for the trade-off.
The 95% interval of k is a profile-likelihood interval (F test on the residual sum of squares). As a check
without a model, each cell well above its floor is matched to the noise that gives the same error from an
earlier start (log-log interpolation along that start's curve), which gives one waiting cost per pair.

The swing amplitude of the runs spans 1.65-1.85, over which sigma_1 doubles (0.075-0.153). The same fit is
therefore repeated on four amplitude bins of the test pool (from each run's saved test predictions), and each
bin's k is compared with sigma_1 and with sigma_bar, the mean growth rate of the growing modes, at the bin's
amplitudes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, data, experiment, floquet, metrics  # noqa: E402

SIGMA1 = 0.118  # fastest Floquet growth rate at A = 1.75 (Phase 1)
KINDS = ("S1", "S2", "S3")
STARTS = (0, 10, 20, 30)
NOISES = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LABELS = ("right", "avg")  # v1's label (Phase 3's runs) and the swing-averaged label
ABOVE_FLOOR = 1.25  # the model-free check uses cells at least 25% above their start's noise-free cell
A_EDGES = (1.65, 1.70, 1.75, 1.80, 1.85)  # amplitude bins of the test pool


def grid():
    """{label: {kind: {(start, noise): record}}} for the phase3_avg table (latest record per config id)."""
    by_id = {}
    with open(experiment.RESULTS) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("config_id"):
                    by_id[rec["config_id"]] = rec
    out = {label: {kind: {} for kind in KINDS} for label in LABELS}
    for cfg in ablations.phase3_avg():
        rec = by_id.get(experiment.config_id(cfg))
        if rec is None:
            continue
        kind = next(k for k in KINDS if cfg["name"].startswith(k + "_"))
        out[cfg.get("label", "right")][kind][(cfg["view"]["start"], cfg["view"]["noise"])] = rec
    return out


def predict(q, sigma, tau):
    floors = np.exp(q[:len(STARTS)])[np.searchsorted(STARTS, tau)]
    ln_a, p, k = q[len(STARTS):]
    return floors + np.exp(ln_a) * (sigma * np.exp(-k * tau)) ** p


def fit(sigma, tau, mae, k=None):
    """Least squares in log space; with k given, k is held fixed. Returns (parameters, residual sum of squares)."""
    floor0 = [np.log(mae[(tau == t) & (sigma == 0)][0]) for t in STARTS]
    top = int(np.argmax(mae))  # start A where the worst cell sits on the predicted curve (p = 1, k = sigma_1)
    excess = max(mae[top] - np.exp(floor0[STARTS.index(tau[top])]), 1e-3)
    q0 = np.r_[floor0, np.log(excess) - np.log(sigma[top] * np.exp(-SIGMA1 * tau[top])), 1.0, SIGMA1]
    lo = np.r_[[-np.inf] * len(STARTS), -np.inf, 0.2, -0.5]
    hi = np.r_[[np.inf] * len(STARTS), np.inf, 5.0, 1.0]
    free = np.ones(len(q0), dtype=bool)
    if k is not None:
        free[-1], q0[-1] = False, k

    def residual(x):
        q = q0.copy()
        q[free] = x
        return np.log(predict(q, sigma, tau)) - np.log(mae)

    best = None
    for p0 in (0.7, 1.0, 1.5):  # a few starting exponents; keep the best fit
        x0 = q0[free].copy()
        x0[len(STARTS) + 1] = p0
        r = optimize.least_squares(residual, x0, bounds=(lo[free], hi[free]))
        if best is None or r.cost < best.cost:
            best = r
    q = q0.copy()
    q[free] = best.x
    return q, float(2 * best.cost)


def profile_interval(sigma, tau, mae, sse, n_params, level=0.95):
    """Values of k whose best fit is not worse than the free fit by more than an F test at `level` allows."""
    dof = len(mae) - n_params
    limit = sse * (1 + stats.f.ppf(level, 1, dof) / dof)
    ks = np.linspace(-0.1, 0.4, 501)
    ok = [k for k in ks if fit(sigma, tau, mae, k)[1] <= limit]
    return (float(min(ok)), float(max(ok))) if ok else (None, None)


def waiting_cost(k):
    return float(np.log(10) / k) if k and k > 0 else None


def collapse_fit(mae_of):
    """The fit above for {(start, noise): mean distance}."""
    sigma = np.array([n for t in STARTS for n in NOISES])
    tau = np.array([t for t in STARTS for n in NOISES], dtype=float)
    mae = np.array([mae_of[(t, n)] for t in STARTS for n in NOISES])
    q, sse = fit(sigma, tau, mae)
    k_lo, k_hi = profile_interval(sigma, tau, mae, sse, len(q))
    _, sse_sigma1 = fit(sigma, tau, mae, SIGMA1)
    _, sse_k0 = fit(sigma, tau, mae, 0.0)
    return {
        "k": float(q[-1]), "k_ci95": [k_lo, k_hi], "p": float(q[-2]), "A": float(np.exp(q[-3])),
        "floors": {t: float(f) for t, f in zip(STARTS, np.exp(q[:len(STARTS)]))},
        "cost_per_decade": waiting_cost(q[-1]),
        "cost_ci95": [waiting_cost(k_hi), waiting_cost(k_lo) if k_lo and k_lo > 0 else None],
        "rms_log_residual": float(np.sqrt(sse / len(mae))),
        "rms_log_residual_k_sigma1": float(np.sqrt(sse_sigma1 / len(mae))),
        "rms_log_residual_k0": float(np.sqrt(sse_k0 / len(mae))),
    }


def iso_error_pairs(mae_of):
    """Model-free check: for each cell well above its floor, the noise that gives the same error from an earlier
    start (log-log interpolation along that start's curve), and the waiting cost per 10x noise that implies."""
    pairs = []
    for tb in STARTS[1:]:
        for nb in NOISES[1:]:
            mae_b = mae_of[(tb, nb)]
            if mae_b < ABOVE_FLOOR * mae_of[(tb, 0.0)]:
                continue
            for ta in STARTS:
                if ta >= tb:
                    break
                curve = np.maximum.accumulate([mae_of[(ta, n)] for n in NOISES[1:]])  # monotone in noise
                if not curve[0] <= mae_b <= curve[-1]:
                    continue
                i = min(max(int(np.searchsorted(curve, mae_b)), 1), len(curve) - 1)
                x, y = np.log(NOISES[1:]), np.log(curve)
                frac = 0.0 if y[i] == y[i - 1] else (np.log(mae_b) - y[i - 1]) / (y[i] - y[i - 1])
                sigma_a = float(np.exp(x[i - 1] + frac * (x[i] - x[i - 1])))
                shift = np.log(nb / sigma_a)
                pairs.append({"tau_b": tb, "sigma_b": nb, "mae_b": mae_b, "tau_a": ta, "sigma_a": sigma_a,
                              "cost_per_decade": float((tb - ta) * np.log(10) / shift) if shift > 0 else None})
    return pairs


def diagonals(mae_of):
    """Cells with nearly the same scaled noise sigma_s * exp(-sigma_1 * tau): noise x ~3.2 per 10 time units."""
    rows = {}
    for i, n0 in enumerate(NOISES[1:]):
        rows[f"{n0:g}"] = {t: {"noise": NOISES[1 + i + j], "scaled": NOISES[1 + i + j] * np.exp(-SIGMA1 * t),
                               "mae": mae_of[(t, NOISES[1 + i + j])]}
                           for j, t in enumerate(STARTS) if i + j < len(NOISES) - 1}
    return rows


def by_amplitude(recs, label, pool):
    """{bin: {(start, noise): mean distance}} on the test runs in each amplitude bin, from saved predictions."""
    truth, amp = pool[f"site_{label}"], pool["A"]
    bins = np.digitize(amp, A_EDGES[1:-1])
    out = {b: {} for b in range(len(A_EDGES) - 1)}
    for key, rec in recs.items():
        pred = np.load(ROOT / rec["run_dir"] / "pred_test.npy").astype(int)
        assert len(pred) == len(truth)
        for b in out:
            keep = (bins == b) & (truth >= 0)
            out[b][key] = float(metrics.ring_distance(pred[keep], truth[keep]).mean())
    return out


def bin_growth(pool):
    """The fastest growth rate sigma_1 and the mean rate of the growing modes, sigma_bar, averaged over the test
    runs in each amplitude bin (and over the whole pool)."""
    grid_a = np.linspace(A_EDGES[0], A_EDGES[-1], 41)
    rates = np.array([floquet.growth_rates(a, np.arange(1, 10)) for a in grid_a])
    s1 = np.interp(pool["A"], grid_a, rates[:, 0])
    s_bar = np.interp(pool["A"], grid_a, [r[r > 1e-6].mean() for r in rates])
    bins = np.digitize(pool["A"], A_EDGES[1:-1])
    out = {b: {"A": [A_EDGES[b], A_EDGES[b + 1]], "runs": int((bins == b).sum()), "sigma1": float(s1[bins == b].mean()),
               "sigma_bar": float(s_bar[bins == b].mean())} for b in range(len(A_EDGES) - 1)}
    out["all"] = {"A": [A_EDGES[0], A_EDGES[-1]], "runs": len(bins), "sigma1": float(s1.mean()), "sigma_bar": float(s_bar.mean())}
    return out


def fmt(c, digits=1):
    return "–" if c is None else f"{c:.{digits}f}"


def main():
    recs = grid()
    pool = data.load_pool(ROOT / "data" / "pool_test.npz")
    growth = bin_growth(pool)
    out = {"sigma1": SIGMA1, "predicted_cost_per_decade": float(np.log(10) / SIGMA1), "amplitude_bins": growth,
           "labels": {}}
    pooled = ["| Label | Setup | Fitted k | Waiting per 10x noise (95% CI) | Model-free pairs: median (n) | "
              "rms log residual: k free / k = σ₁ / k = 0 |", "|---|---|---|---|---|---|"]
    bins = [g for b, g in growth.items() if b != "all"]
    binned = ["| Label | Setup | " + " | ".join(f"A {g['A'][0]:.2f}–{g['A'][1]:.2f}" for g in bins) + " |",
              "|---|---|" + "---|" * len(bins)]
    for key, name in (("sigma1", "σ₁"), ("sigma_bar", "σ̄, mean of growing modes")):
        binned.append(f"| | {name} (its cost) | " + " | ".join(f"{g[key]:.3f} ({np.log(10) / g[key]:.1f})" for g in bins) + " |")
    for label in LABELS:
        out["labels"][label] = {}
        for kind in KINDS:
            cells = recs[label][kind]
            if len(cells) < len(STARTS) * len(NOISES):
                print(f"{label} {kind}: {len(cells)} of {len(STARTS) * len(NOISES)} cells so far, skipped")
                continue
            mae_of = {key: rec["test"]["mae"] for key, rec in cells.items()}
            r = collapse_fit(mae_of)
            r["iso_error_pairs"] = iso_error_pairs(mae_of)
            costs = [p["cost_per_decade"] for p in r["iso_error_pairs"] if p["cost_per_decade"]]
            r["iso_error_cost_median"] = float(np.median(costs)) if costs else None
            r["diagonals"] = diagonals(mae_of)
            r["cells"] = {f"t{t}_noise{n:g}": {m: cells[(t, n)]["test"][m] for m in ("mae", "acc_0", "acc_1", "acc_3", "over_10")}
                          for t in STARTS for n in NOISES}
            r["by_amplitude"] = {b: dict(collapse_fit(m), cells={f"t{t}_noise{n:g}": v for (t, n), v in m.items()})
                                 for b, m in by_amplitude(cells, label, pool).items()}
            out["labels"][label][kind] = r
            pooled.append(f"| {label} | {kind} | {r['k']:.3f} | {fmt(r['cost_per_decade'])} "
                          f"({fmt(r['cost_ci95'][0])}–{fmt(r['cost_ci95'][1])}) | {fmt(r['iso_error_cost_median'])} "
                          f"({len(costs)}) | {r['rms_log_residual']:.3f} / {r['rms_log_residual_k_sigma1']:.3f} / "
                          f"{r['rms_log_residual_k0']:.3f} |")
            binned.append(f"| {label} | {kind} | " + " | ".join(
                f"{a['k']:.3f} ({fmt(a['cost_per_decade'])}; {fmt(a['cost_ci95'][0])}–{fmt(a['cost_ci95'][1])})"
                for a in r["by_amplitude"].values()) + " |")
    print(f"Predicted at A = 1.75: ln 10 / sigma_1 = {np.log(10) / SIGMA1:.1f} time units per 10x noise; whole test "
          f"pool: sigma_1 {growth['all']['sigma1']:.3f} ({np.log(10) / growth['all']['sigma1']:.1f}), sigma_bar "
          f"{growth['all']['sigma_bar']:.3f} ({np.log(10) / growth['all']['sigma_bar']:.1f})\n")
    print("\n".join(pooled))
    print("\nBy amplitude bin of the test runs: fitted k (waiting per 10x noise; 95% CI)\n")
    print("\n".join(binned))
    for label in out["labels"]:
        for kind, r in out["labels"][label].items():
            print(f"\n{label} {kind}: floors " + ", ".join(f"tau {t}: {f:.2f}" for t, f in r["floors"].items())
                  + f"; p = {r['p']:.2f}")
            for n0, row in r["diagonals"].items():
                print(f"  diagonal from {n0:>6} at tau 0: " + "  ".join(
                    f"tau {t} ({c['noise']:g}): {c['mae']:.2f}" for t, c in row.items()))
    path = ROOT / "results" / "noise_tradeoff.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
