"""Two physics checks on the prediction target: is the localization time just a parameter trend, and
is the hotspot really one pendulum? Writes results/hotspot_timing.json.

    python scripts/hotspot_timing.py

When: predictors of the localization time on the test pool, from simplest to the network:
- a constant (the training median)
- the amplitude A alone (cubic fit of log time)
- a power clock: grow the initial offsets in ring modes 1-6 at their Floquet rates until their summed
  power reaches one; calibrated with a linear fit
- a peak clock: the same growth, stopped when the tallest bump of the summed modes reaches one rad,
  so it also knows how the modes line up in space; calibrated with A
The network's time head comes from results/time_head.json (scripts/eval_final.py).

Where: for 2,000 test runs, the energy profile at the moment v1's label fires: the top pendulum's
share, the width of the swing-averaged hump, competing humps, and how far the hump moves afterwards.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, floquet, sim  # noqa: E402

MODES = np.arange(1, 7)
GRID = np.linspace(1.64, 1.86, 23)


def mode_terms(pool, rates):
    A = pool["A"].astype(float)
    d = pool["theta0"] - pool["theta0"].mean(axis=1, keepdims=True)
    c = np.fft.rfft(d, axis=1)[:, 1:7] / d.shape[1]
    sig = np.clip(np.stack([np.interp(A, GRID, rates[:, j]) for j in range(len(MODES))], axis=1), 0, None)
    return A, c, sig


def bisect(stop, n, hi=400.0, steps=50):
    """Smallest t in [0, hi] with stop(t) true, for n runs at once (stop is monotone in t)."""
    lo, hi = np.zeros(n), np.full(n, hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        done = stop(mid)
        lo, hi = np.where(done, lo, mid), np.where(done, mid, hi)
    return (lo + hi) / 2


def power_clock(c, sig):
    power = np.abs(c) ** 2
    return bisect(lambda t: np.sum(power * np.exp(2 * sig * t[:, None]), axis=1) >= 1.0, len(c))


def peak_clock(c, sig, n_sites, chunk=2000):
    phase = np.exp(2j * np.pi * np.outer(MODES, np.arange(n_sites)) / n_sites)
    out = np.empty(len(c))
    for s in range(0, len(c), chunk):
        cc, ss = c[s:s + chunk], sig[s:s + chunk]
        out[s:s + chunk] = bisect(lambda t: np.abs(2 * np.real((cc * np.exp(ss * t[:, None])) @ phase)).max(axis=1) >= 1.0,
                                  len(cc))
    return out


def fit(X_tr, y_tr, X_te):
    w, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return X_te @ w


def time_summary(pred, true):
    rel = np.abs(pred - true) / true
    return {"median_rel_error": float(np.median(rel)), "within_10pct": float(np.mean(rel <= 0.1)),
            "median_abs_error": float(np.median(np.abs(pred - true))),
            "r2_log_time": float(1 - np.var(np.log(true) - np.log(pred)) / np.var(np.log(true)))}


def when():
    rates = np.array([floquet.growth_rates(a, MODES) for a in GRID])
    train = data.load_pool(ROOT / "data" / "pool_train.npz", 20_000)
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    A_tr, c_tr, s_tr = mode_terms(train, rates)
    A_te, c_te, s_te = mode_terms(test, rates)
    t_tr, t_te = train["time_right"].astype(float), test["time_right"].astype(float)
    poly = lambda A: np.stack([np.ones_like(A), A, A ** 2, A ** 3], axis=1)  # noqa: E731
    pw_tr, pw_te = power_clock(c_tr, s_tr), power_clock(c_te, s_te)
    pk_tr, pk_te = peak_clock(c_tr, s_tr, train["theta0"].shape[1]), peak_clock(c_te, s_te, test["theta0"].shape[1])
    lin = lambda p, A: np.stack([np.ones_like(p), p, A, A ** 2], axis=1)  # noqa: E731
    out = {"constant": time_summary(np.full_like(t_te, np.median(t_tr)), t_te),
           "amplitude_only": time_summary(np.exp(fit(poly(A_tr), np.log(t_tr), poly(A_te))), t_te),
           "power_clock": time_summary(fit(lin(pw_tr, A_tr), t_tr, lin(pw_te, A_te)), t_te),
           "peak_clock": time_summary(fit(lin(pk_tr, A_tr), t_tr, lin(pk_te, A_te)), t_te)}
    path = ROOT / "results" / "time_head.json"
    if path.exists():
        runs = json.loads(path.read_text())["runs"]
        out["network_time_head"] = {k: float(np.mean([r[k] for r in runs.values()]))
                                    for k in ("median_rel_error", "within_10pct", "median_abs_error")}
    out["corr_log_time_vs_A"] = float(np.corrcoef(np.log(t_te), A_te)[0, 1])
    return out


def where(n_runs=2000, seed=0):
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    idx = np.sort(np.random.default_rng(seed).choice(len(test["A"]), n_runs, replace=False))
    p = sim.Params()
    n_samples = int(np.nanmax(test["time_right"][idx]) / p.sample) + 61   # 30 time units past the latest
    W, N = p.avg_window, test["theta0"].shape[1]
    stats = {k: [] for k in ("top_share_instant", "top_share_swing", "top3_share_swing", "hump_width_fwhm",
                             "second_hump_ratio", "moved_after_8", "moved_after_30", "swing_vs_v1_label")}

    def ring(a, b):
        return int(min(abs(a - b), N - abs(a - b)))

    for start in range(0, n_runs, 250):
        sub = idx[start:start + 250]
        th, om = sim.evolve(test["theta0"][sub], n_samples, p)
        e_right = sim.site_energy(th, om, p.g, p.h, bond="right")
        e_sym = sim.site_energy(th, om, p.g, p.h, bond="sym")
        csum = np.concatenate([np.zeros_like(e_sym[:, :1]), np.cumsum(e_sym, axis=1)], axis=1)
        for j, run in enumerate(sub):
            k = int(round(test["time_right"][run] / p.sample))

            def swing(kk):  # symmetric energy averaged over the last W samples (one swing)
                kk = min(kk, n_samples - 1)
                return (csum[j, kk + 1] - csum[j, max(kk + 1 - W, 0)]) / min(W, kk + 1)

            stats["top_share_instant"].append(e_right[j, k].max() / e_right[j, k].sum())
            prof = swing(k)
            top = int(prof.argmax())
            stats["top_share_swing"].append(prof[top] / prof.sum())
            stats["top3_share_swing"].append(prof[[(top - 1) % N, top, (top + 1) % N]].sum() / prof.sum())
            above, width = prof >= prof[top] / 2, 1
            for step in (1, -1):
                n = top
                while above[(n + step) % N] and width < N:
                    n, width = (n + step) % N, width + 1
            stats["hump_width_fwhm"].append(width)
            far = np.abs((np.arange(N) - top + N // 2) % N - N // 2) >= 5
            peaks = (prof > np.roll(prof, 1)) & (prof >= np.roll(prof, -1)) & far
            stats["second_hump_ratio"].append(float(prof[peaks].max() / prof[top]) if peaks.any() else 0.0)
            stats["moved_after_8"].append(ring(int(swing(k + 16).argmax()), top))
            stats["moved_after_30"].append(ring(int(swing(k + 60).argmax()), top))
            stats["swing_vs_v1_label"].append(ring(top, int(test["site_right"][run])))
        print(f"where: {start + len(sub)}/{n_runs} runs", flush=True)
    out = {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
           for k, v in stats.items()}
    second = np.asarray(stats["second_hump_ratio"])
    out["share_second_hump_over_half"] = float(np.mean(second >= 0.5))
    out["share_second_hump_over_80pct"] = float(np.mean(second >= 0.8))
    for key in ("moved_after_8", "moved_after_30"):
        m = np.asarray(stats[key])
        out[f"{key}_within1"] = float(np.mean(m <= 1))
        out[f"{key}_over5"] = float(np.mean(m > 5))
    return out


def main():
    result = {"when": when()}
    for k, v in result["when"].items():
        print("when", k, v, flush=True)
    result["where"] = where()
    for k, v in result["where"].items():
        print("where", k, v, flush=True)
    (ROOT / "results" / "hotspot_timing.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
