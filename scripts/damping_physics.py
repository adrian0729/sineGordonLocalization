"""Physics of the damped ring: does the energy still localize, and when?

    python scripts/damping_physics.py --workers 10

Linear damping (-gamma * dtheta/dtau; gamma per time unit, quality factor Q = 1 / gamma) drains the
uniform swing, so its amplitude A(tau) drifts down toward A_c = 1.577, below which no ring mode
grows. A run localizes only if its perturbation has grown enough before that.

Measured on the fixed-damping pools robust_gamma0 ... robust_gamma1e-2 (scripts/make_robustness_pools.py;
5,000 runs that share their initial conditions, labels searched to tau = 150): the share of runs
that localize, the shift of the localization time against the same run undamped, and how often
damping moves the site. The first --long runs are re-simulated to tau = 600 to tell late from never.

Predicted (adiabatic Floquet): the perturbation grows like exp(integral of sigma(A(tau)) dtau), with
sigma the fastest Floquet growth rate at the current amplitude. Damping also slows the perturbation
itself, by gamma / 2, but the swing loses energy about as fast, so the share of energy on the
hotspot feels damping only through A(tau) (the leftover, gamma (1 - r) / 2 with r the swing's mean
squared velocity over its energy, is printed below). Two versions:
- one stage: all of the undamped time T0 is linear growth, so a run needs sigma(A0) T0 e-folds;
- two stages: undamped runs follow T0 = L / sigma(A0) + T_nl (fitted on the gamma = 0 pool), linear
  growth by L e-folds and then a nonlinear stage of T_nl that does not need the swing to be unstable.
  A damped run localizes T_nl after its integral reaches sigma(A0) (T0 - T_nl).

Writes results/damping_physics.json. With --figure it only draws figures/fig20_damping from that file
and results/robustness.jsonl (in the style of scripts/make_figures.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, floquet, metrics, sim  # noqa: E402

GAMMAS = [("0", 0.0), ("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3), ("3e-3", 3e-3), ("1e-2", 1e-2)]
OMEGA0 = (9.81 / 0.2) ** 0.5  # rad/s for a 20 cm pendulum, the report's unit conversion
SAMPLE = sim.Params().sample
EDGES = np.linspace(1.65, 1.85, 11)  # amplitude bins


def growth_curve(a_c, a_max=1.88, step=0.0025, modes=np.arange(1, 9)):
    """Fastest Floquet growth rate on an amplitude grid (zero below the threshold)."""
    grid = np.arange(a_c, a_max + step, step)
    sigma = np.array([max(floquet.growth_rates(a, modes).max(), 0.0) for a in grid])
    return np.concatenate(([0.0, a_c - 1e-9], grid)), np.concatenate(([0.0, 0.0], sigma))


def envelope(a0, gamma, t_final, dt=0.05):
    """Turning-point amplitude of the damped uniform swing (one pendulum), every SAMPLE time units."""
    theta, omega = np.asarray(a0, dtype=float)[:, None].copy(), np.zeros((len(a0), 1))
    n = round(t_final / SAMPLE) + 1
    out = np.empty((len(a0), n))
    for k in range(n):
        energy = 0.5 * omega[:, 0] ** 2 + 1 - np.cos(theta[:, 0])
        out[:, k] = np.arccos(np.clip(1 - energy, -1, 1))
        for _ in range(round(SAMPLE / dt)):
            theta, omega = sim.rk4_step(theta, omega, dt, 0.0, 0.0, None, gamma)  # g, h act only on differences
    return out


def predicted_times(a_grid, s_grid, a0, needed, gamma, t_final):
    """First time the integrated growth reaches `needed` (NaN if not by t_final): the end of the linear stage."""
    if gamma == 0:
        return np.where(needed > 0, needed / np.interp(a0, a_grid, s_grid), np.nan)
    sig = np.interp(envelope(a0, gamma, t_final), a_grid, s_grid)
    grown = np.concatenate([np.zeros((len(a0), 1)), np.cumsum(0.5 * (sig[:, 1:] + sig[:, :-1]) * SAMPLE, axis=1)], axis=1)
    reached = grown >= needed[:, None]
    return np.where(reached.any(axis=1), reached.argmax(axis=1) * SAMPLE, np.nan)


def fit_two_stage(sigma0, t0):
    """Undamped runs: T0 = L / sigma(A0) + T_nl, least squares over runs."""
    (L, t_nl), *_ = np.linalg.lstsq(np.stack([1 / sigma0, np.ones_like(sigma0)], axis=1), t0, rcond=None)
    return float(L), float(t_nl)


def prediction(pred, t0, ok, a_amp, measured_time, n_long):
    """Compare predicted localization times with the simulated outcome."""
    pred_ok = pred <= sim.Params().t_final
    out = {"localized": float(pred_ok.mean()), "agreement": float(np.mean(pred_ok == ok)),
           "localized_long": float(np.mean(np.isfinite(pred[:n_long]))), "a50": a50(a_amp, pred_ok)}
    hit = ok & pred_ok
    if hit.any():
        out["median_shift"] = float(np.median(pred[hit] - t0[hit]))
        out["median_abs_time_error"] = float(np.median(np.abs(pred[hit] - measured_time[hit])))
    return out


def velocity_ratio(a):
    """r = <omega^2> / E over one swing of a pendulum with amplitude a (1 for a harmonic swing)."""
    speed = quad(lambda th: np.sqrt(max(2 * (np.cos(th) - np.cos(a)), 0.0)), 0, a)[0]
    return 4 * speed / float(floquet.period(a)) / (1 - np.cos(a))


def a50(a, ok):
    """Amplitude above which half the runs localize, by linear interpolation over the bins."""
    centres = 0.5 * (EDGES[1:] + EDGES[:-1])
    frac = np.array([ok[(a >= lo) & (a < hi)].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    if (frac >= 0.5).all():
        return f"< {EDGES[0]:.2f}"
    if (frac < 0.5).all():
        return f"> {EDGES[-1]:.2f}"
    i = np.argmax(frac >= 0.5)
    return float(np.interp(0.5, frac[i - 1:i + 1], centres[i - 1:i + 1])) if i else float(centres[0])


def robust_means(prefix, conditions, key="mae"):
    """Mean over seeds of results/robustness.jsonl entries for run dirs starting with `prefix` (latest wins)."""
    latest = {}
    with open(ROOT / "results" / "robustness.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if Path(rec["run_dir"]).name.startswith(prefix):
                latest[(rec["run_dir"], rec["condition"])] = rec["test"]
    out = []
    for cond in conditions:
        vals = [t[key] for (_, c), t in latest.items() if c == cond and t.get("n")]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def figure():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_figures as mf  # noqa: E402  (shared style, colours and save())
    plt = mf.plt
    phys = json.loads((ROOT / "results" / "damping_physics.json").read_text())
    names = [name for name, _ in GAMMAS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), layout="constrained")
    centres = 0.5 * (np.array(phys["by_amplitude"]["edges"][1:]) + np.array(phys["by_amplitude"]["edges"][:-1]))
    shades = dict(zip(names[3:], mf.RAMP[3::2]))  # gamma = 1e-3, 3e-3, 1e-2, light to dark (smaller gamma: all localize)
    for name in names[3:]:
        a.plot(centres, 100 * np.array(phys["by_amplitude"]["measured"][name]), marker="o", markersize=4,
               color=shades[name], label=f"γ = {name}")
        a.plot(centres, 100 * np.array(phys["by_amplitude"]["predicted"][name]), linestyle="--", linewidth=1.0,
               color=shades[name])
    a.set_xlabel("Initial swing amplitude A")
    a.set_ylabel("Runs localizing by τ = 150 (%)")
    a.set_ylim(-3, 103)
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), title="solid: simulated\ndashed: two-stage Floquet",
             title_fontsize=7.5)
    a.set_title("(a) Damping decides whether the energy localizes", loc="left", fontsize=9)
    conds = [f"robust_gamma{n}" for n in names[:-1]]
    x = np.arange(len(conds))
    series = [("v2_final_", conds, mf.SLOT[0], "v2 final, trained undamped", "-"),
              ("v2_final_", ["robust_gamma0"] + [f"{c}_labels_only" for c in conds[1:]], mf.SLOT[2],
               "v2 final fed the undamped frames (outcome change only)", "-"),
              ("damped_trained_s", conds, mf.SLOT[1], "v2 recipe, trained on damped chains", "-"),
              ("v1_seed", conds, mf.MUTED, "v1 rebuilt, trained undamped", "none")]
    curves = [(robust_means(prefix, cs), color, label, style) for prefix, cs, color, label, style in series]
    top = 1.3 * max(np.nanmax(y) for y, _, label, _ in curves if not label.startswith("v1"))
    for y, color, label, style in curves:
        keep = np.isfinite(y)  # v1 was re-scored on fewer pools (markers only)
        if keep.any():
            b.plot(x[keep], np.minimum(y[keep], 0.97 * top), marker="o", markersize=5, color=color, linestyle=style,
                   label=label)
        for xi, yi in zip(x[keep], y[keep]):
            if yi > 0.97 * top:  # off the scale: mark it at the top with its value
                b.annotate(f"{yi:.1f}, off scale", (xi, 0.97 * top), xytext=(-8, 0), textcoords="offset points", ha="right",
                           va="center", color=mf.INK2, fontsize=8)
    b.set_xticks(x, [f"{n}\n(Q {'∞' if g == 0 else f'{1 / g:,.0f}'})" for n, g in GAMMAS[:-1]])
    b.set_xlabel("Damping γ per time unit (quality factor)")
    b.set_ylabel("Mean distance, runs that localize (sites)")
    b.set_ylim(0, top)
    b.legend(loc="upper left", fontsize=7.5)
    b.set_title("(b) Early warning on damped chains", loc="left", fontsize=9)
    fig.suptitle("Linear damping: the swing decays toward A_c = 1.577 (5,000 runs per γ, shared initial conditions)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    mf.save(fig, "fig20_damping")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--long", type=int, default=2000, help="runs re-simulated to tau = --t-long")
    parser.add_argument("--t-long", type=float, default=600.0)
    parser.add_argument("--figure", action="store_true", help="only draw figures/fig20_damping from saved results")
    args = parser.parse_args()
    if args.figure:
        return figure()

    pools = {name: data.load_pool(ROOT / "data" / f"robust_gamma{name}.npz") for name, _ in GAMMAS}
    ref = pools["0"]
    assert all(np.array_equal(p["theta0"], ref["theta0"]) for p in pools.values()), "pools must share initial conditions"
    a_amp, a0 = ref["A"], ref["theta0"].mean(axis=1)  # the drawn amplitude, and the swing's actual one
    a_c = floquet.threshold_amplitude()
    a_grid, s_grid = growth_curve(a_c)
    t0 = ref["time_right"].astype(float)
    sigma0 = np.interp(a0, a_grid, s_grid)
    L, t_nl = fit_two_stage(sigma0, t0)
    needed = {"one_stage": (sigma0 * t0, 0.0), "two_stage": (sigma0 * np.clip(t0 - t_nl, 0, None), t_nl)}
    print(f"A_c = {a_c:.4f}; one stage: sigma(A0) T0 = {np.median(sigma0 * t0):.2f} e-folds (median); "
          f"two stages: T0 = {L:.2f} / sigma(A0) + {t_nl:.1f}; r = <omega^2>/E = {velocity_ratio(1.65):.3f} / "
          f"{velocity_ratio(1.75):.3f} / {velocity_ratio(1.85):.3f} at A = 1.65 / 1.75 / 1.85", flush=True)

    long_runs = ref["theta0"][:args.long]
    rows, by_amp = [], {"edges": EDGES.tolist(), "measured": {}, "predicted": {}}
    for name, gamma in GAMMAS:
        p = pools[name]
        ok = p["site_right"] >= 0
        both = ok & (ref["site_right"] >= 0)
        dist = metrics.ring_distance(p["site_right"][both], ref["site_right"][both])
        both_avg = (p["site_avg"] >= 0) & (ref["site_avg"] >= 0)
        dist_avg = metrics.ring_distance(p["site_avg"][both_avg], ref["site_avg"][both_avg])
        preds = {k: predicted_times(a_grid, s_grid, a0, need, gamma, args.t_long) + extra for k, (need, extra) in needed.items()}
        row = {"gamma": gamma, "Q": 1 / gamma if gamma else None,
               "amplitude_decay_s_20cm": 2 / (gamma * OMEGA0) if gamma else None,  # 1/e time of a harmonic swing
               "localized": float(ok.mean()), "localized_avg": float((p["site_avg"] >= 0).mean()),
               "median_time": float(np.median(p["time_right"][ok])) if ok.any() else None,
               "median_shift": float(np.median(p["time_right"][both] - t0[both])) if both.any() else None,
               "mean_shift": float(np.mean(p["time_right"][both] - t0[both])) if both.any() else None,
               "site_same": float(np.mean(dist == 0)) if both.any() else None,
               "site_within1": float(np.mean(dist <= 1)) if both.any() else None,
               "site_within3": float(np.mean(dist <= 3)) if both.any() else None,
               "avg_site_within1": float(np.mean(dist_avg <= 1)) if both_avg.any() else None,
               "avg_site_within3": float(np.mean(dist_avg <= 3)) if both_avg.any() else None,
               "a50": a50(a_amp, ok),
               "predicted": {k: prediction(v, t0, ok, a_amp, p["time_right"], args.long) for k, v in preds.items()}}
        if gamma:
            env = envelope(np.array([1.75]), gamma, args.t_long)[0]
            below = env < a_c
            row["tau_to_threshold_A1.75"] = float(np.argmax(below) * SAMPLE) if below.any() else None
            sig = np.interp(env, a_grid, s_grid)
            row["growth_available_A1.75"] = float(np.sum(0.5 * (sig[1:] + sig[:-1])) * SAMPLE)
            long = data.localize_parallel(long_runs, sim.Params(t_final=args.t_long), chunk=100, workers=args.workers,
                                          labels=("right",), gamma=gamma)
            late = long["site_right"] >= 0
            row["localized_long"] = float(late.mean())
            row["localized_same_runs"] = float(ok[:len(long_runs)].mean())  # by tau = 150, on the re-simulated runs
            row["n_long"] = int(len(long_runs))
            row["median_time_long"] = float(np.median(long["time_right"][late])) if late.any() else None
        by_amp["measured"][name] = [float(ok[(a_amp >= lo) & (a_amp < hi)].mean()) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        by_amp["predicted"][name] = [float((preds["two_stage"] <= sim.Params().t_final)[(a_amp >= lo) & (a_amp < hi)].mean())
                                     for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {"A_c": a_c, "omega0_20cm": OMEGA0, "runs": int(len(a0)), "t_long": args.t_long,
           "one_stage_growth_median": float(np.median(sigma0 * t0)), "two_stage": {"L": L, "T_nl": t_nl},
           "conditions": rows, "by_amplitude": by_amp,
           "velocity_ratio": {f"{a:.2f}": velocity_ratio(a) for a in (1.65, 1.75, 1.85)}}
    (ROOT / "results" / "damping_physics.json").write_text(json.dumps(out, indent=1))

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def num(x, fmt="{:.1f}"):
        return "n/a" if x is None else (fmt.format(x) if not isinstance(x, str) else x)

    print("\nPredictions in parentheses: two stages; one stage localized by tau = 150: " +
          ", ".join(f"{r['gamma']:g}: {pct(r['predicted']['one_stage']['localized'])}" for r in rows))
    print("\n| gamma | Q | 1/e amplitude decay, 20 cm (s) | Localized by tau = 150 (predicted) | by tau = 600 | "
          "Median shift (predicted) | Same site / ±1 / ±3 | Swing-averaged ±1 | A where half localize (predicted) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        q = r["predicted"]["two_stage"]
        print(f"| {r['gamma']:g} | {num(r['Q'], '{:.0f}')} | {num(r['amplitude_decay_s_20cm'], '{:.0f}')} | "
              f"{pct(r['localized'])} ({pct(q['localized'])}) | {pct(r.get('localized_long'))} "
              f"(τ ≤ 150 on the same runs: {pct(r.get('localized_same_runs'))}) | "
              f"{num(r['median_shift'])} ({num(q.get('median_shift'))}) | "
              f"{pct(r['site_same'])} / {pct(r['site_within1'])} / {pct(r['site_within3'])} | {pct(r['avg_site_within1'])} | "
              f"{num(r['a50'], '{:.3f}')} ({num(q['a50'], '{:.3f}')}) |")


if __name__ == "__main__":
    main()
