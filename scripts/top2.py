"""Two-site output: let a trained model name a second site, for runs where a rival hump competes.

    python scripts/top2.py                                   # the final models on v1's, the averaged and the hump labels
    python scripts/top2.py --runs "runs/deploy_*_s*_*" --out results/top2_deploy.json

The two sites are the two highest peaks of the predicted distribution at least MIN_APART sites apart: the most
likely site, and the highest local maximum of the probability at least MIN_APART sites from it. Each model is
scored on its own label (test pool; runs that never localize on that label are left out). "One of two within
±1" counts a run when either site is within one pendulum of the truth. Also scored: naming the second site only
when the 3 sites centred on it hold at least a given share of the probability (MASSES), and, as the reference
for luck, a second site drawn at random from those at least MIN_APART sites from the first.

Rival humps: the test of scripts/hotspot_timing.py, now for all test runs. Take the symmetric energy averaged
over the swing (16 samples) that ends when v1's label fires; a run has a rival when a local maximum at least 5
sites from the top pendulum holds at least half its energy (18% of runs). The same test at the moment each
model's own label fires is reported too. Results per run and averaged over the seeds of each config, written to
results/top2.json (or --out).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_final import outputs  # noqa: E402
from sgloc import data, metrics, sim  # noqa: E402

MIN_APART = 5
RIVAL = 0.5            # a rival hump holds at least this share of the top pendulum's swing-averaged energy
MASSES = (0.05, 0.1, 0.2, 0.3)  # name the second site only when the 3 sites around it hold this much probability
FINAL = {"right": "runs/v2_final_s*_*", "avg": "runs/final_avg_label_s*_*", "hump": "runs/final_hump_label_s*_*",
         "hump_top": "runs/final_hump_top_label_s*_*"}
FIRES_WITH = {"hump_top": "hump"}  # labels read at another label's moment
LABEL_TIMES = ("right", "avg", "hump")


def swing_profiles(args):
    """Symmetric site energy averaged over the swing (avg_window samples) ending at each run's label times."""
    theta0, times, p = args
    theta, omega = np.array(theta0, dtype=float), np.zeros_like(theta0, dtype=float)
    W, stride = p.avg_window, round(p.sample / p.dt)
    ks = {name: np.where(np.isnan(t), -1, np.round(t / p.sample)).astype(int) for name, t in times.items()}
    last = max(int(k.max()) for k in ks.values())
    window = np.zeros((W, *theta.shape))
    out = {name: np.full(theta.shape, np.nan) for name in times}
    for k in range(last + 1):
        window[k % W] = sim.site_energy(theta, omega, p.g, p.h, "sym")
        for name, kk in ks.items():
            rows = np.nonzero(kk == k)[0]
            if len(rows):
                out[name][rows] = window[:, rows].mean(axis=0) if k >= W - 1 else window[:k + 1, rows].mean(axis=0)
        for _ in range(stride if k < last else 0):
            theta, omega = sim.rk4_step(theta, omega, p.dt, p.g, p.h)
    return out


def second_hump_ratio(prof, min_apart=MIN_APART):
    """Tallest local maximum at least `min_apart` sites from the top pendulum, as a share of the top (0 if none)."""
    n_sites = prof.shape[1]
    top = prof.argmax(axis=1)
    far = metrics.ring_distance(np.arange(n_sites)[None, :], top[:, None], n_sites) >= min_apart
    peaks = (prof > np.roll(prof, 1, axis=1)) & (prof >= np.roll(prof, -1, axis=1)) & far
    return np.where(peaks, prof, 0.0).max(axis=1) / prof[np.arange(len(prof)), top]


def two_sites(probs, min_apart=MIN_APART):
    """The most likely site and the highest local maximum of the probability at least `min_apart` sites from it
    (-1 if there is none), with the probability held by the 3 sites centred on each."""
    n_sites = probs.shape[1]
    first = probs.argmax(axis=1)
    far = metrics.ring_distance(np.arange(n_sites)[None, :], first[:, None], n_sites) >= min_apart
    peaks = (probs > np.roll(probs, 1, axis=1)) & (probs >= np.roll(probs, -1, axis=1)) & far
    second = np.where(peaks.any(axis=1), np.where(peaks, probs, -1.0).argmax(axis=1), -1)
    near = probs + np.roll(probs, 1, axis=1) + np.roll(probs, -1, axis=1)
    rows = np.arange(len(probs))
    return first, second, near[rows, first], np.where(second >= 0, near[rows, second.clip(min=0)], 0.0)


def scores(first, second, mass2, truth, n_sites=100):
    d1 = metrics.ring_distance(first, truth, n_sites)
    d2 = np.where(second >= 0, metrics.ring_distance(second.clip(min=0), truth, n_sites), n_sites)
    # reference: a second site drawn at random from those at least MIN_APART sites from the first
    sites = np.arange(n_sites)[None, :]
    allowed = metrics.ring_distance(sites, first[:, None], n_sites) >= MIN_APART
    lucky = (allowed & (metrics.ring_distance(sites, truth[:, None], n_sites) <= 1)).sum(axis=1) / allowed.sum(axis=1)
    out = {"n": int(len(truth)), "top1_within1": float(np.mean(d1 <= 1)), "top1_within3": float(np.mean(d1 <= 3)),
           "top2_within1": float(np.mean(np.minimum(d1, d2) <= 1)), "second_within1": float(np.mean(d2 <= 1)),
           "top2_random_within1": float(np.mean(np.where(d1 <= 1, 1.0, lucky))),
           "has_second": float(np.mean(second >= 0)), "second_mass_median": float(np.median(mass2))}
    for m in MASSES:  # a second site only when the model gives it this much probability
        named = mass2 >= m
        out[f"mass{m:g}"] = {"named": float(np.mean(named)),
                             "within1": float(np.mean((d1 <= 1) | (named & (d2 <= 1))))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/top2.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    test = data.load_pool(ROOT / "data" / "pool_test.npz")
    p, chunk = sim.Params(), 1000
    times = {name: test[f"time_{name}"].astype(float) for name in LABEL_TIMES}
    jobs = [(test["theta0"][i:i + chunk], {n: t[i:i + chunk] for n, t in times.items()}, p)
            for i in range(0, len(test["theta0"]), chunk)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(swing_profiles, jobs))
    profiles = {name: np.concatenate([part[name] for part in parts]) for name in LABEL_TIMES}
    # the profiles reproduce the swing-averaged labels exactly: the window that fires is the one stored here
    for name, energy in (("avg", lambda e: e), ("hump", sim.hump_energy)):
        ok = test[f"site_{name}"] >= 0
        assert np.array_equal(energy(profiles[name][ok]).argmax(axis=1), test[f"site_{name}"][ok]), name
    ratio = {name: np.zeros(len(prof)) for name, prof in profiles.items()}
    for name, prof in profiles.items():
        ok = ~np.isnan(prof[:, 0])
        ratio[name][ok] = second_hump_ratio(prof[ok])
    rival = ratio["right"] >= RIVAL
    print(f"rival hump (>= {RIVAL:g} of the top, >= {MIN_APART} sites away) when v1's label fires: {rival.mean():.1%} "
          f"of {len(rival)} test runs; at the averaged label's time {np.mean(ratio['avg'] >= RIVAL):.1%}, "
          f"at the hump label's time {np.mean(ratio['hump'] >= RIVAL):.1%}", flush=True)

    runs = [str(r.relative_to(ROOT)) for pattern in (args.runs or FINAL.values())
            for r in sorted(ROOT.glob(pattern) if not Path(pattern).is_absolute() else [Path(pattern)])]
    result = {"min_apart": MIN_APART, "rival": RIVAL, "masses": list(MASSES),
              "rival_share": {name: float(np.mean(r >= RIVAL)) for name, r in ratio.items()}, "runs": {}}
    for run in runs:
        run_dir = ROOT / run
        cfg = json.loads((run_dir / "config.json").read_text())
        label = cfg.get("label", "right")
        probs, pred_time = outputs(run_dir, test)
        first, second, _, mass2 = two_sites(probs)
        truth = test[f"site_{label}"].astype(int)
        own = ratio[FIRES_WITH.get(label, label)] >= RIVAL
        keep = truth >= 0
        groups = {"all": keep, "rival": keep & rival, "no_rival": keep & ~rival,
                  "rival_own_time": keep & own, "no_rival_own_time": keep & ~own}
        true_time = test[f"time_{label}"].astype(float)
        result["runs"][run] = {"name": cfg["name"], "label": label, "time_median_rel_error": float(np.median(
            np.abs(pred_time[keep] - true_time[keep]) / true_time[keep])),
            **{g: scores(first[m], second[m], mass2[m], truth[m]) for g, m in groups.items()}}
        r = result["runs"][run]
        print(f"{run:48s} {label:8s}  ±1 top-1 {r['all']['top1_within1']:.1%} -> top-2 {r['all']['top2_within1']:.1%}   "
              f"rival runs {r['rival']['top1_within1']:.1%} -> {r['rival']['top2_within1']:.1%}   "
              f"no rival {r['no_rival']['top1_within1']:.1%} -> {r['no_rival']['top2_within1']:.1%}", flush=True)
    by_name = {}
    for run, r in result["runs"].items():
        by_name.setdefault(r["name"], []).append(r)
    result["mean"] = {}
    for name, rs in by_name.items():
        mean = {"time_median_rel_error": float(np.mean([r["time_median_rel_error"] for r in rs]))}
        for g in rs[0]:
            if not isinstance(rs[0][g], dict):
                continue
            mean[g] = {k: float(np.mean([r[g][k] for r in rs])) for k in rs[0][g] if not isinstance(rs[0][g][k], dict)}
            for m in MASSES:
                key = f"mass{m:g}"
                mean[g][key] = {k: float(np.mean([r[g][key][k] for r in rs])) for k in ("named", "within1")}
        result["mean"][name] = {"label": rs[0]["label"], "seeds": len(rs), **mean}
    (ROOT / args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
