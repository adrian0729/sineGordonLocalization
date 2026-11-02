"""What the threshold study means for the hump label (paper Secs. 3.5 and 3.6; REPORT.md "Which threshold?"):
training-free checks on the saved pools and on the threshold study's saved predictions. Writes
results/threshold_hump.json.

    python scripts/threshold_hump.py

1. Cross-scores. The v2 runs of the threshold study (runs/<label>_s<seed>_*/pred_test.npy) scored against their
   own first-crossing label, which reproduces results/threshold.json, and against the hump centre of Eq. (6).
2. Swing phase. For each first-crossing label, the phase of the uniform swing at the labelled sample,
   phi = (tau mod T/2) / (T/2), with T the period of the swing at the run's amplitude (0 at a turning point, 0.5 at
   the bottom of the swing), and the share of labels within +-1 of the hump centre near the bottom
   (0.375 <= phi < 0.625) and near a turning point (phi >= 0.875 or phi < 0.125), with where the label lies
   relative to the centre in each (label minus centre, around the ring; positive is to the right). Also for the
   10% label with each spring split evenly (sym10), and for the swing-averaged label (avg).
3. Timing on the main test pool: how the hump label of Eq. (6) follows the first pendulum past the top, run by run,
   and how the site of that pendulum agrees with the hump label and with the event-timed label (eh_k3).
4. Rings of 50, 100, 200 and 400 pendulums (setpool test pools): the growing modes and sigma1 at A = 1.75, the
   continuum cut-off wavelength of the growing band, the clock (median over runs of tau_ott * sigma1(A, N)), and for
   the 4% and 10% first-crossing labels and the hump label of Eq. (6) the share of runs labelled, the median lag
   after the first pendulum past the top and the agreement with the event-timed label. The N = 400 pools stop each
   block of 1,000 runs once its event-timed labels are complete, so labels that come later count as not labelled;
   for those pools, also when the integration ended, per run, after its first pendulum past the top.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import floquet  # noqa: E402

DATA = ROOT / "data"
SHARES = ["thr03", "thr04", "thr05", "thr06", "thr07", "thr08", "thr09", "thr10", "thr12", "thr15"]
RINGS = [("n50", 50), ("main", 100), ("n200", 200), ("n400", 400)]


def ring_dist(a, b, n):
    d = np.abs(np.asarray(a, dtype=int) - np.asarray(b, dtype=int)) % n
    return np.minimum(d, n - d)


def ring_signed(a, b, n):
    """a - b around the ring, in (-n/2, n/2]; positive is to the right (higher index)."""
    d = (np.asarray(a, dtype=int) - np.asarray(b, dtype=int)) % n
    return np.where(d > n // 2, d - n, d)


def agreement(site, ref, n, mask):
    d = ring_dist(site[mask], ref[mask], n)
    return {"n": int(mask.sum()), "exact": float((d == 0).mean()), "pm1": float((d <= 1).mean()),
            "pm3": float((d <= 3).mean()), "mean_distance": float(d.mean())}


def cross_scores(t, threshold):
    """v2 of the threshold study, scored on its own label and on the hump centre (same runs)."""
    out = {}
    for lab in SHARES + ["ott"]:
        dirs = sorted(glob.glob(str(ROOT / "runs" / f"{lab}_s[0-9]_*")))
        y, hump = t[f"site_{lab}"], t["site_hump"]
        mask = (y >= 0) & (hump >= 0)
        own, on_hump = [], []
        for d in dirs:
            pred = np.load(Path(d) / "pred_test.npy")
            assert pred.shape == y.shape, (d, pred.shape)
            own.append(agreement(pred, y, 100, y >= 0)["pm1"])
            on_hump.append(agreement(pred, hump, 100, mask)["pm1"])
        ref = threshold["v2"][lab]["acc_1"]
        assert abs(np.mean(own) - ref) < 1e-6, (lab, np.mean(own), ref)  # the same runs as results/threshold.json
        out[lab] = {"seeds": len(dirs), "own_pm1": float(np.mean(own)), "hump_pm1": float(np.mean(on_hump)),
                    "hump_pm1_by_seed": on_hump}
    return out


def share(ok, sel):
    """Mean of ok over sel, or None when sel is empty."""
    return float(ok[sel].mean()) if sel.any() else None


def offsets(d, sel):
    """Where the label lies relative to the hump centre, over the runs in sel."""
    out = {f"{k:+d}" if k else "0": share(d == k, sel) for k in (-2, -1, 0, 1, 2)}
    out["3 or more"] = share(np.abs(d) >= 3, sel)
    return out


def swing_phase(t):
    """Agreement of each label with the hump centre against the phase of the swing at which it fires."""
    period = floquet.period(t["A"])
    out = {}
    for lab in SHARES + ["sym10", "avg", "ott"]:
        y, tau = t[f"site_{lab}"], t[f"time_{lab}"]
        m = (y >= 0) & (t["site_hump"] >= 0)
        half = period[m] / 2
        phi = np.mod(tau[m], half) / half
        ok = ring_dist(y[m], t["site_hump"][m], 100) <= 1
        d = ring_signed(y[m], t["site_hump"][m], 100)
        bottom = (phi >= 0.375) & (phi < 0.625)
        turning = (phi >= 0.875) | (phi < 0.125)
        eighth = np.clip(np.floor(phi * 8).astype(int), 0, 7)
        out[lab] = {
            "pm1_hump": float(ok.mean()),
            "bottom": {"share_of_runs": float(bottom.mean()), "pm1_hump": share(ok, bottom),
                       "offset_from_centre": offsets(d, bottom)},
            "turning": {"share_of_runs": float(turning.mean()), "pm1_hump": share(ok, turning),
                        "offset_from_centre": offsets(d, turning)},
            "by_eighth": {"share_of_runs": [float((eighth == b).mean()) for b in range(8)],
                          "pm1_hump": [share(ok, eighth == b) for b in range(8)]},
        }
    return out


def timing(t, s):
    """The hump label of Eq. (6) against the first pendulum past the top, on the main test pool."""
    assert np.allclose(t["theta0"], s["theta0"]), "thrpool_full and setpool_main must hold the same runs"
    assert (t["site_hump"] == s["site_hump_k3_s15"]).all(), "site_hump is the label of Eq. (6)"
    assert (s["time_eh_k3"] == s["time_ott"]).all(), "the event-timed label is time-stamped at the event"
    lag = s["time_hump_k3_s15"] - s["time_ott"]
    q = np.percentile(lag, [0, 1, 50, 90, 99, 100])
    every = np.ones(len(lag), dtype=bool)
    return {
        "n": int(len(lag)),
        "eq6_strictly_after_ott": float((lag > 0).mean()),
        "eq6_lag_after_ott": dict(zip(["min", "p1", "median", "p90", "p99", "max"], map(float, q))),
        "median_time_ott": float(np.median(s["time_ott"])),
        "median_time_eq6": float(np.median(s["time_hump_k3_s15"])),
        "ott_vs_eq6": agreement(s["site_ott"], s["site_hump_k3_s15"], 100, every),
        "ott_vs_eh_k3": agreement(s["site_ott"], s["site_eh_k3"], 100, every),
        "eh_k3_vs_eq6": agreement(s["site_eh_k3"], s["site_hump_k3_s15"], 100, every),
    }


def cutoff_wavelength(A=1.75, n=100):
    """Wavelength (in pendulums) at which the growth rate of the band falls to zero, by bisection in the mode
    number treated as continuous (the rate depends on the mode only through sin(pi m / N))."""
    lo, hi = 1.0, n / 2
    assert floquet.growth_rates(A, [lo], n_sites=n)[0] > 1e-9 and floquet.growth_rates(A, [hi], n_sites=n)[0] < 1e-9
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if floquet.growth_rates(A, [mid], n_sites=n)[0] > 1e-9 else (lo, mid)
    return n / hi


def rings():
    out = {"cutoff_wavelength_A175": cutoff_wavelength()}
    for tag, n_expected in RINGS:
        s = np.load(DATA / f"setpool_{tag}_test.npz", allow_pickle=True)
        n = s["theta0"].shape[1]
        assert n == n_expected, (tag, n)
        modes = np.arange(1, n // 2 + 1)
        r = floquet.growth_rates(1.75, modes, n_sites=n)
        growing = int((r > 1e-6).sum())
        # sigma1(A) on a grid over the pool's amplitudes; only the longest waves can grow there
        few = modes[: max(4, int(np.ceil(0.12 * n)))]
        grid = np.linspace(s["A"].min(), s["A"].max(), 41)
        rates = np.array([floquet.growth_rates(a, few, n_sites=n) for a in grid])
        assert (rates.argmax(axis=1) < len(few) - 1).all(), "the fastest mode must lie inside the modes scanned"
        sigma1 = np.interp(s["A"], grid, rates.max(axis=1))
        entry = {
            "N": n, "runs": int(len(s["A"])),
            "growing_modes_A175": growing, "sigma1_A175": float(r.max()), "fastest_mode_A175": int(modes[r.argmax()]),
            "median_time_ott": float(np.median(s["time_ott"])),
            "efoldings_at_ott_median": float(np.median(s["time_ott"] * sigma1)),
            "equipartition_multiple_of_10pct": 0.1 * n,
            "labels": {},
        }
        if json.loads(str(s["meta"])).get("stop_at_event"):
            # setting_labels.py --stop-at-event integrates each block of 1,000 runs only until every run's event-timed
            # window (7 samples, 3.5 time units, past its first pendulum over the top) is complete
            t_ott = s["time_ott"]
            end = np.concatenate([np.full(len(b), b.max() + 3.5) for b in np.split(t_ott, np.arange(1000, len(t_ott), 1000))])
            for key in s.files:
                if key.startswith("time_"):
                    assert not (s[key] > end + 1e-6).any(), f"{tag}: {key} fires after the reconstructed end"
            entry["integration_end"] = {"tau_range": [float(end.min()), float(end.max())],
                                        "after_ott_median": float(np.median(end - t_ott))}
        ref = s["site_eh_k3"]
        for lab, key in [("thr04", "thr04"), ("thr10", "right"), ("eq6", "hump_k3_s15")]:
            y, tau = s[f"site_{key}"], s[f"time_{key}"]
            fired = y >= 0
            entry["labels"][lab] = {
                "share_labelled": float(fired.mean()),
                "median_lag_after_ott": float(np.median(tau[fired] - s["time_ott"][fired])) if fired.any() else None,
                "vs_eh_k3": agreement(y, ref, n, fired) if fired.any() else None,
            }
        out[tag] = entry
    return out


def main():
    threshold = json.loads((ROOT / "results" / "threshold.json").read_text())
    t = np.load(DATA / "thrpool_full_test.npz", allow_pickle=True)
    s = np.load(DATA / "setpool_main_test.npz", allow_pickle=True)
    out = {
        "cross_scores": cross_scores(t, threshold),
        "swing_phase": swing_phase(t),
        "timing": timing(t, s),
        "rings": rings(),
    }
    path = ROOT / "results" / "threshold_hump.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path.relative_to(ROOT))

    print("\nv2 of the threshold study: within +-1 of its own label / of the hump centre (test)")
    for lab, c in out["cross_scores"].items():
        print(f"  {lab:6s} {100 * c['own_pm1']:6.2f} {100 * c['hump_pm1']:6.2f}")
    print("\nlabel within +-1 of the hump centre: all / near the bottom of the swing / near a turning point")
    def pct(x):
        return "     -" if x is None else f"{100 * x:6.1f}"

    for lab, p in out["swing_phase"].items():
        print(f"  {lab:6s} {pct(p['pm1_hump'])} {pct(p['bottom']['pm1_hump'])} "
              f"({100 * p['bottom']['share_of_runs']:4.1f}% of runs) {pct(p['turning']['pm1_hump'])} "
              f"({100 * p['turning']['share_of_runs']:4.1f}% of runs)")
    for lab in ("thr10", "sym10"):
        print(f"  {lab} near a turning point, label minus centre: {out['swing_phase'][lab]['turning']['offset_from_centre']}")
    tm = out["timing"]
    print(f"\nEq. (6) after the event: strictly after in {100 * tm['eq6_strictly_after_ott']:.2f}% of runs; lag "
          f"{tm['eq6_lag_after_ott']}; ott vs Eq. 6 +-1 {100 * tm['ott_vs_eq6']['pm1']:.2f}%, ott vs eh_k3 +-1 "
          f"{100 * tm['ott_vs_eh_k3']['pm1']:.2f}%")
    rg = out["rings"]
    print(f"\ncut-off wavelength at A = 1.75: {rg['cutoff_wavelength_A175']:.2f} pendulums")
    for tag, _ in RINGS:
        e = rg[tag]
        labs = "; ".join(f"{k}: labelled {100 * v['share_labelled']:.1f}%, lag {v['median_lag_after_ott']}, "
                         f"+-1 eh_k3 {100 * v['vs_eh_k3']['pm1']:.1f}%" for k, v in e["labels"].items() if v["vs_eh_k3"])
        print(f"  N={e['N']:3d}: {e['growing_modes_A175']} growing, sigma1 {e['sigma1_A175']:.4f}, t_ott "
              f"{e['median_time_ott']}, e-foldings {e['efoldings_at_ott_median']:.2f} | {labs}")
        if "integration_end" in e:
            print(f"         integrated to tau {e['integration_end']['tau_range']}, a median of "
                  f"{e['integration_end']['after_ott_median']} after the first pendulum past the top")


if __name__ == "__main__":
    main()
