"""What puts one pendulum over a share of the total energy (paper Sec. 3.5; REPORT.md "What the sweep says about the
hump label"). Writes results/crossing_energy.json.

    python scripts/crossing_energy.py        # about five minutes on an idle laptop CPU

The threshold study's test pool (data/thrpool_full_test.npz, 20,000 runs) is integrated again from its initial
angles with the pools' own simulator, and at the sample where each first-crossing label fires (5%, 10% and 15% of
the total energy) the labelled pendulum's energy, Eq. (2) of the preprint (each pendulum owning the spring to its
right-hand neighbour), is split into its four parts:
- kinetic, theta_dot^2 / 2;
- on-site, 1 - cos theta;
- spring, g/2 (theta_{n+1} - theta_n)^2;
- global, h/2 (theta_n - theta_bar)^2, the twist of the global coupling, which grows with the square of a pendulum's
  angle from the ring average and so holds most of the energy of a pendulum that has turned over the top.
The same is done for the hump centre (the site of the hump label of Eq. (6)) at that moment. Also recorded:
- how many pendulums lie beyond the top, |theta| > pi;
- the share of the ring's energy held by the twist and by motion;
- whether the labelled pendulum is the first one to go over the top.
Runs are grouped by the phase of the uniform swing at the crossing, as in scripts/threshold_hump.py:
- near the bottom of the swing (0.375 <= phi < 0.625);
- near a turning point (phi >= 0.875 or phi < 0.125).
Within each group they are split by whether the label lies within +-1 of the hump centre or two or more sites from it
(on a flank). Medians over runs. For each group the energy profile across the hump is also kept: the median share of
the total energy held by the pendulums at offsets -4..+4 from the centre, and how far the labelled pendulum lies from
the centre with its sign (positive to the right, the side whose spring each pendulum owns). Since a median profile
need not look like any one run's, the shape is also checked run by run: the share of runs in which the centre holds
more than both pendulums two sites off, and in which it holds the most of the five pendulums from -2 to +2.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import floquet, sim  # noqa: E402

LABELS = ["thr05", "thr10", "thr15"]
PARTS = ["kinetic", "onsite", "spring", "global"]
OFFSETS = np.arange(-4, 5)
CHUNK = 4000


def ring_dist(a, b, n=100):
    d = np.abs(a - b) % n
    return np.minimum(d, n - d)


def ring_signed(a, b, n=100):
    d = (a - b) % n
    return np.where(d > n // 2, d - n, d)


def parts(theta, omega, sites, p):
    """The four parts of Eq. (2) for one site per run (right-hand spring)."""
    r = np.arange(len(sites))
    th = theta[r, sites]
    stretch = theta - np.roll(theta, -1, axis=-1)
    return np.stack([0.5 * omega[r, sites] ** 2, 1 - np.cos(th), 0.5 * p.g * stretch[r, sites] ** 2,
                     0.5 * p.h * (th - theta.mean(-1)) ** 2], 1)


def snapshots(t, p):
    """Per label, the energy parts and angles of the labelled pendulum and the hump centre at the crossing sample."""
    stride = round(p.sample / p.dt)
    rec = {lab: {k: [] for k in ("idx", "lab", "ctr", "profile", "abs_theta", "beyond_top", "ring_global",
                                 "ring_kinetic")}
           for lab in LABELS}
    for lo in range(0, len(t["theta0"]), CHUNK):
        sl = slice(lo, lo + CHUNK)
        th = t["theta0"][sl].astype(float)
        om = np.zeros_like(th)
        ks = {}
        for lab in LABELS:
            tau = t[f"time_{lab}"][sl]
            ks[lab] = np.where(np.isfinite(tau), np.round(np.nan_to_num(tau, nan=-1) / p.sample), -1).astype(int)
        k_end = max(int(k.max()) for k in ks.values())
        for k in range(k_end + 1):
            for lab in LABELS:
                hit = np.flatnonzero(ks[lab] == k)
                if not len(hit):
                    continue
                y, c = t[f"site_{lab}"][sl][hit], t["site_hump"][sl][hit]
                th_h, om_h = th[hit], om[hit]
                total = sim.site_energy(th_h, om_h, p.g, p.h).sum(-1)[:, None]
                r = rec[lab]
                r["idx"].append(hit + lo)
                r["lab"].append(parts(th_h, om_h, y, p) / total)
                r["ctr"].append(parts(th_h, om_h, np.where(c >= 0, c, 0), p) / total)
                around = (np.where(c >= 0, c, 0)[:, None] + OFFSETS) % th.shape[1]
                e_site = sim.site_energy(th_h, om_h, p.g, p.h)
                r["profile"].append(np.take_along_axis(e_site, around, 1) / total)
                r["abs_theta"].append(np.abs(th_h[np.arange(len(hit)), y]))
                r["beyond_top"].append((np.abs(th_h) > np.pi).sum(-1))
                r["ring_global"].append((0.5 * p.h * (th_h - th_h.mean(-1, keepdims=True)) ** 2).sum(-1) / total[:, 0])
                r["ring_kinetic"].append((0.5 * om_h ** 2).sum(-1) / total[:, 0])
            for _ in range(stride):
                th, om = sim.rk4_step(th, om, p.dt, p.g, p.h)
        print(f"runs {lo}-{min(lo + CHUNK, len(t['theta0']))} done", flush=True)
    return {lab: {k: np.concatenate(v) for k, v in r.items()} for lab, r in rec.items()}


def summary(r, sel):
    lab, ctr, prof = r["lab"][sel], r["ctr"][sel], r["profile"][sel]
    c, m2, p2 = prof[:, 4], prof[:, 2], prof[:, 6]  # offsets 0, -2 and +2
    return {"n": int(sel.sum()),
            "label_share_of_total": float(np.median(lab.sum(1))),
            "label_parts_of_its_energy": {k: float(np.median(lab[:, i] / lab.sum(1))) for i, k in enumerate(PARTS)},
            "label_abs_theta": float(np.median(r["abs_theta"][sel])),
            "centre_share_of_total": float(np.median(ctr.sum(1))),
            "centre_parts_of_its_energy": {k: float(np.median(ctr[:, i] / ctr.sum(1))) for i, k in enumerate(PARTS)},
            "pendulums_beyond_top": float(np.median(r["beyond_top"][sel])),
            "ring_share_global": float(np.median(r["ring_global"][sel])),
            "ring_share_kinetic": float(np.median(r["ring_kinetic"][sel])),
            "profile_around_centre": {str(o): float(v) for o, v in zip(OFFSETS, np.median(prof, 0))},
            "runs_centre_above_both_pm2": float(((c > m2) & (c > p2)).mean()),
            "runs_centre_is_max_of_5": float((np.argmax(prof[:, 2:7], 1) == 2).mean())}


def main():
    t = np.load(ROOT / "data" / "thrpool_full_test.npz", allow_pickle=True)
    p = sim.Params()
    snaps = snapshots(t, p)
    out = {"pool": "thrpool_full_test", "runs": int(len(t["theta0"])), "labels": {}}
    for lab, r in snaps.items():
        idx = r["idx"]
        y, c = t[f"site_{lab}"][idx], t["site_hump"][idx]
        ok = c >= 0
        lab_tau = t[f"time_{lab}"][idx]
        half = floquet.period(t["A"][idx]) / 2
        phi = np.mod(lab_tau, half) / half
        d = ring_dist(y, np.where(ok, c, 0))
        sd = ring_signed(y, np.where(ok, c, 0))
        groups = {"all": ok, "bottom": ok & (phi >= 0.375) & (phi < 0.625),
                  "turning": ok & ((phi >= 0.875) | (phi < 0.125))}
        res = {"check_label_share_min": float(r["lab"].sum(1).min())}
        for name, g in groups.items():
            res[name] = {"share_of_runs": float(g.sum() / ok.sum()), "pm1_hump": float((d[g] <= 1).mean()),
                         "first_over_top": float((y[g] == t["site_ott"][idx][g]).mean()),
                         "offset_from_centre": {str(k): float((d[g] == k).mean()) for k in range(4)},
                         "signed_offset_from_centre": {str(k): float((sd[g] == k).mean()) for k in range(-3, 4)},
                         "all": summary(r, g), "on_centre": summary(r, g & (d <= 1)),
                         "flank": summary(r, g & (d >= 2)) if (g & (d >= 2)).sum() >= 20 else None}
        out["labels"][lab] = res
        b, tu = res["bottom"], res["turning"]
        print(f"{lab}: bottom {b['share_of_runs']:.3f} (pm1 {b['pm1_hump']:.3f}, global {b['all']['label_parts_of_its_energy']['global']:.2f}, "
              f"beyond top {b['all']['pendulums_beyond_top']:.0f}); turning {tu['share_of_runs']:.3f} (pm1 {tu['pm1_hump']:.3f}, "
              f"kinetic {tu['all']['label_parts_of_its_energy']['kinetic']:.2f}, beyond top {tu['all']['pendulums_beyond_top']:.0f})")
    (ROOT / "results" / "crossing_energy.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
