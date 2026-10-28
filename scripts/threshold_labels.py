"""Localization labels for the threshold study (REPORT.md Sec. 10, "Which threshold?").

    python scripts/threshold_labels.py                                   # train, val, test at the stated offsets
    python scripts/threshold_labels.py --noise-scale 0.5 --tag half --n-train 60000

Each run starts from its pool's initial angles (optionally with the offsets about A rescaled) at rest
and is integrated to tau = 150. On the 0.5 sample grid it records:
  thrXX  first sample where one site's energy (spring booked to the right, as in v1) exceeds XX% of the
         total; thr10 is v1's label (site_right).
  symXX  the same with each spring split evenly, to measure how often the bond convention moves the site.
  sepK   first sample where one site's energy exceeds K times the separatrix energy E_sep = 2 (the
         on-site energy of a pendulum at the top), an absolute rather than a relative threshold.
  ott    first sample where a pendulum is past the top, |theta_n| > pi; the site is that pendulum.
Unlocalized runs get site -1 and time NaN. With --tmax, every label also gets a copy cut at that time
(e.g. 99.5 for the students' t_f = 100 grid), named <label>_tf100.

Writes data/thrpool_<tag>_<split>.npz holding A, theta0, meta and (at the stated offsets) the pool's own
labels, so a config can use the new labels through experiment's "pools" key; the shared pools are untouched.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, sim  # noqa: E402

SHARES = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15)  # percent of the total energy
SEPARATRIX = (1.5, 2, 3, 4, 6)              # multiples of E_sep
E_SEP = 2.0


def label_names():
    return ([f"thr{s:02d}" for s in SHARES] + [f"sym{s:02d}" for s in SHARES]
            + [f"sep{k:g}".replace(".", "p") for k in SEPARATRIX] + ["ott"])


def chunk(theta0):
    p = sim.Params()
    theta = np.array(theta0, dtype=float)
    omega = np.zeros_like(theta)
    total = sim.site_energy(theta, omega, p.g, p.h, "right").sum(1)
    rules = ([("right", s / 100 * total) for s in SHARES] + [("sym", s / 100 * total) for s in SHARES]
             + [("right", np.full(len(theta), k * E_SEP)) for k in SEPARATRIX])
    names = label_names()
    site = np.full((len(names), len(theta)), -1, np.int16)
    when = np.full((len(names), len(theta)), np.nan, np.float32)
    stride = round(p.sample / p.dt)
    for k in range(round(p.t_final / p.sample)):
        E = {b: sim.site_energy(theta, omega, p.g, p.h, b) for b in ("right", "sym")}
        top = {b: (E[b].max(1), E[b].argmax(1)) for b in E}
        for j, (bond, limit) in enumerate(rules):
            m, am = top[bond]
            hit = (site[j] < 0) & (m > limit)
            site[j][hit], when[j][hit] = am[hit], k * p.sample
        past = np.abs(theta)
        hit = (site[-1] < 0) & (past.max(1) > np.pi)
        site[-1][hit], when[-1][hit] = past[hit].argmax(1), k * p.sample
        if (site >= 0).all():
            break
        for _ in range(stride):
            theta, omega = sim.rk4_step(theta, omega, p.dt, p.g, p.h)
    return site, when


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--n-train", type=int, default=None, help="keep only the first runs of the training pool")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="rescale the offsets about A (0.5 = half)")
    parser.add_argument("--tmax", type=float, default=None, help="also write copies cut at this time (_tf100)")
    parser.add_argument("--tag", default="full")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=1000)
    args = parser.parse_args()

    names = label_names()
    for split in args.splits:
        start = time.time()
        pool = data.load_pool(ROOT / "data" / f"pool_{split}.npz", args.n_train if split == "train" else None)
        A = pool["A"]
        theta0 = A[:, None] + args.noise_scale * (pool["theta0"] - A[:, None])
        parts = [theta0[i:i + args.chunk] for i in range(0, len(theta0), args.chunk)]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(chunk, parts))
        site = np.concatenate([r[0] for r in res], 1)
        when = np.concatenate([r[1] for r in res], 1)
        out = {"A": A, "theta0": theta0}
        if args.noise_scale == 1.0:  # the pool's own labels still apply
            out.update({k: v for k, v in pool.items() if k.startswith(("site_", "time_"))})
        for j, name in enumerate(names):
            out[f"site_{name}"], out[f"time_{name}"] = site[j], when[j]
            if args.tmax is not None:
                late = ~(when[j] <= args.tmax)
                out[f"site_{name}_tf100"] = np.where(late, -1, site[j]).astype(np.int16)
                out[f"time_{name}_tf100"] = np.where(late, np.nan, when[j]).astype(np.float32)
        if args.noise_scale != 1.0:  # v1's rule (10%, spring to the right) at these offsets, which training also scores
            out["site_right"], out["time_right"] = out["site_thr10"], out["time_thr10"]
        meta = dict(pool["meta"], threshold_study={"labels": names, "shares_percent": list(SHARES),
                    "separatrix_multiples": list(SEPARATRIX), "E_sep": E_SEP, "noise_scale": args.noise_scale,
                    "tmax_copy": args.tmax, "seconds": round(time.time() - start),
                    "created": time.strftime("%Y-%m-%d %H:%M")})
        path = ROOT / "data" / f"thrpool_{args.tag}_{split}.npz"
        np.savez(path, meta=json.dumps(meta), **out)
        loc = {n: float((site[j] >= 0).mean()) for j, n in enumerate(names)}
        print(f"{split}: {len(theta0)} runs in {time.time() - start:.0f}s -> {path.name}; "
              f"localized by tau 150: thr15 {loc['thr15']:.4f}, sep6 {loc['sep6']:.4f}, ott {loc['ott']:.4f}", flush=True)
        if args.noise_scale == 1.0:
            same = np.array_equal(out["site_thr10"], pool["site_right"])
            print(f"  check: thr10 == site_right on all runs: {same}", flush=True)


if __name__ == "__main__":
    main()
