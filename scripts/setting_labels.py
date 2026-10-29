"""Localization labels for the settings study (hump-label knobs, amplitude, ring size N, offset size).

    python scripts/setting_labels.py --tag main --from-pool --n-train 60000                       # the main pools' runs
    python scripts/setting_labels.py --tag off0p5 --from-pool --offset-scale 0.5 --n-train 60000  # offsets about A halved
    python scripts/setting_labels.py --tag amp1p75 --amplitude 1.75 --sizes 60000 5000 10000 --seed 1750
    python scripts/setting_labels.py --tag ampwide --a-range 1.60 1.95 --sizes 60000 5000 10000 --seed 4000
    python scripts/setting_labels.py --tag n200 --n-sites 200 --sizes 60000 5000 10000 --seed 5200

Each run (g = 0.75, h = 0.5, dt = 0.05, samples every 0.5, released from rest) is integrated until every
label has fired or tau = --tmax (default 300) and gets these labels, each stored as site_<name> (int16, -1 if
never) and time_<name> (float32, NaN if never). E_tot is the conserved total energy.
  right        v1's label: first sample where one site's energy (spring booked to the right) exceeds 10% of E_tot.
  thr04        the same at 4%.
  ott          first sample where a pendulum is past the top, |theta_n| > pi; the site is that pendulum.
  hump_kK_sS   threshold-timed hump, K in (3, 5, 7), S in (10, 15, 20): the symmetric site energy averaged over
               the last 16 samples, summed over K neighbouring pendulums, first exceeds S% of E_tot; the site is
               the centre of the hump, the time the end of the window. hump_k3_s15 is sim.localize's "hump"
               (also stored as site_hump / time_hump).
  eh_kK        event-timed hump, K in (1, 3, 5, 7): the same K-pendulum sum of the symmetric site energy averaged
               over the 16 samples centred on the ott sample k0 (k0-8 .. k0+7; from sample 0 if k0 < 8); the time
               is that of ott. A run that crosses the top within 7 samples of tmax gets -1 / NaN.
Diagnostics rival_eh3 and rival_hump (float32, NaN if the label did not fire): for the K = 3 profile H used by
eh_k3 (resp. by hump_k3_s15 at its firing sample), the highest local maximum of H at ring distance >= 5 from the
main peak, over the main peak (0 if there is none).

Writes data/setpool_<tag>_<split>.npz holding A, theta0, the labels above and a json meta. With --from-pool and
--offset-scale 1 the right and hump labels are checked against the pool's own (cut at the pool's t_final).
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

GRID_K = (3, 5, 7)
GRID_S = (10, 15, 20)
EVENT_K = (1, 3, 5, 7)
LAG = 7        # eh needs samples up to k0 + 7
MIN_RIVAL = 5  # ring distance from the main peak at which a second peak counts as a rival

DEFINITIONS = {
    "right": "first sample with one site's energy (spring to the right) > 10% of the total",
    "thr04": "the same at 4%",
    "ott": "first sample with max |theta_n| > pi; site = that pendulum",
    "hump_kK_sS": "16-sample average of the sym site energy, summed over K neighbours, first > S% of the total; "
                  "site = centre, time = end of the window; hump_k3_s15 = sim.localize's hump",
    "eh_kK": "the same K-sum of the sym energy averaged over samples k0-8..k0+7 around the ott sample k0; "
             "time = ott time",
    "rival_eh3 / rival_hump": "highest local maximum of the K=3 profile at ring distance >= 5 from its peak, "
                              "over the peak (0 if none)",
}


def hump_names():
    return [f"hump_k{k}_s{s}" for k in GRID_K for s in GRID_S]


def label_names():
    return ["right", "thr04", "ott"] + hump_names() + [f"eh_k{k}" for k in EVENT_K]


def rival(H):
    """Highest local maximum of each profile (rows of H, periodic) at ring distance >= 5 from the main peak,
    over the main peak; 0 if the profile has no such maximum."""
    n = H.shape[1]
    peak = H.argmax(1)
    top = H[np.arange(len(H)), peak]
    local = (H >= np.roll(H, 1, axis=1)) & (H >= np.roll(H, -1, axis=1))
    gap = np.abs(np.arange(n)[None, :] - peak[:, None])
    far = np.minimum(gap, n - gap) >= MIN_RIVAL
    best = np.where(local & far, H, -np.inf).max(1)
    return np.where(np.isfinite(best), best / top, 0.0)


def chunk(args):
    theta0, tmax, *rest = args
    stop_at_event = bool(rest and rest[0])
    p = sim.Params()
    theta = np.array(theta0, dtype=float)
    omega = np.zeros_like(theta)
    runs, n_sites = theta.shape
    W = p.avg_window
    total = sim.site_energy(theta, omega, p.g, p.h, "sym").sum(axis=1)
    n_samples = int(np.ceil(tmax / p.sample - 1e-9))  # samples k with tau = 0.5 k < tmax
    stride = round(p.sample / p.dt)
    names = label_names()
    site = {n: np.full(runs, -1, np.int16) for n in names}
    when = {n: np.full(runs, np.nan, np.float32) for n in names}
    riv = {n: np.full(runs, np.nan, np.float32) for n in ("rival_eh3", "rival_hump")}
    ott_k = np.full(runs, -1, int)
    buf = np.zeros((W, runs, n_sites))  # the last 16 "sym" site-energy samples

    def fire(name, hit, which, k):
        site[name][hit], when[name][hit] = which[hit], k * p.sample

    for k in range(n_samples):
        Er = sim.site_energy(theta, omega, p.g, p.h, "right")
        Es = sim.site_energy(theta, omega, p.g, p.h, "sym")
        buf[k % W] = Es
        for name, share in (("right", 0.10), ("thr04", 0.04)):
            hit = (site[name] < 0) & (Er.max(1) > share * total)
            fire(name, hit, Er.argmax(1), k)
        past = np.abs(theta)
        hit = (ott_k < 0) & (past.max(1) > np.pi)
        ott_k[hit] = k
        fire("ott", hit, past.argmax(1), k)
        if k >= W - 1:
            avg = buf.mean(axis=0)
            for K in GRID_K:
                H = sim.hump_energy(avg, K)
                top, where = H.max(1), H.argmax(1)
                for S in GRID_S:
                    name = f"hump_k{K}_s{S}"
                    hit = (site[name] < 0) & (top > S / 100 * total)
                    fire(name, hit, where, k)
                    if K == 3 and S == 15 and hit.any():
                        riv["rival_hump"][hit] = rival(H[hit])
        idx = np.nonzero(ott_k == k - LAG)[0]  # the window k0-8 .. k0+7 is now complete
        if len(idx):
            avg = buf[:, idx].sum(axis=0) / min(W, k + 1)  # fewer than 16 samples only when k0 < 8
            for K in EVENT_K:
                H = sim.hump_energy(avg, K)
                site[f"eh_k{K}"][idx] = H.argmax(1)
                when[f"eh_k{K}"][idx] = (k - LAG) * p.sample
                if K == 3:
                    riv["rival_eh3"][idx] = rival(H)
        events_done = ((ott_k >= 0) & ((ott_k + LAG <= k) | (ott_k + LAG >= n_samples))).all()
        if events_done and (stop_at_event or all((site[n] >= 0).all() for n in names if not n.startswith("eh_"))):
            break
        if k < n_samples - 1:
            for _ in range(stride):
                theta, omega = sim.rk4_step(theta, omega, p.dt, p.g, p.h)

    site["hump"], when["hump"] = site["hump_k3_s15"], when["hump_k3_s15"]
    return site, when, riv


def run_split(theta0, tmax, workers, size, stop_at_event=False):
    parts = [(theta0[i:i + size], tmax, stop_at_event) for i in range(0, len(theta0), size)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(chunk, parts))
    out = {}
    for j, kind in enumerate(("site", "time")):
        for name in res[0][j]:
            out[f"{kind}_{name}"] = np.concatenate([r[j][name] for r in res])
    for name in res[0][2]:
        out[name] = np.concatenate([r[2][name] for r in res])
    return out


def cut(site, time_, t_final):
    """Labels as a search over tau < t_final would find them (the pools use t_final = 150)."""
    late = ~(time_ < t_final)
    return np.where(late, -1, site).astype(np.int16), np.where(late, np.nan, time_).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--from-pool", action="store_true", help="use data/pool_<split>.npz initial conditions")
    parser.add_argument("--n-train", type=int, default=None, help="with --from-pool, keep the first runs of train")
    parser.add_argument("--offset-scale", type=float, default=1.0, help="rescale the offsets about A (--from-pool)")
    parser.add_argument("--amplitude", type=float, default=None, help="fixed amplitude A for every new run")
    parser.add_argument("--a-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--n-sites", type=int, default=100)
    parser.add_argument("--offset", type=float, default=data.NOISE, help="full width of the offsets")
    parser.add_argument("--sizes", type=int, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=None, help="split seeds are seed*10 + 1/2/3")
    parser.add_argument("--tmax", type=float, default=300.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--stop-at-event", action="store_true",
                        help="stop each chunk once every run's ott and eh labels are done; share-of-total labels "
                             "that have not fired by then stay unfired (for long rings, where 10%% of the total "
                             "energy is many times one pendulum's share and is rarely reached)")
    args = parser.parse_args()
    if args.amplitude is not None and args.a_range is not None:
        parser.error("give --amplitude or --a-range, not both")
    if args.from_pool and (args.amplitude is not None or args.a_range is not None):
        parser.error("--from-pool takes its amplitudes from the pool")
    if not args.from_pool and (args.sizes is None or args.seed is None):
        parser.error("new draws need --sizes and --seed (or use --from-pool)")
    a_range = (args.amplitude, args.amplitude) if args.amplitude is not None else tuple(args.a_range or data.A_RANGE)

    for split in args.splits:
        i_split = ("train", "val", "test").index(split)
        start = time.time()
        meta = {"tag": args.tag, "split": split, "tmax": args.tmax, "labels": DEFINITIONS,
                "script": "scripts/setting_labels.py"}
        pool = None
        if args.from_pool:
            pool = data.load_pool(ROOT / "data" / f"pool_{split}.npz", args.n_train if split == "train" else None)
            A = pool["A"]
            if args.offset_scale == 1.0:
                theta0 = pool["theta0"]
            else:
                theta0 = A[:, None] + args.offset_scale * (pool["theta0"] - A[:, None])
            pm = pool["meta"]
            meta.update(source_pool=f"data/pool_{split}.npz", pool_seed=pm.get("seed"),
                        a_range=pm.get("a_range"), offset_width=pm.get("noise"), offset_scale=args.offset_scale)
        else:
            size = args.sizes[i_split]
            if size == 0:
                print(f"{split}: size 0, skipped", flush=True)
                continue
            seed = args.seed * 10 + i_split + 1
            A, theta0 = data.initial_conditions(size, seed, args.n_sites, a_range, args.offset)
            meta.update(seed=seed, base_seed=args.seed, a_range=list(a_range), offset_width=args.offset,
                        offset_scale=1.0)
            if args.amplitude is not None:
                meta["amplitude"] = args.amplitude
        meta.update(n_runs=len(theta0), n_sites=theta0.shape[1])

        out = run_split(theta0, args.tmax, args.workers, args.chunk, args.stop_at_event)
        meta["stop_at_event"] = args.stop_at_event
        meta.update(seconds=round(time.time() - start), created=time.strftime("%Y-%m-%d %H:%M"))
        path = ROOT / "data" / f"setpool_{args.tag}_{split}.npz"
        np.savez(path, A=A, theta0=theta0, meta=json.dumps(meta), **out)

        frac = {n: float((out[f"site_{n}"] >= 0).mean()) for n in ("right", "thr04", "ott", "hump", "hump_k7_s20", "eh_k3")}
        med = float(np.nanmedian(out["time_ott"])) if np.isfinite(out["time_ott"]).any() else float("nan")
        print(f"{split}: {len(theta0)} runs in {time.time() - start:.0f}s -> {path.name}; localized: "
              + ", ".join(f"{n} {v:.4f}" for n, v in frac.items()) + f"; median ott time {med:.1f}", flush=True)
        if pool is not None and args.offset_scale == 1.0:
            t_final = pool["meta"].get("params", {}).get("t_final", sim.Params().t_final)
            ok = True
            for name in ("right", "hump"):
                s, t = cut(out[f"site_{name}"], out[f"time_{name}"], t_final)
                same = np.array_equal(s, pool[f"site_{name}"]) and np.array_equal(t, pool[f"time_{name}"], equal_nan=True)
                print(f"  check: {name} == pool's site_{name}/time_{name} on all {len(s)} runs "
                      f"(ours cut at tau < {t_final:g}): {same}", flush=True)
                ok &= same
            assert ok, "labels differ from the pool's stored labels"


if __name__ == "__main__":
    main()
