"""Test pools for the robustness checks of Phase 6, written to data/robust_<name>.npz.

    python scripts/make_robustness_pools.py --workers 3

Each pool is simulated with its own "true" chain; models still observe it through the nominal
model (g = 0.75, h = 0.5) when they compute energies.
- g-10 / g+10 / h-10 / h+10: coupling or global strength off by 10%
- noise2x: twice the initial noise (A + 0.1 (U - 0.5))
- N80 / N150: a shorter or longer ring
- impurity1: one pendulum per run with a 1% weaker gravity term at a random site (paper Sec. 3.5)
- impurity_train / impurity_val / impurity_test: impurity strength U(0, 2%) at a random site,
  for training a model that is given a calibration map of the chain
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, sim  # noqa: E402

CONDITIONS = {
    "g-10": dict(size=5000, seed=301, params=sim.Params(g=0.675)),
    "g+10": dict(size=5000, seed=302, params=sim.Params(g=0.825)),
    "h-10": dict(size=5000, seed=303, params=sim.Params(h=0.45)),
    "h+10": dict(size=5000, seed=304, params=sim.Params(h=0.55)),
    "noise2x": dict(size=5000, seed=305, noise=0.1),
    "N80": dict(size=5000, seed=306, n_sites=80),
    "N150": dict(size=5000, seed=307, n_sites=150),
    "impurity1": dict(size=5000, seed=308, impurity=(0.01, 0.01)),
    "impurity_train": dict(size=60_000, seed=309, impurity=(0.0, 0.02)),
    "impurity_val": dict(size=10_000, seed=310, impurity=(0.0, 0.02)),
    "impurity_test": dict(size=10_000, seed=311, impurity=(0.0, 0.02)),
}


def make(name, spec, workers):
    params = spec.get("params", sim.Params())
    n_sites = spec.get("n_sites", 100)
    A, theta0 = data.initial_conditions(spec["size"], spec["seed"], n_sites=n_sites, noise=spec.get("noise", data.NOISE))
    onsite = None
    extra = {}
    if "impurity" in spec:
        rng = np.random.default_rng(spec["seed"] + 1000)
        site = rng.integers(0, n_sites, spec["size"])
        eps = rng.uniform(*spec["impurity"], spec["size"])
        onsite = np.ones((spec["size"], n_sites))
        onsite[np.arange(spec["size"]), site] -= eps
        extra = {"impurity_site": site.astype(np.int16), "impurity_eps": eps, "onsite": onsite.astype(np.float32)}
    start = time.time()
    labels = data.localize_parallel(theta0, params, chunk=500, workers=workers, labels=("right", "avg"), onsite=onsite)
    meta = {"name": name, "size": spec["size"], "seed": spec["seed"], "n_sites": n_sites,
            "noise": spec.get("noise", data.NOISE), "impurity": spec.get("impurity"), "params": asdict(params),
            "seconds": round(time.time() - start, 1), "created": time.strftime("%Y-%m-%d %H:%M")}
    np.savez(ROOT / "data" / f"robust_{name}.npz", A=A, theta0=theta0, meta=json.dumps(meta), **labels, **extra)
    print(f"{name}: {spec['size']} runs in {meta['seconds']} s, localized {np.mean(labels['site_right'] >= 0):.1%}, "
          f"mean time {np.nanmean(labels['time_right']):.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    for name, spec in CONDITIONS.items():
        if not (ROOT / "data" / f"robust_{name}.npz").exists():
            make(name, spec, args.workers)


if __name__ == "__main__":
    main()
