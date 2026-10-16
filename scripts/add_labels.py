"""Add labels that are not computed by default (sgloc.sim.EXTRA_LABELS, e.g. "hump") to existing data
pools, re-simulated from their stored initial angles (the initial conditions are not regenerated).

    python scripts/add_labels.py hump                          # pool_train, pool_val and pool_test
    python scripts/add_labels.py hump --pools pool_val --workers 4

Every array already in a pool is kept bit-identical, and the pool's meta records how and when the label
was added. Each file is written to a temporary file and then swapped in, so a reader sees the old pool
or the new one, never half of one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, sim  # noqa: E402

DEFINITION = {"hump": {"sites": sim.HUMP_SITES, "threshold": sim.HUMP_THRESHOLD},
              "hump_top": {"fires_with": "hump", "sites": 1, "threshold": sim.HUMP_THRESHOLD}}


def add(path, labels, workers):
    pool = data.load_pool(path)
    if "onsite" in pool:  # stored as float32, while the chains were simulated with the float64 map
        raise SystemExit(f"{path.name}: the impurity map is not stored exactly, so new labels would not match the runs")
    start = time.time()
    new = data.localize_parallel(pool["theta0"], sim.Params(**pool["meta"]["params"]), workers=workers, labels=labels,
                                 gamma=pool.get("gamma", 0.0))
    seconds = round(time.time() - start, 1)
    with np.load(path) as f:  # read again just before writing, so keys another process added meanwhile are kept
        arrays = {key: f[key] for key in f.files}
    meta = json.loads(str(arrays["meta"]))
    added = meta.setdefault("added_labels", {})
    for name in labels:
        added[name] = {**DEFINITION.get(name, {}), "numpy": np.__version__, "seconds": seconds,
                       "created": time.strftime("%Y-%m-%d %H:%M")}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{**arrays, **new, "meta": json.dumps(meta)})
    with np.load(tmp) as f:
        for key, value in arrays.items():
            if key != "meta":
                assert f[key].dtype == value.dtype and np.array_equal(f[key], value, equal_nan=value.dtype.kind == "f"), key
    os.replace(tmp, path)
    return new, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", nargs="+", choices=sim.EXTRA_LABELS)
    parser.add_argument("--pools", nargs="+", default=["pool_val", "pool_test", "pool_train"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    for name in args.pools:
        path = ROOT / "data" / f"{name}.npz"
        new, seconds = add(path, tuple(args.labels), args.workers)
        for label in args.labels:
            site, when = new[f"site_{label}"], new[f"time_{label}"]
            print(f"{name}: {label} for {len(site)} runs in {seconds} s; localized {np.mean(site >= 0):.2%}; "
                  f"median time {np.nanmedian(when):.1f}, mean {np.nanmean(when):.2f}", flush=True)


if __name__ == "__main__":
    main()
