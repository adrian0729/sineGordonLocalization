"""Generate the data pools (initial conditions + labels) in data/.

    python scripts/make_pools.py            # skips pools that already exist
    python scripts/make_pools.py --force    # regenerate everything
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import data, sim  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    (ROOT / "data").mkdir(exist_ok=True)
    for name in ("val", "test", "train"):
        path = ROOT / "data" / f"pool_{name}.npz"
        if path.exists() and not args.force:
            print(f"{name}: {path.name} exists, skipping")
            continue
        meta = data.make_pool(name, path, sim.Params(), workers=args.workers)
        pool = data.load_pool(path)
        print(f"{name}: {meta['size']} runs in {meta['seconds']} s; "
              f"localized {np.mean(pool['site_right'] >= 0):.2%}; "
              f"mean localization time {np.nanmean(pool['time_right']):.1f}", flush=True)


if __name__ == "__main__":
    main()
