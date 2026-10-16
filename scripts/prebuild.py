"""Fill the trajectory cache for every config of a phase (architecture-independent), so a later
sweep starts training immediately.

    python scripts/prebuild.py phase3 --workers 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(ablations.PHASES))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    start, pools = time.time(), {}
    for cfg in ablations.PHASES[args.phase]():
        key = (cfg["n_train"], json.dumps(cfg.get("pools"), sort_keys=True))
        if key not in pools:
            pools[key] = experiment.load_pools(cfg["n_train"], cfg.get("pools"))
        experiment.prebuild(cfg, pools[key], args.workers)
    print(f"{args.phase}: trajectory cache ready in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
