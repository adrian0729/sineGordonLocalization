"""Run a single config from an ablation table (by phase, name and seed) and record it.

    python scripts/run_one.py phase2 early_fusion --seed 0 --threads 4
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(ablations.PHASES))
    parser.add_argument("name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = next(c for c in ablations.PHASES[args.phase]() if c["name"] == args.name and c["seed"] == args.seed)
    if experiment.config_id(cfg) in experiment.completed_ids():
        print("already recorded")
        return
    rec = experiment.run(cfg, workers=args.workers, threads=args.threads)
    t = rec["test"]
    print(f"{args.name} seed {args.seed}: mean {t['mae']:.3f}  ±1 {t['acc_1']:.2%}  ±3 {t['acc_3']:.2%}  "
          f">10 {t['over_10']:.2%}  ({rec['train_seconds']}s, {rec['params']} params)")


if __name__ == "__main__":
    main()
