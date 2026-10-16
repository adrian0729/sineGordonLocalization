"""Run one config (a JSON file written by scripts/sweep.py) in its own process, so all memory,
including the GPU allocator's cache, is returned when it finishes.

    python scripts/run_config.py <config.json> --device mps
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rec = experiment.run(cfg, workers=1, threads=args.threads, device=args.device)
    print("RESULT " + json.dumps({"test": rec["test"], "train_seconds": rec["train_seconds"], "params": rec["params"]}))


if __name__ == "__main__":
    main()
