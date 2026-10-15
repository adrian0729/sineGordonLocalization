"""Run one phase's ablation table, skipping configs already in results/runs.jsonl.

    python scripts/sweep.py phase2 --dry-run
    python scripts/sweep.py phase2 --gpu-workers 2 --cpu-workers 3
    python scripts/sweep.py phase3 --gpu-device cuda --gpu-workers 12 --cpu-workers 0   # on an NVIDIA GPU

Every config runs in a fresh process (scripts/run_config.py), so memory never accumulates.
GPU slots (Apple GPU via MPS) and single-thread CPU slots pull from one job list; heavy configs
(multi-frame front ends, wide or deep models) are kept for the GPU slots.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402

QUEUE = ROOT / "runs" / "_queue"


def heavy(cfg):
    m = cfg["model"]
    return (m.get("front") in ("gru", "conv2d") or m.get("kind") == "v1" or m.get("width", 32) >= 64
            or m.get("kernel", 5) >= 17 or m.get("depth", 2) >= 4 or len(m.get("dilations") or []) >= 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=sorted(ablations.PHASES))
    parser.add_argument("--group", default=None)
    parser.add_argument("--gpu-workers", type=int, default=2)
    parser.add_argument("--gpu-device", default="mps", help="mps (Apple GPU) or cuda (NVIDIA, e.g. an A100)")
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--prebuild-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = ablations.PHASES[args.phase]()
    if args.group:
        configs = [c for c in configs if c["group"] == args.group]
    done, pending, seen = experiment.completed_ids(), [], set()
    for cfg in configs:
        cid = experiment.config_id(cfg)
        if cid not in done and cid not in seen:
            pending.append(cfg)
            seen.add(cid)
    print(f"{args.phase}: {len(configs)} configs, {len(pending)} still to run", flush=True)
    if args.dry_run:
        for cfg in pending:
            print(f"  {cfg['group']:16s} {cfg['name']:32s} seed {cfg['seed']}  n_train {cfg['n_train']}"
                  f"{'  (GPU only)' if heavy(cfg) else ''}")
        return
    if not pending:
        return

    # fill the trajectory cache once, in parallel, before training starts
    start = time.time()
    pools = {}
    for cfg in pending:
        key = (cfg["n_train"], json.dumps(cfg.get("pools"), sort_keys=True))
        if key not in pools:
            pools[key] = experiment.load_pools(cfg["n_train"], cfg.get("pools"))
        experiment.prebuild(cfg, pools[key], args.prebuild_workers)
    del pools
    print(f"trajectory cache ready in {time.time() - start:.0f}s", flush=True)

    QUEUE.mkdir(parents=True, exist_ok=True)
    gpu_queue = [c for c in pending if heavy(c)] if args.gpu_workers else []
    any_queue = sorted([c for c in pending if c not in gpu_queue], key=lambda c: -c["n_train"])
    slots = [args.gpu_device] * args.gpu_workers + ["cpu"] * args.cpu_workers
    running = {}  # slot index -> (process, cfg, log path)
    finished, start = 0, time.time()
    env_cpu = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    while gpu_queue or any_queue or running:
        for i, device in enumerate(slots):
            if i in running:
                continue
            source = (gpu_queue or any_queue) if device != "cpu" else any_queue
            if not source:
                continue
            cfg = source.pop(0)
            cid = experiment.config_id(cfg)
            path = QUEUE / f"{cid}.json"
            path.write_text(json.dumps(cfg))
            log = QUEUE / f"{cid}.log"
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_config.py"), str(path), "--device", device],
                                    stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                    env=env_cpu if device == "cpu" else os.environ)
            running[i] = (proc, cfg, log)
        for i, (proc, cfg, log) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[i]
            finished += 1
            label = f"[{finished}/{len(pending)} {time.time() - start:5.0f}s {slots[i]}] {cfg['group']}/{cfg['name']} seed {cfg['seed']}"
            lines = log.read_text().splitlines()
            result = next((l for l in reversed(lines) if l.startswith("RESULT ")), None)
            if result is None:
                print(f"{label} FAILED\n" + "\n".join(lines[-15:]), flush=True)
                continue
            r = json.loads(result[7:])
            t = r["test"]
            print(f"{label}: mean {t['mae']:.3f}  ±1 {t['acc_1']:.2%}  ±3 {t['acc_3']:.2%}  "
                  f">10 {t['over_10']:.2%}  ({r['train_seconds']}s, {r['params']} params)", flush=True)
            log.unlink()
            (QUEUE / f"{experiment.config_id(cfg)}.json").unlink(missing_ok=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
