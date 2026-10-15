"""Live terminal dashboard for a running sweep: progress, each running job, recent results, leaderboard.

    .venv/bin/python scripts/watch.py            # follows the most recently active phase
    .venv/bin/python scripts/watch.py phase3     # a specific phase

Read-only: it only reads log files and results/runs.jsonl, so it is safe to start and stop
(Ctrl-C) at any time.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sgloc import ablations, experiment  # noqa: E402

DONE_LINE = re.compile(r"^\[(\d+)/(\d+)\s+(\d+)s (\w+)\] (\S+) seed (\d+): mean ([\d.]+)\s+±1 ([\d.]+)%")


def phase_log(phase):
    if phase:
        return phase, ROOT / "results" / f"{phase}_log.txt"
    logs = sorted((ROOT / "results").glob("phase*_log.txt"), key=lambda p: p.stat().st_mtime)
    log = logs[-1]
    return log.stem.replace("_log", ""), log


def running_jobs(phase):
    jobs = []
    ids = {experiment.config_id(c) for c in ablations.PHASES[phase]()}
    for cfg_path in sorted((ROOT / "runs" / "_queue").glob("*.json")):
        cfg = json.loads(cfg_path.read_text())
        if cfg_path.stem not in ids:
            continue
        cid = cfg_path.stem
        run_log = ROOT / "runs" / f"{cfg['name']}_s{cfg['seed']}_{cid}" / "log.txt"
        step, total, best, elapsed = 0, None, None, 0
        if run_log.exists():
            text = run_log.read_text()
            m = re.search(r"steps (\d+)", text)
            total = int(m.group(1)) if m else None
            steps = re.findall(r"^step\s+(\d+).*best ([\d.]+) @ \d+\s+(\d+)s", text, re.M)
            if steps:
                step, best, elapsed = int(steps[-1][0]), float(steps[-1][1]), int(steps[-1][2])
        jobs.append((cfg, step, total, best, elapsed))
    return jobs


def leaderboard(phase, top=10):
    by_id = {}
    with open(experiment.RESULTS) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("config_id"):
                    by_id[r["config_id"]] = r
    rows = defaultdict(list)
    for cfg in ablations.PHASES[phase]():
        r = by_id.get(experiment.config_id(cfg))
        if r:
            rows[(cfg["group"], cfg["name"])].append(r)
    agg = [(g, n, len(rs), sum(r["test"]["mae"] for r in rs) / len(rs), sum(r["test"]["acc_1"] for r in rs) / len(rs),
            rs[0]["params"]) for (g, n), rs in rows.items()]
    return sorted(agg, key=lambda a: a[3])[:top]


def render(phase, log):
    lines = log.read_text().splitlines() if log.exists() else []
    starts = [i for i, l in enumerate(lines) if l.startswith(phase + ":")]
    lines = lines[starts[-1]:] if starts else lines  # only the latest sweep in this log (resumed runs append)
    header = lines[0] if starts else ""
    m = re.search(r"(\d+) still to run", header)
    total = int(m.group(1)) if m else 0
    done = [DONE_LINE.match(l) for l in lines]
    done = [d for d in done if d]
    failed = sum("FAILED" in l for l in lines)
    elapsed = int(done[-1].group(3)) if done else 0
    rate = len(done) / elapsed * 3600 if elapsed else 0
    eta = (total - len(done)) / rate if rate else None

    top = Table.grid(padding=(0, 2))
    bar = ProgressBar(total=max(total, 1), completed=len(done), width=40)
    top.add_row(Text(f"{phase}", style="bold"), bar,
                Text(f"{len(done)}/{total} runs   {rate:.0f}/hour   ETA {eta:.1f} h" if eta is not None else f"{len(done)}/{total} runs"),
                Text(f"{failed} failed" if failed else "", style="red"))

    jobs = Table(title="Running now", title_justify="left", expand=True)
    for col in ("config", "seed", "progress", "best val", "time"):
        jobs.add_column(col)
    for cfg, step, steps_total, best, secs in running_jobs(phase):
        frac = step / steps_total if steps_total else 0
        jobs.add_row(f"{cfg['group']}/{cfg['name']}", str(cfg["seed"]),
                     ProgressBar(total=1.0, completed=frac, width=24) if steps_total else Text("preparing data"),
                     f"{best:.3f}" if best else "", f"{secs // 60}m{secs % 60:02d}s" if secs else "")

    recent = Table(title="Recently finished (test pool)", title_justify="left", expand=True)
    for col in ("config", "seed", "device", "mean distance", "within ±1"):
        recent.add_column(col)
    for d in done[-6:][::-1]:
        recent.add_row(d.group(5), d.group(6), d.group(4), d.group(7), d.group(8) + "%")

    board = Table(title="Leaderboard so far (mean over seeds, test pool; v1 rebuilt = 3.00 / 61.7%)",
                  title_justify="left", expand=True)
    for col in ("group", "config", "seeds", "params", "mean distance", "within ±1"):
        board.add_column(col)
    for g, n, k, mae, acc, params in leaderboard(phase):
        board.add_row(g, n, str(k), f"{params:,}", f"{mae:.3f}", f"{100 * acc:.1f}%")

    return Panel(Group(top, Text(""), jobs, recent, board),
                 title=f"sine-Gordon localization sweep   {time.strftime('%H:%M:%S')}", border_style="blue")


def main():
    phase, log = phase_log(sys.argv[1] if len(sys.argv) > 1 else None)
    with Live(render(phase, log), refresh_per_second=1, screen=False) as live:
        while True:
            time.sleep(5)
            try:
                live.update(render(phase, log))
            except Exception as err:  # files being written mid-read; try again next tick
                live.console.print(f"[dim]refresh skipped: {err}[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
