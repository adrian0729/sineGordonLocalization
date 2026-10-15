#!/bin/zsh
# usage: scripts/chain.sh <pid to wait for> <phase> [sweep args...]
# waits for one process to exit (by PID, never by name), then runs the next sweep
cd "$(dirname "$0")/.."
wait_pid=$1; phase=$2; shift 2
while kill -0 "$wait_pid" 2>/dev/null; do sleep 15; done
.venv/bin/python scripts/sweep.py "$phase" "$@" >> "results/${phase}_log.txt" 2>&1
