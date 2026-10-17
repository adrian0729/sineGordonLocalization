#!/bin/zsh
# Commit and push progress every N minutes (default 30) while experiments run, so GitHub always has a
# recent checkpoint. Only project files are staged; data/, runs/ and .venv/ stay ignored.
#   zsh scripts/checkpoint.sh [minutes]        (stop it with: pkill -f scripts/checkpoint.sh)
cd "$(dirname "$0")/.."
interval=$(( ${1:-30} * 60 ))
while true; do
  sleep $interval
  [ -e .git/index.lock ] && continue
  git add -A sgloc scripts tests docs results figures pilot papers v1_original README.md REPORT.md requirements.txt .gitignore pytest.ini 2>/dev/null
  if ! git diff --cached --quiet; then
    logged=$(cat results/logs/phase*_log.txt 2>/dev/null | grep -c '^\[')
    git commit -q -m "Checkpoint $(date '+%Y-%m-%d %H:%M'): ${logged} sweep runs logged

Automatic checkpoint while experiments run.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    GIT_TERMINAL_PROMPT=0 git push -q origin HEAD 2>/dev/null
  fi
done
