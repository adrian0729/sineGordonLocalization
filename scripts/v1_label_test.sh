#!/bin/zsh
# Does the label definition explain the v1 reproduction gap? Train the rebuilt v1 on the
# period-averaged label (two seeds, one after the other) once the GPU slot given by <pid> frees up.
#   zsh scripts/v1_label_test.sh <pid to wait for>
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
for seed in 0 1; do
  .venv/bin/python scripts/train_v1.py --seed $seed --label avg --tag _avglabel --device mps \
    --note "Phase 6 follow-up: the rebuilt v1 trained and scored on the period-averaged label, to test whether the label definition explains the reproduction gap"
done
