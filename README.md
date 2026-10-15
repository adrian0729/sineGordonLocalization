# Predicting energy localization in the sine-Gordon pendulum ring

Code for v2 of "Machine-learning Prediction of Type-III Instabilities and Spontaneous Energy
Localization" (J. Appl. Nonlinear Dyn. 15(2), 2026). The original Colab exports (the `*.py`
files at the top level) are kept unchanged. The plan for v2 is in
[docs/improvement_plan.md](docs/improvement_plan.md).

## Results so far

Full write-up: [docs/report.md](docs/report.md). Tables: [results/headline.md](results/headline.md).
Test pool of 20,000 runs; mean distance in pendulums around the ring.

| Model | Parameters | Training | Mean distance | Within ±1 | Within ±3 |
|---|---|---|---|---|---|
| v1 as published | 26.5M | | 1.44 | 94.4% | |
| v1 rebuilt from the paper (2 seeds) | 26.5M | 23 min | 3.14 | 60.4% | 90.3% |
| Floquet filter (physics, no training) | 0 | | 3.12 | 52.0% | 87.3% |
| v2: whole-ring dilated CNN (5 seeds) | 22,786 | 6 min | 1.49 | 77.2% | 97.3% |

The published v1 numbers could not be reproduced from the paper's description (see the report, Sec. 2).
v2 also predicts when localization happens (1.5% median error), and degrades about three times
more slowly than v1 under sensor noise.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running the plan

```
.venv/bin/python -m pytest                              # physics and model checks
.venv/bin/python scripts/make_pools.py                  # Phase 0: 140k simulations into data/ (~10 min)
.venv/bin/python scripts/validate_sim.py                # Phase 0: agreement with the paper's odeint pipeline
.venv/bin/python scripts/train_v1.py --seed 0           # Phase 0: the paper's CNN-BiLSTM, rebuilt
.venv/bin/python scripts/phase1.py                      # Phase 1: physics checks and baselines
.venv/bin/python scripts/make_robustness_pools.py       # Phase 6 test sets (mismatch, impurities, other N)
.venv/bin/python scripts/sweep.py phase2 --dry-run      # list an ablation table
.venv/bin/python scripts/sweep.py phase2                # run it (Apple GPU + CPU slots, resumable)
.venv/bin/python scripts/summarize.py phase2            # tables: results/phase2_tables.md
.venv/bin/python scripts/lr_range_test.py               # Phase 5 learning-rate range test
.venv/bin/python scripts/robustness.py --runs runs/<dir> --v1   # Phase 6 shifted-condition tests
.venv/bin/python scripts/eval_final.py --runs runs/<dir>   # time head and prediction sets of final models
.venv/bin/python scripts/headline.py                    # v1 vs v2 tables: results/headline.md
.venv/bin/python scripts/make_figures.py                # report figures into figures/
.venv/bin/python scripts/watch.py phase6                # live terminal dashboard for a running sweep
```

`scripts/chain.sh <pid> <phase>` starts a sweep once another process exits, and
`scripts/checkpoint.sh 30` commits and pushes progress every 30 minutes while sweeps run.

Long jobs can be started detached so they survive the terminal closing, for example
`nohup .venv/bin/python scripts/sweep.py phase3 > results/phase3_log.txt 2>&1 &`.

## Layout

| Path | What it holds |
|---|---|
| `sgloc/sim.py` | Vectorized RK4 for Eq. (4), site energies, localization labels |
| `sgloc/floquet.py` | Growth spectrum of the uniform swing, training-free Floquet filter |
| `sgloc/data.py` | Data pools (initial angles + labels) and model inputs rebuilt from them |
| `sgloc/models.py` | v1 CNN-BiLSTM (Keras-faithful PyTorch) and the v2 ring models |
| `sgloc/losses.py`, `sgloc/metrics.py`, `sgloc/train.py` | Loss, ring-distance metrics, training loop |
| `sgloc/views.py`, `sgloc/experiment.py`, `sgloc/ablations.py` | Model inputs, one-config runner, the ablation tables |
| `scripts/` | Pools, validation, v1 training, phase scripts, sweeps, summaries, figures |
| `tests/` | Symmetry, conservation, convergence and model checks |
| `pilot/` | The pilot scripts behind the numbers in the plan |
| `results/runs.jsonl` | One line per training run (config, why it ran, metrics) |
| `data/`, `runs/` | Generated pools and checkpoints (not tracked) |

## Conventions

- Time is dimensionless (τ = ω₀t). Samples are Δτ = 0.5 apart, as in the paper's code.
- Parameters: g̃ = 0.75, h̃ = 0.5 (the paper's code; Sec. 2.1 of the paper says 0.05 by mistake).
- A run is labeled at the first sample where one site holds more than 10% of the total energy.
  `site_right` uses the paper's code convention (each site owns the spring to its right) and
  is the label used for comparisons with v1. `site_left`, `site_sym` and `site_avg`
  (symmetric, averaged over one swing) are stored for the label ablations.
