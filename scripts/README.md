# Scripts

Run every script from the repository root with the project's environment, for example `.venv/bin/python scripts/phase1.py`. Each script's docstring gives its full usage and options.

## The phases of the study

Each phase answers one question of the study and is written up in the [report](../REPORT.md). Phases 2–10 are ablation tables in [sgloc/ablations.py](../sgloc/ablations.py), run with `sweep.py <table>` and summarized with `summarize.py <table>`.

| Phase | Question | How to run it | Results |
|---|---|---|---|
| 0 | Simulator, data pools and the rebuilt v1 | `make_pools.py`, `validate_sim.py`, `train_v1.py` | `results/validation.json`, `results/runs.jsonl` |
| 1 | Physics checks and baselines | `phase1.py` | `results/phase1.json` |
| 2 | Architecture (Q2) | `sweep.py phase2` | `results/phase2_tables.md` |
| 3 | Inputs, sensor noise and when to watch (Q1) | `sweep.py phase3` | `results/phase3_tables.md` |
| 3, addendum | Phase 3's noise-and-timing grid on the swing-averaged label | `sweep.py phase3_avg`, `noise_tradeoff.py` | `results/phase3_avg_tables.md`, `results/noise_tradeoff.json` |
| 4 | Loss (Q3) | `sweep.py phase4` | `results/phase4_tables.md` |
| 5 | Optimizer (Q4) | `lr_range_test.py`, then `sweep.py phase5` | `results/lr_range.json`, `results/phase5_tables.md` |
| 6 | The final model, its leave-one-out ablations and robustness (Q5) | `sweep.py phase6` (and its parts `phase6_deploy`, `phase6_label`, `phase6_damping`), `make_robustness_pools.py`, `robustness.py`, `damping_physics.py` | `results/phase6_tables.md`, `results/robustness.jsonl`, `results/damping_physics.json` |
| 7 | The key choices on the swing-averaged label | `sweep.py phase7` | `results/phase7_tables.md` |
| 8 | Regularization with little data | `sweep.py phase8`, `regularization.py` | `results/phase8_tables.md`, `results/phase8_regularization.md` |
| 9 | The hump-scale label and a second site | `add_labels.py hump`, `sweep.py phase9`, `top2.py` | `results/phase9_tables.md`, `results/top2.json` |
| 10 | The key choices and sensing setups on the hump label | `add_labels.py hump_top`, `sweep.py phase10`, `hump_label.py` | `results/phase10_tables.md`, `results/hump_label.json` |
| Camera start time | When a coarse camera should start watching, at full budget | `sweep.py deploy_start` | `results/deploy_start_tables.md` |

## Data and checks

| Script | What it does | Writes |
|---|---|---|
| `make_pools.py` | Simulates the training, validation and test pools (100,000 / 20,000 / 20,000 runs of initial angles and labels) | `data/pool_*.npz` |
| `add_labels.py` | Adds labels that are not computed by default (`hump`, `hump_top`) to existing pools, re-simulated from their stored initial angles | `data/pool_*.npz` |
| `make_robustness_pools.py` | Test pools for Phase 6: shifted couplings, more initial noise, other ring lengths, impurities and damping | `data/robust_*.npz` |
| `validate_sim.py` | Checks the simulator against the paper's own `odeint` pipeline, step-size convergence, the paper's box-car number and the Floquet threshold | `results/validation.json` |

## Training

| Script | What it does | Writes |
|---|---|---|
| `sweep.py` | Runs one phase's ablation table on GPU and CPU slots, skipping configs already in `results/runs.jsonl`. It fills the trajectory cache in `data/cache/` first. | `runs/<run>/`, `results/runs.jsonl` |
| `run_config.py` | Runs one config in its own process, so memory is returned when it finishes (called by `sweep.py`) | as above |
| `run_one.py` | Runs a single config of a table, by phase, name and seed | as above |
| `train_v1.py` | Trains the rebuilt v1, the paper's CNN-BiLSTM | `runs/<run>/`, `results/runs.jsonl` |
| `summarize.py` | Mean ± standard deviation over seeds for one table | `results/<table>_tables.md` |

## Analyses

| Script | What it does | Writes |
|---|---|---|
| `phase1.py` | Growth spectrum, best box-car width against amplitude, mode knockouts, sensitivity to the initial state, and the physics baselines | `results/phase1.json` |
| `noise_tradeoff.py` | Fits the trade-off between sensor noise and waiting, on both labels | `results/noise_tradeoff.json` |
| `lr_range_test.py` | Learning-rate range test for each optimizer of Phase 5 | `results/lr_range.json` |
| `robustness.py` | Re-evaluates trained runs on shifted test conditions, without retraining | `results/robustness.jsonl` |
| `damping_physics.py` | Which damped runs still localize, and when | `results/damping_physics.json`, `figures/fig20_damping.*` |
| `eval_final.py` | The final models' time head and prediction sets | `results/time_head.json`, `results/prediction_sets.json` |
| `hotspot_timing.py` | Is the localization time just an amplitude trend, and is the hotspot one pendulum? | `results/hotspot_timing.json` |
| `diagnose_late_frames.py` | How readable the site is from a single later snapshot (a check on the v1 reproduction gap) | `results/late_frames.json` |
| `regularization.py` | How each regularizer changes overfitting with little data | `results/phase8_regularization.md`, `results/phase8_curves.json`, `results/phase8_train.json` |
| `top2.py` | A second site, for runs where a rival hump competes | `results/top2.json`, `results/top2_deploy.json` |
| `hump_label.py` | Why the hump label is easier to predict than the swing-averaged label | `results/hump_label.json` |

## Tables and figures

| Script | What it does | Writes |
|---|---|---|
| `headline.py` | The v1 against v2 headline tables | `results/headline.md` |
| `make_figures.py` | Every report figure whose inputs exist; with `--paper`, only the five data figures of the v2 preprint, in paper style | `figures/*.png`, `figures/*.pdf`; `papers/v2_preprint/figures/paper_fig*.pdf` |
