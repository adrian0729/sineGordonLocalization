# Results

Everything here is written by the scripts in [scripts/](../scripts/README.md) and read by the [report](../REPORT.md). Test-pool numbers are on 20,000 runs. Mean distance is in pendulums, measured around the ring.

## Start with

- [headline.md](headline.md): the v1 against v2 headline tables (`scripts/headline.py`).

## Tables, one per phase

Each `<table>_tables.md` gives, for every config of an ablation table, the mean ± standard deviation over seeds of the validation mean distance (used for choices) and the test-pool metrics. They are written by `scripts/summarize.py <table>`.

| File | Phase |
|---|---|
| [phase2_tables.md](phase2_tables.md) | Phase 2: architecture |
| [phase3_tables.md](phase3_tables.md) | Phase 3: inputs, sensor noise and timing |
| [phase3_avg_tables.md](phase3_avg_tables.md) | Phase 3 addendum: the same grid on the swing-averaged label |
| [phase4_tables.md](phase4_tables.md) | Phase 4: loss |
| [phase5_tables.md](phase5_tables.md) | Phase 5: optimizer |
| [phase6_tables.md](phase6_tables.md) | Phase 6: the final model and its leave-one-out ablations |
| [phase7_tables.md](phase7_tables.md) | Phase 7: the key choices on the swing-averaged label |
| [phase8_tables.md](phase8_tables.md), [phase8_regularization.md](phase8_regularization.md) | Phase 8: regularization with little data |
| [phase9_tables.md](phase9_tables.md) | Phase 9: the hump label |
| [phase10_tables.md](phase10_tables.md) | Phase 10: the key choices and sensing setups on the hump label |
| [deploy_start_tables.md](deploy_start_tables.md) | When a coarse camera should start watching |
| [threshold_tables.md](threshold_tables.md) | The threshold study (written by `scripts/threshold_analysis.py`) |
| [settings_tables.md](settings_tables.md) | The settings and coupling studies (written by `scripts/settings_analysis.py`) |

## Records and analysis outputs

| File | What it holds | Written by |
|---|---|---|
| `runs.jsonl` | One line per training run: its config, the hypothesis it tests (`note`), training details and test metrics | `sgloc/experiment.py`, `scripts/train_v1.py` |
| `robustness.jsonl` | Trained runs re-evaluated on shifted test conditions | `scripts/robustness.py` |
| `validation.json` | Simulator checks and the first baselines | `scripts/validate_sim.py` |
| `phase1.json` | Growth spectrum, best box-car width, mode knockouts and physics baselines | `scripts/phase1.py` |
| `noise_tradeoff.json` | The fitted trade-off between sensor noise and waiting | `scripts/noise_tradeoff.py` |
| `lr_range.json` | Learning-rate range tests | `scripts/lr_range_test.py` |
| `damping_physics.json` | Which damped runs still localize, and when | `scripts/damping_physics.py` |
| `time_head.json`, `prediction_sets.json` | The final models' time head and prediction sets | `scripts/eval_final.py` |
| `hotspot_timing.json` | Predictors of the localization time, and the shape of the hotspot | `scripts/hotspot_timing.py` |
| `late_frames.json` | How readable the site is from one later snapshot | `scripts/diagnose_late_frames.py` |
| `phase8_curves.json`, `phase8_train.json` | Validation curves and training-set scores of the Phase 8 runs | `scripts/regularization.py` |
| `top2.json`, `top2_deploy.json` | Second-site output for the final models and the sensing setups | `scripts/top2.py` |
| `hump_label.json` | Why the hump label is easier to predict | `scripts/hump_label.py` |
| `threshold.json` | The threshold study: label diagnostics, physics baselines, v1 and v2 on every label, correlations and the two threshold choices | `scripts/threshold_analysis.py` |
| `threshold_hump.json` | What the threshold study means for the hump label, and the growing modes and label timing on other ring sizes (no training) | `scripts/threshold_hump.py` |
| `crossing_energy.json` | How the hump holds its energy when a single-pendulum label fires (5/10/15%), by phase of the swing: energy parts of the labelled pendulum and the hump centre, pendulums beyond the top, energy profile across the hump | `scripts/crossing_energy.py` |
| `settings.json` | The settings and coupling studies: label diagnostics, Floquet filter, trained models, cross-scoring and the hump label's decision rule | `scripts/settings_analysis.py` |
| `ring_transfer.json` | Every ring-size network and v2 scored on every ring of 50–400 pendulums, including new pools of 80 and 150 (no training; the matrix of the preprint's Fig. 9b), and the identity behind the transfers to rings whose size divides the training ring's | `scripts/ring_transfer.py` |
