### optimizer

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| O1_adam_constant | 3 | 22,849 | 1.70 ± 0.04 | 1.68 ± 0.05 | 29.5 ± 2.2 | 71.8 ± 1.4 | 96.9 ± 0.1 | 2.0 ± 0.1 | 3 min |
| O2_adamw_cosine | 3 | 22,849 | 1.77 ± 0.07 | 1.75 ± 0.09 | 28.7 ± 1.2 | 71.5 ± 1.8 | 96.7 ± 0.3 | 2.1 ± 0.2 | 4 min |
| O3_adamw_onecycle | 3 | 22,849 | 1.68 ± 0.07 | 1.65 ± 0.08 | 28.1 ± 0.9 | 72.9 ± 0.2 | 97.1 ± 0.2 | 1.9 ± 0.2 | 3 min |
| O4_sgd_nesterov_cosine | 3 | 22,849 | 20.72 ± 2.75 | 20.65 ± 2.62 | 1.7 ± 0.4 | 4.8 ± 0.9 | 11.2 ± 2.1 | 68.6 ± 6.1 | 3 min |
| O5_lion_cosine | 3 | 22,849 | 1.77 ± 0.06 | 1.74 ± 0.01 | 28.1 ± 0.8 | 71.4 ± 0.1 | 96.8 ± 0.1 | 2.1 ± 0.0 | 3 min |

### batch

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| adamw_batch64 | 3 | 22,849 | 1.71 ± 0.16 | 1.68 ± 0.17 | 29.3 ± 1.3 | 72.3 ± 2.7 | 96.9 ± 0.6 | 2.0 ± 0.4 | 6 min |
| adamw_batch1024 | 3 | 22,849 | 1.78 ± 0.03 | 1.74 ± 0.02 | 28.7 ± 1.4 | 71.2 ± 0.7 | 96.8 ± 0.1 | 2.1 ± 0.1 | 2 min |

### weight_decay

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| adamw_wd0 | 0 | | |  |  |  |  |  | pending |
| adamw_wd0.01 | 0 | | |  |  |  |  |  | pending |
