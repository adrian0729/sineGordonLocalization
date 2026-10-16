### final

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| v2_final | 5 | 22,786 | 1.52 ± 0.02 | 1.49 ± 0.03 | 27.7 ± 0.4 | 77.2 ± 0.5 | 97.3 ± 0.1 | 1.5 ± 0.1 | 6 min |

### leave_one_out

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| no_dilation | 3 | 22,786 | 2.54 ± 0.01 | 2.59 ± 0.01 | 23.9 ± 0.4 | 59.0 ± 0.2 | 90.1 ± 0.2 | 3.5 ± 0.0 | 6 min |
| zero_padding | 3 | 22,786 | 2.46 ± 0.06 | 2.42 ± 0.07 | 24.7 ± 0.6 | 66.7 ± 0.1 | 91.7 ± 0.2 | 4.0 ± 0.2 | 6 min |
| dense_head | 3 | 231,013 | 1.78 ± 0.05 | 1.80 ± 0.03 | 26.6 ± 1.1 | 72.0 ± 0.7 | 96.2 ± 0.3 | 2.3 ± 0.1 | 5 min |
| plain_cross_entropy | 3 | 22,753 | 1.56 ± 0.01 | 1.57 ± 0.02 | 40.9 ± 0.5 | 64.3 ± 0.4 | 96.7 ± 0.1 | 1.7 ± 0.1 | 6 min |
| no_soft_labels | 3 | 22,786 | 1.55 ± 0.02 | 1.53 ± 0.01 | 40.1 ± 1.2 | 65.3 ± 0.5 | 96.9 ± 0.1 | 1.6 ± 0.1 | 6 min |
| no_time_head | 3 | 22,753 | 1.51 ± 0.01 | 1.51 ± 0.01 | 27.4 ± 0.2 | 76.8 ± 0.4 | 97.2 ± 0.0 | 1.6 ± 0.0 | 6 min |
| v1_inputs | 3 | 22,882 | 1.51 ± 0.04 | 1.51 ± 0.01 | 27.4 ± 1.0 | 76.4 ± 0.4 | 97.3 ± 0.0 | 1.6 ± 0.0 | 6 min |
| adam_constant | 3 | 22,786 | 1.56 ± 0.04 | 1.56 ± 0.06 | 26.9 ± 1.4 | 75.8 ± 1.0 | 97.0 ± 0.1 | 1.7 ± 0.1 | 6 min |

### data

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| final_n1000 | 2 | 22,786 | 2.46 ± 0.05 | 2.42 ± 0.10 | 25.3 ± 0.1 | 65.2 ± 0.6 | 93.2 ± 0.3 | 3.8 ± 0.3 | 6 min |
| final_n3000 | 2 | 22,786 | 2.12 ± 0.01 | 2.07 ± 0.01 | 25.5 ± 2.3 | 68.5 ± 0.7 | 95.0 ± 0.0 | 2.9 ± 0.1 | 7 min |
| final_n10000 | 2 | 22,786 | 1.81 ± 0.05 | 1.80 ± 0.05 | 27.1 ± 2.4 | 71.2 ± 0.4 | 96.2 ± 0.4 | 2.3 ± 0.1 | 9 min |
| final_n30000 | 2 | 22,786 | 1.61 ± 0.03 | 1.56 ± 0.05 | 27.5 ± 0.8 | 75.7 ± 0.6 | 97.0 ± 0.1 | 1.7 ± 0.1 | 9 min |
| final_n100000 | 2 | 22,786 | 1.47 ± 0.02 | 1.44 ± 0.02 | 27.7 ± 0.6 | 77.8 ± 0.2 | 97.5 ± 0.1 | 1.4 ± 0.0 | 9 min |

### noise

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| final_trained_noise1e-3 | 3 | 22,786 | 1.80 ± 0.03 | 1.72 ± 0.02 | 28.7 ± 1.0 | 73.3 ± 0.3 | 96.8 ± 0.1 | 2.2 ± 0.0 | 8 min |

### impurity

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| impurity_trained | 3 | 22,786 | 1.10 ± 0.07 | 1.06 ± 0.10 | 39.7 ± 9.1 | 86.2 ± 1.1 | 98.5 ± 0.3 | 1.0 ± 0.0 | 9 min |
| impurity_trained_with_map | 3 | 22,882 | 1.10 ± 0.00 | 1.05 ± 0.02 | 33.8 ± 0.7 | 87.0 ± 0.5 | 98.8 ± 0.1 | 0.8 ± 0.0 | 9 min |

### deploy

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| deploy_camera_every2_noise1e-3 | 3 | 22,882 | 1.94 ± 0.02 | 1.93 ± 0.00 | 28.2 ± 0.2 | 72.6 ± 0.6 | 95.9 ± 0.1 | 2.7 ± 0.0 | 6 min |
| deploy_camera_noise1e-2_t20 | 3 | 22,786 | 2.07 ± 0.04 | 2.05 ± 0.01 | 27.5 ± 1.0 | 72.6 ± 0.3 | 95.2 ± 0.1 | 3.1 ± 0.0 | 6 min |
| deploy_energy_t30 | 3 | 21,922 | 1.21 ± 0.03 | 1.24 ± 0.03 | 49.3 ± 2.3 | 82.4 ± 1.1 | 95.6 ± 0.4 | 1.4 ± 0.0 | 6 min |

### label

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| final_avg_label | 3 | 22,786 | 0.58 ± 0.01 | 0.58 ± 0.01 | 70.5 ± 0.2 | 96.4 ± 0.1 | 98.5 ± 0.0 | 0.7 ± 0.0 | 6 min |
| final_sym_label | 3 | 22,786 | 1.53 ± 0.02 | 1.49 ± 0.01 | 32.4 ± 0.5 | 72.5 ± 0.0 | 97.5 ± 0.0 | 1.6 ± 0.0 | 7 min |
