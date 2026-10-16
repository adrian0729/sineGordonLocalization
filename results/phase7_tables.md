### inputs

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| v1_inputs_avg | 3 | 22,882 | 0.60 ± 0.01 | 0.62 ± 0.01 | 68.8 ± 0.5 | 95.7 ± 0.2 | 98.4 ± 0.1 | 0.7 ± 0.0 | 7 min |
| v1_inputs_v1label | 3 | 22,882 | 1.51 ± 0.04 | 1.51 ± 0.01 | 27.4 ± 1.0 | 76.4 ± 0.4 | 97.3 ± 0.0 | 1.6 ± 0.0 | 6 min |
| angle_frames_avg | 3 | 22,786 | 0.58 ± 0.01 | 0.58 ± 0.01 | 70.5 ± 0.2 | 96.4 ± 0.1 | 98.5 ± 0.0 | 0.7 ± 0.0 | 6 min |
| angle_frames_v1label | 3 | 22,786 | 1.53 ± 0.02 | 1.50 ± 0.03 | 27.9 ± 0.3 | 76.9 ± 0.3 | 97.3 ± 0.1 | 1.6 ± 0.1 | 6 min |
| theta0_only_avg | 3 | 21,922 | 0.66 ± 0.02 | 0.68 ± 0.02 | 68.3 ± 1.0 | 95.0 ± 0.2 | 98.1 ± 0.1 | 0.9 ± 0.0 | 8 min |
| theta0_only_v1label | 3 | 21,922 | 1.53 ± 0.02 | 1.52 ± 0.01 | 31.0 ± 1.4 | 74.5 ± 0.4 | 97.1 ± 0.3 | 1.6 ± 0.0 | 7 min |

### receptive_field

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| rf17_avg | 3 | 13,474 | 1.73 ± 0.00 | 1.77 ± 0.00 | 41.5 ± 0.5 | 78.2 ± 0.2 | 93.4 ± 0.0 | 2.4 ± 0.0 | 2 min |
| rf17_v1label | 3 | 13,474 | 2.32 ± 0.01 | 2.37 ± 0.02 | 25.7 ± 0.3 | 61.2 ± 0.1 | 91.8 ± 0.0 | 3.1 ± 0.1 | 2 min |
| rf33_avg | 3 | 16,578 | 0.70 ± 0.02 | 0.70 ± 0.03 | 66.0 ± 0.8 | 93.8 ± 0.4 | 98.1 ± 0.1 | 0.8 ± 0.1 | 3 min |
| rf33_v1label | 3 | 16,578 | 1.67 ± 0.02 | 1.70 ± 0.01 | 28.1 ± 0.8 | 70.9 ± 0.1 | 96.5 ± 0.0 | 1.9 ± 0.0 | 3 min |
| rf129_avg | 3 | 22,786 | 0.58 ± 0.01 | 0.58 ± 0.01 | 70.5 ± 0.2 | 96.4 ± 0.1 | 98.5 ± 0.0 | 0.7 ± 0.0 | 6 min |
| rf129_v1label | 3 | 22,786 | 1.53 ± 0.02 | 1.50 ± 0.03 | 27.9 ± 0.3 | 76.9 ± 0.3 | 97.3 ± 0.1 | 1.6 ± 0.1 | 6 min |

### soft_labels

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| soft_on_avg | 3 | 22,786 | 0.58 ± 0.01 | 0.58 ± 0.01 | 70.5 ± 0.2 | 96.4 ± 0.1 | 98.5 ± 0.0 | 0.7 ± 0.0 | 6 min |
| soft_on_v1label | 3 | 22,786 | 1.53 ± 0.02 | 1.50 ± 0.03 | 27.9 ± 0.3 | 76.9 ± 0.3 | 97.3 ± 0.1 | 1.6 ± 0.1 | 6 min |
| soft_off_avg | 3 | 22,786 | 0.60 ± 0.03 | 0.59 ± 0.02 | 71.6 ± 1.4 | 96.2 ± 0.1 | 98.4 ± 0.0 | 0.8 ± 0.0 | 7 min |
| soft_off_v1label | 3 | 22,786 | 1.55 ± 0.02 | 1.53 ± 0.01 | 40.1 ± 1.2 | 65.3 ± 0.5 | 96.9 ± 0.1 | 1.6 ± 0.1 | 6 min |

### deploy

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| deploy_camera_every2_noise1e-3_avg | 3 | 22,882 | 1.14 ± 0.01 | 1.18 ± 0.03 | 62.4 ± 0.6 | 92.9 ± 0.4 | 96.8 ± 0.1 | 2.2 ± 0.1 | 4 min |
| deploy_camera_every2_noise1e-3_v1label | 3 | 22,882 | 1.94 ± 0.02 | 1.93 ± 0.00 | 28.2 ± 0.2 | 72.6 ± 0.6 | 95.9 ± 0.1 | 2.7 ± 0.0 | 6 min |
| deploy_camera_noise1e-2_t20_avg | 3 | 22,786 | 1.33 ± 0.02 | 1.27 ± 0.03 | 59.2 ± 0.8 | 91.2 ± 0.1 | 96.5 ± 0.1 | 2.3 ± 0.1 | 4 min |
| deploy_camera_noise1e-2_t20_v1label | 3 | 22,786 | 2.07 ± 0.04 | 2.05 ± 0.01 | 27.5 ± 1.0 | 72.6 ± 0.3 | 95.2 ± 0.1 | 3.1 ± 0.0 | 6 min |
| deploy_energy_t30_avg | 3 | 21,922 | 0.55 ± 0.01 | 0.55 ± 0.01 | 71.7 ± 0.2 | 96.3 ± 0.1 | 98.6 ± 0.0 | 0.7 ± 0.0 | 4 min |
| deploy_energy_t30_v1label | 3 | 21,922 | 1.21 ± 0.03 | 1.24 ± 0.03 | 49.3 ± 2.3 | 82.4 ± 1.1 | 95.6 ± 0.4 | 1.4 ± 0.0 | 6 min |
