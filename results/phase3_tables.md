### information

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| I0_v1_inputs | 3 | 22,849 | 1.77 ± 0.07 | 1.75 ± 0.09 | 28.7 ± 1.2 | 71.5 ± 1.8 | 96.7 ± 0.3 | 2.1 ± 0.2 | 4 min |
| I1_theta0 | 3 | 21,889 | 1.72 ± 0.07 | 1.66 ± 0.08 | 29.2 ± 0.5 | 71.7 ± 0.9 | 97.1 ± 0.4 | 1.9 ± 0.2 | 3 min |
| I2_E0 | 3 | 21,889 | 1.73 ± 0.05 | 1.70 ± 0.05 | 28.6 ± 0.2 | 71.8 ± 0.3 | 97.0 ± 0.1 | 2.0 ± 0.1 | 4 min |
| I3_theta_frames | 3 | 22,753 | 1.57 ± 0.01 | 1.56 ± 0.01 | 26.1 ± 0.2 | 76.7 ± 0.1 | 96.9 ± 0.3 | 1.7 ± 0.0 | 3 min |
| I4_E_frames | 3 | 22,753 | 1.65 ± 0.04 | 1.63 ± 0.01 | 28.6 ± 0.3 | 72.6 ± 0.5 | 97.2 ± 0.1 | 1.8 ± 0.0 | 3 min |
| I5_theta_omega_random_start | 3 | 21,985 | 2.02 ± 0.02 | 2.03 ± 0.01 | 28.0 ± 0.2 | 68.9 ± 0.6 | 95.8 ± 0.2 | 2.8 ± 0.0 | 3 min |
| I6_dtheta0_amplitude | 3 | 21,985 | 1.57 ± 0.03 | 1.57 ± 0.03 | 42.0 ± 0.5 | 64.3 ± 0.4 | 96.8 ± 0.2 | 1.7 ± 0.1 | 3 min |

### phase

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| theta_random_start_w1 | 2 | 21,889 | 2.80 ± 0.07 | 2.87 ± 0.04 | 25.0 ± 0.8 | 62.0 ± 0.0 | 90.9 ± 0.3 | 4.8 ± 0.1 | 3 min |
| theta_random_start_w2 | 2 | 21,985 | 2.13 ± 0.03 | 2.12 ± 0.01 | 27.6 ± 0.3 | 67.4 ± 0.3 | 95.1 ± 0.0 | 3.0 ± 0.0 | 3 min |
| theta_random_start_w4 | 2 | 22,177 | 2.11 ± 0.05 | 2.14 ± 0.12 | 27.5 ± 0.5 | 68.1 ± 0.2 | 95.2 ± 0.6 | 3.0 ± 0.3 | 3 min |
| theta_random_start_w8 | 2 | 22,561 | 2.10 ± 0.01 | 2.10 ± 0.03 | 26.9 ± 0.7 | 68.5 ± 0.3 | 95.3 ± 0.1 | 2.9 ± 0.1 | 3 min |
| theta_random_start_w16 | 2 | 23,329 | 1.89 ± 0.02 | 1.85 ± 0.01 | 28.3 ± 0.4 | 69.8 ± 0.2 | 96.5 ± 0.1 | 2.3 ± 0.0 | 3 min |
| E_random_start_w1 | 2 | 21,889 | 1.82 ± 0.07 | 1.80 ± 0.09 | 30.4 ± 3.1 | 69.4 ± 3.4 | 96.5 ± 0.3 | 2.3 ± 0.2 | 3 min |
| theta_omega_random_start_w1 | 2 | 21,985 | 2.01 ± 0.00 | 2.03 ± 0.00 | 28.1 ± 0.1 | 69.3 ± 0.1 | 95.9 ± 0.1 | 2.8 ± 0.0 | 3 min |

### noise_time

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| S1_t0_noise0 | 1 | 22,561 | 1.58 | 1.60 | 36.9 | 65.5 | 96.3 | 1.7 | 3 min |
| S1_t0_noise0.0001 | 1 | 22,561 | 1.62 | 1.67 | 36.5 | 67.0 | 96.2 | 1.9 | 3 min |
| S1_t0_noise0.0003 | 1 | 22,561 | 1.70 | 1.72 | 40.3 | 64.1 | 96.4 | 2.1 | 3 min |
| S1_t0_noise0.001 | 1 | 22,561 | 2.21 | 2.20 | 34.2 | 66.2 | 94.9 | 3.6 | 3 min |
| S1_t0_noise0.003 | 1 | 22,561 | 2.96 | 2.97 | 26.5 | 66.7 | 92.2 | 5.7 | 3 min |
| S1_t0_noise0.01 | 1 | 22,561 | 6.93 | 6.98 | 17.9 | 47.5 | 74.4 | 17.2 | 3 min |
| S1_t10_noise0 | 1 | 22,561 | 1.61 | 1.55 | 40.3 | 64.9 | 96.9 | 1.6 | 3 min |
| S1_t10_noise0.0001 | 1 | 22,561 | 1.62 | 1.53 | 40.6 | 65.7 | 96.8 | 1.6 | 3 min |
| S1_t10_noise0.0003 | 1 | 22,561 | 1.63 | 1.59 | 39.9 | 66.0 | 96.9 | 1.8 | 3 min |
| S1_t10_noise0.001 | 1 | 22,561 | 1.74 | 1.68 | 39.4 | 65.1 | 96.5 | 2.0 | 3 min |
| S1_t10_noise0.003 | 1 | 22,561 | 2.23 | 2.21 | 28.4 | 70.2 | 94.6 | 3.6 | 3 min |
| S1_t10_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S1_t30_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S2_t0_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S2_t10_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S2_t20_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S2_t30_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S3_t0_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S3_t10_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S3_t20_noise0.01 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0.0001 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0.0003 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0.001 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0.003 | 0 | | |  |  |  |  |  | pending |
| S3_t30_noise0.01 | 0 | | |  |  |  |  |  | pending |

### sparse

Validation mean distance (used for choices) and test-pool metrics (20,000 runs; accuracy in %).

| Config | Seeds | Params | Val mean distance | Mean distance | Exact | ±1 | ±3 | >10 off | Train time |
|---|---|---|---|---|---|---|---|---|---|
| S1_t0_noise0_every1 | 1 | 22,561 | 1.58 | 1.60 | 36.9 | 65.5 | 96.3 | 1.7 | 3 min |
| S1_t0_noise0_every2 | 0 | | |  |  |  |  |  | pending |
| S1_t0_noise0_every4 | 0 | | |  |  |  |  |  | pending |
| S1_t0_noise0_every8 | 0 | | |  |  |  |  |  | pending |
| S1_t0_noise0.001_every1 | 1 | 22,561 | 2.21 | 2.20 | 34.2 | 66.2 | 94.9 | 3.6 | 3 min |
| S1_t0_noise0.001_every2 | 0 | | |  |  |  |  |  | pending |
| S1_t0_noise0.001_every4 | 0 | | |  |  |  |  |  | pending |
| S1_t0_noise0.001_every8 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0_every1 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0_every2 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0_every4 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0_every8 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.001_every1 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.001_every2 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.001_every4 | 0 | | |  |  |  |  |  | pending |
| S1_t20_noise0.001_every8 | 0 | | |  |  |  |  |  | pending |
