# Progress log

Results by phase of [the improvement plan](improvement_plan.md), written so they can move straight into the report. The machine-readable records are:
- `results/validation.json`: simulator checks and baselines
- `results/runs.jsonl`: one line per training run

Distances are measured around the ring. "±1" is the share of runs whose predicted site is within one pendulum of the true site.

---

## Phase 0: simulator, data pools, v1 baseline (15 October 2026)

### What was built

- **`sgloc/`**, containing:
  - a vectorized RK4 simulator for Eq. (4)
  - site energies under three spring conventions
  - four localization labels
  - the Floquet tools
  - data pools
  - a PyTorch re-implementation of v1 that keeps the Keras defaults
  - the training loop and ring-distance metrics
- **17 tests**, checking:
  - Rotating or mirroring the ring moves every label exactly as it should.
  - Energy drifts only 3.4×10⁻⁶ over τ = 150, and the drift shrinks 33× when the step is halved.
  - Labels don't depend on the step size.
  - The Floquet threshold and unstable band come out right.
  - The custom LSTM matches PyTorch's built-in one.
  - The parameter count is 26,499,205, matching the paper's architecture.

### Data pools (`data/`, not tracked)

The pools store initial angles and labels only; model inputs are rebuilt from them on demand.

| Pool | Runs | Seed | Localized by τ = 150 | Mean localization time | Generation time (10 cores) |
|---|---|---|---|---|---|
| train | 100,000 | 1 | 100% | 57.0 | 7.4 min |
| val | 20,000 | 2 | 100% | 57.0 | 1.5 min |
| test | 20,000 | 3 | 100% | 56.9 | 1.5 min |

### Simulator validation (`scripts/validate_sim.py`)

- **Agreement with the paper's own pipeline.** That pipeline is its `odeint` setup, the notebooks' energy formula and the first-crossing loop. It gives the same site and the same time as the new simulator on 1,000 of 1,000 test runs.
- **Step size.** Halving the step leaves all four label conventions unchanged on 1,000 of 1,000 runs.
- **Fig. 5 of the paper.** With A = 1.75, K = 17 at sample 5, I get a mean distance of 5.83 on 2,000 new runs; the paper reports 5.41. Each number has a standard error of about 0.2, and the pilot got 5.38 on a different random set, so the two agree within noise.
- **Floquet.** The threshold amplitude is 1.5767, and modes 1–5 are unstable at A = 1.75.

### Physics baselines on the test pool (20,000 runs, v1 label)

| Predictor | Trained parameters | Mean distance | ±1 | ±3 | More than 10 off |
|---|---|---|---|---|---|
| Box-car K = 17 on E at τ = 0 | 0 | 5.96 | 33.6% | 66.3% | 11.5% |
| Box-car K = 17 on E at τ = 4.5 | 0 | 5.14 | 37.7% | 70.6% | 9.2% |
| Floquet filter, 6 periods | 0 | 3.11 | 52.0% | 87.4% | 4.6% |

### v1 reproduction (`scripts/train_v1.py`)

The setup follows Sec. 2.3 and the Appendix:
- **Data:** 60,000 training runs, 20,000 validation, 20,000 test.
- **Inputs:** E at τ = 0…4.5 plus θ(0).
- **Training:** Adam at 10⁻³ with batch size 64; early stopping with patience 10 on validation loss, restoring the best weights.

There is one deliberate change: the input scaler is fit on the training split only.

**Result (test pool, 20,000 runs)**

| | Paper | v1 rebuilt, seed 0 | Floquet filter (no training) |
|---|---|---|---|
| Parameters | 26.5M | 26,499,205 | 0 |
| Mean distance (ring) | 1.44 | 3.00 | 3.11 |
| Mean distance (no wrap) | (1.44, definition unstated) | 4.42 | 5.00 |
| Within ±1 | 94.4% | 61.7% | 52.0% |
| Within ±3 | n/a | 90.9% | 87.4% |
| More than 10 off | n/a | 5.5% | 4.6% |
| Training | n/a | 19 epochs, best at epoch 9; 24 min on the M3 Pro GPU | none |

Seeds 1 and 2 are queued to measure seed-to-seed spread.

**Reproduction gate: not met.** The plan asked for v1 to land within about 10% of the paper's numbers. It is far off: 61.7% vs 94.4% within ±1, and a mean distance of 3.00 vs 1.44.
- The simulator matches the paper's own pipeline exactly (above), so the gap comes from the ML side.
- The rebuilt network overfits early. From epoch 9 the training loss keeps falling (1.84 → 1.38) while the validation loss rises (2.01 → 2.30).
- As described, v1 ends up about as good as the training-free Floquet filter on mean distance, and better on ±1.

**Diagnostic: is the answer readable from later snapshots?** (`scripts/diagnose_late_frames.py`, 5,000 test runs)

| τ of the snapshot | Runs already localized | Raw energy peak: ±1 | Box-car 5: ±1 | Box-car 17: ±1 |
|---|---|---|---|---|
| 0 | 0% | 5.3% | 17.7% | 34.4% |
| 20 | 0% | 29.1% | 42.6% | 46.5% |
| 40 | 0.2% | 54.7% | 59.2% | 51.2% |
| 50 | 30.3% | 59.6% | 62.9% | 48.7% |
| 60 | 65.5% | 48.3% | 51.9% | 40.1% |

Even right around localization, a single snapshot gives only about 60% within ±1, and later snapshots do worse because the hotspot keeps moving. So inputs reaching later than stated would not by themselves explain 94%.

**Open question for v1.** Three possible explanations remain:
1. The original pipeline differs from the paper's description, for example in label or feature definitions.
2. Training and test runs overlap; for example, the threaded data generation could have repeated random seeds. That would let the 26.5M-parameter network score high by memorizing, and it would also stop the validation loss from turning back up.
3. Something in this rebuild that the paper's description doesn't pin down.

The original notebook, or the ~3 GB HDF5 dataset, would settle it. With the dataset, duplicates across splits can be checked directly by hashing the initial conditions.

---

## Phase 1: physics checks and baselines (15 October 2026)

Script: `scripts/phase1.py`. Results: `results/phase1.json` and `results/phase1_log.txt`. Figures (made with
`scripts/make_figures.py`):

- [Fig. 1](../figures/fig1_growth_spectrum.png): growth spectrum
- [Fig. 2](../figures/fig2_best_width.png): best box-car width
- [Fig. 3](../figures/fig3_label_sensitivity.png): label sensitivity
- [Fig. 4](../figures/fig4_baselines.png): baselines

### Best box-car width follows the Floquet band edge

- **Setup:** 2,000 runs at each fixed amplitude. Only runs that localize count; all of them did, with τ up to 300.
- **Prediction:** K* = N / m_c, where m_c is the first stable ring mode.
- **Measurement:** the best odd K for E(τ = 0). The 16–84% range comes from 300 bootstrap resamples.

| A | Unstable modes | Predicted K* | Measured K* | 16–84% range |
|---|---|---|---|---|
| 1.60 | 1–2 | 33.3 | 35 | 35–41 |
| 1.65 | 1–3 | 25.0 | 25 | 23–25 |
| 1.70 | 1–4 | 20.0 | 21 | 19–23 |
| 1.75 | 1–5 | 16.7 | 17 | 17–17 |
| 1.80 | 1–6 | 14.3 | 15 | 15–15 |
| 1.85 | 1–6 | 14.3 | 15 | 15–15 |
| 1.90 | 1–7 | 12.5 | 15 | 11–15 |

The paper's empirical optimum, K = 17 at A = 1.75, is the Floquet band edge. The rule holds from just above the threshold (A_c = 1.577) to A = 1.90.

### Which parts of the initial noise decide the site (2,000 test runs, re-simulated)

| Change to the initial angles | Mean shift of the site | Moved more than 1 | Moved more than 3 | Moved more than 10 |
|---|---|---|---|---|
| Keep only m ≤ 2 | 8.83 | 81.0% | 58.8% | 19.0% |
| Keep only m ≤ 3 | 4.99 | 60.9% | 31.2% | 8.5% |
| Keep only m ≤ 5 (the unstable band) | 1.71 | 25.0% | 8.6% | 2.6% |
| Keep only m ≤ 6 | 1.46 | 19.6% | 6.7% | 2.3% |
| Keep only m ≤ 12 | 1.20 | 17.2% | 5.7% | 1.7% |
| Keep only m ≤ 25 | 1.24 | 17.0% | 5.8% | 1.8% |
| Remove m ≤ 5 | 26.3 (random) | 97.5% | 94.3% | 82.1% |

The unstable band sets the coarse site. The finer modes move about a fifth of the sites by more than one, and rarely by more than three.

### How precisely the initial state must be known (white Gaussian error per pendulum)

| Error (rad, rms) | 10⁻⁹ | 10⁻⁶ | 10⁻⁵ | 10⁻⁴ | 3×10⁻⁴ | 10⁻³ | 3×10⁻³ | 10⁻² |
|---|---|---|---|---|---|---|---|---|
| Labels moved by more than 1 | 0% | 0% | 0.6% | 4.4% | 9.8% | 24.5% | 46.5% | 75.1% |
| Labels moved by more than 3 | 0% | 0% | 0.2% | 1.7% | 3.4% | 8.1% | 19.4% | 50.1% |

The same 10⁻³ rms error moves 50% of labels when it is confined to m ≤ 5, but only 4.5% when it is confined to m ≥ 6.

### Baselines on the test pool (v1 label, 20,000 runs)

| Predictor | Trained parameters | Mean distance | Exact | ±1 | ±3 | More than 10 off |
|---|---|---|---|---|---|---|
| Box-car K = 17 on E(τ = 4.0), width chosen on validation | 0 | 5.07 | n/a | 38.0% | 71.1% | 9.0% |
| Floquet filter, 6 periods (chosen on validation) | 0 | 3.12 | n/a | 52.0% | 87.3% | 4.6% |
| v1 CNN-BiLSTM, rebuilt (seed 0) | 26,499,205 | 3.00 | 23.2% | 61.7% | 90.9% | 5.5% |
| Learned linear ring filter on δθ(0) | 100 | 3.07 | 23.1% | 59.9% | 89.6% | 5.0% |
| Learned linear, filter changes with amplitude | 200 | 2.17 | 27.3% | 67.4% | 95.3% | 3.1% |
| Learned linear on v1's 11 input channels | 1,100 | 2.02 | 27.6% | 68.2% | 95.7% | 2.7% |

The learned-linear models are multinomial logistic regression with one full-ring circular filter per channel. They are convex, fit with L-BFGS in 7–236 s, and the L2 strength was chosen on validation (0 won every time).

**Takeaways**
- A 200-parameter linear filter beats the 26.5M-parameter v1 rebuild on every metric.
- Most of what v1 learns is the linear growth stage plus its dependence on amplitude.
- The remaining gap to a perfect score is what the nonlinear models in Phase 2 have to explain.

---

## Phase 2: architecture (Q2), 15 October 2026

**Setup**
- Sweep: `scripts/sweep.py phase2`. Tables: `results/phase2_tables.md`.
- Figures:
  - [Fig. 5](../figures/fig5_receptive_field.png): receptive field
  - [Fig. 6](../figures/fig6_data_efficiency.png): data efficiency
  - [Fig. 7](../figures/fig7_architectures.png): all architectures
- Every run uses the same data (60,000 training runs, v1's information: E at τ = 0…4.5 plus θ(0)) and the same budget: AdamW at 3×10⁻³, cosine schedule, 7,050 steps, batch 256.
- Choices are made on the validation pool; the numbers below are on the test pool (20,000 runs).
- "Mean distance" is to the true site in pendulums, averaged over seeds; ± is the spread between seeds.

**H2.1 (how to combine the 10 frames): partly refuted.**

| Front end | Params | Mean distance | ±1 |
|---|---|---|---|
| Single frame | 12k | 1.81 | 71.5% |
| 10 frames stacked (early fusion) | 17k | 1.75 | 73.5% |
| 2D (time × site) convolution | 46k | 1.69 ± 0.10 | 71.5% |
| Per-site GRU | 22k | 1.59 | 74.2% |

- Later frames do add information (1.81 → 1.75).
- A model that processes each site's frame history nonlinearly does better still, so "the LSTM is unnecessary" was too strong. What matters is per-site nonlinear processing over time, not the flatten + LSTM design of v1.
- Pending: a cheaper per-site MLP front end and GRU + dilated combinations (below).

**H2.2 (a site must see most of the ring): supported.** Local-only dilated circular CNNs, by how many sites each output sees:

| Sites seen | 9 | 17 | 33 | 65 | 129 (whole ring, 3 seeds) |
|---|---|---|---|---|---|
| Mean distance | 4.79 | 2.75 | 1.82 | 1.63 | **1.54** |

- The 9-site stack (v1's convolution stack without its dense head) is worse than the 200-parameter linear filter (2.17).
- The whole-ring dilated stack (23k parameters) is the best model in the table.

**H2.3 (a global Fourier branch plus a local branch): refined.**
- Both branches together (1.75) beat the Fourier branch alone (1.94) and the local branch alone (4.90).
- The number of kept ring modes behaves as the Floquet band predicts:

  | Modes kept | 1–3 | 1–6 | 1–12 | 1–25 |
  |---|---|---|---|---|
  | Mean distance | 2.58 | 1.75 | 1.77 | 1.81 |

  Modes 1–3 miss modes 4–6, which grow at the larger amplitudes; beyond 6 nothing is gained.
- But a deep nonlinear whole-ring stack beats the one linear Fourier layer (1.54 vs 1.75). Deciding which growing hump wins is a nonlinear competition across the ring.

**H2.4 (small kernels and widths suffice): supported.**
- Kernel {3, 5, 9, 17} × width {8, 16, 32, 64}, depth 2 or 4: everything from 2k to 115k parameters lands between 1.68 and 2.01.
- Width 8 with kernel 3 or 5 (2k parameters) reaches 1.83–1.85.
- Depth 4 helps a little (1.70–1.72).

**H2.5 (respecting ring symmetry saves data): strongly supported.** Mean distance against training-set size:

| Training runs | 1k | 3k | 10k | 30k | 60k |
|---|---|---|---|---|---|
| Circular padding, one score per site | 2.46 | 2.00 | 1.90 | 1.76 | 1.75 |
| Zero padding | 2.61 | 2.12 | 1.90 | 1.81 | 1.80 |
| Flatten + dense head (like v1) | 11.3 | 7.40 | 4.43 | 3.19 | 2.73 |

- The equivariant model trained on 1,000 runs beats v1 trained on 60,000 (3.0).
- Zero padding costs a little at every size.

**H2.6 (the Floquet field as an input channel helps when data is scarce): refuted.**
- With it: 2.02 at 3k runs and 1.75 at 60k. Without it: 2.00 and 1.75.
- The network learns the linear growth stage from data alone.

**Follow-ups (run after the first results)**

| Variant | Params | Mean distance | Seeds |
|---|---|---|---|
| Per-site MLP over the stacked frames (no recurrence) | 30k | 1.66 | 3 |
| Whole-ring dilated + Fourier branch | 30k | 1.58 | 3 |
| Whole-ring dilated + per-site MLP front | 26k | 1.60 | 3 |
| Whole-ring dilated + per-site GRU front | 25k | 1.57 | 2+ |
| Whole-ring dilated, width 64 | 89k | 1.60 | 1+ |

- The MLP front recovers part of the GRU's gain (1.75 → 1.66 → 1.59), so recurrence over the frames does add something.
- None of these combinations beats the plain whole-ring dilated stack (1.54). The pieces don't add up.

**Decision (Q2).**

Chosen on validation mean distance, with ties broken by cost. Three models tie within one standard error:

| Model | Validation mean distance | Parameters | GPU training time |
|---|---|---|---|
| Plain whole-ring dilated CNN | 1.557 ± 0.012 (3 seeds) | 23k | 5 min |
| Dilated + Fourier | 1.570 ± 0.020 | 30k | 7 min |
| Per-site GRU | 1.571 ± 0.005 | 22k | 15 min |

The plain whole-ring dilated CNN is the cheapest and simplest of the three, so it becomes the v2 architecture for Phases 3–6:
- 7 circular convolutions, kernel 3, dilations 1-2-4-8-16-32-1, width 32
- one score per site
- no Fourier layer and no recurrence

Physics reading:
- The deciding modes span the whole ring (H2.2), so the model needs a global view.
- Which hump wins is a nonlinear competition, so that global view has to be nonlinear (H2.3, refined).
- The ring's symmetry, built into the model, is worth 20× in data (H2.5).

---

## Phase 3: which inputs to measure (Q1), 15 October 2026

**Setup**
- Model: the Phase 2 architecture (whole-ring dilated CNN).
- Training: 15-epoch screening runs.
- Numbers below are test mean distance in pendulums, averaged over seeds; ± is the spread between seeds.
- Tables: `results/phase3_tables.md`.

**H1.1 (the initial state is enough): supported, with a twist.** Observation from release, noise-free:

| Inputs | Mean distance | ±1 |
|---|---|---|
| v1's inputs: E at τ = 0…4.5 + θ(0) | 1.75 ± 0.09 | 71.5% |
| θ(0) only, one frame | 1.66 ± 0.08 | 71.7% |
| E(0) only | 1.70 ± 0.05 | 71.8% |
| E at τ = 0…4.5 | 1.63 ± 0.01 | 72.6% |
| **θ at τ = 0…4.5 (camera only)** | **1.56 ± 0.01** | **76.7%** |
| θ(0) minus chain mean, plus amplitude | 1.57 ± 0.03 | 64.3% |

- Raw angles are the best input. A single frame of θ(0) already beats v1's inputs.
- Several angle frames are better still, even though θ(0) is the complete state in clean data. The extra frames act as an easier-to-learn version of the same information, not new information.
- Energies never beat angles, and they cost more to measure (they need velocities and the model's parameters).
- The mean-subtracted input lands most often on the exact site (42% vs 26–29%), but its misses fall two sites off, so it scores lower within ±1. This is a reminder that ±1 is sensitive to small systematic offsets.

**H1.2 (an unknown start point in the swing hurts angle inputs): supported.**

When the observation starts at a random point within the first swing:

| Inputs | Mean distance |
|---|---|
| Angles, 1 frame | 2.87 |
| Angles, 2 frames | 2.12 |
| Angles, 4 frames | 2.14 |
| Angles, 8 frames | 2.10 |
| Angles, 16 frames (8 time units, about one full swing) | 1.85 |
| Angles and velocities, 1 frame | 2.03 |
| **Energy, 1 frame** | **1.80** |

- The energy barely changes along the swing, so it removes the unknown phase. With one frame it does better than angles and velocities together, even though both contain the same information.
- With a camera only, about one full swing of frames is needed. Half a swing, as predicted, is not enough.

**H1.3 (sensor precision decides ±1 accuracy): partly supported.**

Three sensing setups, each trained and tested with the same white sensor noise on every pendulum and frame (one seed each):
- S1, camera: angles, 8 frames (4 time units)
- S2, camera + gyroscopes: angles and velocities, 1 frame
- S3, energies computed from S2's noisy measurements, 1 frame

Mean distance for the camera setup (S1), by sensor noise and by when watching starts ([Fig. 8](../figures/fig8_noise_vs_time.png)):

| Noise (rad) | from τ = 0 | τ = 10 | τ = 20 | τ = 30 |
|---|---|---|---|---|
| none | 1.60 | 1.55 | 1.53 | 1.53 |
| 10⁻⁴ | 1.67 | 1.53 | 1.54 | 1.47 |
| 3×10⁻⁴ | 1.72 | 1.59 | 1.52 | 1.52 |
| 10⁻³ | 2.20 | 1.68 | 1.54 | 1.55 |
| 3×10⁻³ | 2.97 | 2.21 | 1.68 | 1.53 |
| 10⁻² | 6.98 | 4.01 | 2.29 | 1.69 |

Single-frame setups from τ = 0, noise none / 10⁻⁴ / 3×10⁻⁴ / 10⁻³ / 3×10⁻³ / 10⁻²:
- S2: 1.60 / 1.75 / 1.81 / 2.46 / 4.95 / 12.20
- S3: 1.64 / 1.67 / 1.92 / 2.44 / 4.91 / 12.01

What this shows:
- Precision matters, but the threshold is about three times higher than predicted.
  - From release, the camera keeps 95% within ±3 up to 10⁻³ rad, and is badly hurt only at 10⁻² rad (mean 7.0).
  - The single-frame setups start losing at 10⁻³ rad and collapse at 3×10⁻³. The 8 frames average the noise down by about √8.
- Noise first causes far misses, not one-site slips. For the camera, ±1 stays at 64–67% up to 3×10⁻³ rad, while runs more than 10 sites off grow from 1.7% to 5.7%. Noise flips the choice between competing humps before it blurs the position of the right one.

**H1.4 (watching longer buys robustness): supported, quantitatively for the camera.**

If waiting helps only because the growing modes rise above a fixed sensor noise, the error should depend on noise σ_s and start time τ only through σ_s·e^{−σ₁τ}, with σ₁ = 0.118 the fastest growth rate ([Fig. 13](../figures/fig13_noise_collapse.png)).
- For the camera this holds:
  - 10⁻³ rad from τ = 0, 3×10⁻³ from τ = 10 and 10⁻² from τ = 20 all give 2.2–2.3.
  - 10⁻³ from τ = 10, 3×10⁻³ from τ = 20 and 10⁻² from τ = 30 all give 1.68–1.69.
- Each 10× worse sensor costs about 20 time units of waiting. The prediction was ln 10 / σ₁ = 19.5.
- The single-frame setups gain less from waiting than this. At the same scaled noise, later frames do worse than τ = 0 once it passes 10⁻⁴. Two likely reasons (not tested separately):
  - At τ = 0 the chain is at rest, so velocity noise barely matters. Later it enters at first order.
  - One frame at a fixed later time lands at a different point of the swing for each amplitude, because the period depends on A. This is the H1.2 problem again.
- Waiting also helps without noise: energies at τ = 30 give 1.24 (82% within ±1), the best single result of Phase 3, against 1.64 from release. There is less left to extrapolate.
- The cost is warning time. In the test pool the earliest run localizes at τ = 37 and the median at τ = 55.
  - The camera window starting at τ = 30 ends at τ = 33.5.
  - That leaves a median warning of 21.5 time units (about 3 s for a 20 cm pendulum), and at least 3.5.

**H1.5 (sparse sensors are enough for the coarse location): partly supported.**

The camera tracks only every s-th pendulum; a mask channel marks where the sensors are ([Fig. 12](../figures/fig12_sparse_sensors.png)). Mean distance:

| Pendulums tracked | τ = 0, no noise | τ = 0, 10⁻³ rad | τ = 20, no noise | τ = 20, 10⁻³ rad |
|---|---|---|---|---|
| all 100 | 1.60 | 2.20 | 1.53 | 1.54 |
| every 2nd (50) | 1.68 | 2.06 | 1.40 | 1.49 |
| every 4th (25) | 1.86 | 3.29 | 1.65 | 2.49 |
| every 8th (13) | 5.27 | 10.25 | 3.60 | 4.19 |

- Every 2nd pendulum costs nothing beyond seed noise in all four conditions.
- Every 4th keeps ±3 at 96.3% (the same as full coverage, noise-free from release), as predicted. Under 10⁻³ rad of noise it costs more (3.29 vs 2.20): fewer sensors average less noise out of the long waves.
- Every 8th fails (78% within ±3, mean 5.3), even though 13 sensors satisfy Nyquist for m ≤ 6.
  - Nyquist is necessary but not enough. At release the stable modes hold about 90% of the perturbation, and with 13 sensors each growing mode shares its sampled wavenumber with about 8 aliased stable modes.
  - Waiting until τ = 20 partly rescues it (5.27 → 3.60): the growing modes rise above their aliases just as they rise above sensor noise.

**Decision (Q1).**
- The final model uses angles from all pendulums at τ = 0…4.5. This is the cheapest best input, and it matches v1's observation window for a fair comparison.
- Recommendation for a real chain:
  - One camera; every 2nd pendulum is enough.
  - With ≲10⁻³ rad precision, predict right after release.
  - With a coarser sensor, start watching about ln(σ_s / 10⁻³)/σ₁ ≈ 8.5·ln(σ_s / 10⁻³) time units later.

---

## Phase 4: loss function (Q3), 15 October 2026

**Setup**
- Base: the Phase 2 architecture with v1's inputs, 15-epoch screening runs, 3 seeds each.
- Numbers are test mean distance; ± is the spread between seeds.
- Tables: `results/phase4_tables.md`. Figure: [Fig. 10](../figures/fig10_losses.png).

| Loss | Mean distance | ±1 | ±3 | Verdict |
|---|---|---|---|---|
| Cross-entropy (v1's site loss) | 1.75 ± 0.09 | 71.5% | 96.7% | baseline |
| Circular regression (angle) | 1.94 ± 0.11 | 63.2% | 92.0% | H3.1 supported: averaging competing humps is worse |
| Soft labels, σ = 0.5 | 1.61 ± 0.04 | 73.3% | 97.2% | |
| **Soft labels, σ = 1** | **1.58 ± 0.02** | **75.0%** | 97.2% | H3.2 supported: better, and steadier between seeds |
| Soft labels, σ = 2 | 1.58 ± 0.02 | 73.7% | 97.4% | |
| σ = 1 + 0.03 × expected distance | 1.57 ± 0.04 | 73.6% | 97.4% | H3.3: no gain at small weight |
| σ = 1 + 0.1 × expected distance | 1.60 ± 0.02 | 73.5% | 97.3% | |
| σ = 1 + 0.3 × expected distance | 1.65 ± 0.01 | 71.1% | 97.2% | larger weight makes the model hedge, as predicted |
| σ = 1 + 0.1 × distance + 0.1 × localization-time head | 1.55 ± 0.02 | 74.6% | 97.4% | H3.4 supported |
| σ = 1 + 0.1 × distance + 0.1 × later-energy-profile head | 1.52 ± 0.04 | 74.7% | 97.4% | H3.4 supported |
| σ = 1 + 0.1 × distance + v1's binary "localizes" head | 1.61 ± 0.06 | 72.6% | 97.3% | H3.5 supported: adds nothing (every run localizes) |

**Decision (Q3).** The loss is soft labels (σ = 1) + 0.1 × expected distance + 0.1 × localization-time head.
- It ties the profile-head variant within one validation standard error (1.54 vs 1.52).
- It is simpler: the time target comes free with every label, while the profile target needs an extra simulation for each training run.
- Physics reading: the hotspot spans a few sites, so neighbouring sites are partly right, and soft labels encode that. When and where localization happens are set by the same growing modes, so predicting the time helps predict the site.

---

## Phase 5: optimizer (Q4), 15 October 2026

**Setup**
- Base: the Phase 2 architecture with v1's inputs and plain cross-entropy; 15-epoch screening runs, 3 seeds each.
- Learning rates come from a range test (`results/lr_range.json`): a 2.3×10⁻³ peak for one-cycle and 7.9×10⁻⁴ for Lion. The test was inconclusive for SGD (its loss never fell during the sweep), so SGD used 0.03 with gradients clipped at 1.
- Tables: `results/phase5_tables.md`. Figure: [Fig. 11](../figures/fig11_optimizers.png).

| Optimizer or setting | Mean distance | ±1 | ±3 |
|---|---|---|---|
| Adam, constant 10⁻³ (v1's optimizer) | 1.68 ± 0.05 | 71.8% | 96.9% |
| AdamW + cosine (the default) | 1.75 ± 0.09 | 71.5% | 96.7% |
| AdamW + one-cycle | 1.65 ± 0.08 | 72.9% | 97.1% |
| SGD-Nesterov + cosine | 20.65 ± 2.62 | 4.8% | 11.2% |
| Lion + cosine | 1.74 ± 0.01 | 71.4% | 96.8% |
| Batch 64 (learning rate × ½) | 1.68 ± 0.17 | 72.3% | 96.9% |
| Batch 1024 (learning rate × 2) | 1.74 ± 0.02 | 71.2% | 96.8% |
| Weight decay 0 | 1.72 ± 0.04 | 72.1% | 96.8% |
| Weight decay 10⁻² | 1.67 ± 0.08 | 72.5% | 97.0% |

**H4.1 (the schedule matters more than the optimizer): refuted at this budget.**
- Every adaptive optimizer and schedule lands at 1.65–1.75, inside the seed-to-seed spread of 0.05–0.09. v1's constant-rate Adam does as well as AdamW + cosine.
- The second half of the prediction held, in a stronger form: at the one learning rate tried, SGD with Nesterov momentum barely trains in 15 epochs (20.7; chance is 25). Per-parameter step sizes matter for this network.

**H4.2 (bigger batches are free): supported.**
- Batch 1024 with the learning rate scaled by √(1024/256) matches batch 256 (1.74 vs 1.75), varies less between seeds, and trains about 2× faster in wall time. The predicted 3–5× did not materialize; with a model this small, data handling and evaluation take a large share of each run.
- Batch 64 is 1.5× slower and three times noisier between seeds.

**Weight decay** (0, 10⁻⁴, 10⁻²) makes no difference.

**Decision (Q4).** Keep AdamW + cosine (3% warmup, peak 3×10⁻³, weight decay 10⁻⁴, batch 256).
- No alternative beat it beyond seed noise. Phase 6 re-checks the choice at the full budget by swapping v1's constant-rate Adam into the final model.
- As the plan expected, physics does not pick the optimizer. Once the inputs, architecture and loss respect the problem's structure (Phases 2–4, worth 0.2–3 sites), any reasonable adaptive optimizer finds the same solution (differences ≤ 0.1 sites).
