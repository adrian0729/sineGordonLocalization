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

Seed-to-seed spread:
- Seed 1 reaches 59.1% within ±1 and a mean distance of 3.27 (16 epochs, best at epoch 6).
- Seed 2 never trained. Its loss stayed at ln 100 = 4.61, the value of a uniform guess, for 14 epochs. A rerun followed the identical trajectory, so this is a property of the seed and not a glitch.
- v1 statistics below therefore use seeds 0 and 1.

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
- A cheaper per-site MLP front end and the GRU + dilated combinations are tested below, under "Follow-ups".

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

---

## Phase 6: the final model, robustness and real-world setups (Q5), 15 October 2026

**Setup**
- Final model: the Phase 2 architecture with angle frames at τ = 0…4.5, the Phase 4 loss and AdamW + cosine. It trains for 30 epochs on 60,000 runs, with 5 seeds.
- Leave-one-out, impurity, noise-trained and real-world groups use 3 seeds; data efficiency uses 2.
- Tables: `results/phase6_tables.md` and `results/headline.md` (from `scripts/headline.py`).
- Robustness re-evaluates the trained models without retraining (`scripts/robustness.py`). The time head and prediction sets come from `scripts/eval_final.py`.

**Headline (test pool).** v2 final: mean distance 1.49 ± 0.03, 77.2% within ±1, 97.3% within ±3, 1.5% more than 10 off. That uses 22,786 parameters and 6 minutes of training. The rebuilt v1 scores 3.14 ± 0.19, 60.4%, 90.3% and 5.7%, with 26.5M parameters and 23 minutes.

**Leave-one-out** (Δ mean distance against the final model):

| Removed | Δ |
|---|---|
| dilation | +1.10 |
| circular padding | +0.93 |
| per-site head (dense head instead) | +0.31 |
| the chosen loss (plain cross-entropy) | +0.08 |
| AdamW + cosine (v1's Adam) | +0.07 |
| soft labels | +0.04 |
| time head | +0.02 |
| angle inputs (v1's inputs) | +0.02 |

Soft labels matter more for ±1: 77% with them, 64–65% without.

**Data efficiency.** Mean distance by number of training runs:

| Training runs | 1k | 3k | 10k | 30k | 60k | 100k |
|---|---|---|---|---|---|---|
| Mean distance | 2.42 | 2.07 | 1.80 | 1.56 | 1.49 | 1.44 |

v2 beats v1 from 1k runs on.

**Time head.** It predicts the localization time with a 1.5% median error (0.8 time units). The amplitude alone gives 5.1%, a constant 13.4%.

**Prediction sets.** About 5.6 sites hold 90% of the probability and contain the true site 98% of the time; the sets are conservative.

**Robustness without retraining** (v1 → v2):

| Test condition | v1 | v2 |
|---|---|---|
| g ±10% | 3.26 / 3.39 | 1.97 / 1.97 |
| h −10% / +10% | 4.36 / 5.91 | 4.25 / 5.53 |
| twice the initial spread | 3.62 | 1.96 |
| N = 80 / 150 | n/a | 1.31 / 6.32 |
| one pendulum 1% weaker | 18.9 | 22.9 |
| sensor noise 10⁻³ rad | 6.24 | 2.26 |

- The damage follows the Floquet spectrum (`floquet.growth_rates`). A 10% change in g moves σ₁ by under 0.3%, while a 10% change in h moves it by +21% / −33% and changes the number of growing modes.

**Noise-trained.** Trained and tested at 10⁻³ rad, the v2 recipe reaches 1.72.

**Impurities** (0–2% weaker pendulum, impurity test pool):
- Trained on uniform chains, both models fail: v1 14.9, v2 18.9. The weak pendulum pins the hotspot, within ±3 of it in 98% of runs above 1%.
- Trained with impurities, v2 reaches 1.06 from angles alone and 1.05 with a measured map of the chain. The map adds nothing.

**Real-world setups (final recipe).**

| Setup | Mean distance |
|---|---|
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 |
| Coarse camera (10⁻²) from τ = 20 | 2.05 |
| Energies at τ = 30 | 1.24 |

**The label** (error analysis, then a follow-up group).
- With plain cross-entropy, misses land two sites off more often than one, and half of those predictions sit on the swing-averaged hotspot. So the label "first pendulum over 10%" jitters within a swing.
- The swing-averaged label (`site_avg`) agrees with it exactly in only 33% of test runs.

| Model and target | Mean distance | ±1 |
|---|---|---|
| v2, v1's label | 1.49 | 77.2% |
| v2, symmetric label (not averaged) | 1.49 | 72.5% |
| v2, swing-averaged label | 0.58 | 96.4% |
| v1 rebuilt, v1's label | 3.14 | 60.4% |
| v1 rebuilt, swing-averaged label (`scripts/v1_label_test.sh`) | 2.25 | 83.6% |

- The jitter comes from the within-swing exchange of energy, not from the spring bookkeeping.
- A smoother label would explain about two-thirds of the v1 reproduction gap in ±1 (60.4% → 83.6%, against the published 94.4%).

---

## Phase 6 addendum: damping, 16 October 2026

**Setup**
- `sgloc/sim.py` adds linear damping, −γ dθ/dτ, to Eq. (4): a friction torque proportional to each pendulum's angular velocity. γ is per time unit, so a free pendulum's quality factor is Q = 1/γ. It can take one value per run.
- The default γ = 0 is the paper's chain: the stored test-pool labels reproduce exactly, and no config id changes. A new test checks that the energy a damped chain loses equals γ ∫ Σθ̇² dτ (to 10⁻⁵).
- With damping, a run localizes when one site holds 10% of the energy the chain still holds, not 10% of its initial energy.
- **Pools** (`scripts/make_robustness_pools.py`):
  - six fixed-damping pools of 5,000 runs (γ = 0, 10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻²) with the same initial conditions, so their differences are damping alone
  - train / val / test pools of 60,000 / 10,000 / 10,000 runs with γ log-uniform in [10⁻⁴, 10⁻²]; 80% of these runs localize by τ = 150
- **Scripts:**
  - physics: `scripts/damping_physics.py` (`results/damping_physics.json`, [Fig. 20](../figures/fig20_damping.png))
  - trained models re-scored without retraining: `scripts/robustness.py`
  - retraining: the Phase 6 `damping` group (`phase6_damping`, 4 configs × 3 seeds, ~7 min each). Runs that never localize are left out of the site and time losses (`ignore_unlocalized`).
  - tables: `results/headline.md`, Sec. 9

**Physics** (each run compared with itself undamped)

| γ | Q | Swing's 1/e decay (2/γω₀), 20 cm pendulum | Localize by τ = 150 | Median delay | v1's label: same site / ±1 / ±3 | Swing-averaged label ±1 |
|---|---|---|---|---|---|---|
| 10⁻⁴ | 10,000 | 48 min | 100% | 0 | 62% / 80% / 96% | 97% |
| 3×10⁻⁴ | 3,300 | 16 min | 100% | +1 | 32% / 60% / 94% | 95% |
| 10⁻³ | 1,000 | 4.8 min | 99.9% | +3 | 24% / 54% / 93% | 90% |
| 3×10⁻³ | 330 | 95 s | 71% | +9 | 20% / 52% / 87% | 82% |
| 10⁻² | 100 | 29 s | 0% | n/a | n/a | n/a |

- **The swing decays toward A_c.** A run localizes only if its perturbation grows enough before the swing gets there.
  - At Q = 330 this becomes an amplitude threshold: below A ≈ 1.69 almost no run localizes, and above 1.75 every run does (half at 1.71).
  - At Q = 100 a swing released at A = 1.75 falls below A_c after 17.5 time units (2.5 s), with 1.3 e-folds of growth available.
- **Late is the same as never.** Re-simulated to τ = 600, the same 2,000 runs localize exactly as often as by τ = 150 (100% at Q = 1,000, 73.3% at Q = 330).
- **A two-stage Floquet picture predicts the delay.** Undamped runs fit T₀ = 5.13/σ(A₀) + 11.3: about 5 e-folds of linear growth at the fastest Floquet rate σ, then about 11 time units of nonlinear focusing.
  - Integrating σ along the decaying swing gives the median delay: +3.8 vs +3.0 measured at Q = 1,000, and +9.3 vs +9.0 at Q = 330.
  - It predicts too few survivors at Q = 330: 41% vs 71%, with the threshold at A = 1.77 instead of 1.71.
  - Counting all of T₀ as linear growth does worse (21%).
  - At Q = 330 the survivors finish localizing at a median swing amplitude of 1.60, and 42% of them below A_c. A perturbation that is already large keeps focusing after the linear instability has stopped.
  - Damping slows the perturbation itself by γ/2, but the swing loses energy almost as fast (⟨θ̇²⟩/E = 0.89 at A = 1.75), so the energy share on the hotspot feels damping almost only through the amplitude.
- **Damping moves v1's label much more than the swing-averaged one.** At Q = 10,000, only 62% of runs keep v1's exact site, but 97% keep the averaged site within ±1. A slightly later crossing changes which pendulum is caught first within the swing.
- **Damping removes the shorter growing modes.** At the moment of localization the median swing amplitude is 1.75 undamped (4–6 growing ring modes), 1.69 at Q = 1,000 (3–5) and 1.60 at Q = 330 (0–3).

**Trained undamped, tested damped** (mean distance over the runs that localize, against each model's own label)

"Undamped frames" feeds a model the first frames of the same runs without damping, and scores it against the damped outcome. It separates a changed outcome from changed inputs.

| γ | v1 rebuilt (2 seeds) | v2 final (5 seeds) | v2 final, undamped frames | v2, swing-averaged label (3 seeds) | the same, undamped frames |
|---|---|---|---|---|---|
| 0 | 3.22 | 1.56 | n/a | 0.62 | n/a |
| 10⁻⁴ | not run | 1.49 | 1.51 | 0.63 | 0.61 |
| 3×10⁻⁴ | not run | 1.49 | 1.50 | 0.64 | 0.60 |
| 10⁻³ | 5.43 | 2.48 ± 0.28 | 1.68 | 4.23 ± 0.73 | 0.84 |
| 3×10⁻³ | 12.40 | 4.92 ± 0.37 | 2.47 | 5.21 ± 0.29 | 1.34 |

- **Up to Q ≈ 3,000, damping costs nothing.**
- **At Q = 1,000, most of the loss comes from the inputs, not the outcome.**
  - With the undamped frames, v2 final scores 1.68, not 2.48. On the averaged label it scores 0.84, not 4.23.
  - A slowly decaying swing looks unlike any training run, and both models rely on details of the frames that damping changes.
- **v1 degrades faster** (3.22 → 5.43 → 12.4), so v2 trained undamped stays ahead.

**Retrained on damped chains** (γ log-uniform in [10⁻⁴, 10⁻²]; mean distance at each γ, over the runs that localize; 3 seeds)

| γ | Recipe | + localizes head | + head, swing-averaged label | + head, camera (every 2nd pendulum, 10⁻³ rad) | Head accuracy |
|---|---|---|---|---|---|
| 0 | 1.55 | 1.53 | 0.69 | 2.09 | 100% |
| 10⁻⁴ | 1.53 | 1.48 | 0.63 | 2.13 | 100% |
| 3×10⁻⁴ | 1.44 | 1.42 | 0.53 | 2.07 | 100% |
| 10⁻³ | 1.19 | 1.16 | 0.49 | 1.86 | 99.9% |
| 3×10⁻³ | 0.84 (92% ±1) | 0.85 | 0.38 (98% ±1) | 1.35 | 99.1% |
| 10⁻² | n/a | n/a | n/a | n/a | 100% |

On the damped test pool:

| Model | Mean distance | ±1 |
|---|---|---|
| v1 rebuilt, trained undamped | 6.22 | 46.8% |
| v2 final, trained undamped | 2.57 | 64.1% |
| v2 recipe, trained damped | 1.30 ± 0.02 | 82.0% |
| + localizes head | 1.31 ± 0.01 | 82.0% |
| + head, swing-averaged label | 0.50 ± 0.02 | 97.2% |
| + head, camera | 1.82 ± 0.01 | 78.0% |

- **Retrained, the recipe loses nothing on undamped chains** (1.55 vs 1.56 for v2 final). **It gets better with damping:** 1.19 at Q = 1,000 and 0.84 at Q = 330.
  - That is not because only easy runs survive. Without damping, v2 final (seed 0) does worse on the runs that localize at Q = 330 than on the rest (1.64 vs 1.31).
  - Near threshold fewer and longer ring modes grow, so the site depends less on local detail. The coarse answer lives in those modes (Phase 1).
- **The localizes head tells from the first 4.5 time units whether a run will localize at all.**
  - On the damped test pool it is right 99.5% of the time, with a ROC area of 0.9999. It misses 0.25% of the runs that localize and raises a false alarm on 1.4% of the rest.
  - It is still right in 99.1% of runs at Q = 330, where 29% do not localize.
  - It costs nothing on the site (1.31 vs 1.30).
- **On the swing-averaged label the retrained recipe reaches 0.50 on damped chains (97.2% within ±1)**, and 0.38 at Q = 330. On undamped chains it costs 0.07 against the undamped-trained model (0.69 vs 0.62).
- **The recommended camera keeps its accuracy on damped chains:** 1.82 on the damped test pool, against 1.93 for the same setup on undamped chains (Phase 6). Its localizes head is right 99.3% of the time.

---

## Phase 7: the key choices on the swing-averaged label, 16 October 2026

**Question.** Phases 2–6 made every choice on v1's label, which jitters by 1–3 sites within a swing (Phase 6). On the swing-averaged label (`site_avg`) the final recipe reaches 0.58 instead of 1.49. Do the input, receptive-field and loss choices still win on a target that does not jitter?

**Setup**
- `ablations.phase7`: the final recipe at the full budget (30 epochs, 60,000 runs, 3 seeds), with one choice changed at a time:
  - inputs: v1's energy frames + θ(0), angle frames (the final recipe), or θ(0) alone
  - receptive field: 17, 33 or 129 sites (Phase 2's dilations 1-2-4-1, 1-2-4-8-1 and 1-2-4-8-16-32-1)
  - soft labels on (the final recipe) or off (hard labels, other loss terms kept)
  - the three recommended setups of Phase 6
- Every config runs twice: on the swing-averaged label (`label="avg"`) and, as its matched control, on v1's label.
  - Controls that Phase 6 already trained have the same config id and are reused: the final recipe (seeds 0–2), v1's inputs, hard labels and the three setups. The final recipe on the averaged label is Phase 6's `final_avg_label`.
  - That leaves 33 new runs: 24 on the averaged label, and 9 controls (θ(0) alone, 17 and 33 sites).
- Each model is scored on the label it was trained on. On the averaged label, 51 of the 20,000 test runs never cross 10% before τ = 150; as in Phase 6, they are left out.
- **Same code as Phase 6.** The damping study edited the training code while this phase ran. One of its edits briefly changed how the 0.25% of averaged-label runs that never localize enter the loss.
  - So Phase 7 ran from a snapshot of commit 28f8428: the Phase 6 code plus this table.
  - Three runs that had started on the edited code were stopped before they wrote a record, then repeated.
- Tables: `results/phase7_tables.md`. Figure: [Fig. 19](../figures/fig19_avg_label_ablations.png).

**Choices** (test pool; mean of 3 seeds; Δ against the final recipe on the same label, ± its standard error; × is the ratio to the final recipe's mean distance)

| Final recipe, but … | v1's label | Δ | × | Averaged label | Δ | × | Exact, v1's / averaged | ±1, v1's / averaged |
|---|---|---|---|---|---|---|---|---|
| nothing (the final recipe) | 1.50 | | | 0.58 | | | 27.9% / 70.5% | 76.9% / 96.4% |
| v1's inputs (energy frames + θ(0)) | 1.51 | +0.01 ± 0.02 | 1.01 | 0.62 | +0.04 ± 0.01 | 1.08 | 27.4% / 68.8% | 76.4% / 95.7% |
| θ(0) only | 1.52 | +0.01 ± 0.02 | 1.01 | 0.68 | +0.11 ± 0.02 | 1.19 | 31.0% / 68.3% | 74.5% / 95.0% |
| sees 33 sites, not the whole ring | 1.70 | +0.19 ± 0.02 | 1.13 | 0.70 | +0.12 ± 0.02 | 1.21 | 28.1% / 66.0% | 70.9% / 93.8% |
| sees 17 sites | 2.37 | +0.87 ± 0.02 | 1.58 | 1.77 | +1.20 ± 0.01 | 3.08 | 25.7% / 41.5% | 61.2% / 78.2% |
| hard labels (other loss terms kept) | 1.53 | +0.03 ± 0.02 | 1.02 | 0.59 | +0.01 ± 0.01 | 1.02 | 40.1% / 71.6% | 65.3% / 96.2% |

On v1's label the final recipe's 1.50 is the mean of seeds 0–2; all 5 seeds give 1.49.

The predictions below were written into `ablations.phase7` before the runs started.
- **Q1, inputs: holds, and now shows.**
  - On v1's label, the three input sets tie at the full budget.
  - On the averaged label, angle frames win: v1's inputs cost 0.04 (8% of the error) and θ(0) alone 0.11 (19%).
  - As predicted, the lower error floor (0.58 instead of 1.49) reveals differences the jitter hid. θ(0) holds the whole initial state, but the frames make it easier to learn (Phase 3).
- **H2.2, seeing the whole ring: holds, and matters more.**
  - 17 sites cost 1.20 on the averaged label (×3.1), against 0.87 (×1.6) on v1's label. Exact-site accuracy falls from 70.5% to 41.5%.
  - 33 sites cost 21% of the error, against 13%.
  - The growing modes are the five longest waves, whichever label marks the hotspot.
- **H3.2, soft labels: their gain was the label's.**
  - On v1's label, hard labels trade ±1 for exact hits: 40.1% exact (27.9% with soft labels), but only 65.3% within ±1 (76.9%).
  - On the averaged label they change nothing beyond seed noise: 0.59 vs 0.58, 71.6% exact, 96.2% within ±1.
  - As predicted, soft labels mainly hedged between the sites the label jitters between. They do no harm on the averaged label, so the recipe keeps them.
- **v1's label understates the choices.**
  - Relative to the final recipe, every change of inputs or receptive field, and every noisy setup, costs more on the averaged label (tables above and below).
  - On v1's label the jitter adds 0.6–1.0 sites to every model, and this hides part of each difference.

**Recommended setups** (final recipe; test pool; 3 seeds)

| Setup | v1's label | Averaged label | × final, v1's / averaged | ±1, v1's / averaged |
|---|---|---|---|---|
| Every pendulum, noise-free, τ = 0…4.5 | 1.50 | 0.58 | 1 / 1 | 76.9% / 96.4% |
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 | 1.18 | 1.28 / 2.04 | 72.6% / 92.9% |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 20 | 2.05 | 1.27 | 1.36 / 2.20 | 72.6% / 91.2% |
| Energies at τ = 30, noise-free | 1.24 | 0.55 | 0.82 / 0.96 | 82.4% / 96.3% |

- **The ranking holds.** The cheap camera is within ±1 of where the energy settles in 92.9% of runs, from release.
- **Noise doubles the error on the averaged label** (×2.0–2.2, against ×1.3–1.4 on v1's label). Noise picks the wrong hump (Phase 3), and no label removes those misses.
- **Without noise, waiting stops paying.**
  - Energies at τ = 30 beat the final recipe by 0.27 on v1's label (exact site 49.3% vs 27.9%). On the averaged label they win by only 0.03 ± 0.01 (exact 71.7% vs 70.5%).
  - Watching closer to the event mostly helps predict the moment in the swing at which v1's label fires. Where the energy settles can already be read at release.
  - So Phase 3's "waiting helps without noise" is mostly about v1's label.

**Side note.** Scored on v1's label, the models trained on the averaged label get 1.78–1.80 (60% within ±1), against 1.50 for the model trained on v1's label. The two labels mark different pendulums, so each model is judged on its own label.

**Decision.** The inputs and architecture chosen on v1's label stand on the swing-averaged label, with larger margins. Soft labels become optional.
- Without noise, watching from release is as good as waiting until τ = 30 on that target (0.58 vs 0.55), and leaves about 25 time units more warning.
- Whether waiting pays against sensor noise on that target is untested: the noise-and-timing grid was not repeated.
