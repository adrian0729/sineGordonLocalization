# Predicting where energy localizes in a sine-Gordon pendulum ring: a physics-guided v2

## Summary

A ring of 100 coupled pendulums, released in a nearly uniform swing, suddenly concentrates its energy on one pendulum after about 57 time units on average (about 8 s for 20 cm pendulums). The published model, v1, is a 26.5M-parameter CNN-BiLSTM that predicts which pendulum from the first 4.5 time units.

This v2 study rebuilds the problem from its physics, the Floquet growth spectrum of the uniform swing. Each design question gets a physics-based hypothesis, tested by ablations: about 340 training runs, 33 GPU-hours on a laptop.

- **v1's published accuracy could not be reproduced.** Rebuilt from the paper, v1 reaches 60.4% within ±1 pendulum, not 94.4%.
  - The label explains part of the gap: which pendulum first passes 10% of the energy jitters by 1–3 sites within a swing.
  - On a swing-averaged label, the rebuilt v1 reaches 83.6%.
- **v2 halves v1's error with 1/1,163 of the parameters.** It is a whole-ring dilated circular CNN with 22,786 parameters, trained in 6 minutes.
  - Mean distance 1.49 vs 3.14; within ±1, 77% vs 60%; within ±3, 97% vs 90%.
  - It beats v1 with 60 times less data, and also predicts *when* localization happens (1.5% median error).
- **On the swing-averaged label, v2 is within ±1 pendulum in 96% of runs** (mean distance 0.58).
- **Physics explains what matters.**
  - The network must see the whole ring, because the growing modes are its five longest waves.
  - Waiting divides sensor noise by the Floquet growth: each 10× worse sensor costs about 20 time units of waiting (predicted: 19.5).
  - Robustness to wrong parameters follows the growth spectrum.
  - One pendulum 1% weaker pins the hotspot, and v2 learns to find it from the motion alone.
- **For a real chain:** one camera tracking every 2nd pendulum to 10⁻³ rad predicts right after release with a mean distance of 1.93, against 3.14 for v1 with perfect data.

## 1. Motivation and what v1 did

### The problem

A ring of N = 100 torsionally coupled pendulums with a weak global (mean-field) coupling is released from rest. The pendulums start in a nearly uniform swing, A ≈ 1.65–1.85 rad, with small random offsets (±0.025 rad).

The uniform swing is unstable (a type-III instability). After about 57 time units, one pendulum suddenly holds more than 10% of the chain's energy. The question is whether the early behaviour of the chain predicts which pendulum it will be.

### What v1 did

The published model (v1) was a CNN-BiLSTM with 26.5M parameters. Its inputs were the per-site energy at τ = 0, 0.5, …, 4.5 plus the initial angles. It reported:
- 94.4% of predictions within ±1 pendulum
- a mean error of 1.44 pendulums
- versus 5.4 for the best physics-inspired smoothing (a box-car average of width 17)

### What v2 does

v2 asks five questions, each answered with a hypothesis from the physics followed by ablations that could refute it:
- **Q1:** which inputs to measure
- **Q2:** what architecture
- **Q3:** what loss
- **Q4:** what optimizer
- **Q5:** which choices matter in the final model

## 2. What this work corrects

| Finding | Consequence |
|---|---|
| All code uses h̃ = 0.5; the paper's Sec. 2.1 says 0.05 | Typo. The Floquet analysis confirms 0.5: with 0.05 the instability has no threshold near 1.59 and is not type III |
| The time axes of Figs. 2–5 and 7 are sample indices (Δτ = 0.5) | Localization happens at τ ≈ 57, not 100. The box-car comparison point "t = 5" is τ = 2.5, earlier than the end of v1's input window |
| Each site's energy includes the whole spring to its right neighbour | The label's exact site depends on this bookkeeping in 21% of runs |
| The input scaler was fit before the train/test split | Minor leakage; v2 fits it on the training pool only |
| `odeint` was run one simulation at a time | Replaced by a vectorized RK4 that is about 100× faster and gives identical labels |

### Reproducing v1

The original training notebook and dataset are not available, so v1 was rebuilt from the paper's description in PyTorch, keeping the Keras defaults:
- the same layers, initializers, dropout masks, loss weights and optimizer
- early stopping on validation loss
- the same 60/20/20 data sizes

The data come from a simulator that matches the paper's own pipeline exactly: its `odeint` setup, energy formula and detection loop give the same localization site and time on 1,000 of 1,000 test runs.

**The rebuilt v1 does not reproduce the reported accuracy.**

| | Paper | v1 rebuilt (2 seeds) |
|---|---|---|
| Within ±1 | 94.4% | 61.7% / 59.1% |
| Mean distance | 1.44 | 3.00 / 3.27 |

A third seed stayed at chance level and never trained.

Later input frames would not explain the gap. A single snapshot of the energy even right at localization only gets about 60% within ±1, because the hotspot keeps moving after it forms.

Three explanations remain, and cannot be distinguished without the original code:
1. The original pipeline differs from its description, for example in label or feature definitions.
2. Training and test runs overlapped (for example, repeated random seeds in the threaded data generation), letting the 26.5M-parameter network score high by memorizing.
3. Something in this rebuild that the paper does not specify.

**Testing the first explanation: the label.** Sec. 10 shows that v1's label jitters. Which pendulum first passes 10% of the energy moves by 1–3 sites within a swing, while the swing-averaged hotspot does not. Trained on the swing-averaged label instead, the rebuilt v1 reaches 83.6% within ±1 and a mean distance of 2.25 (2 seeds).
- That is about two-thirds of the way from 60.4% to the published 94.4%.
- So a smoother label in the original training pipeline would explain much of the gap, but not all of it. The rest could come from a different smoothing, or from explanations 2 and 3.
- This is a consistency check, not an identification: the original labelling code is not available.

**All v1 vs v2 comparisons below use the rebuilt v1 on the same test pool and v1's label as the code defines it.**

## 3. Physics of the type-III instability

### Growth spectrum

Linearizing the dynamics about the uniform swing gives one Hill equation per Fourier mode of the ring ([Fig. 1](../figures/fig1_growth_spectrum.png)).
- The uniform swing becomes unstable at A_c = 1.577. The paper's Fig. 3b shows an asymptote near 1.59.
- At A = 1.75 only the five longest waves grow (wavelengths 100, 50, 33, 25 and 20 sites).
- The fastest grows by a factor of e every 8.4 time units.
- v1's 10-frame window (τ ≤ 4.5) sees at most about 2× growth.

### The best smoothing width is the Floquet band edge

A box-car of width K first cancels ring mode N/K. The best width should cancel the first stable mode, which gives K* = N/m_c.
- Measured on 2,000 runs at each of seven amplitudes, the prediction matches from just above threshold (A = 1.60) to A = 1.90 ([Fig. 2](../figures/fig2_best_width.png)).
- That includes the paper's empirical optimum of 17 at A = 1.75.

### The label is deterministic but needs a precise initial state

([Fig. 3](../figures/fig3_label_sensitivity.png))
- Perturbing the initial angles by 10⁻⁶ rad never changes the label, and neither does halving the integration step.
- The accuracy ceiling is therefore close to 100%. Chaos does not limit it; measurement precision does.
- A white error of 10⁻⁴ rad moves 4% of labels by more than one site; 10⁻³ rad moves 25%.
- The same 10⁻³ rad error placed only in the five growing modes moves 50%. Placed only in the stable modes, it moves 4.5%.
- Deleting every initial mode except the growing ones keeps the site within ±3 in 91% of runs. Deleting the growing modes makes the site random.

The coarse answer lives in about ten numbers (the five growing modes); local detail decides the last site or two.

## 4. Methods

See [report_methods_draft.md](report_methods_draft.md) for:
- the simulator
- data pools of 100k/20k/20k runs with no overlap
- the four label conventions
- the metrics
- the v1 rebuild
- the v2 models and training recipe

In short:
- Every model trains on the same 60,000 runs, picks its checkpoint and design on the 20,000-run validation pool, and is reported on the 20,000-run test pool.
- Phase 2 runs use 30 epochs. Phases 3–5 use 15-epoch screening runs. The final model is retrained at 30 epochs.
- Every run is logged with a note naming the hypothesis it tests.

## 5. Baselines

([Fig. 4](../figures/fig4_baselines.png); test pool, mean distance in sites)

| Predictor | Parameters | Mean distance | ±1 | ±3 |
|---|---|---|---|---|
| Box-car, width 17, best frame | 0 | 5.07 | 38.0% | 71.1% |
| Floquet filter: initial noise evolved 6 periods through the linear dynamics | 0 | 3.12 | 52.0% | 87.3% |
| v1 rebuilt (2 seeds) | 26.5M | 3.00–3.27 | 59.1–61.7% | 89.7–90.9% |
| Learned linear ring filter, amplitude-dependent | 200 | 2.17 | 67.4% | 95.3% |
| Learned linear ring filters on v1's inputs | 1,100 | 2.02 | 68.2% | 95.7% |

A convex, 200-parameter model built on the physics beats the 26.5M-parameter v1 rebuild on every metric.

## 6. Architecture (Q2)

All models below see v1's information and get the same training budget ([Fig. 7](../figures/fig7_architectures.png)).

| Hypothesis | Verdict | Evidence (test mean distance) |
|---|---|---|
| H2.1: the 10 frames need no recurrent model | Partly refuted | Stacked frames 1.75; per-site MLP 1.66; per-site GRU 1.59. Single frame 1.81 |
| H2.2: a site must see most of the ring | Supported ([Fig. 5](../figures/fig5_receptive_field.png)) | Receptive field 9 / 17 / 33 / 65 / 129 sites: 4.79 / 2.75 / 1.82 / 1.63 / **1.54** |
| H2.3: global Fourier branch + local branch | Refined | Together 1.75 beats Fourier-only (1.94) and local-only (4.90); modes 1–3 / 6 / 12 / 25 give 2.58 / 1.75 / 1.77 / 1.81, saturating at the unstable band. A deep nonlinear whole-ring stack does better (1.54) |
| H2.4: small kernels and widths suffice | Supported | 2k–115k parameters all give 1.68–2.01 |
| H2.5: ring symmetry saves data | Strongly supported ([Fig. 6](../figures/fig6_data_efficiency.png)) | Equivariant model at 1k runs: 2.46; v1-style dense head at 60k runs: 2.73 |
| H2.6: the Floquet field as an input helps with little data | Refuted | 2.02 vs 2.00 at 3k runs |

**Choice.** Three models tie on validation within one standard error: the whole-ring dilated CNN (1.557, 3 seeds), dilated + Fourier (1.570) and the per-site GRU (1.571). v2 uses the cheapest and simplest of them:
- the **whole-ring dilated CNN**: 7 circular convolutions, kernel 3, dilations 1-2-4-8-16-32-1, width 32
- one score per site
- 23k parameters, about 1/1,000 of v1
- about 6 minutes to train on a laptop GPU

Combining it with the Fourier branch, the MLP front or the GRU front, or doubling its width, does not improve it (1.57–1.61).

## 7. Which inputs to measure (Q1)

Every input set was tested with the Phase 2 architecture on 15-epoch screening runs ([Fig. 9](../figures/fig9_inputs.png), [Fig. 8](../figures/fig8_noise_vs_time.png)).

### Information (noise-free, observed from release)

| Inputs | Mean distance | ±1 |
|---|---|---|
| **Angles at τ = 0…4.5** | **1.56** | **76.7%** |
| θ(0) only | 1.66 | 71.7% |
| v1's energy frames + θ(0) | 1.75 | 71.5% |

Energies never beat angles, and they need velocities plus the model's parameters to compute. In clean data θ(0) is the complete state; the extra angle frames simply make the same information easier to learn.

### Unknown start in the swing

If observation starts at an unknown point in the swing, raw angles need about one full swing of frames (16 frames, 8 time units) to recover: 1.85, against 2.87 with one frame.

A single frame of energy reaches 1.80. Energy barely changes along the swing, so it removes the unknown phase. The network does worse with the angle-and-velocity pair that contains the same information (2.03).

### Sensor noise and when to watch

- **Camera alone (8 angle frames), watching from release:** fine up to about 10⁻³ rad of noise (about 95% within ±3, mean distance 2.2), then degrades. At 10⁻² rad it reaches a mean distance of 7.0.
- **Noise first causes far misses, not one-site slips.** ±1 accuracy stays at 64–67% up to 3×10⁻³ rad, while the share of runs more than 10 sites off triples. Noise picks the wrong hump before it blurs the right one.
- **Later observation recovers the loss.** At 10⁻² rad, watching from τ = 10, 20 and 30 gives 4.0, 2.3 and 1.7, against 1.55 noise-free.
- **The trade-off follows the Floquet growth rate exactly** ([Fig. 13](../figures/fig13_noise_collapse.png)):
  - For the camera, the error depends on noise σ_s and start time τ only through σ_s·e^{−σ₁τ}, the noise relative to the grown signal.
  - Each 10× worse sensor costs about 20 time units of waiting; the prediction was ln 10 / σ₁ = 19.5.
- **Single-frame setups (angles + velocities, or energies) degrade faster with noise** than the 8-frame camera window, and gain less from waiting.
- **Waiting also helps without noise.** Energies at τ = 30 reach 1.24 (82% within ±1), the best Phase 3 result, because less is left to extrapolate.
- Watching from τ = 30 still leaves a median of 21.5 time units of warning (about 3 s for a 20 cm pendulum). The earliest test run localizes at τ = 37.

### Fewer sensors

([Fig. 12](../figures/fig12_sparse_sensors.png); camera from release, noise-free)

| Pendulums tracked | Mean distance | ±3 |
|---|---|---|
| all 100 | 1.60 | 96.3% |
| every 2nd (50) | 1.68 | 96.3% |
| every 4th (25) | 1.86 | 96.3% |
| every 8th (13) | 5.27 | 78.0% |

- Every 2nd pendulum costs nothing, with or without noise and at either start time.
- Every 8th fails even though 13 sensors satisfy Nyquist for the five growing modes. At release the stable modes hold about 90% of the perturbation, and they alias onto the growing modes' wavenumbers.
- Watching later partly rescues the sparse camera (5.27 → 3.60 from τ = 20), as the growing modes rise above their aliases.

### Recommendation

One camera tracking every pendulum, or every 2nd, at 30–60 frames per second, predicting from a short window of angle frames.
- With a precise camera (≲10⁻³ rad), predict right after release.
- With a coarser one, start about 8.5·ln(σ_s / 10⁻³) time units later, once the long-wave modes have grown above the sensor noise.

## 8. Loss function (Q3)

([Fig. 10](../figures/fig10_losses.png); 15-epoch screening runs, 3 seeds, v1's inputs)

| Loss | Mean distance | ±1 |
|---|---|---|
| Cross-entropy (v1) | 1.75 ± 0.09 | 71.5% |
| Circular regression | 1.94 ± 0.11 | 63.2% |
| Soft labels, σ = 1 | 1.58 ± 0.02 | 75.0% |
| + 0.1 × expected distance + localization-time head | **1.55 ± 0.02** | 74.6% |
| + later-energy-profile head (instead of the time head) | 1.52 ± 0.04 | 74.7% |
| + v1's binary "localizes" head | 1.61 ± 0.06 | 72.6% |

Each hypothesis maps onto a row:
- **H3.1, supported.** Circular regression averages competing humps and does worst.
- **H3.2, supported.** Soft labels that credit neighbouring sites help most, and cut the seed-to-seed spread fourfold.
- **H3.3, refuted as a gain.** An explicit expected-distance term adds nothing at small weight, and at 0.3 it makes the model hedge.
- **H3.4, supported.** Predicting when localization happens helps predict where.
- **H3.5, supported.** v1's binary head adds nothing: every run localizes.

**Choice:** soft labels + 0.1 × expected distance + time head. It ties the profile variant and needs no extra simulation.

## 9. Optimizer (Q4)

([Fig. 11](../figures/fig11_optimizers.png); 15-epoch screening runs, 3 seeds, v1's inputs, plain cross-entropy)

| Optimizer or setting | Mean distance | ±1 |
|---|---|---|
| Adam, constant 10⁻³ (v1) | 1.68 ± 0.05 | 71.8% |
| AdamW + cosine | 1.75 ± 0.09 | 71.5% |
| AdamW + one-cycle | 1.65 ± 0.08 | 72.9% |
| Lion + cosine | 1.74 ± 0.01 | 71.4% |
| SGD-Nesterov + cosine | 20.65 ± 2.62 | 4.8% |
| Batch 64 / 1024 (learning rate scaled by √batch) | 1.68 ± 0.17 / 1.74 ± 0.02 | 72.3% / 71.2% |
| Weight decay 0 / 10⁻² | 1.72 ± 0.04 / 1.67 ± 0.08 | 72.1% / 72.5% |

- **H4.1, refuted at this budget.** Every adaptive optimizer and schedule lands within the seed-to-seed spread; v1's constant-rate Adam is as good as AdamW + cosine. SGD barely trains at the one learning rate tried, as the plan warned it would without a finer search.
- **H4.2, supported.** Batch 1024 matches batch 256, varies less between seeds, and trains about 2× faster (the plan predicted 3–5×).
- **Weight decay** makes no difference.

**Choice:** AdamW + cosine, kept because nothing beat it; Phase 6 re-checks v1's optimizer on the final model. As expected, physics does not choose the optimizer. The physics-guided choices of Phases 2–4 moved the error by 0.2–3 sites; the optimizer moves it by at most 0.1.

## 10. The final model against v1 (Q5)

The final model puts together the winners of Phases 2–5:
- the whole-ring dilated CNN (Q2)
- angle frames at τ = 0…4.5, the same observation window as v1 (Q1)
- soft labels + 0.1 × expected distance + 0.1 × localization-time head (Q3)
- AdamW + cosine (Q4)

It trains for 30 epochs on the same 60,000 runs as v1.

### Headline

([results/headline.md](../results/headline.md); test pool of 20,000 runs; mean ± standard deviation over seeds)

| Model | Parameters | Training | Mean distance | ±1 | ±3 | >10 off |
|---|---|---|---|---|---|---|
| v1 as published | 26.5M | | 1.44 | 94.4% | | |
| v1 rebuilt (2 seeds) | 26.5M | 23 min | 3.14 ± 0.19 | 60.4% | 90.3% | 5.7% |
| Floquet filter (physics, no training) | 0 | | 3.12 | 52.0% | 87.3% | 4.6% |
| Learned linear ring filter | 200 | seconds | 2.17 | 67.4% | 95.3% | 3.1% |
| **v2 final (5 seeds)** | **22,786** | **6 min** | **1.49 ± 0.03** | **77.2%** | **97.3%** | **1.5%** |

Against the rebuilt v1, v2:
- halves the mean distance (3.14 → 1.49) and cuts far misses (more than 10 sites off) from 5.7% to 1.5%
- raises ±1 from 60% to 77%, and ±3 from 90% to 97%
- uses 1/1,163 of the parameters and trains about 4× faster, on a laptop GPU
- varies six times less between seeds, and no seed fails to train (one of three v1 seeds never left chance level)

v2 does not reach the paper's reported 94.4% within ±1 (Sec. 2 discusses why that number could not be reproduced). The paper's mean distance of 1.44 does not say whether it wraps around the ring; v2 scores 1.49 around the ring and 2.43 without wrapping.

### Which choices matter

([Fig. 14](../figures/fig14_leave_one_out.png); each row retrains the final model with one choice reverted, 3 seeds; Δ is against the final model's 1.49, ± its standard error)

| Final model, but … | Mean distance | Δ | ±1 |
|---|---|---|---|
| no dilation (each site sees 17 pendulums) | 2.59 | +1.10 ± 0.01 | 59.0% |
| zero padding instead of circular | 2.42 | +0.93 ± 0.04 | 66.7% |
| flatten + dense head (like v1) | 1.80 | +0.31 ± 0.02 | 72.0% |
| plain cross-entropy (v1's loss) | 1.57 | +0.08 ± 0.02 | 64.3% |
| v1's optimizer (Adam, constant 10⁻³) | 1.56 | +0.07 ± 0.04 | 75.8% |
| hard labels (other loss terms kept) | 1.53 | +0.04 ± 0.01 | 65.3% |
| no localization-time head | 1.51 | +0.02 ± 0.01 | 76.8% |
| v1's inputs (energy frames + θ(0)) | 1.51 | +0.02 ± 0.02 | 76.4% |

- **The ring's physics carries the architecture.** Seeing the whole ring is worth 1.1 sites, since the growing modes are the five longest waves. Circular padding is worth 0.9: the ring has no ends, and the hotspot can form anywhere. One score per site is worth 0.3, with a tenth of the parameters.
- **The loss sets ±1 more than the mean distance.** At the full budget, plain cross-entropy or hard labels cost only 0.04–0.08 in mean distance, but ±1 falls from 77% to 64–65%. Soft labels place predictions between the two candidate sites of the jittery label (see below).
- **v1's optimizer and v1's inputs cost little** (≤ 0.07) once the architecture is right. In clean data, every input set carries the same initial state; inputs matter for real sensors (Sec. 7).

### Regularization: why v2 adds so little

v2 uses little explicit regularization. Each choice was tested or follows from the physics.

| | v1 rebuilt | v2 final |
|---|---|---|
| Dropout | 0.2 on the LSTM's inputs and recurrent state | none |
| Weight decay | none (Adam) | 10⁻⁴ (AdamW) |
| Stopping | early stopping, patience 10 on validation loss | a fixed 30 epochs, keeping the best validation checkpoint |
| Label smoothing | none | soft labels, σ = 1 site (Sec. 8) |
| Auxiliary task | binary "localizes" head | localization-time head |
| Built-in symmetry | none | the same filters at every pendulum, one score per site |

Mean distance on the 60,000 training runs, against the test pool:

| Model | Train | Test | Gap | ±1, train / test |
|---|---|---|---|---|
| v2 final (5 seeds) | 1.35 | 1.49 | +0.14 | 78.4% / 77.2% |
| v2 backbone with v1's flatten + dense head (3 seeds) | 1.06 | 1.80 | +0.74 | 79.7% / 72.0% |
| v1 rebuilt (seed 0) | 1.62 | 3.00 | +1.38 | 72.1% / 61.7% |

- **The ring's symmetry is the main regularizer.** The dynamics are the same at every pendulum, so v2 applies the same small filters everywhere and scores every site the same way.
  - It has 22,786 parameters and none tied to a particular pendulum, so it has little to memorize with.
  - At 60,000 runs its train–test gap is small, and its best checkpoints come late in training (steps 4,465–7,050 of 7,050).
- **Where overfitting appears, a structural fix beats a penalty.** v1 overfits despite dropout and early stopping. v1's dense head overfits on v2's backbone too: its best checkpoint comes about a third of the way through training. Switching to one score per site cuts the gap from 0.74 to 0.14 and the test error by 0.31.
- **Weight decay was tested** (Sec. 9). 0, 10⁻⁴ and 10⁻² are indistinguishable, so it stays at a small default.
- **Data is the cheaper regularizer.** The simulator makes more runs for free; going from 60,000 to 100,000 lowers the test error from 1.49 to 1.44.
- **With little data, v2 does overfit.**
  - At 1,000 and 3,000 runs the best checkpoint is the first one evaluated (step 235 of 7,050), and the validation error then rises by about 50%. At 10,000 runs the best checkpoint comes at steps 705–1,880.
  - So the small-data numbers in Fig. 15 are pessimistic. Regularization there is untested: earlier stopping, stronger weight decay or dropout, and mirror-image augmentation, which the ring's reflection symmetry allows (v1's one-sided label turns into the left-bond label under reflection).

### Data efficiency

([Fig. 15](../figures/fig15_final_data_efficiency.png); final recipe, 2 seeds per size, same number of gradient steps)

| Training runs | 1k | 3k | 10k | 30k | 60k | 100k |
|---|---|---|---|---|---|---|
| v2 mean distance | 2.42 | 2.07 | 1.80 | 1.56 | 1.49 | 1.44 |

- Trained on 1,000 runs, v2 (2.42) already beats v1 trained on 60,000 (3.14): 60 times less data.
- Beyond 60,000 runs the gains are small (1.44 at 100,000).

### When, as well as where

The time head was added only as a training aid (H3.4). Is the time worth predicting, or is it just a trend in the parameters? The predictors below go from the simplest rule to the network ([results/hotspot_timing.json](../results/hotspot_timing.json), `scripts/hotspot_timing.py`; test pool):

| Predictor of the localization time | Median error | Within 10% | Log-time variance explained |
|---|---|---|---|
| A constant (the training median, τ = 55) | 13.4% (7.0 time units) | 39.7% | 0% |
| The amplitude A alone (cubic fit) | 5.0% (2.7) | 82.9% | 82% |
| Floquet clock: the five growing modes' initial size, grown at their own rates until order one | 4.2% (2.3) | 90.7% | 88% |
| Floquet peak clock: the same, stopped when the tallest bump of the summed modes reaches 1 rad | 4.1% (2.2) | 94.3% | 89% |
| **v2 time head (5 seeds)** | **1.5% (0.8)** | **99.4%** | |

- **Most of the timing is a trend in the amplitude.** A larger swing makes the unstable modes grow faster, so it localizes sooner. The amplitude alone explains 82% of the variance of the log-time.
- **Linear Floquet physics adds little.** Growing the actual initial noise in the five unstable modes, even accounting for how the modes line up in space, only takes the error from 5.0% to 4.1%.
- **The network is about three times more precise**, so the noise pattern carries real timing information beyond the linear stage, presumably from the nonlinear focusing of the hump. In absolute terms the gain is small: 2.7 → 0.8 time units, or 0.4 s → 0.12 s for a 20 cm pendulum.
- **"When" is not a research target of its own.** As a training aid it barely helps the site (removing it costs 0.02). As an output it is free. For most uses the amplitude rule is enough; the network's timing matters only if warnings need sub-second precision. v2 keeps the head because it costs nothing.

### What is left of the error is mostly the label's own jitter

v1's label is the first pendulum whose instantaneous energy passes 10% of the total, with each spring booked to the pendulum on its left. Within one swing, a forming hotspot's energy moves between a pendulum's motion and the springs on either side, so which pendulum crosses 10% first depends on the moment in the swing ([Fig. 18](../figures/fig18_label_jitter.png)).
- With plain cross-entropy, v2's misses land two sites away more often than one. Half of those two-site misses, and three in four of the ones to the left, sit exactly on the hotspot's swing-averaged position.
- A label without the jitter is the first pendulum whose energy, with springs split evenly and averaged over one swing (8 time units), passes 10%. It agrees exactly with v1's label in only 33% of test runs, usually sitting 1–3 sites to the left.

Same inputs, architecture and training; only the target changes (test pool; seeds in brackets):

| Model and target | Mean distance | Exact | ±1 | ±3 |
|---|---|---|---|---|
| v2, v1's label (first instantaneous crossing) [5] | 1.49 ± 0.03 | 27.7% | 77.2% | 97.3% |
| v2, symmetric label: springs split evenly, not averaged [3] | 1.49 ± 0.01 | 32.4% | 72.5% | 97.5% |
| **v2, swing-averaged label** [3] | **0.58 ± 0.01** | **70.5%** | **96.4%** | **98.5%** |
| v1 rebuilt, v1's label [2] | 3.14 ± 0.19 | 22.5% | 60.4% | 90.3% |
| v1 rebuilt, swing-averaged label [2] | 2.25 ± 0.00 | 42.0% | 83.6% | 94.1% |

- **Where the energy settles is predictable** to within one pendulum in 96% of runs, from 4.5 time units of watching.
- **The jitter comes from energy moving within each swing**, not from how the springs are booked: splitting them evenly without averaging does not help (1.49).
- **Most of v2's remaining error on v1's label is the label's own jitter**, not a failure to predict where the energy goes.
- **The rebuilt v1 also gains** from the averaged label (60.4% → 83.6% within ±1), but stays far behind v2 on the same target (Sec. 2 discusses what this means for the reproduction gap).
- For applications, the swing-averaged label is the better target.

### Is one pendulum the right target?

What the hotspot looks like at the moment v1's label fires (2,000 test runs; [results/hotspot_timing.json](../results/hotspot_timing.json)):

| Property | Median (10th–90th percentile) |
|---|---|
| Energy on the top pendulum, instantaneous | 10.3% (10.1–10.9%): just past the threshold, by definition |
| Energy on the top pendulum, averaged over a swing | 5.5% (4.8–6.9%) |
| Energy on the top 3 pendulums, averaged over a swing | 16% (14–20%) |
| Width of the swing-averaged hump at half height | 8 pendulums (6–9) |
| How far the hump moves over the next swing (8 time units) | 0 sites (within ±1 in 88% of runs) |
| How far it moves over the next 30 time units | 6 sites (more than 5 in 50% of runs) |
| A rival hump at least 5 sites away with half the energy or more | 18% of runs (7% with 80% or more) |

- **The hotspot is a hump about 8 pendulums wide, not one pendulum.** Averaged over a swing, its top pendulum holds about 5% of the energy (five times its 1% share in the uniform swing). The 10% on a single pendulum is the instantaneous peak of a breathing hump.
- **The hump is well placed when it forms, then drifts.** It stays put for about a swing, but within 30 time units half of the humps have moved more than 5 sites.
- **About one run in six has a rival hump** with at least half the energy. In those runs, any single-site answer is a guess between two places.

**The best way to frame the target**, given these facts:
1. **Predict the hump's centre when it forms**, not the first pendulum over 10%. That is the swing-averaged label above: 96.4% within ±1.
2. **Keep the output a probability over all pendulums** rather than one regressed position. Regression averages rival humps (circular regression did worst, Sec. 8), while a distribution keeps both and gives prediction sets (below).
3. **Optionally predict the hump's shape** as a per-site energy profile. The profile head tied for best in Sec. 8, and it gives the width and any rival hump for free.
4. **Score by ring distance and calibration** (mean distance, ±1 / ±3, coverage of prediction sets), not by exact-site accuracy.
5. **Define the event at the hump's scale**, for example when the swing-averaged energy of 3 neighbouring pendulums first passes a threshold. This is untested, and is listed as a next step.

### A set of sites instead of one

The model outputs a probability for every site, so it can name the smallest set of sites that holds a chosen share of the probability ([results/prediction_sets.json](../results/prediction_sets.json); 5 seeds, test pool):

| Probability held by the set | Sites in the set (mean) | True site inside |
|---|---|---|
| 50% | 2.5 | 66.6% |
| 80% | 4.4 | 93.3% |
| 90% | 5.6 | 98.0% |
| 95% | 6.6 | 99.3% |

- The sets are conservative: each covers more often than its nominal level. This is expected from training on soft labels, which spread probability over neighbouring sites.
- For a warning system, about five sites (5% of the ring) contain the true site 98% of the time.

## 11. Robustness and real-world sensing

### Shifted chains, without retraining

([Fig. 16](../figures/fig16_robustness.png); both models trained on the nominal chain; shifted pools of 5,000 runs)

| Test condition | v1 rebuilt | v2 final | What changes in the growth spectrum at A = 1.75 |
|---|---|---|---|
| Nominal chain | 3.14 | 1.49 | 5 growing modes, σ₁ = 0.1185 |
| Neighbour coupling g −10% / +10% | 3.26 / 3.39 | 1.97 / 1.97 | σ₁ within 0.3%; the same 5 modes grow |
| Twice the initial spread | 3.62 | 1.96 | Nothing; perturbations start twice as large |
| Global coupling h −10% | 4.36 | 4.25 | σ₁ +21%; 6 modes grow; A_c falls to 1.50 |
| Global coupling h +10% | 5.91 | 5.53 | σ₁ −33%; 3 modes grow; A_c rises to 1.68 |
| 80 pendulums | n/a | 1.31 | 4 modes grow |
| 150 pendulums | n/a | 6.32 | 8 modes grow |
| One pendulum 1% weaker (about 2 mm longer) | 18.9 | 22.9 | The weak pendulum pins the site (within ±3 of it in 92% of runs) |

The pattern follows the Floquet spectrum:
- **Shifts that leave the spectrum alone cost v2 about half a site** (g ±10%, a doubled initial spread). It stays far ahead of v1.
- **Shifts that change the spectrum hurt both models alike.** A 10% change in the global coupling h moves the fastest growth rate by +21% or −33% and changes how many modes grow, and both models lose 3–4 sites. In a real chain, h must be known to a few percent, or the model trained over a range of h.
- **v2 works on rings of other sizes; v1's dense head cannot.** At 80 pendulums it is as accurate relative to the ring's size (1.31 of 80 vs 1.49 of 100). At 150 it degrades (6.3): eight modes compete instead of five, and the ring outgrows the 129-site receptive field.
- **A single slightly weaker pendulum defeats both models.** A 1% weaker restoring force pins the localization site, which neither model saw in training (see Chains with an impurity, below).

### Sensor noise

Test-time sensor noise on every pendulum and frame, mean distance:

| Noise (rad) | v1 rebuilt | v2 final, trained noise-free |
|---|---|---|
| none | 3.14 | 1.49 |
| 10⁻⁴ | 3.20 | 1.51 |
| 3×10⁻⁴ | 3.58 | 1.62 |
| 10⁻³ | 6.24 | 2.26 |
| 3×10⁻³ | 13.86 | 4.63 |

- v2 degrades about three times more slowly than v1. Even untrained for noise, v2 at 10⁻³ rad (2.26) beats the noise-free v1 (3.14).
- Training with the same noise helps further: trained and tested at 10⁻³ rad, the v2 recipe reaches 1.72 ± 0.02.

### Chains with an impurity

(One random pendulum's restoring force 0–2% weaker; 10,000 test runs; [results/headline.md](../results/headline.md))

| Model | Mean distance | ±1 | ±3 |
|---|---|---|---|
| v1, trained on uniform chains | 14.85 | 24.7% | 41.7% |
| v2 final, trained on uniform chains | 18.90 | 19.8% | 28.7% |
| v2 recipe, trained with impurities, angles only | 1.06 ± 0.10 | 86.2% | 98.5% |
| v2 recipe, trained with impurities, plus a measured map of every pendulum | 1.05 ± 0.02 | 87.0% | 98.8% |

- **A slightly weaker pendulum pins the hotspot.** Above a 1% difference the site lands within ±3 of the weak pendulum in 98% of runs. Below 0.5% it does so in only 32%, because the random initial offsets win.
- **Models trained on uniform chains fail** because they have never seen this cue.
- **Trained with impurities, v2 finds the weak pendulum from the motion alone.** A weaker pendulum swings slightly slower and falls behind its neighbours. With an impurity, v2 is more accurate than on uniform chains (1.06 vs 1.49), since the impurity makes the outcome less random.
- **A measured map of every pendulum adds nothing** beyond seed noise (1.05 vs 1.06). For a single impurity of this size, 4.5 time units of angles already reveal it. Chains with small differences at every pendulum remain untested (Sec. 12).

### The recommended setups, at full budget

([results/headline.md](../results/headline.md); median warning is the median localization time, τ = 55, minus the end of the observation; seconds for a 20 cm pendulum)

| Setup | Mean distance | ±1 | ±3 | Median warning |
|---|---|---|---|---|
| v2 final: every pendulum, noise-free, τ = 0…4.5 | 1.49 | 77.2% | 97.3% | 50.5 (7.2 s) |
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 | 72.6% | 95.9% | 50.5 (7.2 s) |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 20 | 2.05 | 72.6% | 95.2% | 30.5 (4.4 s) |
| Energies at τ = 30, noise-free (needs gyroscopes) | 1.24 | 82.4% | 95.6% | 25 (3.6 s) |
| v1 rebuilt: every pendulum, noise-free, from release | 3.14 | 60.4% | 90.3% | 50.5 (7.2 s) |

- **A realistic, cheap setup works.** One camera tracking every 2nd pendulum to 10⁻³ rad predicts right after release with a mean distance of 1.93, well ahead of the noise-free v1 (3.14).
- **A ten times coarser camera** gets nearly the same accuracy (2.05) by starting at τ = 20, and still leaves about 4 s of warning ([Fig. 17](../figures/fig17_warning_tradeoff.png)).
- **Energies at τ = 30** give the best accuracy (1.24), but they need gyroscopes and leave less warning.

## 12. Limitations and next steps

### Limitations

- **The v1 comparison is against a rebuild.** The original notebook and data are not available. The rebuild reaches 60.4% within ±1 (83.6% on a swing-averaged label), not the published 94.4%. If the published pipeline differs in a way that matters, the real gap between v1 and v2 could be smaller than reported here. The physics baselines (Floquet filter 3.12, learned linear filter 2.17) do not depend on the rebuild, and v2 beats them too.
- **Simulation only, without damping.** Real pendulums lose energy to friction and air drag. A damped uniform swing drifts toward the instability threshold (A_c = 1.577), and below it nothing localizes. This study keeps the paper's conservative model.
- **Idealized sensor noise.** Noise was white and independent across pendulums and frames. Real cameras add correlated errors (lens distortion, motion blur, a fixed calibration offset per pendulum). A fixed offset acts like a wrong initial state, which the dynamics amplify.
- **Manufacturing tolerances.** The impurity tests detune one pendulum per chain, and a single 1% impurity already pins the site within ±3 in 92% of runs. A model must therefore be trained with the tolerances it will meet. For one impurity the motion alone reveals it (Sec. 11). A real chain has small differences at every pendulum, which may make a measured map of the chain necessary.
- **Screening budget and seeds.** Phases 3–5 used 15-epoch runs, and the noise/time grid one seed per cell. Differences below about 0.1 sites in those phases are within noise.
- **The label.** v1's label (the first pendulum over 10%) jitters by 1–3 sites within a swing (Sec. 10). Exact-site and ±1 accuracy on it partly measure the moment in the swing rather than where the energy goes. The swing-averaged label is the better target, but it was tested only on the final recipe and on v1, not through every phase.

### Next steps

1. Add damping to the simulator, retrain, and test whether early warning survives.
2. Build a bench-top chain and test the recommended setup: one camera, every 2nd pendulum, a short window of frames after release.
3. Calibrate the prediction sets (Sec. 10), which are conservative, for example with a temperature fitted on the validation pool.
4. Train with small differences at every pendulum, plus a measured map of each pendulum's frequency.
5. Train over a range of the global coupling h, which moves the growth spectrum (Sec. 11), or estimate it from the chain first.
6. Repeat the input and architecture choices on the swing-averaged label.
7. Define localization at the hump's scale (for example, 3 neighbouring pendulums over a swing), and let the model name a second site when a rival hump competes (Sec. 10).
8. If the original v1 notebook resurfaces, rerun the comparison against it, check its labels, and check its data for train/test overlap.
