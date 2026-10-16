# Predicting where energy localizes in a sine-Gordon pendulum ring: a physics-guided v2

## Summary

A ring of 100 coupled pendulums, released in a nearly uniform swing, suddenly concentrates its energy on one pendulum after about 57 time units on average (about 8 s for 20 cm pendulums). The published model, v1, is a 26.5M-parameter CNN-BiLSTM that predicts which pendulum from the first 4.5 time units.

This v2 study rebuilds the problem from its physics, the Floquet growth spectrum of the uniform swing. Each design question gets a physics-based hypothesis, tested by ablations: about 600 training runs, 44 GPU-hours on a laptop. Negative results are listed alongside the positive ones.

### How v2 locates the hotspot

1. **Watch** the angle of every pendulum (every 2nd is enough) in 10 frames over the first 4.5 time units after release: one camera at 30–60 frames per second.
2. **Read** the frames with a whole-ring dilated circular CNN: 7 convolutions with kernel 3, dilations 1-2-4-8-16-32-1 and width 32, so every pendulum's features see the whole ring. The same filters apply at every pendulum, and a small head gives one score per pendulum (22,786 parameters).
3. **Answer** with the most likely pendulum from the resulting probability over the ring. Optionally add a second site when the model gives a rival hump 5% or more, or name the smallest set of sites holding 90% of the probability. A second head predicts when localization happens.
4. **Target.** For the comparison with v1, the label is v1's: the first pendulum whose instantaneous energy passes 10% of the total. The recommended target is the hump label: at the first moment the swing-averaged energy of 3 neighbouring pendulums passes 15% of the total, the centre of those 3. It fires as early as v1's label and is predicted far better.

### v1, and what this work corrects

- **v1's published accuracy could not be reproduced.** Rebuilt from the paper, v1 reaches 60.4% within ±1 pendulum, not 94.4%, and one of three seeds never trained.
  - The label explains part of the gap. Which pendulum first passes 10% of the energy jitters by 1–3 sites within a swing; on a swing-averaged label the rebuilt v1 reaches 83.6%, about two-thirds of the way to 94.4%.
  - Without the original code this is a consistency check, not proof. Train/test overlap, or a detail the paper does not specify, could explain the rest.
- **The paper's code and text disagree in places.**
  - The global coupling is h̃ = 0.5 in all code, not 0.05 as printed; the Floquet analysis confirms 0.5.
  - The figures' time axes are sample indices, so localization happens at τ ≈ 57, not 100.
  - Each spring's energy is booked to one neighbour, which moves the labelled site in 21% of runs.
  - The input scaler was fit before the train/test split (minor leakage).
- **A vectorized simulator** gives the paper pipeline's exact labels on 1,000 of 1,000 runs, about 100 times faster.

### Physics of the instability

- **The uniform swing becomes unstable at A_c = 1.577.** At A = 1.75 only the five longest waves grow, the fastest by a factor of e every 8.4 time units, so v1's input window sees at most about 2× growth.
- **The best smoothing width is the Floquet band edge**, K* = N/m_c. It matches measurements at seven amplitudes, including the paper's empirical 17 at A = 1.75.
- **The outcome is deterministic.** 10⁻⁶ rad of initial error never changes the site, so the limit is measurement precision, not chaos; 10⁻³ rad moves a quarter of the labels. The coarse answer lives in the five growing modes, and local detail decides the last site or two.
- **Physics alone gets far.** A training-free Floquet filter matches the rebuilt v1 (3.12 vs 3.14 mean distance), and a 200-parameter learned linear ring filter beats it (2.17).

### Architecture (Q2)

- **Each pendulum must see the whole ring.** Widening the receptive field from 9 to 129 pendulums takes the error from 4.79 to 1.54. In the final model, losing it costs 1.1 sites, losing circular padding 0.9, and losing the one-score-per-site head 0.3.
- **The ring's symmetry saves data.** The symmetric model trained on 1,000 runs beats a v1-style dense head trained on 60,000.
- **Small models suffice.** Models from 2k to 115k parameters land within 1.68–2.01.
- **What did not pay off:**
  - A recurrent network over each pendulum's frames helps a small CNN, but the whole-ring CNN ties it without one, so v1's LSTM is unnecessary.
  - A global Fourier branch beats a local CNN on its own, but adds nothing to the dilated stack.
  - Feeding the Floquet-evolved field as an input does not help, even with little data.

### Inputs and sensing (Q1)

- **Angles are the best input.** Energies never beat them, and they need velocities and the model's parameters to compute. The initial angles alone hold the full state, but a few frames make it easier to learn.
- **An unknown start point in the swing** needs about one full swing of angle frames, or one frame of energies.
- **Noise first causes far misses**, a wrong hump rather than a one-site slip. A camera is fine up to about 10⁻³ rad when watching from release.
- **Waiting divides sensor noise by the growth of the unstable modes.**
  - Each 10× worse sensor costs about 25 time units of waiting, on either label, as the mean growth rate of the five growing modes predicts (24.8). The fastest mode alone would predict 19.5.
  - Single-frame setups gain less from waiting (30–34 time units per 10×).
- **Sparse sensors:** every 2nd pendulum costs nothing and every 4th a little. Every 8th fails even though it satisfies Nyquist, because the stable modes alias onto the growing ones.
- **Without noise, waiting helps only on v1's jittery label**; where the energy settles can already be read at release.

### Loss (Q3) and optimizer (Q4)

- **On v1's label, soft labels win** by hedging between the sites the label jitters between: within ±1 rises from 65% to 77%. On the swing-averaged label they stop mattering, and on the hump label hard labels win.
- **Circular regression does worst**, because it averages rival humps.
- **An expected-distance term and v1's "localizes" head add nothing**; a time head helps slightly.
- **No optimizer beats AdamW + cosine** beyond seed noise, and v1's constant-rate Adam does as well. SGD failed to train at the one learning rate tried, batch 1024 trains twice as fast at equal accuracy, and weight decay makes no difference.

### The final model (Q5)

- **v2 halves v1's error with 1/1,163 of the parameters.**
  - Mean distance 1.49 vs 3.14; within ±1, 77% vs 60%; within ±3, 97% vs 90%; more than 10 sites off, 1.5% vs 5.7%.
  - It trains in 6 minutes, about 4 times faster, with six times less spread between seeds.
- **It beats v1 with 60 times less data** (2.42 from 1,000 runs). More data helps only a little past 60,000 runs (1.44 at 100,000).
- **It predicts when localization happens** to a 1.5% median error. But the amplitude alone gives 5.0%, and linear Floquet clocks only 4.1%, so the timing is mostly an amplitude trend; it comes for free.
- **Its probabilities give prediction sets.** About 5.6 sites contain the true site 98% of the time; the sets are conservative.

### The target: one pendulum or a hump

- **The hotspot is a hump about 8 pendulums wide.** It stays put for about a swing, then drifts, and about 1 run in 6 has a rival hump with at least half its energy.
- **Most of v2's error on v1's label is the label's own jitter.**
  - On the swing-averaged label, v2 reaches 0.58 (96.4% within ±1).
  - Splitting each spring evenly without averaging over a swing does not help.
- **The hump label is the best target:** 0.27, with 99.4% within ±1. It fires as early as v1's label.
  - Reading it when the hump forms and locating it by a 3-pendulum centre each gain about half of its edge over the swing-averaged label.
  - Adding a second site in the 2.3% of runs where the model sees a rival raises within ±1 to 99.8%.
- **The same design choices win on the cleaner labels**, often by more. Seeing the whole ring and using angle frames matter more there: v1's label adds 0.6–1.0 sites of jitter error to every model, which hides part of each difference.

### Regularization

- **At 60,000 runs v2 barely overfits** (train 1.35, test 1.49). The ring symmetry does the job: v1 (1.62 vs 3.00) and a v1-style dense head (1.06 vs 1.80) overfit.
- **With little data it does, and the usual penalties fail.** Weight decay up to 10⁻¹ changes nothing, and dropout makes every size worse.
- **What works with little data:**
  - Mirror images of each run, the one ring symmetry the network does not build in, are worth about twice the data.
  - Shorter schedules help, and weight decay 1 helps once mirror images slow the overfitting.
  - Combined, the remedies are worth 2–4 times the data: 1k runs 2.42 → 2.04, 3k 2.07 → 1.75, 10k 1.80 → 1.63.
  - At 60,000 runs mirror images add nothing.

### Robustness

- **Damage follows the Floquet spectrum.**
  - A 10% error in the neighbour coupling, or twice the initial spread, leaves the spectrum alone and costs v2 about half a site.
  - A 10% error in the global coupling changes the growth rates (+21% / −33%) and costs both models 3–4 sites.
- **Other chain lengths:** v2 handles 80 pendulums (1.31) without retraining, but degrades at 150 (6.3). v1 cannot take other lengths at all.
- **Sensor noise:** even untrained for it, v2 at 10⁻³ rad (2.26) beats the noise-free v1 (3.14); trained with the noise, it reaches 1.72.
- **Impurities:** one pendulum 1% weaker pins the hotspot and defeats both models trained on uniform chains (mean distance 19–23). Trained with impurities, v2 finds the weak pendulum from the motion alone (1.06), and a measured map of the chain adds nothing.
- **Damping:**
  - Nearly every release localizes only if the chain's quality factor is about 1,000 or more; at Q = 330, 71% do, and at Q = 100 none do. The Floquet growth rate predicts the delay.
  - Trained on damped chains, v2 still warns early (0.84 at Q = 330). A "localizes" head says from the first 4.5 time units whether a run will localize at all (99.5% right).

### For a real chain

- **One camera tracking every 2nd pendulum to 10⁻³ rad, predicting right after release**, scores 1.93, against 3.14 for v1 with perfect data. On the hump label it is within ±1 of where the hump forms in 96.8% of runs, and in 98.3% when it may name a second site.
- **A coarser camera should start watching about 11·ln(σ_s / 10⁻³) time units later**, where σ_s is its noise. A 10⁻² rad camera catches up when it starts at τ ≈ 25–26, leaving about 3.6 s of warning for a 20 cm pendulum.
- **Energies at τ = 30** are the most accurate on v1's label (1.24), but they need gyroscopes, and their edge vanishes on the cleaner labels.
- **Still open** (Sec. 12): a real chain's damping law, small differences at every pendulum, a range of the global coupling, and a bench-top test.

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
- **The trade-off follows the Floquet growth of the unstable modes** ([Fig. 13](../figures/fig13_noise_collapse.png), [Fig. 13b](../figures/fig13b_noise_avg_label.png)):
  - For the camera, the error depends on noise σ_s and start time τ only through σ_s·e^{−kτ}, the noise relative to the grown signal.
  - The rate k is about the mean growth rate of the growing modes, 0.09–0.10, not the fastest one's σ₁ = 0.118. So each 10× worse sensor costs about 23–26 time units of waiting, not ln 10 / σ₁ = 19.5. Phase 3's cleanest cells suggested about 20; a fit of all of them gives 26.5.
  - Split by swing amplitude, the rate rises as the growth rates do.
- **The label does not create the trade-off.** Rerun on the swing-averaged label with the same inputs and seeds (Phase 3 addendum), the camera's waiting cost is 25.5 time units per 10× noise (95% CI 23–29), against 26.5 on v1's label. Noise adds about the same error on both labels (+0.7 sites at 10⁻³ rad from release), which doubles the error on the averaged label's lower floor.
- **Single-frame setups (angles + velocities, or energies) degrade faster with noise** than the 8-frame camera window, and gain less from waiting (30–34 time units per 10× noise, on either label).
- **Waiting also helps without noise.** Energies at τ = 30 reach 1.24 (82% within ±1), the best Phase 3 result, because less is left to extrapolate. Most of that gain belongs to v1's label: on the swing-averaged label, waiting until τ = 30 gains only 0.03 (Sec. 10).
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
- With a coarser one, start about 11·ln(σ_s / 10⁻³) time units later (25 per 10× noise), once the long-wave modes have grown above the sensor noise. At the full budget a 10⁻² rad camera catches up with the 10⁻³ rad one from release when it starts at τ ≈ 25–26, on either label (Sec. 11).

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
- **v1's optimizer and v1's inputs cost little** (≤ 0.07) once the architecture is right. In clean data, every input set carries the same initial state; inputs matter for real sensors (Sec. 7) and on the swing-averaged label (below).

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
- **With little data, v2 does overfit.** At 1,000 and 3,000 runs the best checkpoint is the first one evaluated (step 235 of 7,050), and the validation error then rises by about 50%. At 10,000 runs the best checkpoint comes at steps 705–1,880.

Phase 8 tries the usual remedies with little data ([Fig. 21](../figures/fig21_regularization.png); [results/phase8_regularization.md](../results/phase8_regularization.md); final recipe, 7,050 steps unless a schedule is named; test pool; 3 seeds, 2 for the first row, weight decay ≤ 10⁻¹ and dropout; every row below the first checkpoints every 47 steps instead of every 235):

| Training runs | 1k | 3k | 10k |
|---|---|---|---|
| Final recipe (Fig. 15) | 2.42 | 2.07 | 1.80 |
| Checkpoint every 47 steps | 2.30 | 2.06 | 1.79 |
| + weight decay 10⁻² / 10⁻¹ | 2.30 / 2.31 | 2.10 / 2.10 | 1.79 / 1.77 |
| + weight decay 1 | 2.32 | 2.04 | 1.64 |
| + dropout 0.1 / 0.2 | 2.66 / 2.68 | 2.14 / 2.18 | 1.93 / 1.89 |
| + mirror images | 2.18 | 1.88 | 1.65 |
| A shorter schedule: 235 steps (1k, 3k), 705 (10k) | 2.26 | 1.92 | 1.70 |
| Mirror images + weight decay 1 | 2.16 | **1.75** | **1.63** |
| Mirror images + the shorter schedule | **2.04** | 1.89 | 1.65 |

(Bold: the best on the validation pool at each size.)

- **Penalties do little.**
  - AdamW shrinks the weights by the learning rate × the weight decay at each step, and at 1,000 runs the model overfits within about 150 steps, before the warmup ends. So weight decay up to 10⁻¹ changes nothing at any size.
  - Weight decay 1 helps only at 10,000 runs, where the best checkpoint comes late enough (steps 2,900–4,800) for the decay to act. There it cuts the train–test gap from 0.37 to 0.13, as small as at 60,000 runs.
  - Dropout makes every size worse (+0.1 to +0.4). A likely reason: the site depends on small differences in the initial state (Sec. 3), and dropout's random masks blur them.
- **The mirror images are worth about twice the data** (2.0–2.4 times, read off Fig. 15's curve).
  - v2 builds in the ring's rotations (the same filters at every pendulum) but not its reflection: its filters are not mirror-symmetric, and v1's label is one-sided. Reflecting a run (pendulum n → 99 − n) gives a new valid run, labeled with the mirror of the stored left-bond label.
  - They also slow the overfitting: at 10,000 runs the best checkpoint moves from steps 750–1,550 to 3,950–4,700.
  - At 60,000 runs they change nothing (1.50 vs 1.50, 3 seeds each).
- **Stopping earlier helps once the learning rate comes down with it.** Finer checkpoints recover 0.12 at 1,000 runs and nothing at larger sizes. A short cosine schedule, which has annealed by the time the model would overfit, gains up to 0.14. Its last checkpoint is within 1–2% of its best, so it needs no validation set to stop.
- **Together, the remedies are worth 2–4 times the data.**
  - At 3,000 and 10,000 runs the best is mirror images + weight decay 1, which work together. The images slow the overfitting enough for the decay to act. At 3,000 runs the best checkpoint moves from steps 560–750 to 4,040–4,470, and the error falls to 1.75, worth 4.3 times the data, where weight decay alone gains nothing.
  - At 1,000 runs the model overfits too fast even with the images, and the short schedule works better: 2.04 with mirror images, worth 3.6 times the data.
- **The lesson from 60,000 runs holds with little data.** A structural fix beats a penalty, here the one symmetry the architecture leaves out, and more data beats both.
  - Checkpoints were chosen on 5,000 validation runs at every size, more than the training set at 1,000–3,000 runs. With a smaller validation set, the short schedule is the safer choice, since it needs no selection.
  - Small data matters only when runs are expensive, for example when training or fine-tuning on a real chain.

### Data efficiency

([Fig. 15](../figures/fig15_final_data_efficiency.png); final recipe, 2 seeds per size, 7,050 gradient steps at every size; Phase 8 rows, 3 seeds)

| Training runs | 1k | 3k | 10k | 30k | 60k | 100k |
|---|---|---|---|---|---|---|
| v2 mean distance | 2.42 | 2.07 | 1.80 | 1.56 | 1.49 | 1.44 |
| + mirror images (Phase 8) | 2.18 | 1.88 | 1.65 | | 1.50 | |
| Best regularization at each size (Phase 8, above) | 2.04 | 1.75 | 1.63 | | | |

- Trained on 1,000 runs, v2 (2.42) already beats v1 trained on 60,000 (3.14): 60 times less data. Regularized for little data, it reaches 2.04.
- Beyond 60,000 runs the gains are small (1.44 at 100,000). At 60,000 runs the mirror images add nothing.

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
- For applications, a label averaged over a swing is the better target. The hump label (below) does best.

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
1. **Predict the hump's centre when it forms**, not the first pendulum over 10%. The swing-averaged label above comes close (96.4% within ±1), but it is read about 5 time units after the hump forms (below).
2. **Keep the output a probability over all pendulums** rather than one regressed position. Regression averages rival humps (circular regression did worst, Sec. 8), while a distribution keeps both and gives prediction sets (below).
3. **Optionally predict the hump's shape** as a per-site energy profile. The profile head tied for best in Sec. 8, and it gives the width and any rival hump for free.
4. **Score by ring distance and calibration** (mean distance, ±1 / ±3, coverage of prediction sets), not by exact-site accuracy.
5. **Define the event at the hump's scale**, for example when the swing-averaged energy of 3 neighbouring pendulums first passes a threshold. Tested below: 99.4% within ±1.

### The event at the hump's scale, and a second site

Phase 9 tests recommendation 5 and the two-site output. The hump label fires at the first sample where the swing-averaged energy of 3 neighbouring pendulums passes 15% of the total, and names the window's centre. The threshold makes it fire with v1's label (median τ 54.5 against 55), about 5 time units before the swing-averaged label. Same final recipe; only the target changes (test pool; seeds in brackets; [results/phase9_tables.md](../results/phase9_tables.md)):

| Target | Fires at (median τ) | Mean distance | Exact | ±1 | >10 off |
|---|---|---|---|---|---|
| v1's label [5] | 55 | 1.49 ± 0.03 | 27.7% | 77.2% | 1.5% |
| swing-averaged label [3] | 60 | 0.58 ± 0.01 | 70.5% | 96.4% | 0.7% |
| **hump label** [3] | **54.5** | **0.27 ± 0.01** | **84.7%** | **99.4%** | **0.4%** |

- **The hump label is the best target.** It halves the error of the swing-averaged label and warns as early as v1's label. Its time is predicted best too (1.1% median error, against 1.5% on v1's label).

**Why it is easier.** It is read when the hump forms, and it locates the hump by the centre of 3 pendulums. Phase 10 separates the two with a control label, `hump_top`, which fires with the hump label but names the top pendulum, as the swing-averaged label does ([results/phase10_tables.md](../results/phase10_tables.md), [results/hump_label.json](../results/hump_label.json)):

| Label [3 seeds] | Read | Located by | Mean distance | Exact | ±1 | >10 off |
|---|---|---|---|---|---|---|
| swing-averaged | about 5 time units after the hump forms | top pendulum | 0.58 | 70.5% | 96.4% | 0.69% |
| `hump_top` | when the hump forms | top pendulum | 0.43 | 76.2% | 98.9% | 0.57% |
| hump | when the hump forms | centre of 3 pendulums | 0.27 | 84.7% | 99.4% | 0.36% |

- **Reading it when the hump forms** gains 0.15, and most of the ±1 gain (2.5 of 3.0 points).
  - In the 5 time units before the swing-averaged label fires, the hump's top moves 2 or more sites in 8% of runs.
  - On those runs the averaged-label model is within ±1 only 70% of the time (99% elsewhere), and 3 in 4 of its misses point at where the hump formed.
- **Locating it by the centre of 3 pendulums** gains another 0.16, mostly in exact hits (+8.5 points) and far misses.
  - The top pendulum flips between neighbours when the hump's top is flat, and in 0.3% of runs it sits in another hump than the 3-pendulum centre. So it is a noisy marker: the model trained on `hump_top` scores better on the hump label (0.33) than on its own.
- **The first 4.5 time units decide where the hump forms.** Even the model trained on v1's label is within ±1 of the hump label in 91% of runs, against 77% of its own label.

**A second site** is the highest peak of the predicted distribution at least 5 sites from the most likely one ([results/top2.json](../results/top2.json)). The rival-hump test above, applied to all 20,000 test runs, flags 16% of them (18% in its 2,000-run sample).

| Model (its own label) | ±1, one site | ±1, either of two | Rival-hump runs: one → two | Second site only at ≥ 5% probability: runs with two, ±1 |
|---|---|---|---|---|
| v1's label | 77.2% | 78.8% | 73.6% → 82.4% | 4.9%, 78.5% |
| swing-averaged label | 96.4% | 97.1% | 90.5% → 94.2% | 2.7%, 96.9% |
| hump label | 99.4% | 99.8% | 97.1% → 99.4% | 2.3%, 99.8% |

- **A second site helps only where a rival hump competes:** 2–9 points on those runs, at most 0.2 elsewhere. A second site drawn at random adds at most 0.4, so the gain comes from the model's own second peak, not from luck.
- **The model knows when to name it.** A second site only when the 3 sites around it hold at least 5% of the probability goes to 2–5% of runs and keeps 75–97% of the gain.
- **Recommended output:** the hump label's site, plus a second site when the model gives it 5% or more. One of them is within ±1 in 99.8% of runs, and only 2.3% of runs get two sites.

### The same choices win on the swing-averaged label

Every choice in Secs. 6–9 was made on v1's jittery label. Phase 7 repeats the key ones on the swing-averaged label. Each run is the final recipe at the full budget with one choice changed, beside a matched control on v1's label ([Fig. 19](../figures/fig19_avg_label_ablations.png); 3 seeds; test pool; Δ against the final recipe on the same label, ± its standard error):

| Final recipe, but … | v1's label | Δ | Averaged label | Δ | ±1, v1's / averaged |
|---|---|---|---|---|---|
| nothing (the final recipe, seeds 0–2) | 1.50 | | 0.58 | | 76.9% / 96.4% |
| v1's inputs (energy frames + θ(0)) | 1.51 | +0.01 ± 0.02 | 0.62 | +0.04 ± 0.01 | 76.4% / 95.7% |
| θ(0) only | 1.52 | +0.01 ± 0.02 | 0.68 | +0.11 ± 0.02 | 74.5% / 95.0% |
| sees 33 sites, not the whole ring | 1.70 | +0.19 ± 0.02 | 0.70 | +0.12 ± 0.02 | 70.9% / 93.8% |
| sees 17 sites | 2.37 | +0.87 ± 0.02 | 1.77 | +1.20 ± 0.01 | 61.2% / 78.2% |
| hard labels (other loss terms kept) | 1.53 | +0.03 ± 0.02 | 0.59 | +0.01 ± 0.01 | 65.3% / 96.2% |

- **Seeing the whole ring matters even more.** With 17 sites the error triples on the averaged label, against ×1.6 on v1's label. The growing modes are the five longest waves, whichever label marks the hotspot.
- **The inputs now matter.**
  - On v1's label the three input sets tie.
  - On the averaged label, angle frames beat v1's inputs by 8% of the error, and θ(0) alone by 19%.
  - θ(0) holds the whole initial state, but the frames make it easier to learn (Sec. 7).
- **Soft labels stop mattering.** On v1's label they raise ±1 from 65% to 77%, by hedging between the sites the label jitters between. On the averaged label, hard labels do as well (96.2% vs 96.4% within ±1).
- **v1's label understates the choices.** Its jitter adds 0.6–1.0 sites of error to every model, which hides part of each difference.

The recommended setups (Sec. 11) keep their order:

| Setup (final recipe) | v1's label | Averaged label | ±1, v1's / averaged |
|---|---|---|---|
| Every pendulum, noise-free, τ = 0…4.5 | 1.50 | 0.58 | 76.9% / 96.4% |
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 | 1.18 | 72.6% / 92.9% |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 20 | 2.05 | 1.27 | 72.6% / 91.2% |
| Energies at τ = 30, noise-free | 1.24 | 0.55 | 82.4% / 96.3% |

- **The cheap camera is within ±1 of where the energy settles in 93% of runs**, predicting right after release.
- **Noise doubles the error on the averaged label**, against ×1.3–1.4 on v1's label. Noise picks the wrong hump (Sec. 7), and no label removes those misses.
- **Without noise, waiting no longer pays.** Energies at τ = 30 beat the final recipe by 0.27 on v1's label, but by only 0.03 ± 0.01 on the averaged label. Watching later mostly helps predict the moment in the swing at which v1's label fires; where the energy settles can already be read at release.

### The key choices and setups on the hump label

Phase 10 repeats Phase 7 on the hump label: the final recipe at the full budget, 3 seeds, with Phase 7's runs as the controls ([results/phase10_tables.md](../results/phase10_tables.md); Δ against the final recipe on the same label, ± its standard error):

| Final recipe, but … | Averaged label | Δ | Hump label | Δ | ±1, averaged / hump |
|---|---|---|---|---|---|
| nothing | 0.58 | | 0.27 | | 96.4% / 99.4% |
| v1's inputs (energy frames + θ(0)) | 0.62 | +0.04 ± 0.01 | 0.30 | +0.03 ± 0.01 | 95.7% / 99.3% |
| θ(0) only | 0.68 | +0.11 ± 0.02 | 0.38 | +0.11 ± 0.02 | 95.0% / 98.9% |
| sees 33 sites | 0.70 | +0.12 ± 0.02 | 0.42 | +0.15 ± 0.01 | 93.8% / 97.8% |
| sees 17 sites | 1.77 | +1.20 ± 0.01 | 1.48 | +1.21 ± 0.01 | 78.2% / 83.1% |
| hard labels (other loss terms kept) | 0.59 | +0.01 ± 0.01 | **0.25** | **−0.03 ± 0.01** | 96.2% / **99.5%** |

- **Each choice costs about the same number of sites on both labels**, so a larger share of the hump label's lower error. What is hard is reading the initial state, whichever label marks the hump. Seeing the whole ring is worth 1.2 sites on both (×5.4 on the hump label).
- **Hard labels now win** (88.3% exact, against 84.7%). Soft labels hedged between the sites v1's label jitters between. The hump label does not jitter, so the hedge only blurs the answer. On this label, use hard labels.

| Setup (final recipe) | v1's label | Averaged | Hump | ±1, hump: one site | Two sites | Second site at ≥ 5% only (runs with two) |
|---|---|---|---|---|---|---|
| Every pendulum, noise-free, τ = 0…4.5 | 1.50 | 0.58 | 0.27 | 99.4% | 99.8% | 99.8% (2.3%) |
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 | 1.18 | 0.92 | 96.8% | 98.8% | 98.3% (5.6%) |
| Coarse camera (10⁻² rad), from τ = 20 | 2.05 | 1.27 | 1.06 | 94.9% | 96.8% | 96.2% (4.6%) |
| Energies at τ = 30, noise-free | 1.24 | 0.55 | 0.26 | 99.4% | 99.8% | 99.7% (1.8%) |

- **The cheap camera is within ±1 of where the hump forms in 96.8% of runs**, from release. With a second site in 5.6% of runs, 98.3% ([results/top2_deploy.json](../results/top2_deploy.json)).
- **Noise costs about the same number of sites on both labels** (+0.65 and +0.79 here, +0.60 and +0.69 on the averaged label). Its far misses (2% of runs more than 10 sites off) are the wrong hump, whatever the label.
  - Those are the misses a second site catches. On rival-hump runs the two cameras go from 86–87% to 96.5–97.4% within ±1 with two sites.
- **Without noise, waiting does not pay on this label either** (energies at τ = 30: 0.26 against 0.27).

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

### Damped chains

(Linear damping −γ dθ/dτ, so a free pendulum's quality factor is Q = 1/γ. Six pools of 5,000 runs share their initial conditions and differ only in γ. Sites are scored over the runs that localize by τ = 150. [Fig. 20](../figures/fig20_damping.png); [results/headline.md](../results/headline.md))

| Q (γ) | Swing's 1/e decay, 20 cm pendulum | Localize by τ = 150 | Median delay | v1 rebuilt, trained undamped | v2 final, trained undamped | v2 recipe, trained on damped chains |
|---|---|---|---|---|---|---|
| ∞ (0) | never | 100% | – | 3.22 | 1.56 | 1.55 |
| 10,000 (10⁻⁴) | 48 min | 100% | 0 | – | 1.49 | 1.53 |
| 3,300 (3×10⁻⁴) | 16 min | 100% | +1 | – | 1.49 | 1.44 |
| 1,000 (10⁻³) | 4.8 min | 99.9% | +3 | 5.43 | 2.48 | 1.19 |
| 330 (3×10⁻³) | 95 s | 71% | +9 | 12.4 | 4.92 | 0.84 |
| 100 (10⁻²) | 29 s | 0% | – | – | – | – |

- **Damping decides whether the energy localizes at all.**
  - The uniform swing decays toward A_c, and a run localizes only if its perturbation has grown enough before then.
  - At Q ≥ 1,000 almost every run localizes, a few time units late. At Q = 330 only swings released above A ≈ 1.71 do. At Q = 100 none do: the swing falls below A_c within 18 time units (2.5 s).
  - A run that misses τ = 150 does not localize later (checked to τ = 600).
- **The Floquet spectrum predicts the delay.** Undamped runs follow T₀ = 5.1/σ(A₀) + 11: about 5 e-folds of linear growth, then about 11 time units of nonlinear focusing.
  - Integrating the growth rate along the decaying swing predicts the median delay: +3.8 vs +3.0 measured at Q = 1,000, and +9.3 vs +9.0 at Q = 330.
  - It predicts too few survivors at Q = 330 (41% vs 71%). 42% of the survivors finish localizing after the swing has dropped below A_c, so a perturbation that is already large keeps focusing without the linear instability.
- **Trained undamped, v2 tolerates Q ≥ 3,000 at no cost.**
  - At Q = 1,000 its error grows to 2.48, still well ahead of v1 (5.43).
  - Most of that comes from the inputs, not the outcome. Fed the same runs' undamped frames, v2 scores 1.68 against the damped outcome. A slowly decaying swing looks unlike any training run.
- **Trained on damped chains (Q from 100 to 10,000), early warning survives, and gets better with damping.**
  - The retrained recipe loses nothing on undamped chains (1.55 vs 1.56). It reaches 1.19 at Q = 1,000 and 0.84 at Q = 330 (92% within ±1).
  - Near the threshold fewer and longer ring modes grow (4–6 when the run localizes undamped, 0–3 at Q = 330), so the site depends less on local detail.
  - A "does it localize" head tells from the first 4.5 time units whether a run will localize at all. On the damped test pool it is right 99.5% of the time (false alarms on 1.4% of the runs that do not localize), and it costs nothing on the site.
  - On the swing-averaged label the retrained recipe reaches 0.50 (97% within ±1) on the damped test pool.
  - The recommended camera (every 2nd pendulum, 10⁻³ rad) scores 1.82 on the damped test pool, against 1.93 on undamped chains.

### The recommended setups, at full budget

([results/headline.md](../results/headline.md) and, for the later coarse-camera starts, [results/deploy_start_tables.md](../results/deploy_start_tables.md); median warning is the median localization time, τ = 55, minus the end of the observation; seconds for a 20 cm pendulum)

| Setup | Mean distance | ±1 | ±3 | Median warning |
|---|---|---|---|---|
| v2 final: every pendulum, noise-free, τ = 0…4.5 | 1.49 | 77.2% | 97.3% | 50.5 (7.2 s) |
| Camera on every pendulum, 10⁻³ rad, from release | 1.72 | 73.3% | 96.8% | 50.5 (7.2 s) |
| Camera on every 2nd pendulum, 10⁻³ rad, from release | 1.93 | 72.6% | 95.9% | 50.5 (7.2 s) |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 20 | 2.05 | 72.6% | 95.2% | 30.5 (4.4 s) |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 24 | 1.77 | 76.2% | 95.2% | 26.5 (3.8 s) |
| Coarse camera (10⁻² rad) on every pendulum, from τ = 28 | 1.67 | 76.2% | 96.2% | 22.5 (3.2 s) |
| Energies at τ = 30, noise-free (needs gyroscopes) | 1.24 | 82.4% | 95.6% | 25 (3.6 s) |
| v1 rebuilt: every pendulum, noise-free, from release | 3.14 | 60.4% | 90.3% | 50.5 (7.2 s) |

- **A realistic, cheap setup works.** One camera tracking every 2nd pendulum to 10⁻³ rad predicts right after release with a mean distance of 1.93, well ahead of the noise-free v1 (3.14).
- **On the hump label** (Sec. 10) the same camera is within ±1 of where the hump forms in 96.8% of runs, and in 98.3% when it may name a second site (5.6% of runs).
- **A ten times coarser camera** gets nearly the same accuracy (2.05) by starting at τ = 20, and still leaves about 4 s of warning ([Fig. 17](../figures/fig17_warning_tradeoff.png)).
  - To match a 10⁻³ rad camera on every pendulum (1.72), it must wait until τ ≈ 25–26 (1.77 from τ = 24, 1.67 from τ = 28; 3 seeds). That leaves about 3.6 s of warning.
  - On the swing-averaged label it catches up at τ ≈ 25 too (1.05 from τ = 24 and 0.84 from τ = 28, against 0.99), as the start-time rule of Sec. 7 predicts.
- **Energies at τ = 30** give the best accuracy (1.24), but they need gyroscopes and leave less warning. On the swing-averaged label their edge nearly vanishes (0.55 vs 0.58, Sec. 10).

## 12. Limitations and next steps

### Limitations

- **The v1 comparison is against a rebuild.** The original notebook and data are not available. The rebuild reaches 60.4% within ±1 (83.6% on a swing-averaged label), not the published 94.4%. If the published pipeline differs in a way that matters, the real gap between v1 and v2 could be smaller than reported here. The physics baselines (Floquet filter 3.12, learned linear filter 2.17) do not depend on the rebuild, and v2 beats them too.
- **Damping is simulated as linear and the same at every pendulum.** Real pendulums lose energy to bearing friction, which is partly a constant (Coulomb) torque, and to air drag, which grows with the square of the speed. No two pendulums lose energy at quite the same rate.
  - With linear damping, nearly every release (A = 1.65–1.85) localizes only if the chain's quality factor is about 1,000 or more. For 20 cm pendulums, that means the swing takes at least 5 min to decay by 1/e (Sec. 11).
  - At Q ≈ 300, 29% of releases never localize, and at Q = 100 none do.
  - Trained with damping, v2 still warns early and says whether a run will localize. But a real chain's Q and damping law must be measured first, and a chain with much lower Q would need a drive to replace the lost energy.
- **Idealized sensor noise.** Noise was white and independent across pendulums and frames. Real cameras add correlated errors (lens distortion, motion blur, a fixed calibration offset per pendulum). A fixed offset acts like a wrong initial state, which the dynamics amplify.
- **Manufacturing tolerances.** The impurity tests detune one pendulum per chain, and a single 1% impurity already pins the site within ±3 in 92% of runs. A model must therefore be trained with the tolerances it will meet. For one impurity the motion alone reveals it (Sec. 11). A real chain has small differences at every pendulum, which may make a measured map of the chain necessary.
- **Screening budget and seeds.** Phases 3–5 used 15-epoch runs, and the noise/time grid one seed per cell. Differences below about 0.1 sites in those phases are within noise.
- **The label.** v1's label (the first pendulum over 10%) jitters by 1–3 sites within a swing (Sec. 10). Exact-site and ±1 accuracy on it partly measure the moment in the swing rather than where the energy goes. Labels averaged over a swing are better targets, and the hump label, read when the hump forms, is the best (Sec. 10). The inputs, receptive field, soft labels and recommended setups were re-tested on the swing-averaged and hump labels (Sec. 10), and the noise-and-timing grid on the swing-averaged label (Sec. 7). The sparse sensors and the optimizer were not.

### Next steps

1. Measure a real chain's quality factor and damping law (linear, Coulomb or quadratic), and retrain with the measured law. Linear damping is done (Sec. 11): trained with damping, v2 warns early wherever the energy still localizes, but below Q ≈ 1,000 a growing share of releases never localize.
2. Build a bench-top chain and test the recommended setup: one camera, every 2nd pendulum, a short window of frames after release.
3. Calibrate the prediction sets (Sec. 10), which are conservative, for example with a temperature fitted on the validation pool.
4. Train with small differences at every pendulum, plus a measured map of each pendulum's frequency.
5. Train over a range of the global coupling h, which moves the growth spectrum (Sec. 11), or estimate it from the chain first.
6. The noise-and-timing grid on the swing-averaged label is done (Sec. 7): waiting costs about 25 time units per 10× noise on either label, at about the mean growth rate of the growing modes. Still open: predict the rate from each mode's share of the localization instead of a plain mean over the growing modes, and repeat the grid with more than one seed per cell.
7. The key choices and sensing setups on the hump label are done (Sec. 10). Each choice costs about the same number of sites as on the swing-averaged label, and hard labels beat soft ones there. A second site lifts the cheap camera from 96.8% to 98.3% within ±1, named in 5.6% of runs. Still open: the noise-and-timing and sparse-sensor grids on the hump label.
8. If the original v1 notebook resurfaces, rerun the comparison against it, check its labels, and check its data for train/test overlap.
