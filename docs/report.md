# Predicting where energy localizes in a sine-Gordon pendulum ring: a physics-guided v2

*Draft. Sections 7–11 are filled in as their phases finish.*

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

**All v1 vs v2 comparisons below use the rebuilt v1 on the same test pool.**

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
- about 5 minutes to train on a laptop GPU

Combining it with the Fourier branch, the MLP front or the GRU front, or doubling its width, does not improve it (1.57–1.61).
