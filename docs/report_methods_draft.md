# Methods (draft for the v2 report)

## Model system

We simulate the discrete sine-Gordon ring with a global term, Eq. (4) of the paper:

```
θ̈ₙ = −sin θₙ + g̃ (θₙ₊₁ + θₙ₋₁ − 2θₙ) + h̃ (θ̄ − θₙ)
```

- **Parameters:** N = 100 pendulums, periodic boundaries, g̃ = 0.75 and h̃ = 0.5. The paper's code uses h̃ = 0.5; its Sec. 2.1 says 0.05, which is a typo.
- **Initial conditions:** each run starts at rest with θₙ(0) = A + ηₙ.
  - A ~ U(1.65, 1.85) is drawn once per run.
  - ηₙ ~ U(−0.025, 0.025) is drawn independently for every pendulum.
- **Time:** dimensionless throughout (τ = ω₀t).

## Simulator

**Integrator.**
- Fourth-order Runge–Kutta, Δτ = 0.05, vectorized across thousands of runs at once (`sgloc/sim.py`).
- Total energy drifts by 3.4×10⁻⁶ over τ = 150, and the drift falls 33× when the step is halved.
- Halving the step changes no label on 1,000 of 1,000 test runs.

**Agreement with the paper's own pipeline.** We reran 1,000 test runs through the paper's pipeline: its `odeint` setup, energy formula and first-crossing loop. It gives exactly the same localization site and time as the new simulator on every run.

**Speed.** 140,000 runs take about 10 minutes on 10 CPU cores, against hours for one `odeint` call per run.

## Localization labels

A run is labeled at the first sample (Δτ = 0.5) where one site holds more than 10% of the total energy. The label is that site.

- **Primary label:** the paper's convention, where each site owns the spring to its right neighbour. This is used for every v1 vs v2 comparison.
- **Variants:** three alternative conventions are stored for the label analyses:
  - left spring
  - symmetric split
  - symmetric energy averaged over one swing

## Data pools

Each pool stores the initial angles, the amplitude and the labels.

| Pool | Runs | Seed | Use |
|---|---|---|---|
| train | 100,000 | 1 | Training. The first 60,000 match v1's budget. |
| val | 20,000 | 2 | Checkpoint selection and model choices |
| test | 20,000 | 3 | Reported numbers only |

All pools are generated independently, so no run appears in two pools.

**Inputs.** Every model input is rebuilt deterministically from the stored initial angles (`sgloc/views.py`):
- any quantity (θ, θ̇, site energy, θ minus the chain mean)
- any set of frames and start time
- random start within the first swing
- Gaussian sensor noise
- instrumenting only every s-th pendulum

The resulting trajectories are cached.

**Standardization.** Inputs are standardized per channel with training-pool statistics only.

## Metrics

All distances are measured around the ring. We report:
- the mean distance between predicted and true site
- accuracy at exactly the site, within ±1, and within ±3
- the share of runs more than 10 sites off
- the negative log-likelihood of the true site

The 20,000-run test pool resolves accuracy to about ±0.3% (95%).

## v1 (rebuilt)

The paper's training notebook is not available, so v1 was rebuilt from Sec. 2.3 and the Appendix in PyTorch, keeping the Keras defaults.

**Architecture** (26,499,205 parameters):
1. Time-distributed Conv1D ×4 (32/64/128/256 filters, kernel 3, zero "same" padding, ReLU).
2. Flatten.
3. Bidirectional LSTM (128 units, dropout 0.2, recurrent dropout 0.2).
4. Dense 64 with ReLU.
5. A sigmoid "localizes" head and a 100-way softmax site head.

**Training:**
- Loss: 0.5 × binary cross-entropy + 1.0 × masked categorical cross-entropy.
- Adam, learning rate 10⁻³, batch size 64.
- Early stopping on validation loss with patience 10, restoring the best weights.

**Inputs:** site energy at τ = 0, 0.5, …, 4.5 plus θ(0) repeated in every frame, as in the paper.

**Changes from the paper:** none, except that the scaler is fit on the training pool only.

## v2 models

**Backbone: a ring model (`sgloc/models.py`).**
- A local branch of circular convolutions.
- A global Fourier branch that keeps ring modes 1..M and mixes channels with one complex weight per mode. This is the form of the linear Floquet growth stage.
- A per-site head that gives one score per pendulum, so the model is shift-equivariant on the ring.

**Multi-frame front ends:**
- **Early fusion:** frames stacked as input channels.
- **Per-site GRU:** a GRU over the frames at each site.
- **2D convolution:** one convolution over (time, site).

**Training recipe for all ablations:**
- AdamW (learning rate 3×10⁻³, weight decay 10⁻⁴) with cosine decay and 3% warmup.
- Batch size 256 for 7,050 steps (30 epochs of 60,000 runs).
- The checkpoint with the lowest validation mean distance is kept. It is checked every 235 steps on 5,000 validation runs.

**Hardware.** Models train on the Apple M3 Pro GPU (MPS) or on single CPU threads.

**Record keeping.** Every run is appended to `results/runs.jsonl` with:
- its full configuration
- a note naming the hypothesis it tests
- its metrics

## Physics baselines

- **Box-car:** a moving average of the site energy, with the width and frame chosen on validation.
- **Floquet filter:**
  1. Evolve the mean-subtracted initial angles through the linearized dynamics for p swing periods. Each ring mode is multiplied by the (1,1) entry of the p-th power of its one-period monodromy matrix, computed for the run's estimated amplitude.
  2. Take the largest site. p = 6 is chosen on validation.
- **Learned linear:** multinomial logistic regression whose site scores are a full-ring circular convolution of the inputs. It is convex and fit to its global optimum with L-BFGS. Variants:
  - plain
  - with an amplitude-dependent filter
  - on v1's inputs
