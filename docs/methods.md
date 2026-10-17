# Methods

The full methods behind [the report](../REPORT.md): model system, simulator, labels, data pools, metrics, the v1 rebuild and the v2 training recipe.

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
- **Damping** (Sec. 11 only): an optional term −γ θ̇ₙ, linear friction with γ per time unit, so a free pendulum's quality factor is Q = 1/γ. Everything else uses γ = 0, the paper's conservative chain. With damping, a run is labeled when one site holds 10% of the energy the chain still holds.

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

## Final model and Phase 6 tests

**Final model.** The whole-ring dilated CNN: 7 circular convolutions with kernel 3, dilations 1-2-4-8-16-32-1 and width 32, and one score per site (22,786 parameters). Its input is the angles of all pendulums at τ = 0, 0.5, …, 4.5. Its loss is circular-Gaussian soft labels (σ = 1 site) + 0.1 × expected ring distance + 0.1 × squared error of a log-localization-time head. It trains with the recipe above for 30 epochs, with five seeds.

**Leave-one-out.** Each removal retrains the final model with one choice reverted, with three seeds:
- no dilation
- zero padding instead of circular
- a flatten + dense head instead of one score per site
- plain cross-entropy
- hard labels (other loss terms kept)
- no time head
- v1's inputs
- v1's optimizer

**Data efficiency.** The final recipe on 1k–100k training runs, two seeds each, with the same number of gradient steps.

**Regularization with little data** (Phase 8, `ablations.phase8`). The final recipe at 1k, 3k and 10k training runs, 2–3 seeds, checkpointed every 47 steps instead of every 235. One change is made at a time:
- AdamW weight decay 10⁻², 10⁻¹ or 1
- dropout 0.1 or 0.2 after every hidden activation (`RingNet(dropout=...)`)
- the mirror image of every training run (`augment: "mirror"` in the config)
- a 235-, 705- or 2,350-step cosine schedule instead of 7,050 steps

The mirror image reverses the site axis (pendulum n → N − 1 − n), which turns angle inputs into those of the mirrored chain exactly. Under reflection a spring on a pendulum's right moves to its left, so the mirrored run's v1 label is the mirror of the stored left-bond label (`site_left`, `time_left`); the symmetric and swing-averaged labels are simply mirrored. A test simulates the mirrored chains and checks that inputs and labels match. The training set holds both, so an epoch is twice as long at the same number of steps; the input scaling is unchanged. Energy channels (one-sided springs) and sparse sensors do not reverse exactly, and the code refuses them. Two combinations were chosen from these results on the validation pool, and the mirror images were also tested at 60,000 runs (3 seeds).

**Robustness pools** (`scripts/make_robustness_pools.py`). Each is simulated with its own true chain. Trained models are applied without retraining, and any energies are still computed with the nominal g̃ and h̃.

| Pool | Runs | What differs from the nominal chain |
|---|---|---|
| g −10% / +10% | 5,000 each | g̃ = 0.675 / 0.825 |
| h −10% / +10% | 5,000 each | h̃ = 0.45 / 0.55 |
| Twice the initial spread | 5,000 | offsets U(−0.05, 0.05) rad |
| N = 80 / 150 | 5,000 each | a shorter or longer ring |
| One weak pendulum | 5,000 | one random pendulum's gravity term 1% weaker |
| Impurity train / val / test | 60,000 / 10,000 / 10,000 | one random pendulum's gravity term weaker by U(0, 2%) |
| Fixed damping | 5,000 each | γ = 0, 10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³ or 10⁻²; all six share their initial conditions |
| Damped train / val / test | 60,000 / 10,000 / 10,000 | γ log-uniform in [10⁻⁴, 10⁻²], one value per run |

Sensor noise at test time is added to the nominal test pool (`scripts/robustness.py`).

**Impurities with and without a map.** The final recipe is retrained on the impurity pools, with three seeds:
- with angles only
- with an extra input channel that maps each pendulum's gravity term, as a one-time calibration of a real chain would

**Real-world sensing setups.** The final recipe is retrained on three setups from Phase 3, with three seeds each:
- a camera on every 2nd pendulum with 10⁻³ rad of noise, from release
- a coarse camera (10⁻² rad) watching from τ = 20
- energies at τ = 30, noise-free

**Damped chains** (`scripts/damping_physics.py`, the `damping` group of Phase 6). The final recipe is retrained on the damped pools with three seeds, alone and with v1's "does it localize" head (weight 0.5), on v1's label, on the swing-averaged label, and with the camera setup. Runs that never localize by τ = 150 are left out of the site and time losses.

**Time head and prediction sets** (`scripts/eval_final.py`). The predicted log-time is compared with the true localization time on the test pool. Baselines are the training median and the median time within each of 20 amplitude bins.
