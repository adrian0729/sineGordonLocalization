# Improvement plan: predicting where energy localizes (v2)

Draft, 15 October 2026. Every pilot number below comes from the scripts in `pilot/` (outputs saved in `pilot/results/`). Unless stated otherwise, a pilot uses 2,000 simulated runs with the same setup as the paper's ML dataset: N = 100, g̃ = 0.75, h̃ = 0.5, A ~ U(1.65, 1.85), initial noise U(−0.025, 0.025), released from rest.

"τ" means dimensionless time (the paper's t). "Sample index" means the row number in the code's arrays, which are spaced Δτ = 0.5 apart.

---

## 0. Summary

**What the audit and pilot changed**

1. **h̃ is 0.5, not 0.05.** Every script uses h = 0.5. The paper's "h̃ = 0.05" in Sec. 2.1 is a typo. The physics confirms it: with h̃ = 0.05 the chain is already unstable at A = 1.50, so there is no threshold near 1.59, and the fastest-growing wavelength is about 12 sites. That is type-I behavior, not type III.
2. **The "time" axes are sample indices.** Localization happens at τ ≈ 57 on average; that is sample index ≈ 114, the paper's "≈ 100". The box-car point "t = 5" in Fig. 5 is really τ = 2.5. My simulator reproduces Fig. 5 at that point: K = 17 gives a mean error of 5.38, versus 5.41 in the paper.
3. **Only five long waves grow.** A linear-stability (Floquet) analysis of the uniform swing at A = 1.75 shows that only five modes grow. Their wavelengths are 100, 50, 33, 25 and 20 sites, and the fastest one grows by a factor of e every 8.4 time units. The paper's 10-frame ML window (τ ≤ 4.5) sees at most 1.7× growth at A = 1.75, and about 2× even at A = 1.85.
4. **The same analysis explains the paper's best smoothing width.** A box-car of width K first cancels mode m = N/K. The best K is the one whose first cancellation lands on the first stable mode: K* = N/m_c.

   | A | Predicted K* | Measured K* |
   |---|---|---|
   | 1.65 | 25 | 25 |
   | 1.75 | 16.7 | 17 |
   | 1.85 | 14.3 | 13 |

5. **A training-free predictor halves the box-car error.** Evolve the initial noise through the linearized dynamics for six swing periods, then take the peak. This predictor has zero trained parameters and gets a mean error of 3.0 sites; the best box-car gets 5.3–5.9.
6. **The labels are deterministic, so sensor precision is the real limit.**
   - In 500 twin runs, halving the integration step never moved a label, and neither did perturbing the initial angles by 10⁻⁶ rad. So the accuracy ceiling is close to 100%, not 94%.
   - An error of 10⁻³ rad (rms) in the initial angles moves 24% of labels by more than one site. When that error sits in the five growing modes, it moves 52%.
7. **"The localization site" is fuzzy at the ±1 level.** When the threshold is first crossed, two reasonable ways of assigning bond energy to sites disagree on the site in 21% of runs. The hotspot also keeps moving for a while after it forms. Report ±3 alongside ±1.

**Direction for v2.** A small ring-aware CNN: circular padding, one output score per pendulum, and a global Fourier branch that keeps only the few growing modes. That is about 10–25k parameters instead of 26.5M, it needs no LSTM, and it should train in minutes on the M3 Pro. Each design question below starts from a physics hypothesis and ends with an ablation that could prove it wrong.

---

## 1. Audit findings (fix before any v1 vs v2 comparison)

| # | Finding | Where | Effect | Fix |
|---|---|---|---|---|
| 1 | h̃ typo | Paper Sec. 2.1 says 0.05; all scripts use `h = 0.5` | Text only; the Floquet analysis confirms 0.5 | Erratum note; use h̃ = 0.5 in v2 |
| 2 | Time units | `locTime()` returns a sample index; Figs. 2–5 and 7 plot indices | Localization is at τ ≈ 57, not 100. The box-car comparison point is τ = 2.5, earlier than the end of the ML window (τ = 4.5). Later in the window the box-car does better (5.3 at τ = 4.5 with random A; 4.5 at τ = 5 with A = 1.75), so v1's 3.75× improvement claim is a bit generous | Use τ everywhere; compare methods at matched τ |
| 3 | The ML code is not in the folder | The six files contain no TensorFlow code | v1 cannot be rerun exactly | Add the original notebook, or re-implement it from Sec. 2.3 and the Appendix |
| 4 | Box-car indexing | `boxcar()` returns 101 values (the wrap duplicates one point), and even K is off-centre by half a site | Small (well under 0.1 site) | Use a true circular filter with odd K |
| 5 | One-sided bond energy | `g*(theta - np.roll(theta,-1,axis=1))**2` gives each site the whole bond to its right | The label site changes in 21% of runs under a symmetric split (Sec. 2.4) | Keep v1's label for comparisons; add a symmetric, period-averaged label |
| 6 | Scripts don't run as files | Colab exports with undefined names (`final_data`, `energyMax`, `localize_t`, `noiseLevel`, `averageTime`) and bare text lines | Not reproducible | One small package (Sec. 10) |
| 7 | `alternateenergylocalization.py` compares two predictors with each other, not with the true site | Its main loop | The script behind Fig. 4 (prediction vs actual site) is not in the folder | Rebuild from the package |
| 8 | Slow simulator | `odeint` runs one simulation at a time | Hours for 100k runs | Vectorized RK4 (pilot): 2,000 runs in 13 s on one core, so 100k runs take about 11 min. Labels were identical at Δτ = 0.05 and 0.025 |
| 9 | v1 ML issues (from the paper) | Sec. 2.3 and Appendix | Scaler fit on all data before the split; ring wrap not stated for the ±n metric and MAE; the binary head is trivial (100% of pilot runs localize by τ = 150); zero padding on a ring; 99% of the 26.5M parameters sit in one LSTM input matrix | Handled by the shared protocol (Sec. 3) and Q2–Q3 |

---

## 2. The physics picture behind every hypothesis

### 2.1 Growth spectrum (Floquet analysis)

**Method.** Linearize Eq. (4) about the uniform swing θ₀(τ), which obeys θ₀'' = −sin θ₀. Each Fourier mode m ≠ 0 of the ring then obeys a Hill equation (a harmonic oscillator whose stiffness varies periodically in time):

```
δ'' = −[ cos θ₀(τ) + h̃ + 4 g̃ sin²(π m / N) ] δ
```

Integrating over one swing period T gives a 2×2 "monodromy" matrix per mode. Its largest eigenvalue fixes the mode's growth rate σ_m. Code: `pilot/floquet.py`.

**Growth at A = 1.75 (period T = 7.75):**

| m | Wavelength (sites) | σ_m | e-fold time | Growth over τ ≤ 4.5 | Growth over 6 periods (τ ≈ 46.5) |
|---|---|---|---|---|---|
| 1 | 100 | 0.118 | 8.4 | 1.70× | 247× |
| 2 | 50 | 0.113 | 8.9 | 1.66× | 191× |
| 3 | 33 | 0.103 | 9.8 | 1.59× | 118× |
| 4 | 25 | 0.085 | 11.7 | 1.47× | 52× |
| 5 | 20 | 0.053 | 18.9 | 1.27× | 12× |
| ≥ 6 | ≤ 17 | 0 (stable) | n/a | 1× | 1× |

**Across the data range:**
- A = 1.65: modes 1–3 are unstable (fastest e-fold time 13.3).
- A = 1.75: modes 1–5 (e-fold 8.4).
- A = 1.85: modes 1–6 (e-fold 6.6).
- The instability switches on at A_c = 1.577. The paper's Fig. 3b asymptote is ≈ 1.59.

### 2.2 Four facts the hypotheses rely on

- **P1. The coarse location lives in at most six global modes.**
  - Keep only m ≤ 5 of the initial noise and re-simulate: the site stays within ±3 in 91% of runs.
  - Delete m ≤ 5 instead: the site becomes random (mean error 25, chance level).
  - So the model needs a whole-ring view, and about ten numbers carry the coarse answer.
- **P2. The exact position depends somewhat on local structure.**
  - Deleting all of m > 5 still moves 24% of sites by more than one.
  - Small (10⁻³ rad) perturbations confined to m ≥ 6 move only 4.8%.
  - So a small local branch should refine the coarse answer.
- **P3. Early frames add little new information in clean data.**
  - Growth over the 10-frame window is at most about 2× (1.7× at A = 1.75).
  - Disturbances travel at most about 0.45 sites per time unit (a small-amplitude estimate), so about 2 sites within the window.
  - θ(0) with θ'(0) = 0 is the complete state.
  - So extra frames mainly help against sensor noise.
- **P4. Precision, not chaos, limits ±1 accuracy.**
  - Per-pendulum sensor noise of 3×10⁻⁴ rad moves 10% of labels; 10⁻³ rad moves 24%.
  - Training data should include realistic sensor noise.
  - Observing later, after the growing modes have amplified, trades warning time for robustness.

### 2.3 Baselines from the pilot (v1's label definition; the same 2,000 runs unless noted)

| Predictor | Trained parameters | Mean error (sites) | Within ±1 | Within ±3 | More than 10 off |
|---|---|---|---|---|---|
| Box-car K = 17 on E at τ = 0 | 0 | 5.88 | 31.9% | 64.8% | 11.5% |
| Box-car K = 17 on E at τ = 4.5 | 0 | 5.34 | 37.2% | 69.6% | 9.7% |
| Floquet filter, 6 periods | 0 (one tuned integer) | 3.04 | 52.8% | 87.8% | 4.5% |
| v1 CNN-BiLSTM (paper; different test set) | 26.5M | 1.44 | 94.4% | n/a | n/a |

**How the Floquet filter works.**
1. Subtract the chain mean from θ(0).
2. Fourier transform it.
3. Multiply mode m by its 6-period linear response.
4. Transform back and take the peak.

Equal weighting of modes 1–5 does much worse (mean error 9.7), so the growth weighting is what matters. The box-car's falling frequency response is a crude version of the same weighting, which is why the box-car works at all.

### 2.4 How well defined is "the localization site"?

- **Bookkeeping.** Splitting each bond's energy equally between its two pendulums, instead of giving it all to the left one, changes the first-crossing site in 21% of runs (by two or more sites in 16%).
  - The two definitions cross the threshold at essentially the same moment (0.14 time units apart on average).
  - So at the crossing there are often several comparable peaks a few sites apart.
- **Drift.** After it forms, the period-averaged energy peak moves by more than one site within 10 time units in 31% of runs (mean shift 3.4 sites).

Both labels are deterministic and learnable. But ±1 accuracy partly rewards learning a bookkeeping convention, so the plan reports ±1, ±3 and mean error throughout.

---

## 3. Shared protocol for every ablation

**Data**
- Vectorized RK4 with Δτ = 0.05; a subset is checked against Δτ = 0.025.
- Fixed pools generated with separate random seeds: 100k training runs, plus 20k validation and 20k test.
- Store only the initial angles, A and the labels: about 120 MB for 140k runs. The simulator is deterministic and fast, so any input view (frames, channels, observation time, sensor noise, missing sensors) can be regenerated exactly. New ablations never need a new dataset.

**Labels**
- Primary: v1's definition (first time any pendulum holds more than 10% of the energy, with v1's bookkeeping), for apples-to-apples comparison.
- Secondary: symmetric bookkeeping, averaged over one swing period. This is the physically robust version.
- Also store the localization time.

**Metrics** (all distances measured around the ring)
- Mean error, accuracy at exactly 0, ±1 and ±3, and the share of runs more than 10 sites off.
- Negative log-likelihood, to check calibration.
- Parameter count, training time and inference time.

**Statistics**
- Three seeds per configuration (five for finalists); report mean ± standard deviation.
- Compare two configurations with a paired bootstrap 95% confidence interval on the shared test set.
- The 20k test set resolves about ±0.3% in accuracy and about ±0.05–0.1 sites in mean error. Paired comparisons on the same test runs are tighter.

**Hygiene**
- Fit any normalization on the training split only.
- Change one factor at a time, in the phase order of Sec. 9. The leave-one-out pass in Q5 catches interactions between factors.

**Logging**
- Every run appends one JSON line to `results/runs.jsonl`: configuration, a free-text `note` explaining why the run exists, metrics, time and git hash.
- The report's tables and its "things I tried and why" section are generated from this file.

---

## 4. Q1: Which inputs should be measured?

### 4.1 What each input costs on a real chain

Rough scale, assuming a 20 cm pendulum:
- One time unit is about 0.14 s.
- Localization takes about 8 s.
- v1's window is about 0.64 s.
- One sample step (Δτ = 0.5) is about 0.07 s, so a 30–60 fps camera is fast enough.
- 10⁻³ rad at the bob is 0.2 mm.

| Input | How to measure it | Cost | Precision limit |
|---|---|---|---|
| Angles θ_n(τ) | One camera, markers on the bobs, sub-pixel tracking | Low: one sensor for all 100 pendulums | Optics. About 10⁻³ rad is a realistic floor for one camera viewing the whole chain (estimate) |
| Velocities θ'_n(τ) | A MEMS gyroscope per pendulum, or differencing video frames | High (100 sensors), or noisy | Differencing amplifies noise in proportion to the frame rate |
| Energy E_n(τ) | Computed from θ, θ', the neighbors, the chain mean, g̃ and h̃ | Highest: needs both sensors plus model parameters | Errors from every ingredient add up. It changes little over a swing, which helps |
| Sparse angles | Track only every s-th pendulum | Lower | Nyquist: capturing m ≤ 6 needs s ≤ 8 |
| One-time calibration | Natural frequency of each pendulum | Low, done once | Matters only when impurities exist (Sec. 8.3) |

### 4.2 Hypotheses

**H1.1 The initial state is enough.**
- *Physics:* the system is deterministic and released from rest, so θ(0) is the complete state; every later frame is a function of it.
- *Prediction:* in noise-free data, θ(0) alone (1 frame, 1 channel) matches v1's inputs (10 energy frames plus θ(0)) within seed-to-seed noise.

**H1.2 The swing phase matters for raw angles.**
- *Physics:* in a real chain you do not know where in the swing you start watching. θ swings between −A and +A every period, while E_n changes little.
- *Prediction:* with a random start time, angle-only inputs need a window of about half a period (≈ 3.9 time units, about 8 frames at Δτ = 0.5) to match energy, or (θ, θ'), taken from a single frame.

**H1.3 Sensor precision decides ±1 accuracy.**
- *Physics (P1, P4):* the coarse answer lives in a few global modes. White sensor noise of σ_s per pendulum puts only about √(10/100) ≈ 0.32 σ_s into those modes, so averaging over many pendulums helps. Even so, 10⁻³ rad in that band moves half the labels.
- *Prediction:* at τ = 0, ±1 accuracy collapses once σ_s ≳ 3×10⁻⁴ rad, while ±3 holds up until about 10⁻³.

**H1.4 Watching longer buys robustness.**
- *Physics:* the growing modes are amplified by e^{σ_m τ}, and sensor noise is not.
- *Prediction:* under sensor noise, accuracy improves with observation time τ_obs.
  - The τ_obs needed for a target ±3 accuracy grows roughly like 8.4 · ln(σ_s / σ_ref). Each 10× worse sensor costs about 19 time units (≈ 2.7 s) of warning.
  - ±1 accuracy recovers only partly, because the local structure from P2 does not grow.

**H1.5 Sparse sensors are enough for the coarse location.**
- *Physics:* the coarse location lives in m ≤ 6, so by Nyquist one sensor every ≤ 8 pendulums captures it.
- *Prediction:* tracking every 2nd, 4th or 8th pendulum keeps ±3 accuracy close to full but loses ±1.

**H1.6 Perturbation coordinates train better.**
- *Physics:* the uniform swing carries no position information. Subtracting the chain mean leaves only the perturbation field.
- *Prediction:* inputs δθ = θ − θ̄ plus a scalar A channel train faster and more stably than raw θ with one global scaler, at equal accuracy.

### 4.3 Ablation grid

| ID | Inputs | Frames | Tests |
|---|---|---|---|
| I0 | E(τ = 0…4.5) + θ(0), as in v1 | 10 | baseline |
| I1 | θ(0) | 1 | H1.1 |
| I2 | E(0) | 1 | H1.1 |
| I3 | θ(0…4.5) | 10 | H1.1, H1.2 |
| I4 | E(0…4.5) | 10 | H1.1 |
| I5 | (θ, θ') at a random start time | 1 | H1.2 |
| I6 | δθ(0) + A | 1 | H1.6 |

**Realism sweeps** on the best three or four input sets:
- random start time; window length ∈ {1 frame, T/4, T/2, T} (H1.2)
- σ_s ∈ {0, 10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻²} rad, with matched noise in training and test (H1.3)
- τ_obs ∈ {0, 10, 20, 30} (H1.4)
- sparsity s ∈ {1, 2, 4, 8}, with a mask channel marking where sensors are (H1.5)

**Deliverable**
- A Pareto plot of accuracy (±3 and ±1) against measurement cost and remaining warning time (τ_loc − τ_obs).
- *Decision rule:* pick the cheapest input set within one standard error of the best, at the target sensor noise.

---

## 5. Q2: Space-time architecture (multi-frame kernels, the LSTM, kernel sizes and widths)

### 5.1 Hypotheses

**H2.1 The LSTM is not needed.**
- *Physics (P3):* within 4.5 time units, information moves about 2 sites and grows at most about 2×. All temporal information is short and local.
- *Prediction:* "early fusion" matches CNN-BiLSTM within seed noise at under 1% of the parameters. Early fusion means stacking the frames as input channels, so every kernel sees all frames at once.
- A 2D convolution over (time, space) adds nothing on top of early fusion.

**H2.2 A whole-ring view is required.**
- *Physics (P1):* modes 1 and 2 (wavelengths 100 and 50) dominate.
- *Prediction:* with one output per site, accuracy climbs steeply with receptive field up to about 50–100 sites, then flattens. v1's convolution stack sees 9 sites, which is not enough on its own; v1's dense head was doing the global mixing.

**H2.3 Two scales: a global branch plus a local branch.**
- *Physics (P1 + P2):* the coarse answer comes from linear growth of global modes; the fine answer from local nonlinear focusing.
- *Predictions:*
  - The global Fourier branch alone lands near the Floquet filter (mean error about 2–3, high ±3).
  - The local branch alone cannot tell which hump wins.
  - The two together are best.
  - The number of kept Fourier modes M saturates near 6 (the unstable band), with no gain beyond 12.

**H2.4 Kernel size and width can stay small.**
- *Physics:* breathers, and the local structure that nudges them, span a few sites. The information is low-dimensional: about 10 global numbers plus local texture.
- *Prediction:* once a global branch exists, local kernels of 3–5 suffice, and 16–32 channels saturate accuracy.

**H2.5 Respecting the ring's symmetry saves data.**
- *Physics:* the equations do not change under rotating or reflecting the ring.
- *Predictions:*
  - Circular padding plus one score per site (shift-equivariant by construction) needs 3–10× less training data than flatten-plus-dense for the same accuracy.
  - It has no edge effects. Zero padding shows larger errors for sites near 0 and 99.

**H2.6 Feeding in the physics prior helps.**
- *Prediction:* adding the Floquet-evolved field as an extra input channel lets a tiny local CNN learn only the nonlinear correction. This should give the best accuracy when training data is scarce.
- The field costs one FFT per sample, not a simulation.

### 5.2 Ablation grid

| Block | Variants | Tests |
|---|---|---|
| Time handling | T1 v1 CNN-BiLSTM · T2 per-frame CNN + small GRU · T3 early fusion · T4 2D conv (k = 3 in time, 5 in space) · T5 single frame | H2.1 |
| Receptive field | Dilated circular stacks with RF ∈ {9, 17, 33, 65, 129} · a global Fourier layer | H2.2 |
| Scales | Global only · local only · both; M ∈ {3, 6, 12, 25} | H2.3 |
| Size | Kernel {3, 5, 9, 17} × width {8, 16, 32, 64}, depth {2, 4}, on the two-branch model | H2.4 |
| Symmetry | v1 head vs one-score-per-site with zero padding vs circular padding; learning curves at 1k, 3k, 10k, 30k, 60k runs; error by site index | H2.5 |
| Physics prior | With and without the Floquet channel, at 3k and 60k runs | H2.6 |
| Learned linear | Logistic regression on the ring (one circular convolution per site, then softmax). It is convex, so L-BFGS finds its global optimum | Splits the gain into linear vs nonlinear parts |

### 5.3 Are there better options than a CNN?

| Family | Fit with the physics | Verdict |
|---|---|---|
| Circular CNN + Fourier layer | Matches the ring symmetry and local coupling. The linear growth stage is exactly a per-mode multiplication in Fourier space (the idea behind Fourier neural operators) | Main candidate |
| CNN + LSTM (v1) | Time modelling is not needed for 10 frames (P3), and flattening throws away the ring symmetry | Baseline only |
| Graph neural network | On a ring with nearest-neighbor edges, it reduces to a 1D CNN | Later: useful for irregular lattices, impurities and real structures |
| Transformer over sites | Global, but has no built-in locality and needs a lot of data, when the global part is only about 6 modes | Optional small baseline |
| Reservoir computing (refs 17–18) | Built to forecast trajectories, not to classify one final outcome | Skip |
| Logistic regression on Fourier features | The exact form of the linear stage; convex and interpretable | Include (the learned-linear row above) |

### 5.4 v2 candidate (untested sketch, PyTorch)

```python
import torch
import torch.nn as nn


def ring_conv(c_in, c_out, k):
    return nn.Conv1d(c_in, c_out, k, padding=k // 2, padding_mode="circular")


class SpectralBranch(nn.Module):
    """Global part: keep ring modes m = 1..M and mix channels per mode (the linear Floquet stage)."""

    def __init__(self, c_in, c_out, M=6):
        super().__init__()
        self.M = M
        self.w = nn.Parameter(torch.randn(M, c_in, c_out, dtype=torch.cfloat) / (c_in * c_out) ** 0.5)

    def forward(self, x):  # x: (batch, channels, N)
        X = torch.fft.rfft(x, dim=-1)[..., 1:self.M + 1]
        Y = torch.einsum("bcm,mcd->bdm", X, self.w)
        out = torch.zeros(x.shape[0], Y.shape[1], x.shape[-1] // 2 + 1, dtype=Y.dtype, device=x.device)
        out[..., 1:self.M + 1] = Y
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class RingNet(nn.Module):
    """One score per pendulum; shift-equivariant on the ring."""

    def __init__(self, c_in, width=32, M=6, k=5):
        super().__init__()
        self.local = nn.Sequential(ring_conv(c_in, width, k), nn.GELU(), ring_conv(width, width, k), nn.GELU())
        self.glob = SpectralBranch(c_in, width, M)
        self.head = nn.Sequential(ring_conv(2 * width, width, 3), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, x):  # (batch, channels, N) -> logits (batch, N)
        return self.head(torch.cat([self.local(x), self.glob(x)], dim=1)).squeeze(1)
```

**Size:** about 12k parameters with 2 input channels, and about 22k with 20 (10 frames × 2), versus 26.5M for v1.

**Notes:**
- CPU is fast enough at this size. Use the M3 GPU (MPS) only for the v1 baseline.
- If inputs are mean-subtracted (H1.6), feed A as a constant extra channel.

---

## 6. Q3: Loss function

**H3.1 The answer is often two-humped, so keep a full distribution over sites.**
- *Physics:* when two humps grow at similar rates, the winner is decided late. v1's errors look bimodal: 94% are within ±1, and from the paper's numbers the remaining ~6% average roughly 17–26 sites off.
- *Prediction:* circular regression (predicting an angle with a von Mises loss) averages the two humps and lands between them, so it has more outliers than any loss that keeps a distribution over sites.

**H3.2 Neighbors matter.**
- *Physics:* the hotspot spans about 3 sites, and the exact first-crossing site depends on bookkeeping in about 21% of runs (Sec. 2.4).
- *Prediction:* soft labels from a circular Gaussian (σ ≈ 1 site) reduce near-miss errors and improve calibration, without lowering exact-site accuracy.

**H3.3 Optimize the metric you report.**
- *Prediction:* adding λ × (expected ring distance under the predicted distribution) lowers mean error and outliers. (With a one-hot target this is the Wasserstein-1 distance on the circle.)
- Too large a λ makes the model hedge and lowers ±1 accuracy.

**H3.4 Auxiliary physics targets help.**
- *Physics:* the same growing modes set both the time and the site. Roughly, τ_loc ≈ ln(threshold / initial amplitude) / σ.
- *Prediction:* an extra head predicting log τ_loc improves data efficiency. So does an extra head predicting the period-averaged energy profile at τ ≈ 30.

**H3.5 The binary head can go.**
- *Physics:* every run above A_c = 1.577 localizes (100% in the pilot for A ∈ [1.65, 1.85]).
- *Prediction:* removing the head changes nothing. Keep it only if the data range extends below A_c; then treat it as time-to-localization with a cutoff.

| ID | Loss | Tests |
|---|---|---|
| L1 | Cross-entropy (v1's site loss) | baseline |
| L2 | Von Mises circular regression | H3.1 |
| L3 | Soft labels, σ ∈ {0.5, 1, 2} | H3.2 |
| L4 | L3 + λ × expected distance, λ ∈ {0.03, 0.1, 0.3} | H3.3 |
| L5 | Best of L3/L4 + 0.1 × log-time head | H3.4 |
| L6 | Best of L3/L4 + 0.1 × energy-profile head | H3.4 |
| L7 | With vs without the binary head | H3.5 |

**Expected winner:**

```
L = CE(soft labels, σ ≈ 1) + λ · E[ring distance] + 0.1 · (log-time error)²
```

---

## 7. Q4: Optimizer and schedule

Physics does not choose the optimizer, so this question is mostly engineering. The physics still frames it in two ways:
- The linear stage is a convex problem, so the learned-linear baseline is fit to its global optimum with L-BFGS; there is no optimizer question there.
- The CNN is small, the inputs are standardized, and the model respects the ring symmetry, so the problem is well conditioned.

**H4.1 The schedule matters more than which optimizer.**
- *Prediction:* AdamW with cosine decay (2–5% warmup) slightly beats v1's constant-rate Adam and varies less between seeds.
- SGD with Nesterov momentum matches it only after a much finer learning-rate search.

**H4.2 Bigger batches are free here.**
- *Reason:* data is cheap and the model is small.
- *Prediction:* batches of 256–1024 with a scaled learning rate train 3–5× faster at equal accuracy.

**Runs**
- Start with a learning-rate range test.
- Optimizers:
  - O1: Adam, constant 10⁻³ (v1)
  - O2: AdamW + cosine
  - O3: AdamW + one-cycle
  - O4: SGD-Nesterov + cosine
  - O5 (optional): Lion
- Batch size {64, 256, 1024} × weight decay {0, 10⁻⁴, 10⁻²}.
- Three seeds each. Report final metrics and time-to-target.

---

## 8. Q5: The final model, and which parts actually matter

### 8.1 Assemble the winners from Q1–Q4, then remove one piece at a time

- the global Fourier branch
- circular padding (use zero padding instead)
- one score per site (use a flatten + dense head instead)
- soft labels (use plain cross-entropy instead)
- the auxiliary time head
- the Floquet input channel
- AdamW + cosine (use constant-rate Adam instead)

Report the change in mean error, ±1, ±3 and outliers for each removal.

### 8.2 Headline comparison: v1 vs v2 on the same test set

| | v1 (reproduced) | v2 |
|---|---|---|
| Parameters | | |
| Training time | | |
| Mean error / ±1 / ±3 / more than 10 off | | |
| Training runs needed for mean error < 2 | | |
| ±3 accuracy at σ_s = 10⁻³ rad | | |
| Warning time left at that accuracy | | |

### 8.3 Robustness (sim-to-real)

- Sensor noise that differs between training and test.
- Parameter mismatch: g̃ and h̃ off by ±10% at test time.
- Twice the initial noise level.
- Different chain lengths (N = 80, 150). A fully convolutional net runs unchanged, and the physics predicts the band edge m_c scales with N.
- A random impurity (ties back to Sec. 3.5 of the paper), with and without a one-time calibration channel.

---

## 9. Phases and compute

All compute numbers are rough estimates for the M3 Pro.

| Phase | Work | Runs | Compute |
|---|---|---|---|
| 0 | Package, simulator, labels, data pools; reproduce v1 | 3 v1 seeds | About 25 min for data. v1 is the slowest item, probably tens of minutes per run |
| 1 | Physics baselines: box-car, Floquet filter, learned linear; K*(A); mode knockouts | About 10 | Under 1 h |
| 2 | Q2 architecture | About 40 configs × 3 seeds | 4–6 h |
| 3 | Q1 inputs and sensing | About 80 | 3–4 h |
| 4 | Q3 loss | About 15 × 3 | About 2 h |
| 5 | Q4 optimizer | About 15 × 3 | About 2 h |
| 6 | Final model, leave-one-out, robustness | About 40 | About 3 h |
| 7 | Report | | |

**Decision gates**
- After Phase 0: v1 must reproduce within about 10% of the paper's numbers before anything is compared to it. **Status (15 Oct): not met.** The rebuild gets 61.7% within ±1 against the paper's 94.4%; see [progress.md](progress.md).
- After Phase 2: if no ring model matches v1, investigate before moving on.

---

## 10. Code layout (simple on purpose)

```
sgloc/
  sim.py       vectorized RK4, energies, both label definitions
  floquet.py   growth spectrum and the Floquet filter
  data.py      generate and store initial conditions + labels; build any input view on the fly
  models.py    v1 CNN-BiLSTM, ring CNN, spectral branch
  losses.py    cross-entropy, soft labels, ring distance, von Mises, auxiliary heads
  train.py     one config in, one JSON line of results out
  sweep.py     runs one ablation table
  report.py    tables and figures from results/runs.jsonl
pilot/         the scripts behind the numbers in this plan
```

**Framework.** PyTorch is recommended because circular padding and FFTs are built in. Keras works too, but needs a small custom layer for the circular padding.

**Reproduce the pilot** (needs numpy and scipy):

```bash
python pilot/floquet.py
```

```bash
python pilot/pilot.py pilot_data.npz
```

```bash
python pilot/pilot2.py pilot pilot_data.npz
```

---

## 11. Report outline (v1 → v2)

1. **What v1 did and what this work corrects.** Sec. 1 of this plan, plus the reproduction gap. Rebuilt from the paper's description (the original notebook and dataset are not available), v1 reaches 61.7% within ±1 and a mean distance of 3.0. The paper reports 94.4% and 1.44. State the three candidate explanations from [progress.md](progress.md), and say that every v1 vs v2 comparison uses the rebuilt v1 on the same test pool.
2. **Physics of the type-III instability.**
   - Figure: growth rate vs mode and amplitude.
   - Figure: predicted vs measured best box-car width, explaining Fig. 5 of the paper.
   - Figure: twin runs and band-limited perturbations (determinism and the precision limit).
3. **Baselines.** Box-car → Floquet filter → learned linear → v1.
4. **One section per question (Q1–Q4).** Each has hypothesis → design → result → verdict (supported, refuted or partly) → what was kept.
5. **Final model vs v1.**
   - The headline table (Sec. 8.2).
   - Error histograms showing v1's bimodal errors next to v2's.
   - Leave-one-out table.
6. **Real-world sensing recommendation.** The Pareto plot, and sensor precision vs warning time.
7. **Limitations and next steps.** Driven-damped chains, impurities, and a test on the physical pendulum array.
