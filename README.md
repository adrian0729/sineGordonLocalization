# Predicting where energy localizes in a sine-Gordon pendulum ring

A ring of 100 coupled pendulums, released in a nearly uniform swing, suddenly concentrates its energy on a few pendulums after about 57 time units. Which pendulums is decided by tiny differences in the starting state. This repository asks how early, and from which measurements, that site can be predicted.

It is the follow-up (v2) to our paper:

> A. Mittal, F. Grishanov, A. Pham, N. Lape and L. Q. English, "Machine-learning prediction of type-III instabilities and spontaneous energy localization", *J. Appl. Nonlinear Dyn.* **15**(2), 277–290 (2026), [doi:10.5890/JAND.2026.06.001](https://doi.org/10.5890/JAND.2026.06.001).

## Start here

| | |
|---|---|
| **[REPORT.md](REPORT.md)** | The full write-up: methods, what was tried, why each choice was made, and every finding (including the ones that did not pay off), with figures and tables |
| **[papers/](papers/)** | Both papers as LaTeX: the published paper (arXiv version, typos corrected) and a draft of the v2 preprint |
| [results/headline.md](results/headline.md) | The headline tables, v1 against v2 |

## Results

Test pool of 20,000 simulated runs. Mean distance is in pendulums, measured around the ring.

| Model | Parameters | Training | Mean distance | Within ±1 | Within ±3 |
|---|---|---|---|---|---|
| v1 as published | 26.5M | | 1.44 | 94.4% | |
| v1 rebuilt from the paper (2 seeds) | 26.5M | 23 min | 3.14 | 60.4% | 90.3% |
| Floquet filter (physics, no training) | 0 | | 3.12 | 52.0% | 87.3% |
| **v2: whole-ring dilated CNN (5 seeds)** | **22,786** | **6 min** | **1.49** | **77.2%** | **97.3%** |

- **Physics first.** The uniform swing becomes unstable at A_c = 1.577, and at A = 1.75 only the five longest ring modes grow. This explains the paper's best smoothing width (17 pendulums). It also makes the site a deterministic function of the starting state, so prediction is limited by measurement precision, not by chaos.
- **A small model built around the ring.** Every pendulum sees the whole ring through dilated circular convolutions, and the same filters score every pendulum. v2 halves the rebuilt v1's error with 1/1,163 of its parameters, and beats it with 60 times less training data.
- **The published v1 accuracy could not be reproduced** from the paper's description: the rebuild reaches 60.4% within ±1, not 94.4%. [REPORT.md, Sec. 2](REPORT.md#2-what-this-work-corrects) discusses why.
- **The hotspot is a hump about 8 pendulums wide.** Asked where the hump forms, rather than which pendulum first passes 10% of the energy, v2 is within ±1 pendulum in 99.4% of runs.
- **For a real chain,** one camera tracking every second pendulum to 10⁻³ rad, predicting right after release, is within ±1 of where the hump forms in 96.8% of runs.

## Repository layout

| Path | What it holds |
|---|---|
| [REPORT.md](REPORT.md) | The report |
| [papers/](papers/) | The two papers, as LaTeX ready for Overleaf and arXiv |
| [sgloc/](sgloc/) | The Python package: simulator, Floquet tools, data pools, models, losses, metrics, training loop and the ablation tables |
| [scripts/](scripts/) | Command-line entry points for data, sweeps, analyses, tables and figures ([scripts/README.md](scripts/README.md)) |
| [tests/](tests/) | Physics and model checks (`pytest`, about 30 s) |
| [results/](results/) | Tables, analysis outputs and the record of every training run ([results/README.md](results/README.md)) |
| [figures/](figures/) | The report's figures, as PNG and PDF ([figures/README.md](figures/README.md)) |
| [v1_original/](v1_original/) | The original Colab code behind the published paper, unchanged |
| `data/`, `runs/` | Generated data pools and model checkpoints (not tracked; several GB) |

## Setup

Python 3.12. Training runs on an Apple-silicon GPU (PyTorch MPS), an NVIDIA GPU (CUDA) or the CPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Reproducing the results

```bash
.venv/bin/python scripts/make_pools.py          # 140,000 simulated runs into data/ (about 10 min on 10 cores)
.venv/bin/python scripts/validate_sim.py        # the simulator against the paper's own pipeline
.venv/bin/python scripts/phase1.py              # physics checks and baselines
.venv/bin/python scripts/train_v1.py --seed 0   # the paper's CNN-BiLSTM, rebuilt
.venv/bin/python scripts/sweep.py phase6        # the final v2 model (5 seeds) and its leave-one-out ablations
.venv/bin/python scripts/headline.py            # results/headline.md
.venv/bin/python scripts/make_figures.py        # figures/
```

- Each phase of the study is one ablation table in [sgloc/ablations.py](sgloc/ablations.py). Run it with `scripts/sweep.py <phase>` and summarize it with `scripts/summarize.py <phase>`. [scripts/README.md](scripts/README.md) lists every script and the phase it belongs to.
- `results/runs.jsonl` already records every run of this study, and `sweep.py` skips configs it finds there. To retrain a phase from scratch, move that file aside first.
- `sweep.py` uses the Apple GPU by default. On an NVIDIA GPU, pass `--gpu-device cuda`; on CPU only, pass `--gpu-workers 0`.
- The whole study (about 750 training runs) took 47 hours on an M3 Pro laptop, plus short sessions on cloud NVIDIA L40 GPUs for the settings study and the rebuilt v1 at other thresholds.

## Conventions

- Time is dimensionless (τ = ω₀t). Samples are Δτ = 0.5 apart, as in the paper's code.
- Parameters: g̃ = 0.75 and h̃ = 0.5, as in the paper's code (Sec. 2.1 of the paper prints 0.05 by mistake).
- Damping is off by default (γ = 0, the paper's conservative chain). With `gamma`, Eq. (4) gains −γ dθ/dτ (γ per time unit, quality factor Q = 1/γ), and the 10% threshold applies to the energy the chain still holds.
- **Labels.** A run is labeled at the first sample where one site holds more than 10% of the total energy.
  - `site_right` uses the paper's code convention (each site owns the spring to its right) and is the label used for comparisons with v1.
  - `site_left`, `site_sym` and `site_avg` (symmetric, averaged over one swing) are stored for the label ablations.
  - `site_hump` is the centre of the 3 neighbouring pendulums that first hold 15% of the energy, averaged over a swing. `scripts/add_labels.py hump` adds it to the pools.

## Citation

If you use this code, please cite the published paper:

```bibtex
@article{mittal2026jand,
  author  = {Mittal, Adrian and Grishanov, Fedya and Pham, Anne and Lape, Noah and English, L. Q.},
  title   = {Machine-learning Prediction of Type-{III} Instabilities and Spontaneous Energy Localization},
  journal = {Journal of Applied Nonlinear Dynamics},
  volume  = {15},
  number  = {2},
  pages   = {277--290},
  year    = {2026},
  doi     = {10.5890/JAND.2026.06.001}
}
```

## License

The code is released under the [MIT License](LICENSE). The papers in [papers/](papers/) are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The published paper is © 2026 The Authors.
