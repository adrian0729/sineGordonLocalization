# Original v1 code

These are the Colab notebooks behind the published paper, exported to Python and kept exactly as they were. They are not maintained, and some cells only run inside Colab.

The v2 simulator in [sgloc/](../sgloc/) reproduces this code. `scripts/validate_sim.py` re-runs its `odeint` right-hand side, energy formula and first-crossing loop, and gets the same localization site and time as the new simulator on 1,000 of 1,000 runs.

| File | What it does |
|---|---|
| `sinegordon_otherresults.py` | Simulates the ring with `odeint` and draws the angle and energy color map of one run (`colorMap.pdf`, as in Fig. 2 of the paper). It also computes the average localization time against the initial noise and against the amplitude (Fig. 3). |
| `sinegordon_alternateresults.py` | For one run, compares the box-car of the energy with the energy computed from box-car-smoothed angles and velocities, and draws the same color map |
| `alternateenergylocalization.py` | Repeats that comparison over kernel sizes 1–30 and sample indices 0–45. It compares the two predictors with each other, not with the true site ([improvement plan](../docs/improvement_plan.md), item 7). |
| `graph_designer.py` | Draws paper figures from hard-coded results: `energyRoughSmooth.pdf` (Fig. 1), `finalResults.pdf` (Fig. 4), `impurityPlot.pdf` (Fig. 7), and `5_45_timePlot.pdf`, an earlier two-panel version of Fig. 5 |
| `movingaverage.py` | Demonstrates the box-car on a synthetic trace (`averageConfirmation.pdf`, used in an early draft) |

The notebook that trained the paper's CNN-BiLSTM, and its dataset, are not available. [sgloc/models.py](../sgloc/models.py) rebuilds that network from the paper's description, and [REPORT.md, Sec. 2](../REPORT.md#2-what-this-work-corrects) compares the rebuild with the published numbers.
