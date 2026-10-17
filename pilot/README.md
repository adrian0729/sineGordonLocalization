# Pilot

The quick experiments run before the main study. They produced the numbers in the [improvement plan](../docs/improvement_plan.md), and their console output is saved in [results/](results/).

| Script | What it does | Output |
|---|---|---|
| `pilot.py` | A first vectorized simulator, the robustness of the label (twin runs), and physics-only predictors | `results/pilot_output.txt` |
| `pilot2.py` | How the label definition changes the site, Fourier-band knockouts, and the best box-car width against amplitude | `results/pilot2_output.txt` |
| `floquet.py` | The Floquet growth spectrum of the uniform swing | `results/floquet_output.txt` |

The maintained versions of these tools are in [sgloc/](../sgloc/) (`sim.py` and `floquet.py`).
