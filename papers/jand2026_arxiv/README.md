# The published paper, arXiv version

A. Mittal, F. Grishanov, A. Pham, N. Lape and L. Q. English, "Machine-learning prediction of type-III instabilities and spontaneous energy localization", *J. Appl. Nonlinear Dyn.* **15**(2), 277–290 (2026), [doi:10.5890/JAND.2026.06.001](https://doi.org/10.5890/JAND.2026.06.001). Open access under CC BY 4.0, © 2026 The Authors.

This folder rebuilds the paper from its Overleaf source in the standard `article` class:
- the published text, with its typographical errors corrected; each change is listed under "Changes from the published version" at the end of the paper
- a note under the abstract giving the journal reference and the license, as CC BY requires
- the journal's reference style (`cj.bst`), with the capitalization in `refs.bib` fixed

[ERRATA.md](ERRATA.md) lists every error found in the published version. It also lists the points where the text and the original code disagree, which this version leaves unchanged. [REPORT.md, Sec. 2](../../REPORT.md#2-what-this-work-corrects) discusses them.

## arXiv submission form

| Field | Value |
|---|---|
| License | CC BY 4.0, matching the journal (arXiv's license cannot be changed later) |
| Primary category | nlin.CD (Chaotic Dynamics) |
| Cross-list | cs.LG (Machine Learning) |
| Title | Machine-learning prediction of type-III instabilities and spontaneous energy localization |
| Authors | Adrian Mittal, Fedya Grishanov, Anne Pham, Noah Lape, L. Q. English |
| Journal-ref | J. Appl. Nonlinear Dyn. 15(2), 277-290 (2026) |
| DOI | 10.5890/JAND.2026.06.001 |
| Comments | 16 pages, 7 figures. Published in J. Appl. Nonlinear Dyn. under CC BY 4.0. This version corrects typographical errors in the published text, including the global coupling in Sec. 2.1 (0.5, printed as 0.05); the changes are listed at the end. |

Abstract, as plain text:

> Modulational instability of a plane-wave mode is known to lead to spontaneous energy localization in nonlinear lattices without the presence of impurities. Where the energy dynamically localizes in the system is highly sensitive to initial conditions. Here we numerically investigate sine-Gordon-type lattices and first show that spatial smoothing of the observed dynamical variables on the lattice can already substantially improve predictions of the eventual localization site from early-time data traces alone, suggesting that the type-III instability that forms the energy hotspot is sensitive to energetic clusters. We then show that machine learning leads to additional dramatic improvement in prediction accuracy. Finally, we also examine the role of chain impurities in determining the localization site.
