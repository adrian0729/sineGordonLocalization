# v2 preprint: notes and open TODOs

arXiv preprint of the physics-guided v2 study, built from `REPORT.md` (including its Methods, Sec. 4),
`results/headline.md` and the results tables. Written in the voice of the v1 paper (first person plural,
narrative physics prose; Fig./Eq./Ref. conventions; no bullet lists or run-in headings in the body).

## Build

```bash
cd papers/v2_preprint && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- Files: `main.tex`, `main.bbl`, `fig_schematic.tex` (TikZ, read by `\input`), `figures/paper_fig*.pdf`.
  The five PDFs are drawn by `scripts/make_figures.py --paper`; the schematic is plain TikZ.
- Current state (2026-10-21): 16 pages (11pt article, 1 in margins), the same as v1 in this layout;
  about 6,400 words of text and 700 of captions, 6 figures, 4 tables. No LaTeX warnings or overfull boxes.
- `main.bbl` is committed. arXiv does not run BibTeX, so regenerate and recommit `main.bbl`
  whenever `refs.bib` or the citations change.
- Upload to arXiv: `main.tex`, `main.bbl`, `fig_schematic.tex`, `figures/*.pdf` (see `../README.md`).
- Set `\draftnotesfalse` in `main.tex` to hide the red TODO-AUTHOR-REVIEW box once Sec. 4 is agreed.

## Length and what was left out

The 2026-10-17 draft was 20 pages, about 7,500 words with roughly twice v1's density of numbers, and
8 tables; it read like a condensed `REPORT.md`. The 2026-10-21 rewrite brings it to v1's length, halves the
number density, and gives the reason behind each design decision before its result, a schematic of both
networks (Fig. 3) and a table of v2's design choices with their physical reasons and costs (Table 2).
It builds on v1 rather than repeating it: v1's background, architecture details and the notes on its
published text are left to v1 and to `papers/jand2026_arxiv/ERRATA.md`, and the reasons we moved away from v1's
LSTM take one paragraph of Sec. 3.2, whose focus is the network v2 built; Sec. 2.4 describes v1 briefly. Left to `REPORT.md`, with at most a sentence
here: the regularization grid with little data, the key choices repeated on three labels, the damping table
and figure, the sensing-setup table and coarse-camera start times, the second-site rule and prediction sets,
the hump-top control, the Floquet clocks, the unknown-start numbers, and the optimizer list.

## Open TODOs

### Before posting (blocking)

1. **TODO-AUTHOR-REVIEW: Sec. 4, "Reproduction of the earlier network".** Under discussion with
   L.Q. English (wording; whether the original training notebook or dataset can be recovered). A reply
   that stresses the discrepancy is drafted in `overleaf/arxiv_submission/email_to_lars_2.md` (local, not in
   git). It raises further points about the published numbers that are deliberately not in the paper; keep
   them out until Lars has replied. Keep every mention of the published 94.4% consistent with the agreed
   wording:
   - abstract ("falls short of the accuracy originally reported")
   - Introduction, paragraphs 2 and 5
   - Table 1 and its caption (wrap-around of the 1.44)
   - Sec. 3.4, second paragraph
   - Sec. 4
   - Conclusions, limitations paragraph
2. **Authors.** Mittal (Department of Physics, Carnegie Mellon University), Grishanov, Pham, Lape and English,
   in v1's order with v1's affiliations for the coauthors. Each coauthor still has to agree (asked in the
   reply to Lars). Add a contact e-mail.
3. **Acknowledgements** are a placeholder `[TODO]`.
4. **Before uploading to arXiv** (which publishes the TeX source): delete the `\authorreview` box and its
   macro, set or remove `\draftnotes`, and strip every `%` comment from `main.tex` (e.g. with
   `arxiv_latex_cleaner`); check with `grep -n 'TODO' main.tex`.
5. **Data and code availability** gives the repository URL. Add a version tag or DOI (e.g. a Zenodo archive).

### Should check

6. **v1 cross-references** ("Sec. 2.1", "Sec. 2.3 and the Appendix", "Figs. 2–5 and 7", "Eq. (5)",
   "Fig. 6" in the email) were checked against the published JAND PDF.
7. **References added from memory** (not in `papers/jand2026_arxiv/refs.bib`) should be verified:
   `sievers1988intrinsic`, `flach2008discrete`, `magnus1966hill`, `virtanen2020scipy`,
   `karniadakis2021physics`, `cohen2016group`, `yu2016multiscale`, `oord2016wavenet`, `li2021fourier`,
   `cho2014learning`, `hendrycks2016gelu`, `kingma2015adam`, `loshchilov2019decoupled`,
   `loshchilov2017sgdr`, `szegedy2016rethinking`, `srivastava2014dropout`, `liu1989limited`,
   `paszke2019pytorch`. `smith2019super`, `chen2023symbolic` and `guo2017calibration` are still in
   `refs.bib` but no longer cited.
8. **Edits to reused v1 references:** `thakur2007driven` says 2008 (J. Phys. D 41, 015503) and lists the
   author as "Basu Thakur, R."; `cuevas2014sine` is a `@book` with editors; SciPy is cited as Virtanen et al.
   (2020). Revert if the v1 forms are preferred.
9. **Interpretive sentences to confirm:**
   - Sec. 3.1: "The energetic clusters identified in Ref. [v1] thus appear to be the growing long waves,
     seen through a low-pass filter."
   - Sec. 3.2: the contrast with the CNN-BiLSTM design (one frame and nine pendulums per convolutional
     feature, 98.9% of the parameters in position-tied LSTM input weights).
   - Sec. 3.6: the reason offered for the waiting rate (white noise in every mode, the site depending on all
     growing modes) is an interpretation, not a derivation.
   - Conclusions: "the features that decide the localization site are largely the projections of the early
     state onto the few growing long waves".
10. **Title** is a proposal: "Physics-guided machine-learning prediction of spontaneous energy
   localization in a sine-Gordon pendulum ring".
11. Once the v1 preprint has an arXiv ID, consider adding it to the `mittal2026jand` entry.

## Traceability

Every number in `main.tex` was checked against `REPORT.md`, `results/headline.md`, `results/*.md` and
`*.json`, `results/runs.jsonl`, the plan and progress logs (git 7641a57), `sgloc/` and the v1 source. An
independent verification pass (2026-10-21) found and corrected these, among others: "both networks lose
3–4 sites" for a 10% error in h̃ (v1 loses 1.2–2.8; also corrected in `REPORT.md`); two different "17-site"
networks (Table 2: the seven layers without dilation, 15 sites; Sec. 3.5: a four-layer stack with dilations
1, 2, 4, 1); weight decay, which does help with little data once combined with mirror images; and a wrong
physical argument that energies lose the sign of the perturbation (to first order, E_n(0) is a rescaled copy
of θ_n(0)). Notes:

- 26.2 million of v1's 26,499,205 parameters (98.9%) are the BiLSTM's input weights (25,600 × 512 per
  direction, two directions), from instantiating the rebuild in `sgloc/models.py`.
- A disturbance travels at most about 0.45 pendulums per time unit (largest linear group velocity), so
  about two pendulums in the 4.5-unit window; growth over the window is 1.70× at A = 1.75, 1.99× at 1.85.
- The waiting rate per tenfold noise: 24.8 predicted from the mean growth rate before any of the swing-averaged
  grid's later-start runs had finished, 25.5 measured there; 26.5 on the first-crossing label, where the rate
  was first found. The noise and sparse-sensor grids have one seed per cell.
- The damped-chain claim that the Q ≈ 330 survivors are not the easy runs (v2 undamped: 1.64 on them
  against 1.31 on the rest, seed 0) is from the progress log, Phase 6 damping addendum.
