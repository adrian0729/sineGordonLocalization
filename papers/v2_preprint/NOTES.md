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
- Current state (2026-10-21): 17 pages (11pt article, 1 in margins), about 7,000 words of text and
  700 of captions, 6 figures, 4 tables, 36 references. No LaTeX warnings or overfull boxes.
- `main.bbl` is committed. arXiv does not run BibTeX, so regenerate and recommit `main.bbl`
  whenever `refs.bib` or the citations change.
- Upload to arXiv: `main.tex`, `main.bbl`, `fig_schematic.tex`, `figures/*.pdf` (see `../README.md`).
- Set `\draftnotesfalse` in `main.tex` to hide the red TODO-AUTHOR-REVIEW box once Sec. 4 is agreed.

## Length and what was left out

The 2026-10-17 draft was 20 pages, about 7,500 words with roughly twice v1's density of numbers, and
8 tables; it read like a condensed `REPORT.md`. The 2026-10-21 rewrite brings it to about v1's length
(v1 is 16 pages in the same layout), halves the number density, and adds the reasons behind each design
decision (moving away from the LSTM, angles instead of energies, the ring symmetry, the loss, the time head,
regularization, the hump label, damping), a schematic of both networks (Fig. 3) and a table pairing each
v1-to-v2 change with its physical reason and its cost (Table 2). Left to `REPORT.md`, with at most a sentence
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
   - Sec. 3.5, second paragraph
   - Sec. 4
   - Conclusions, limitations paragraph
   - Appendix B
2. **Authors and affiliation.** Adrian Mittal only for now (`% TODO-AUTHORS` in `main.tex`). Decide
   coauthors and order. The affiliation is still the one used in v1; update it (the author is now at CMU).
3. **Acknowledgements** are a placeholder `[TODO]`. If the v1 coauthors are not authors here, credit them,
   e.g. for building the CNN-BiLSTM studied here and for the simulation code that served as the reference.
4. **Data and code availability** gives the repository URL. Add a version tag or DOI (e.g. a Zenodo archive).

### Should check

5. **v1 cross-references** ("Sec. 2.1", "Sec. 2.3 and the Appendix", "Figs. 2–5 and 7", "Eq. (5)",
   "Fig. 6" in the email) were checked against the published JAND PDF.
6. **References added from memory** (not in `papers/jand2026_arxiv/refs.bib`) should be verified:
   `sievers1988intrinsic`, `flach2008discrete`, `magnus1966hill`, `virtanen2020scipy`,
   `karniadakis2021physics`, `cohen2016group`, `yu2016multiscale`, `oord2016wavenet`, `li2021fourier`,
   `cho2014learning`, `hendrycks2016gelu`, `kingma2015adam`, `loshchilov2019decoupled`,
   `loshchilov2017sgdr`, `szegedy2016rethinking`, `srivastava2014dropout`, `liu1989limited`,
   `paszke2019pytorch`. `smith2019super`, `chen2023symbolic` and `guo2017calibration` are still in
   `refs.bib` but no longer cited.
7. **Edits to reused v1 references:** `thakur2007driven` says 2008 (J. Phys. D 41, 015503) and lists the
   author as "Basu Thakur, R."; `cuevas2014sine` is a `@book` with editors; SciPy is cited as Virtanen et al.
   (2020). Revert if the v1 forms are preferred.
8. **Interpretive sentences to confirm:**
   - Sec. 3.1: "The energetic clusters identified in Ref. [v1] thus appear to be the growing long waves,
     seen through a low-pass filter."
   - Sec. 3.3: the critique of the CNN-BiLSTM design (one frame and nine pendulums per convolutional
     feature, 98.9% of the parameters in position-tied LSTM input weights, zero padding, bidirectionality).
   - Sec. 3.7: the reason offered for the waiting rate (white noise in every mode, the site depending on all
     growing modes) is an interpretation, not a derivation.
   - Conclusions: "the features that decide the localization site are largely the projections of the early
     state onto the few growing long waves".
9. **Title** is a proposal: "Physics-guided machine-learning prediction of spontaneous energy
   localization in a sine-Gordon pendulum ring".
10. Once the v1 preprint has an arXiv ID, consider adding it to the `mittal2026jand` entry.

## Traceability

Every number in `main.tex` was checked against `REPORT.md`, `results/headline.md`, `results/*.md` and
`*.json`, `results/runs.jsonl`, the plan and progress logs (git 7641a57), `sgloc/` and the v1 source. An
independent verification pass (2026-10-21) found and corrected these, among others: "both networks lose
3–4 sites" for a 10% error in h̃ (v1 loses 1.2–2.8; also corrected in `REPORT.md`); two different "17-site"
networks (Table 2: the seven layers without dilation, 15 sites; Sec. 3.6: a four-layer stack with dilations
1, 2, 4, 1); weight decay, which does help with little data once combined with mirror images; and a wrong
physical argument that energies lose the sign of the perturbation (to first order, E_n(0) is a rescaled copy
of θ_n(0)). Notes:

- 26.2 million of v1's 26,499,205 parameters (98.9%) are the BiLSTM's input weights (25,600 × 512 per
  direction, two directions), from instantiating the rebuild in `sgloc/models.py`.
- A disturbance travels at most about 0.45 pendulums per time unit (largest linear group velocity), so
  about two pendulums in the 4.5-unit window; growth over the window is 1.70× at A = 1.75, 1.99× at 1.85.
- The waiting rate per tenfold noise: 24.8 predicted from the mean growth rate before the swing-averaged
  grid was run, 25.5 measured there; 26.5 on the first-crossing label, where the rate was first found.
- The damped-chain claim that the Q ≈ 330 survivors are not the easy runs (v2 undamped: 1.64 on them
  against 1.31 on the rest, seed 0) is from the progress log, Phase 6 damping addendum.
