# v2 preprint: notes and open TODOs

Draft arXiv preprint of the physics-guided v2 study, built from `REPORT.md`,
`REPORT.md` Sec. 4, `results/headline.md` and the results tables. Written in the
voice of the v1 paper (first person plural, narrative physics prose; Fig./Eq./Ref. conventions).

## Build

```bash
cd papers/v2_preprint && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- Current state: 20 pages (11pt article, 1 in margins), no LaTeX warnings, no overfull boxes.
- `main.bbl` is committed. arXiv does not run BibTeX, so regenerate and recommit `main.bbl`
  whenever `refs.bib` or the citations change.
- Checked: `main.tex` + `main.bbl` + `figures/` alone compile cleanly with two pdflatex passes.
- Upload to arXiv: `main.tex`, `main.bbl`, `figures/*.pdf`. Not `main.pdf`, `refs.bib` is optional.
- Set `\draftnotesfalse` in `main.tex` to hide the red TODO-AUTHOR-REVIEW box once Sec. 5 is agreed.

## Open TODOs

### Before posting (blocking)

1. **TODO-AUTHOR-REVIEW: Sec. 5, "Reproduction of the earlier model".** Still being discussed with
   L.Q. English (wording of the discrepancy; whether the original training notebook can be
   recovered). The section states only what `REPORT.md` Sec. 2 supports, worded as "could not
   be reproduced from the published description". Keep every other mention consistent with the
   agreed wording:
   - abstract ("a rebuild of the earlier network from its published description")
   - Introduction, 4th paragraph
   - footnote in Sec. 2.1 (the printed h̃ = 0.05)
   - Table 1 caption and the first paragraph of Sec. 4.5 (94.4%; wrap-around of the 1.44)
   - Conclusions, limitations paragraph
2. **Authors.** Adrian Mittal only for now (`% TODO-AUTHORS` in `main.tex`). Decide coauthors,
   order and affiliations; confirm the affiliation (copied from v1); add a contact e-mail if wanted.
3. **Acknowledgements** are a placeholder `[TODO]`.
4. **Data and code availability** now gives the repository URL. Add a version tag or DOI
   (e.g. a Zenodo archive of this repo) when posting.

### Should check

5. **v1 cross-references in Sec. 5** ("Sec. 2.1", "Sec. 2.3 and the Appendix", "Figs. 2–5 and 7",
   "Eq. (5)"): checked against the published JAND PDF on 2026-10-17, and the numbering matches.
6. **New references** (not in `papers/jand2026_arxiv/refs.bib`) were added from memory and should be verified:
   `sievers1988intrinsic`, `flach2008discrete`, `magnus1966hill`, `virtanen2020scipy`,
   `karniadakis2021physics`, `cohen2016group`, `yu2016multiscale`, `oord2016wavenet`,
   `li2021fourier`, `cho2014learning`, `hendrycks2016gelu`, `kingma2015adam`,
   `loshchilov2019decoupled`, `loshchilov2017sgdr`, `smith2019super`, `chen2023symbolic`,
   `szegedy2016rethinking`, `srivastava2014dropout`, `liu1989limited`, `paszke2019pytorch`,
   `guo2017calibration`. Three use "and others" for long author lists (Virtanen, Paszke, Chen).
7. **Edits to reused v1 references:** `thakur2007driven` now says 2008 (J. Phys. D 41, 015503;
   the v1 bibliography had 2007) and lists the author as "Basu Thakur, R."; `cuevas2014sine` is now a
   `@book` with editors (the v1 bibliography had it as an article); the SciPy citation is now Virtanen et al.
   (2020) instead of the v1 bibliography's `scipy` web entry. Revert if the v1 forms are preferred.
8. **Interpretive sentences to confirm** (they frame v1's findings; numbers are from the report):
   - Sec. 4.1: "the energetic clusters identified there are the growing long waves, seen through
     a low-pass filter"
   - Conclusions: "To answer the question left open in Ref. [v1], the features that decide the
     localization site are largely the projections of the early state onto the few growing long
     waves"
9. **Title** is a proposal: "Physics-guided machine-learning prediction of spontaneous energy
   localization in a sine-Gordon pendulum ring".
10. Once the v1 preprint has an arXiv ID, consider adding it to the `mittal2026jand` entry.

### Optional

11. **Figures** are copies of `figures/*.pdf`. They carry slide-style titles inside the plots and
    some small fonts; for a journal version, re-render without in-plot titles. Eight figure files
    are used, in seven floats (fig1, fig2, fig3, fig5 + fig6 side by side, fig13b, fig18, fig20).
    Dropped for length, with their numbers kept in the text or
    tables: 4, 7, 8, 10, 11, 12, 14, 15, 16, 17, 19, 21. They could go into supplementary material.
12. Details from the report left out for length: the hump-anatomy table, the timing and
    prediction-set tables, the second-site table, the full Phase 3 input list, checkpoint-step
    details of Phase 8, and the later coarse-camera start times beyond Table 8.
13. arXiv metadata: abstract is about 1,670 characters (limit 1,920); planned primary nlin.CD with
    cross-list cs.LG.

## Traceability

Every number in `main.tex` was checked against `REPORT.md` (including its Methods, Sec. 4),
`results/headline.md`, `results/*.md`, `results/*.json`, the v1 source (for quoted v1 values) and
`sgloc/` (for model constants such as the dilations). An automated pass found every numeric token
in those sources, and each was then checked in context by hand. Notes:

- Table 4's first-crossing column uses seeds 0–2 of the final model (1.50), as in report Sec. 10
  (Phase 7); everywhere else the five-seed value 1.49 is used.
- "Floquet clocks ... 4.1–4.2%" covers the two clock variants in report Sec. 10 (4.2% and 4.1%).
- The h̃ = 0.5 footnote, the τ ≈ 57 localization time, the spring bookkeeping (21% of runs) and
  the scaler leakage are from report Secs. 1–2.
