# v2 preprint: notes and open TODOs

arXiv preprint of the physics-guided v2 study, built from `REPORT.md` (including its Methods, Sec. 4),
`results/headline.md` and the results tables. Written in the voice of the v1 paper (first person plural,
narrative physics prose; Fig./Eq./Ref. conventions; no bullet lists or run-in headings in the body).

## Build

```bash
cd papers/v2_preprint && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

- Files: `main.tex`, `main.bbl`, `fig_schematic.tex` (TikZ, read by `\input`), `figures/paper_fig*.pdf`.
  The figure PDFs are drawn by `scripts/make_figures.py --paper`; the schematic is plain TikZ.
- Current state (2026-11-02, second pass): 23 pages (11pt article, 1 in margins), 9 figures, 5 tables, two
  appendices. No LaTeX warnings or overfull boxes. The abstract is 1,902 characters as plain text (1,909 raw);
  arXiv's limit is 1,920.
- `main.bbl` is committed. arXiv does not run BibTeX, so regenerate and recommit `main.bbl`
  whenever `refs.bib` or the citations change.
- Upload to arXiv: `main.tex`, `main.bbl`, `fig_schematic.tex`, `figures/*.pdf` (see `../README.md`).

## Length and what was left out

The 2026-10-17 draft was 20 pages, about 7,500 words with roughly twice v1's density of numbers, and
8 tables; it read like a condensed `REPORT.md`. The 2026-10-21 rewrite brought it to v1's length, halved the
number density, and gave the reason behind each design decision before its result, a schematic of both
networks (Fig. 3) and a table of v2's design choices with their physical reasons and costs (Table 2).
The threshold, settings and coupling studies (2026-10-22 to 11-02) brought it back to 21 pages.

The 2026-11-02 review (an independent referee-style read, then edits):
- The settings study (clock, offsets, couplings; Fig. 7) moved out of Sec. 3.1 into its own Sec. 3.6,
  after the target is defined, since it uses the event-timed hump label, the first pendulum over the top and
  rival humps. Sec. 3.7 now runs noise, sensors, parameter errors and impurities, damping, then the training
  conditions (amplitude, offsets, coupling, ring size).
- Sec. 4 on the published accuracy became Appendix B, with a bridge in the threshold paragraph of Sec. 3.5.
- Secs. 3.2 to 3.4 were condensed (the two-branch network, GRU, loss variants, regularization; the time head
  moved into the clock paragraph). v1's architecture is described once in Sec. 2.4 and in Fig. 3.
- Table 3 gained the event-timed hump label as the bold recommended target; Table 4 lost the
  doubled-offsets and ring-size rows (the offsets are covered in Sec. 3.7, and the first-crossing label
  fails on other ring sizes for its own reasons).
- Kept on purpose: the Spearman correlations of the threshold study (the link to the physical baselines),
  and the statements of what was predicted before each test.

The 2026-11-02 second pass, after the question "why keep the 3–15% sweep if we have the hump label, and where
are the other ring sizes?". A three-judge panel agreed to keep the sweep in the main text but restructure it:
- Sec. 3.5 now uses the sweep as a probe of the hump label. A bridge asks why the 10% label (77%) and the hump
  label (99.4%) differ so much when both are read about five time units after the first pendulum goes over
  the top. Two findings follow:
  - what v2 learns from any single-pendulum label up to 12% is where the hump forms;
  - one pendulum marks the hump only around that event, since later the crossing depends on the swing phase.
  The baselines come next, then the physics rule fixed before training, which selected no share (now stated
  in full), and finally the event-timed label, with its adoption rule stated.
- Fig. 6(b) is now v2's accuracy against the label's median lag after the event. It replaces the
  hump-agreement and spring-split panel.
- Ring size has its own physics paragraph in Sec. 3.6: about N/18 growing modes, rival humps, the clock in
  e-foldings, the Floquet filter against v2, and why a share of the total energy fails on other rings.
  Sec. 3.7 keeps only the transfer between rings. Fig. 9 is pinned before the Conclusions with `\FloatBarrier`
  (`placeins`).
- The abstract, the Introduction (paragraph 4) and the Conclusions (paragraphs 1 and 3) each gained a
  ring-size clause. Appendix B no longer says energy "passes to the neighbors".

The paper builds on v1 rather than repeating it: v1's background, architecture details and the notes on its
published text are left to v1 and to `papers/jand2026_arxiv/ERRATA.md`. Left to `REPORT.md`, with at most a
sentence here: the regularization grid with little data, the key choices repeated on three labels, the damping
table and figure, the sensing-setup table and coarse-camera start times, the second-site rule and prediction
sets, the hump-top control, the Floquet clocks, the unknown-start numbers, and the optimizer list.

## Open TODOs

### Before posting (blocking)

**Audit fixes not yet applied (2026-11-02).** An adversarial audit of the second pass confirmed about 170
numbers against `results/` and found these, still to fix in `main.tex` before the final review:
- Sec. 2.4 says the Floquet filter runs "for six swing periods"; the number is chosen on validation (six on the
  main pools, four to nine on the further pools of Sec. 3.6). Its Fig. 3(a) citation comes before Fig. 2.
- The N = 400 pools stop integrating once the event-timed labels are complete (τ = 75.5–83.0, a median of
  32.5 time units after the event; `threshold_hump.json`, `rings` → `integration_end`). Secs. 2.3 and 3.6 should
  say so, since it is why the 10% and Eq. (6) labels fire there in only 2.9% and 23% of runs.
- Sec. 3.5: "the event" also names the hump label firing (line ~260); use one verb for the event ("goes over
  the top", not "past" or "passes"); state in the text that the hump-center predictor is shifted one site at
  7–12%; "a median of at most two time units", not "within two"; the 63% is 79% once shifted; explain 0.27
  against 0.28 (0.28 each when trained side by side); hedge the drift at 20% (the extra misses are mostly by
  two or three sites); "none of the shares we computed", naming the E_sep labels tested.
- Sec. 3.6: "a wider version of v2"; its fall from 99.9% to 98.8% (far misses 0.1% to 0.9%); the rival-hump
  share at N = 100 (20%).
- Sec. 3.7: "carry over reliably only from longer rings"; the wide network's dilations 64 and 128 wrap on the
  100-pendulum ring (128 ≡ 28), another possible reason it fails on longer rings; the standard network's 92%
  and 89% come with 6% and 10% far misses; "on a ring at least as long, preferably of its own length" (also in
  the Conclusions).
- Fig. 6: say why the sweep's 10% scores differ slightly from Table 1 (retrained for the study) and that the
  1.1% of runs not crossing 15% by τ = 150 are left out; in (b), the diamonds sit at the end of their one-swing
  average and the stars at zero lag; axis labels "goes over the top" (`make_figures.py`).
- Fig. 9: define the rival hump at the event; move the figure before "These tests applied a network..." (about
  a fifth of p. 18 is blank).
- Abstract: restore "the size of the initial offsets". Introduction: "the 10% threshold of Ref. [v1]", and the
  ring clause as its own sentence. Conclusions: "from 3% to 12%"; "only when it fires close to ...".
- `REPORT.md`: "read 3, 5 and 7 time units after the event" should say where the one-swing averages end; the
  drift wording near the k3_s20 line (about line 951).

1. **Author.** Adrian Mittal alone (decided 2026-11-02), Department of Physics, Carnegie Mellon University;
   almittal@andrew.cmu.edu. v2 is new work that cites Ref. [v1] for the earlier results.
2. **Appendix B and the published accuracy.** It treats the retrained CNN-BiLSTM as v1's network and states
   that the published accuracy was obtained at an effective threshold of 4–5% and offsets of ±0.0125
   (decided 2026-11-02; the original training code never came). Every mention of the published 94.4% follows
   it: abstract, Introduction (paragraphs 2 and 5), Table 1 caption, Sec. 3.4, Sec. 3.5 (the sweep's baselines paragraph),
   Appendix B, Conclusions (limitations). `REPORT.md` (Sec. 2) keeps the hedged account of the rebuild.
3. **Acknowledgements** (added 2026-11-02, after the Conclusions as in v1) thank L.Q. English and the other
   coauthors of Ref. [v1] for the collaboration this work builds on. Add funding, or anything more specific
   about the earlier work, before posting if wanted.
4. **Before uploading to arXiv** (which publishes the TeX source): strip every `%` comment from `main.tex`
   (e.g. with `arxiv_latex_cleaner`); check with `grep -n 'TODO' main.tex`.
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
   - Sec. 3.7: the reason offered for the waiting rate (white noise in every mode, the site depending on all
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
- Sec. 3.5's sweep paragraphs and Sec. 3.6's ring paragraph trace to `results/threshold.json` (v2, v1,
  baselines, diagnostics, `spearman`, and `choice` for the physics rule), `results/settings.json` (part A for
  the hump and event-timed labels and their `decision`; part C for the rings) and
  `results/threshold_hump.json` (`scripts/threshold_hump.py`, no training). The last of these holds:
  - `cross_scores`: the sweep's v2 runs re-scored against the hump center;
  - `swing_phase`: agreement with the hump center near the bottom of the swing (0.375 ≤ φ < 0.625) and near a
    turning point;
  - `timing`: the Eq. (6) label against the event;
  - `rings`: modes, σ1, e-foldings and label lags at N = 50–400.
- The hump-center predictor of Fig. 6(a) is the Eq. (6) site shifted by the constant offset that suits each
  label best on validation (one site at 7–12%, none otherwise; `baselines.*.hump_oracle`).
- The ring numbers come from the wider network (dilations up to 128, 28,994 parameters), including at
  N = 100 (99.5%, 0.24). They are not Table 3's standard network (99.4%, 0.28).
- Sec. 3.6's Floquet accuracies (52/73/87% against the offset scale, 17% at A = 1.60) are scores against
  the event-timed hump label (`settings_analysis.py`, `floquet_baseline` with L = eh_k3). An earlier draft
  called them scores against the first pendulum over the top.
