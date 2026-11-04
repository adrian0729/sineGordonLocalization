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
- Current state (2026-11-03, after the cuts): 23 pages (11pt article, 1 in margins), 9 figures (Fig. 9 one
  panel, at 0.6 of the text width), 5 tables, two appendices. No LaTeX warnings or overfull boxes. The abstract
  is 1,606 characters raw, 314 under arXiv's limit of 1,920.
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

The audit of the second pass (2026-11-02; about 170 numbers confirmed against `results/`) led to these changes:
- Disclosures. The Floquet filter's number of periods is chosen per pool on validation (Sec. 2.4). The N = 400
  pools stop integrating once the event-timed labels are complete (Sec. 2.3); Sec. 3.6 now leads with the
  uncensored N = 50 to 200 lags. The hump-center predictor's one-site shift is stated in the text, and Fig. 6
  leaves out the 1.1% of runs that never cross 15%.
- Sec. 3.5 explains the partial recovery at 15%: crossed 6–12 time units after the event, the label marks the
  hump center in 99% of runs, and the misses are runs that cross 14 or more units after it, where most of v2's
  gain over the hump center comes from. It also gives the flank (near a turning point the 10% label most often
  lies two sites from the center).
- Sec. 3.7 gives two reasons for the failed transfer to longer rings, the wide network's dilations wrapping on
  the 100-pendulum ring and the continuous band, and says that whether a long-ring model must see its whole
  ring is untested (the third check, below, separated the two reasons).
- One name for the event ("over the top"); "the label" where "the event" meant the label firing; Fig. 9 moved up;
  Fig. 3 no longer cited before Fig. 2. The paper grew to 24 pages.

A check of those changes (2026-11-02, all numbers confirmed) led to a second set:
- The fixed-lag check was confounded: the first pendulum goes over the top just after a turning point in 96% of
  runs, so a later label's phase follows from its lag, and in the 6–8 bin the crossings near the bottom are all
  at lag 6.0 and those near a turning point at 6.5–7.5. Sec. 3.5 now shows the agreement rising and falling
  with each half swing (30%, 99%, 19% and 85% at lags 3.5, 5.5, 7.5 and 9), from `by_sample_lag`.
- Claims made per threshold, not per run: a single-pendulum label marks the hump reliably only at thresholds
  passed around the event (the 15% label crossed 6–12 units after it marks the hump in 99%). The phase matters
  only where the pendulums on the flanks of the hump can also reach the threshold; near the event the 5% label
  is the evidence (the first pendulum over the top is picked by its angle). The 20% hump label: "in some runs
  the hump has moved", since in general it stays put for about a swing.
- Sec. 3.7 says the transfers to shorter rings ask little of a circular network, which on a ring whose size
  divides its training ring's computes what it computes on its own ring with the state repeated (every shorter
  ring tested then did divide; the third check, below, tested others).
- The N = 400 sentence no longer reads the integration end (32.5) as the 10% label's lag (23.5). Two long
  paragraphs split. Fig. 9(a)'s legend no longer says "wide model" (the wider network is the one of 9(b)), and
  9(b) draws the two networks scored at N = 100 to either side of it. Fig. 6 moved up to the page of its first
  citation.

A third check (2026-11-02, numbers confirmed) led to the last set:
- Near the event the phase does not matter, shown per run: the 5% label lies within ±1 of the hump center in
  99% of the 3% of runs that reach 5% in the quarter around a turning point, and in 98% of those that reach it
  2–4 time units after the event (near the next turning point), against 28% for the 10% label; the first
  pendulum over the top lies within ±1 in 99% of the 62% of runs in which it goes over in the quarter around a
  turning point. "Around the event" became "close to the event", the even spring split's contrast (92% and
  34%) is stated, one more paragraph split, and the Fig. 9 caption says where the triangles are drawn.
- "Networks carry over from longer rings to shorter ones" overclaimed: every shorter ring tested divided the
  training ring. New test pools at N = 80 and 150 (`scripts/ring_transfer.py`) show that the wider network
  carries over only when trained on 400 pendulums (98% at 150); trained on 50–200 it places 0–62% within ±1.
  The standard network trained at N = 100, which differs only in lacking the dilations of 64 and 128, keeps 97%
  at 80 and loses accuracy gradually on longer rings (94%, 92%, 89% at 150, 200, 400). The wider network's
  failures thus point to its dilations wrapping around its training ring, not to the 100-pendulum ring's
  global modes. Sec. 3.7's transfer paragraph became two, and the advice for a real chain (Sec. 3.7 and the
  Conclusions) changed from "a ring at least as long, preferably of its own length" to "a ring of its own
  length".

A check of that rewrite (2026-11-02, all numbers confirmed) led to a fourth set, after every network was scored on
every ring (`ring_transfer.py` now fills the whole matrix, five networks on six rings):
- Fig. 9(b) is now that matrix (rows: the wider network trained at 50–400 and v2 at 100; columns: rings of 50–400;
  outlined the training ring, asterisks the rings that divide it). It replaces the line plot whose transfers were
  all to rings dividing the training ring or to longer rings, so that it suggested that long rings carry over to
  short ones. Its rival-hump labels remain in Sec. 3.6's text (2.4% to 71%).
- Transfers to rings that divide the training ring no longer "ask little": the repeated state is one the network
  never met, and the more often it repeats, the more the network loses (under a point twice, 4–6 points four or
  five times, 16 points eight times). The 93% at N = 80, the 13% at 150 and the network trained at 50 are
  reported, and "large differences between seeds" is limited to the networks trained at 100 and 200.
- "The standard network" is v2, with its own-ring 99.4% and 0.4% far misses as the baseline. Two caveats qualify
  the dilation explanation (the extra layers also widen the view, 513 against 129; networks trained on longer
  rings saw more modes), and "what matters seems to be the ring a network was trained on" rests on 400 → 150
  working and 100 → 400 failing. v2's loss on longer rings has two candidate causes, the modes it never saw and a
  view that no longer spans the ring, and "the hump forms by a local process" became a hedged "suggests ...
  perhaps". The expectation that was refuted is stated in full (seeing the whole ring would let a network carry
  over to rings of other sizes).
- The closing advice is its own paragraph and says "a ring of the chain's size"; the Conclusions say "to larger
  offsets" (toward smaller ones the network keeps 99.3% at half and 98.2% at a quarter of the offsets). Sec. 2.3
  lists the pools of 80 and 150 pendulums. The paper grew to 25 pages.

A final check of that rewrite and of the new Fig. 9(b) (2026-11-02; every number confirmed, only wording and
precision) led to a fifth set:
- Sec. 3.7's ring paragraphs open by naming the networks scored. A transfer to a ring that divides the training
  ring "does not test whether" a network carries over, and the losses on repeated states are stated against the
  network's own ring (against the one trained on the test ring, the eight-fold loss would read 17 points). The
  wider network trained at 100 is quoted at N = 80 too (17%, against v2's 97.4%), and v2's row is given to one
  decimal, as in the figure (89.5%, which an integer 89% contradicted on rounding).
- The second ring paragraph is split, and the argument from N = 100 is scoped to the network trained there.
  "What matters" is now whether a dilation exceeds half of the training ring rather than the test ring. The view
  caveat covers every training ring: a view spans 4 × the largest dilation + 1, so the largest dilation exceeds
  half a ring just when the view goes more than twice around it, and the wider network's view goes 10, 5, 2.6
  and 1.3 times around its training rings (the one that carries over, 400, has v2's 1.3). The caveat on modes
  includes the network trained at 50, which saw fewer.
- v2's gradual loss gets its missing step (on rings of 150–400 only one to three of the 8 to 22 growing modes
  are longer than its view), and the loss at N = 80, which neither candidate explains, may act on longer rings
  too. The caption of Fig. 9(b) names the four training rings and says "shorter rings" for the asterisks;
  Sec. 2.3 says "other amplitudes and couplings" and is split in two; the Conclusions give the reasons in the
  order of the requirements.

The paper builds on v1 rather than repeating it: v1's background, architecture details and the notes on its
published text are left to v1 and to `papers/jand2026_arxiv/ERRATA.md`. Left to `REPORT.md`, with at most a
sentence here: the regularization grid with little data, the key choices repeated on three labels, the damping
table and figure, the sensing-setup table and coarse-camera start times, the second-site rule and prediction
sets, the hump-top control, the Floquet clocks, the unknown-start numbers, and the optimizer list.

The 2026-11-03 round:
- The abstract was cut from 1,918 to 1,770 characters; its ring-size clause went (ring size stays in Sec. 3.6,
  Sec. 3.7, the Introduction and the Conclusions).
- Sec. 3.5 now explains why a single-pendulum label leaves the hump center, from `crossing_energy.json`: after
  the event the hump's pendulums swing over the top and back together; near the bottom of the swing the global
  coupling's twist holds the energy and peaks at the most-turned pendulum, the center; near a turning point the
  energy is kinetic and flat across five pendulums, so the label often lands two sites off. The spring
  convention only picks the side. Paragraph 1's "motion and the springs on either side" was wrong and went; the
  15% paragraph's "presumably" became the measured narrowing of the hump; the Conclusions gained one clause.
- The Floquet filter of Sec. 3.6 and Fig. 7(c): periods searched from 3 to 20, per amplitude bin, at the
  estimated amplitude (Traceability, below). 17% at A = 1.60 became 25%, and "stays below 65% even with small
  offsets" near threshold went (per bin, the lowest bin reaches 89%).
- An independent read of Sec. 3.5 then led to: the 5.5% "averaged over a swing" is the trailing swing, so it
  includes growth (now said; "breathing" became "changes form within each swing, as we show below"); the rival
  hump no longer motivates the hump label (it has rivals too: 14.4% against 16.3% in `top2.json`
  `rival_share`) and is stated as a limit neither label removes; "libration turns into rotation" became "past
  the inverted position", since the hump's pendulums come back; the labeled pendulum "turned through about a
  full revolution", with why one turn holds about 8% (the global term is not periodic); the flat top is given
  with the dip at the center on the off-center runs instead of "small differences in speed"; the 5%-vs-10%
  comparison at the next turning point names its 291 and 2,405 runs; the 85% at lag 9 gives its 94 runs; why the
  right edge wins; the 5–6% window of the physics rule; "thus" and the three-pendulum window reworded.
  Untraced by the reader in `results/` (they are in REPORT.md): "agree exactly in only a third", "half of those
  two-site misses", the 17-site stack factors, the soft- and hard-label remarks, the adoption-rule margins and
  the side-by-side 0.28. The last two were cut; the others are traced to the data since (Traceability).

The 2026-11-03 cuts, after the user asked to cut what is not important and does not lead anywhere, in the
abstract and the paper (about 1,700 words, 25 to 23 pages; backup of the text before them in the session
scratchpad only):
- Abstract, 1,770 to 1,606 characters raw: "We close with what this means for a real chain" (says nothing), the 60
  times less training data (not one of the Introduction's three findings), "the initial offsets set this clock",
  and "revisit the problem starting from" became "start from".
- Ring size. Secs. 3.6 and 3.7 each keep one paragraph: in Sec. 3.6 the growing modes and rival humps against
  N, the Floquet filter against a version of v2 trained on each ring, and the energy share that changes meaning
  with N while the event does not; in Sec. 3.7 v2's transfer from N = 100 (97.4% at 80 to 89.5% at 400) and
  the wider network's failure from N = 100 (13–30%), with "presumably" for the dilations that reach more than
  halfway around its ring. Gone: Fig. 9(b), the transfer to rings whose size divides the training ring's, the
  other training rings, both caveats, the e-foldings on other rings, the 10% and 4% crossings at N = 400, and
  the unexplained loss at N = 80. The full matrix and the dilation-wrap analysis stay in `REPORT.md`.
- Side results with no follow-up: mirror images (they add nothing at 60,000 runs; also the mirror map in
  App. A), energies at an unknown start, the rival-hump fraction at N = 100 and the twins' rival humps, the
  hump label's gain split between timing and window, the side-by-side 0.28, the three-pendulum window at 20%
  with its ±3 numbers, the Floquet periods near threshold (Sec. 2.4 says it), and App. B's pointer to the
  report for v1's time axes.
- Compressed: the threshold rule (169 to 118 words, keeping the rule and its outcome; the margins of the
  adoption rule became "It did neither"), the last-pendulum paragraph (cosh and the linear-theory gap went),
  the damping detail, and the 15% loose end (one sentence in the 15% paragraph).
- Kept on purpose: the single-pendulum to hump mechanism (untouched), the threshold sweep and its baselines,
  the statements of what was predicted before each test, the two-branch network and the Floquet input (the
  Introduction promises the choices that did not pay off), and every real-chain result except ring transfer.

## Open TODOs

### Before posting (blocking)

1. **Author.** Adrian Mittal alone (decided 2026-11-02), Department of Physics, Carnegie Mellon University;
   almittal@andrew.cmu.edu. v2 is new work that cites Ref. [v1] for the earlier results.
2. **Appendix B and the published accuracy.** It treats the retrained CNN-BiLSTM as v1's network and states
   that the published accuracy was obtained at an effective threshold of 4–5% and offsets of ±0.0125
   (decided 2026-11-02; the original training code never came). Every mention of the published 94.4% follows
   it: abstract, Introduction (paragraphs 2 and 5), Table 1 caption, Sec. 3.4, Sec. 3.5 (the sweep's baselines paragraph),
   Appendix B, Conclusions (limitations). `REPORT.md` (Sec. 2) keeps the hedged account of the rebuild.
3. **Acknowledgements** (added 2026-11-02, after the Conclusions as in v1) thank L.Q. English and the other
   coauthors of Ref. [v1] for the collaboration this work builds on. No funding to add (the user, 2026-11-04).
4. **Before uploading to arXiv** (which publishes the TeX source): strip every `%` comment from `main.tex`
   (e.g. with `arxiv_latex_cleaner`); check with `grep -n 'TODO' main.tex`.
5. **Data and code availability** gives the repository URL. The repository is public (checked 2026-11-04),
   and its default branch `main` was fast-forwarded to `v2-physics-guided` on 2026-11-04 (it had stopped at
   2026-10-21). Fast-forward `main` again (`git push origin v2-physics-guided:main`) after any later change,
   so the link in the paper shows the posted version. Optionally add a version tag or a Zenodo DOI, which archives a frozen copy of a tagged
   release, so the link keeps pointing at the code as it was when the paper was posted.

### Should check

6. **v1 cross-references** ("Sec. 2.1", "Sec. 2.3 and the Appendix", "Figs. 2–5 and 7", "Eq. (5)",
   "Fig. 6" in the email) were checked against the published JAND PDF.
7. **References added from memory** (not in `papers/jand2026_arxiv/refs.bib`), verified 2026-11-03 against
   live records: Crossref for `sievers1988intrinsic`, `flach2008discrete`, `virtanen2020scipy`,
   `karniadakis2021physics`, `liu1989limited`, `szegedy2016rethinking` (10.1109/CVPR.2016.308) and
   `cho2014learning` (10.3115/v1/D14-1179); Crossref's 1967 review for `magnus1966hill`; PMLR v48 for
   `cohen2016group`; jmlr.org for `srivastava2014dropout`; the NeurIPS BibTeX for `paszke2019pytorch`, which
   gives no pages, so the unverifiable 8024–8035 went; the arXiv API (authors in full, and the ICLR year in
   the arXiv comment) for `yu2016multiscale`, `kingma2015adam`, `loshchilov2017sgdr`,
   `loshchilov2019decoupled`, `oord2016wavenet`, `hendrycks2016gelu` and `li2021fourier`. Only the venue of
   `li2021fourier` (ICLR 2021) is from memory: OpenReview and DBLP answer scripts with a browser challenge.
   `smith2019super`, `chen2023symbolic` and `guo2017calibration` are still in `refs.bib` but not cited, so
   they do not reach `main.bbl`.
8. **Edits to references copied from v1's bibliography:** `thakur2007driven` says 2008 and "Basu Thakur, R."
   (Crossref, 10.1088/0022-3727/41/1/015503: family name Basu Thakur, online 2007-12-19, in the 2008 print
   volume 41), against v1's 2007 and "Thakur, R Basu"; `cuevas2014sine` drops the subtitle; SciPy, not in
   v1's list, is cited as Virtanen et al. (2020). The v2 forms are kept.
9. **Interpretive sentences**, checked 2026-11-03:
   - Sec. 3.1, "The energetic clusters identified in Ref. [v1] thus appear to be the growing long waves, seen
     through a low-pass filter": rests on Fig. 2(a) (the best width follows the band edge at seven amplitudes)
     and Fig. 2(c) (the five growing modes alone keep the site within ±3 in 91%), and says "appear".
   - Sec. 3.2, the CNN-BiLSTM design: confirmed in `sgloc/models.py`. Four convolutions of kernel 3 act on each
     frame separately, so a feature sees nine pendulums of one frame; each direction's LSTM input kernel is
     512 × 25,600 = 13,107,200 weights, so the two hold 26,214,400 of the 26,499,205 parameters (98.9%), each
     weight tied to one pendulum and channel of the flattened frame.
   - Sec. 3.7, the waiting rate: an interpretation, worded "A likely reason"; it predicted 24.8 time units on
     the swing-averaged label before those runs finished, against 25.5 measured.
   - Sec. 3.7, the wider network's failure from N = 100, "presumably because its two extra layers, with
     dilations of more than half the ring, connect pendulums in a way specific to the ring it was trained on":
     hedged. `REPORT.md` has the controlled comparison with v2 (the two networks differ only in those layers)
     and the caveat the paper no longer states: the same layers also make the view go around that ring more
     than twice, which the data cannot separate from the dilations.
   - Conclusions, "largely the projections of the early state onto the few growing long waves": rests on
     Fig. 2(c), the Floquet filter and the linear ring filter, and says "largely".
   - Sec. 3.5, the median profiles, checked run by run (`crossing_energy.json`, `runs_centre_above_both_pm2`
     and `runs_centre_is_max_of_5`). When the 5% label fires near a turning point, the center holds more than
     both pendulums two sites off in 98.8% of the 642 runs, so "not yet spread over its top" holds run by run.
     When the 10% label fires near a turning point, the center holds the most of the five pendulums from −2 to
     +2 in only 13.7% of the 7,199 runs and less than one of the pendulums two sites off in 75%, against 78.8%
     and 9.8% near the bottom, so the top is flat or dips at the center run by run, as the median says. Why the
     top is flat in the kinetic phase is still not derived. The 15% sentence said "no pendulum on a flank comes
     close", but run by run the center holds more than both pendulums two sites off in only 72% (bottom) and
     87% (turning point) of runs; it now gives the medians alone ("far above the 8.6–8.7%").
10. **Title** is a proposal: "Physics-guided machine-learning prediction of spontaneous energy
   localization in a sine-Gordon pendulum ring".
11. Once the v1 preprint has an arXiv ID, consider adding it to the `mittal2026jand` entry.

## Traceability

- 2026-11-03, the numbers an independent reader found only in `REPORT.md`, now traced to the data:
  "agree exactly in only a third" is 32.9% of the 19,949 test runs in which the swing-averaged label fires
  (`data/pool_test.npz`, `site_avg` against `site_right`), the averaged site lying 1–3 sites to the left in
  45.5% and to the right in 17.6%; with plain cross-entropy v2 misses by two sites in 26.2–27.0% and by one
  in 22.5–24.3% (three seeds, `runs/plain_cross_entropy_*/pred_test.npy`), and 51–53% of the two-site misses
  sit on the swing-averaged site; the 17-site stack (`runs.jsonl`: `rf17_v1label` 2.37 against 1.49, ×1.6;
  `rf17_avg` 1.78 against 0.58, ×3.1; `rf17_hump` 1.48 against 0.27, ×5.4); soft labels (`soft_off_avg`
  0.59 against 0.58, `soft_off_hump` 0.245 against 0.273, exact 88.3% against 84.7%).
- 2026-11-03, after the cuts: the ring paragraphs read `settings.json` `C.per_n` (growing modes 2 to 22,
  rival share 2.44% to 71.2%, 19.6% at N = 100; the wider network 99.85% to 98.84%), `threshold_hump.json`
  `rings` (10% lags 2.0, 5.5, 20.5; 48.6% and 82% within ±1 of the event-timed label at N = 200) and
  `ring_transfer.json` (v2 from N = 100: 97.37%, 99.44%, 94.31%, 89.47% at N = 80, 100, 150, 400, far misses
  0.42% to 9.51%; the wider network from N = 100: 16.6%, 12.8%, 29.7%, 14.5%).

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
- `lag_strata` in `results/threshold_hump.json` holds Sec. 3.5's lag-and-phase strata: by two-unit bins, at each
  sampled lag (`by_sample_lag`: the 30/99/19/85% at 10%), and the 15% label crossed 6–12 or 14 or more time units
  after the event, with v2 on its own label there. The event's own phase (96% in the quarter of a half period
  after a turning point) is `swing_phase` → `ott` → `by_eighth`, the first two eighths.
- Sec. 3.5's mechanism paragraph is `results/crossing_energy.json` (`scripts/crossing_energy.py`, about 4–8
  minutes on CPU, no training): the test runs integrated again and, at each 5/10/15% crossing, the labelled
  pendulum's energy split into its four parts of Eq. (2) (right-hand spring), pendulums beyond the top, and
  `profile_around_centre` (median share of the total at offsets −4..+4 from the hump centre), by swing phase.
  At 10%: near the bottom, 7 beyond the top, the label at |θ| = 6.43, global 0.80 of its energy, profile
  10.3% at the centre and 7.1–7.6% at ±2; near a turning point, 0 beyond the top, kinetic 0.89, profile 9.2%
  at the centre and 9.4–10.0% at ±1 and ±2; signed offsets +2 44.8%, −2 8.2% (even split: 28.7% and 27.7%,
  from `site_sym10` in `data/thrpool_full_test.npz`). The 5% label near a turning point: 5.1% at the centre,
  4.1% at ±2. The 15% label: 15.4% and 8.6–8.7%. Its 94/32/36% match `swing_phase` in `threshold_hump.json`.
  Before 2026-11-03 the paper said the energy "moves between a pendulum's motion and the springs on either
  side"; the springs hold under 5% of the labelled pendulum's energy, so that was replaced.
- The ring paragraph's label lags and agreements come from `rings` (runs in which the label fired; at N = 400
  censored at the end of the integration, `integration_end`).
- Sec. 3.6 and Fig. 7(c) use the Floquet filter with the number of periods chosen on validation from 3 to 20,
  per pool and, in Fig. 7(c), per amplitude bin (`periods` in `settings.json` and `floquet_periods` in
  `floquet_vs_ratio`: 5, 6 and 8 for 4×, 1× and 1/8× the offsets; 15 at A = 1.60, 4 from 1.85 up; 7, 6, 5, 5 on
  the rings; 3–15 overall, none at the top of the range), each run evolved at its estimated amplitude, the mean of
  its initial angles, as Sec. 2.4 says. Until 2026-11-03 the search stopped at 9 (A = 1.60 sat at that cap and
  scored 17%; it is 25% at 15), the bins reused their pool's period count (the lowest bin then stayed at 49–65%
  with small offsets; tuned per bin it reaches 89%), and the pools used their true amplitude.
- The hump-center predictor of Fig. 6(a) is the Eq. (6) site shifted by the constant offset that suits each
  label best on validation (one site at 7–12%, none otherwise; `baselines.*.hump_oracle`).
- The ring numbers come from the wider network (dilations up to 128, 28,994 parameters), including at
  N = 100 (99.5%, 0.24). They are not Table 3's standard network (99.4%, 0.28), which is Fig. 9(b)'s bottom
  row and the subject of Sec. 3.7's second ring paragraph.
- The views of 129 and 513 pendulums are what a pendulum's features see (Sec. 3.2's receptive field, 4 × the
  largest dilation + 1). The head's kernel-3 convolution adds one pendulum on each side to what a site score
  sees (131 and 515); both were measured on the saved weights. On rings of 150–400 pendulums, 1–3 of the 8–22
  growing modes (cutoff 18.1 pendulums at A = 1.75, `threshold_hump.json` rings) are longer than 129.
- Sec. 3.7's transfers and Fig. 9(b) are `results/ring_transfer.json` (`scripts/ring_transfer.py`): the wider network
  trained at 50, 100, 200 and 400 and v2 trained at 100, each scored on rings of 50, 80, 100, 150, 200 and 400 (the own
  rings repeat `settings.json` C.per_n and the pairs in C.cross repeat C.cross, exactly), each seed's accuracy, and
  the check that a network on a ring whose size divides its training ring's computes what it computes there with
  the state repeated (saved weights, 64 random states, seven pairs: scores within about 1e-4 on scores of up to about
  1,200, the same predicted site in every state). The losses on repeated states (under a point twice, 4–6 points
  four or five times, 16 points eight times) are differences between cells of that matrix. Its new test pools
  (`data/setpool_n80_test.npz`, `data/setpool_n150_test.npz`, 10,000 runs each) are git-ignored; the commands that
  rebuild them are in the script's docstring.
- Sec. 3.6's Floquet accuracies (52/73/87% against the offset scale; 89–99% where η_max/(A − A_c) < 0.04 and
  25–35% where it is about one, 25% at A = 1.60) are scores against
  the event-timed hump label (`settings_analysis.py`, `floquet_baseline` with L = eh_k3). An earlier draft
  called them scores against the first pendulum over the top.
