# Errors in the published version

Mittal, Grishanov, Pham, Lape and English, J. Appl. Nonlinear Dyn. 15(2), 277–290 (2026), doi:10.5890/JAND.2026.06.001.

Checked against the journal PDF, the Overleaf source and the original simulation scripts. The arXiv version in this folder corrects everything in sections 1 and 4, and the references in section 6 except [5]. Sections 2 and 3 need a decision from the authors, so it leaves them unchanged. Section 5 concerns only the journal's typeset PDF.

## 1. Errors that change the meaning

| Where | Published | Should be |
|---|---|---|
| Sec. 2.1, parameter values | h̃ = 0.05 | h̃ = 0.5. Sec. 2.3 and every simulation script use 0.5. With 0.05 the uniform swing has no threshold near A = 1.59 and the instability is not type III. |
| Sec. 2.2, Eq. (6) | Σ from −k to +k of f[i+k] | Σ over j from −k to +k of f[i+j]. k is used as both the half-width and the summation index. |
| Sec. 3.3, first paragraph | "each of the 6 different kernel sizes" | 7. Fig. 4 shows kernel sizes 1, 5, 10, 15, 20, 25 and 30. |
| Sec. 3.4 | "(Section III.C)" | Section 3.3 |
| Sec. 1, ML paragraph | "a CNN and a LSTM transformer" | a CNN and a bidirectional LSTM. The model has no transformer. |
| Sec. 4, second paragraph | "training a CNN … (from 0 ≤ t ≤ 5)" | a CNN-LSTM, with input window 0 ≤ t ≤ 4.5 (as in Secs. 2.3 and 3.4) |

## 2. Where the text and the code disagree

These are not typos. They need a decision from the authors before anything is changed.

- **Time axes.** Figs. 2–5 and 7 plot sample indices (Δτ = 0.5), not τ. So localization happens at τ ≈ 57, not "t ≈ 100", and the box-car comparison at "t = 5" is τ = 2.5.
- **Eq. (5) and the code book each spring's energy to opposite neighbours.** Eq. (5) books it to the left neighbour (θ_n − θ_{n−1}). The code books it to the right (θ_n − θ_{n+1}). The statistics are the same by symmetry, but the labelled site can differ. The code's energy is also 2 × Eq. (5), which does not change the 10% threshold.
- **Even kernel sizes.** Eq. (6) describes odd windows (2k + 1). Kernel sizes 10, 20 and 30 are even. The code averages K sites and shifts the result by ⌊K/2⌋.
- **Input scaler.** The StandardScaler was fit before the train/test split, as Sec. 2.3 describes. This leaks a little test-set information.
- **ML accuracy.** The rebuilt model does not reproduce the reported accuracy: about 60% within ±1, against the published 94.4%. See [REPORT.md, Sec. 2](../../REPORT.md#2-what-this-work-corrects).

## 3. Inconsistent numbers

- **Best kernel size.** It is given as 12–18 (Sec. 3.3), 15–20 and 17 (Fig. 5 and Sec. 3.3), and about 15 (Sec. 4). Sec. 3.3 calls the cluster size "10–15 pendulums".
- **Noise level.** Sec. 2.1 gives the noise amplitude as 0.025, which is the half-width. Fig. 7's legend gives "Noise Level = 0.05 / 0.1", which is the full width. The text then gives the same two levels as 1.4% and 2.8%.
- **Repeated sentence.** Sec. 3.4 gives "94.38% … within ±1" twice in consecutive sentences.

## 4. Typographical errors

- Fig. 3 caption: "average localized time" should be "average localization time".
- Fig. 5 caption: "We circles represent" should be "The circles represent". "box-cart" should be "box-car". The last sentence is garbled: "The x-axis represents different kernel sizes experimentation for box-cart smoothing only, which is irrelevant to the solid line."
- Sec. 3.3:
  - "at the timestamps of 5" should be "at a timestamp of 5".
  - "most efficient kernel size" should be "best" or "most accurate".
- Sec. 3.4: "1.44 pendula" should be "pendulums", as used everywhere else.
- Sec. 3.5: "Impurities of the opposite sign, repel energy" has a stray comma.
- Sec. 4:
  - "choatic" should be "chaotic".
  - "This strongly suggest" should be "suggests".
- Sec. 2.2: "Python's odeint()". odeint is SciPy's.
- Sec. 2.3:
  - Code names have mismatched quotes, e.g. ‘odeint‘ and ‘StandardScaler‘.
  - There are stray spaces before commas: "library ,", "‘StandardScaler‘ ,", and in the Appendix "localization accuracy ,".
  - Items 3 and 4 of the model list run together on one line.
  - "Location Accuracy(with tolerance)" is missing a space.
- Appendix: the two output-layer bullets were typed as "-" and print inline.

## 5. Front matter

- **Copyright line.** The abstract box says "©2026 L&H Scientific Publishing, LLC. All rights reserved." The page footer and the article's web page say "© 2026 The Authors", under CC BY 4.0. The journal should fix the abstract box, since it contradicts the license.

## 6. References

- **Proper nouns printed in lowercase.**
  - [3] "tacoma bridge"
  - [4] "tacoma narrows bridge"
  - [11] and [14] "sine-gordon"
  - [14] "lugiato-lefever"
  - [15] "josephson"
  - [17] "lyapunov"
  - [20] "II. influence"
  - [23] "python"
- **[11]** is a Springer book: Cuevas-Maraver, Kevrekidis and Williams (eds.), *The sine-Gordon Model and its Applications*, Nonlinear Systems and Complexity vol. 10, Springer, 2014. It is printed as a journal article, "Nonlinear Systems and Complexity, 10".
- **[5]** has no article number or DOI. It is still missing in the arXiv version, because the article could not be found to look them up.
