# Papers

Each paper is a self-contained LaTeX folder: `main.tex`, its bibliography, a pre-built `main.bbl` and `figures/`. Each compiles with pdfLaTeX on its own, locally, on Overleaf or on arXiv.

| Folder | Paper | Status |
|---|---|---|
| [jand2026_arxiv/](jand2026_arxiv/) | Mittal, Grishanov, Pham, Lape and English, "Machine-learning prediction of type-III instabilities and spontaneous energy localization", *J. Appl. Nonlinear Dyn.* **15**(2), 277–290 (2026) | Published, open access under CC BY 4.0. This folder holds the published text with its typographical errors corrected. It will not be posted to arXiv; the v2 preprint cites the journal version. |
| [v2_preprint/](v2_preprint/) | Mittal, "Physics-guided machine-learning prediction of spontaneous energy localization in a sine-Gordon pendulum ring" | Draft, ready for a final review. Open items are listed in [NOTES.md](v2_preprint/NOTES.md). |

## Building

```bash
cd papers/v2_preprint        # or papers/jand2026_arxiv
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload this folder as a zip (New Project → Upload Project). Then choose the paper to compile under Menu → Main document: `jand2026_arxiv/main.tex` or `v2_preprint/main.tex`.

## Posting to arXiv

Only the v2 preprint will be posted. arXiv compiles with pdfLaTeX but does not run BibTeX, so the upload is `main.tex`, `main.bbl`, `fig_schematic.tex` (a TikZ figure) and `figures/`:

```bash
(cd papers/v2_preprint && tar -czf ../v2_preprint.tar.gz main.tex main.bbl fig_schematic.tex figures)
```

After changing a citation, run BibTeX again and commit the new `main.bbl`. The open items before posting are in [v2_preprint/NOTES.md](v2_preprint/NOTES.md).

## License

Both papers are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The published paper is © 2026 The Authors. The code in the rest of the repository is under the [MIT License](../LICENSE).
