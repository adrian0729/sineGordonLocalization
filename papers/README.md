# Papers

Each paper is a self-contained LaTeX folder: `main.tex`, its bibliography, a pre-built `main.bbl` and `figures/`. Each compiles with pdfLaTeX on its own, locally, on Overleaf or on arXiv.

| Folder | Paper | Status |
|---|---|---|
| [jand2026_arxiv/](jand2026_arxiv/) | Mittal, Grishanov, Pham, Lape and English, "Machine-learning prediction of type-III instabilities and spontaneous energy localization", *J. Appl. Nonlinear Dyn.* **15**(2), 277–290 (2026) | Published, open access under CC BY 4.0. This is the arXiv version: the published text with its typographical errors corrected. Not yet posted to arXiv. |
| [v2_preprint/](v2_preprint/) | "Physics-guided machine-learning prediction of spontaneous energy localization in a sine-Gordon pendulum ring" | Draft. Open items are listed in [NOTES.md](v2_preprint/NOTES.md). Sec. 5, on reproducing the published model, is still being discussed with the published paper's coauthors. |

## Building

```bash
cd papers/v2_preprint        # or papers/jand2026_arxiv
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload this folder as a zip (New Project → Upload Project). Then choose the paper to compile under Menu → Main document: `jand2026_arxiv/main.tex` or `v2_preprint/main.tex`.

## Posting to arXiv

arXiv compiles with pdfLaTeX but does not run BibTeX, so each upload is `main.tex`, `main.bbl` and `figures/`:

```bash
cd papers/jand2026_arxiv && tar -czf ../jand2026_arxiv.tar.gz main.tex main.bbl figures
```

After changing a citation, run BibTeX again and commit the new `main.bbl`. The values for the arXiv form are in [jand2026_arxiv/README.md](jand2026_arxiv/README.md) and [v2_preprint/NOTES.md](v2_preprint/NOTES.md).

## License

Both papers are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The published paper is © 2026 The Authors. The code in the rest of the repository is under the [MIT License](../LICENSE).
