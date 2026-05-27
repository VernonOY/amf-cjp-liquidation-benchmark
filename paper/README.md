# Manuscript Source

This directory contains the LaTeX source and compiled PDF for the Applied Mathematical Finance submission:

**Sample Complexity of Calibration versus Model-Free Learning in Cartea--Jaimungal--Penalva Optimal Liquidation**

## Main Files

| File or directory | Purpose |
|---|---|
| `main.tex` | Main manuscript source |
| `main.pdf` | Compiled manuscript PDF |
| `refs.bib` | BibTeX bibliography |
| `figures/` | Figure PDFs included by the manuscript |
| `tables/` | Generated LaTeX tables included by the manuscript |
| `interact.cls` and bundled `.sty` files | Taylor & Francis Interact LaTeX support files |

## Compile

The local build uses Tectonic:

```bash
tectonic -X compile main.tex
```

From the repository root:

```bash
cd paper
tectonic -X compile main.tex
```

The root project README describes how to regenerate figures and data.
