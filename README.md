# Sprint M04 — Semantic Axes

Build a **semantic map**: pick terms, design two semantic axes from opposing word sets, produce one publication-quality 2D scatterplot.

> [!IMPORTANT]
> **Fork first.** Without forking, your Codespace opens on the original repo and you can't push. Fork → clone your fork → work on it.

## To get started 

1. Go [Molab notebook](https://molab.marimo.io/notebooks/nb_9zEp2dqxRbXDrrbQcieFDK). Or clone and open `assignment.py` with `uvx marimo edit --sandbox assignment.py`. This is a **worked example** on the cities dataset — not the deliverable.
2. Pick a case study, build your own pipeline.


You can choose one of the three datasets, or bring your own.

| File | Case study | N | Extra columns |
|---|---|---|---|
| `data/universities.csv` | U.S. higher-ed institutions | 157 | `type`, `region` |
| `data/sp500.csv` | S&P 500 (sample) | 203 | `sector` |
| `data/chemicals.csv` | Chemicals / materials | 179 | `class` |

If you bring your own, document the source and curation process. Requirements: ≥ 100 terms, ≥ 1 categorical attribute for color/shape.

## Tasks

### 1. Two semantic axes

Each axis: 3–6 words for **+ pole**, 3–6 for **− pole**. Good axis:

- Well-separated poles (cosine distance between centroids ≥ 0.3).
- Spreads the data, not piled at midpoint.
- One-sentence interpretation.

Axes should capture different, ideally orthogonal aspects — redundant axes waste half the plot.

### 2. One scatterplot

Plot each term at `(axis1, axis2)`. Encode categorical/ordinal attributes with **color** and **shape**. Follow data-viz principles:

- Clarity: readable symbols/text and no overlapping labels. Redundantly encode with shape
- Colorblind-friendly: No green and red colors in the same plot.
- Pre-attentive attention: use color/size/position to pull the eye to your story.
- Gestalt: proximity, similarity; zero lines, quadrant annotations, text anchors help.

### 3. Observations

2–4 paragraphs in notebook or `NOTE.md` placed at the repository root:

- What separates along each axis?
- Most **surprising** point/group — what does it say about the embedding?
- What would a **third axis** capture?

## Deliverable

Your repo must have:

- Code for the figure (marimo / Jupyter / `.py`).
- Reproducible pipeline — `run.sh` / Makefile / Snakemake that regenerates the figure from scratch.
- Raw data (CSV in `data/`).
- Final figure (PNG/PDF in `figs/` folder).
- Observations inline or in `NOTE.md` in the project root.

Submit by pushing to github and posting the URL to Brightspace.

## Submission (Completed)

- Student: Abdullah Mamun
- Final observations: `NOTE.md`
- Reproducible pipeline: `bash run.sh`
- Final figure output: `figs/scatter.png`

## Evaluation

| Criterion | What we look for |
|---|---|
| Atomic git history | Small focused commits, meaningful messages. Not `final`, `final2`, `final-real`. |
| Reproducible pipeline | `bash run.sh` regenerates data + figure on a fresh clone (or Snakefile or Makefile). No manual steps. |
| Documentation | Explains *why* each axis and *what* the figure shows. |
| Viz quality | Clear, separated, colorblind-friendly, deliberate. |
| Task completion | Two axes, one scatterplot, observations — all present. |

## FAQ

**Teams?** Yes. Team repo, list members in `NOTE.md`, all submit same URL.

**Embedding model?** You can use any models. Smaller is ideal in light of reproducibility. Larger OK, document the model used.

**I could not find meaningful patterns. Get Unimodal/boring axis** That's information. Try named entities over abstractions, or poles that *should* separate your data. 
