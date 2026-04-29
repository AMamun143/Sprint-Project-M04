#!/usr/bin/env python3
"""Generate semantic-axis scatter plot for Sprint M04.

Case study: U.S. universities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def make_axis(
    positive_words: list[str], negative_words: list[str], vectorizer: TfidfVectorizer
) -> np.ndarray:
    """Return unit semantic axis vector."""
    pos = vectorizer.transform(positive_words).toarray()
    neg = vectorizer.transform(negative_words).toarray()
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)
    neg = neg / (np.linalg.norm(neg, axis=1, keepdims=True) + 1e-12)
    vec = pos.mean(axis=0) - neg.mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1e-12)


def score_terms(
    terms: list[str], axis: np.ndarray, vectorizer: TfidfVectorizer
) -> np.ndarray:
    emb = vectorizer.transform(terms).toarray()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb @ axis


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "universities.csv"
    figs_dir = root / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_path = figs_dir / "scatter.png"

    df = pd.read_csv(data_path)
    terms = df["name"].astype(str).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))

    # Axis 1: prestige/selective (+) vs access/teaching-focused (-)
    axis1_pos = [
        "elite university",
        "highly selective admissions",
        "prestigious private college",
        "top ranked research institution",
    ]
    axis1_neg = [
        "open enrollment college",
        "teaching focused school",
        "community focused campus",
        "accessible local college",
    ]

    # Axis 2: technical/engineering (+) vs arts/humanities (-)
    axis2_pos = [
        "engineering and technology",
        "computer science heavy curriculum",
        "applied STEM research",
        "technical institute",
    ]
    axis2_neg = [
        "liberal arts education",
        "humanities and fine arts",
        "classical studies focus",
        "arts and letters tradition",
    ]

    corpus = (
        terms
        + axis1_pos
        + axis1_neg
        + axis2_pos
        + axis2_neg
        + ["public university", "private college", "state school", "research university"]
    )
    vectorizer.fit(corpus)

    axis1 = make_axis(axis1_pos, axis1_neg, vectorizer)
    axis2 = make_axis(axis2_pos, axis2_neg, vectorizer)

    # Diagnostics for axis quality.
    pos1 = vectorizer.transform(axis1_pos).toarray()
    neg1 = vectorizer.transform(axis1_neg).toarray()
    pos2 = vectorizer.transform(axis2_pos).toarray()
    neg2 = vectorizer.transform(axis2_neg).toarray()
    pos1 = (pos1 / (np.linalg.norm(pos1, axis=1, keepdims=True) + 1e-12)).mean(axis=0)
    neg1 = (neg1 / (np.linalg.norm(neg1, axis=1, keepdims=True) + 1e-12)).mean(axis=0)
    pos2 = (pos2 / (np.linalg.norm(pos2, axis=1, keepdims=True) + 1e-12)).mean(axis=0)
    neg2 = (neg2 / (np.linalg.norm(neg2, axis=1, keepdims=True) + 1e-12)).mean(axis=0)
    pole_dist_1 = 1 - np.dot(pos1, neg1) / (np.linalg.norm(pos1) * np.linalg.norm(neg1))
    pole_dist_2 = 1 - np.dot(pos2, neg2) / (np.linalg.norm(pos2) * np.linalg.norm(neg2))
    axis_cos = float(np.dot(axis1, axis2))

    df["x"] = score_terms(terms, axis1, vectorizer)
    df["y"] = score_terms(terms, axis2, vectorizer)

    # Colorblind-friendly palette (avoid red+green pairing).
    region_palette = {
        "Northeast": "#0072B2",  # blue
        "Midwest": "#E69F00",  # orange
        "South": "#CC79A7",  # purple
        "West": "#56B4E9",  # sky blue
    }
    default_color = "#595959"

    # Shape by institutional type.
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8", "p", "d"]
    marker_map = {}
    for i, t in enumerate(sorted(df["type"].dropna().unique().tolist())):
        marker_map[t] = marker_cycle[i % len(marker_cycle)]
    default_marker = "o"

    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)

    for (inst_type, region), g in df.groupby(["type", "region"], dropna=False):
        color = region_palette.get(region, default_color)
        marker = marker_map.get(inst_type, default_marker)
        ax.scatter(
            g["x"],
            g["y"],
            s=48,
            c=color,
            marker=marker,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )

    # Add non-overlapping anchor labels: only extremes.
    anchors = pd.concat(
        [
            df.nsmallest(3, "x"),
            df.nlargest(3, "x"),
            df.nsmallest(3, "y"),
            df.nlargest(3, "y"),
        ]
    ).drop_duplicates(subset=["name"])
    for _, row in anchors.iterrows():
        ax.annotate(
            row["name"],
            (row["x"], row["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.95,
        )

    ax.axhline(0, color="#8a8a8a", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#8a8a8a", linewidth=0.8, linestyle="--")

    ax.set_title("Semantic map of U.S. universities", fontsize=16, weight="bold")
    ax.set_xlabel("Access/teaching  <----->  Prestige/selective", fontsize=12)
    ax.set_ylabel("Arts/humanities  <----->  Technical/STEM", fontsize=12)
    ax.grid(alpha=0.15)

    # Build legends: region for color, type for marker.
    region_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=8,
            label=r,
        )
        for r, c in region_palette.items()
    ]
    type_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=m,
            color="#4a4a4a",
            linestyle="None",
            markersize=7,
            label=t,
        )
        for t, m in marker_map.items()
    ]

    leg1 = ax.legend(handles=region_handles, title="Region (color)", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, title="Type (shape)", loc="lower right")

    footnote = (
        f"Axis diagnostics - pole distance: axis1={pole_dist_1:.3f}, axis2={pole_dist_2:.3f}; "
        f"axis cosine={axis_cos:.3f}"
    )
    fig.text(0.01, 0.01, footnote, fontsize=9, color="#444444")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    print(footnote)


if __name__ == "__main__":
    main()
