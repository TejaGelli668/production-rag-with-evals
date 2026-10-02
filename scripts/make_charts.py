"""Render README charts from committed run summaries (evals/results/*.json).

Each chart is written in a light and a dark variant (docs/img/<name>-{light,dark}.svg)
so the README can serve the one matching the viewer's GitHub theme.

    uv run python scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
OUT = ROOT / "docs" / "img"

# Reference palette tokens: one accent hue for the configuration that was kept,
# a muted neutral for everything else. Every bar carries a direct value label.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "accent": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "accent": "#3987e5",
    },
}


def load(run: str) -> dict | None:
    path = RESULTS / f"{run}.json"
    return json.loads(path.read_text()) if path.exists() else None


def bar_chart(
    name: str, title: str, subtitle: str, rows: list[tuple[str, dict, bool]], xlabel: str
) -> None:
    """Horizontal bars with 95% CI whiskers. rows: (label, metric stat, is_accent)."""
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.3), dpi=100)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        ys = range(len(rows))[::-1]
        for y, (_label, stat, accent) in zip(ys, rows, strict=True):
            mean, (lo, hi) = stat["mean"], stat["ci95"]
            ax.barh(
                y,
                mean,
                height=0.5,
                color=t["accent"] if accent else t["muted"],
                edgecolor=t["surface"],
                linewidth=2,
                zorder=2,
            )
            ax.plot([lo, hi], [y, y], color=t["text2"], linewidth=1.2, zorder=3)
            ax.text(
                hi + 0.012,
                y,
                f"{mean:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color=t["text"],
                fontweight="bold" if accent else "normal",
            )
        ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=9, color=t["text"])
        upper = max(stat["ci95"][1] for _, stat, _ in rows)
        ax.set_xlim(0, min(1.0, round(upper + 0.15, 1)))
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", colors=t["text2"], labelsize=8)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.set_xlabel(xlabel, fontsize=8, color=t["text2"])
        # Title block sits a fixed distance (in inches) from the top at any chart height.
        height = fig.get_figheight()
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / height))
        fig.text(
            0.02,
            1 - 0.12 / height,
            title,
            ha="left",
            va="top",
            fontsize=11,
            color=t["text"],
            fontweight="bold",
        )
        fig.text(
            0.02, 1 - 0.38 / height, subtitle, ha="left", va="top", fontsize=8, color=t["text2"]
        )
        fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=t["surface"])
        plt.close(fig)
    print(f"wrote {name}-{{light,dark}}.svg")


def retrieval_chart() -> None:
    configs = [
        ("E1 dense baseline", "e1_full", False),
        ("E2 page-bounded chunks", "e2_page", False),
        ("E2 250-token chunks", "e2_fixed250", False),
        ("E5 company filter", "e5_company", False),
        ("E5 company + year filter", "e5_company_year", False),
        ("E6 rerank top-50", "e6_rerank", False),
        ("E4 hybrid (BM25)", "e4_hybrid", False),
        ("filter + hybrid", "stack_filter_hybrid", False),
        ("filter + rerank (kept)", "stack_filter_rerank", True),
        ("filter + hybrid + rerank", "stack_full", False),
    ]
    rows = [
        (label, s["metrics"]["page_hit@5"], accent)
        for label, run, accent in configs
        if (s := load(f"{run}__dev__retrieval"))
    ]
    bar_chart(
        "retrieval_page_hit",
        "Finding the right page: page hit@5",
        "FinanceBench dev (n=50), all 360 filings · whiskers: bootstrap 95% CI",
        rows,
        "share of questions with a gold evidence page in the top 5 chunks",
    )


def end_to_end_chart() -> None:
    configs = [
        ("closed-book (no retrieval)", "e0_closed_book__dev", False),
        ("E1 baseline (full corpus)", "e1_full__dev", False),
        ("filter + rerank (kept)", "stack_filter_rerank__dev", True),
        ("oracle (gold pages given)", "e0_oracle__dev", False),
    ]
    rows = [
        (label, s["metrics"]["correct"], accent)
        for label, run, accent in configs
        if (s := load(run))
    ]
    if len(rows) < len(configs):
        print("end_to_end: some runs missing, skipped")
        return
    bar_chart(
        "end_to_end_correct",
        "Answer accuracy",
        "FinanceBench dev (n=50) · generator and judge: qwen3:14b · whiskers: bootstrap 95% CI",
        rows,
        "share of questions answered correctly",
    )


def test_chart() -> None:
    configs = [
        ("E1 baseline (dense)", "e1_full__test", False),
        ("filter + rerank (final)", "stack_filter_rerank__test", True),
    ]
    rows = [
        (label, s["metrics"]["correct"], accent)
        for label, run, accent in configs
        if (s := load(run))
    ]
    if len(rows) < len(configs):
        print("test: some runs missing, skipped")
        return
    bar_chart(
        "test_correct",
        "Held-out test: answer accuracy",
        "FinanceBench test (n=100), run once after all tuning · whiskers: bootstrap 95% CI",
        rows,
        "share of questions answered correctly",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    retrieval_chart()
    end_to_end_chart()
    test_chart()
