"""Human labels for calibrating the correctness judge.

`rag label <run>` shows each answer from an eval run next to the gold answer and
records your verdict in evals/judge/human_labels.jsonl (committed). Judge grades
are hidden while you label, so they can't anchor you. `rag calibrate <run>` then
reports how often the judge agrees with you, with Cohen's kappa.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from rag.data.financebench import PROJECT_ROOT

LABELS_PATH = PROJECT_ROOT / "evals" / "judge" / "human_labels.jsonl"
_KEYS = {"c": "correct", "i": "incorrect", "s": "skip", "q": "quit"}


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_labels(path: Path = LABELS_PATH) -> dict[tuple[str, str], str]:
    """(prompt_id, answer text) -> human verdict. Keyed on the answer, so a label
    stays attached to the exact text that was judged."""
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            labels[(row["prompt_id"], row["answer"])] = row["label"]
    return labels


def label_run(run_dir: Path, path: Path = LABELS_PATH, input_fn=input) -> int:
    """Interactively label the non-refused answers of a run. Returns how many were added."""
    labels = load_labels(path)
    todo = [
        r
        for r in _rows(run_dir)
        if r["grade"].get("refused") == 0.0
        and "correct" in r["grade"]
        and (r["prompt_id"], r["answer"]) not in labels
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    for i, row in enumerate(todo, start=1):
        print(f"\n{'─' * 100}\n[{i}/{len(todo)}] {row['prompt_id']}\n")
        for title, text in (
            ("QUESTION", row["prompt"]),
            ("GOLD", row["gold"]),
            ("ANSWER", row["answer"]),
        ):
            print(f"{title}:")
            print(textwrap.indent(textwrap.fill(text, 96, replace_whitespace=False), "  "))
        key = ""
        while key not in _KEYS:
            key = input_fn("\nIs the ANSWER correct? [c]orrect / [i]ncorrect / [s]kip / [q]uit: ")
            key = key.strip().lower()[:1]
        if key == "q":
            break
        if key == "s":
            continue
        with path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "prompt_id": row["prompt_id"],
                        "answer": row["answer"],
                        "label": _KEYS[key],
                        "run": run_dir.name,
                    }
                )
                + "\n"
            )
        added += 1
    return added


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    categories = {x for pair in pairs for x in pair}
    expected = sum(
        (sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n)
        for c in categories
    )
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def calibrate(run_dir: Path, path: Path = LABELS_PATH) -> dict[str, Any]:
    """Agreement between the run's correctness judge and human labels on the same answers."""
    labels = load_labels(path)
    pairs, disagreements = [], []
    for row in _rows(run_dir):
        human = labels.get((row["prompt_id"], row["answer"]))
        if human is None or "correct" not in row["grade"] or row["grade"].get("refused"):
            continue
        judge = "correct" if row["grade"]["correct"] == 1.0 else "incorrect"
        pairs.append((human, judge))
        if human != judge:
            disagreements.append(
                {
                    "prompt_id": row["prompt_id"],
                    "human": human,
                    "judge": judge,
                    "judge_reasoning": row["explanation"].get("correct", ""),
                }
            )
    return {
        "run": run_dir.name,
        "labeled": len(pairs),
        "agreement": sum(h == j for h, j in pairs) / len(pairs) if pairs else float("nan"),
        "cohens_kappa": cohens_kappa(pairs),
        "disagreements": disagreements,
    }
