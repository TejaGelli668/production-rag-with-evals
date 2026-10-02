"""Error analysis: attribute each answer to the pipeline stage where it went wrong.

Stages are checked in pipeline order, and each case lands in the first one that failed:

    wrong_filing        no top-k chunk from the gold filing
    wrong_page          right filing, but no chunk overlapping a gold page
    evidence_missing    a gold page was retrieved, but under half the evidence text was
    refused_with_evidence   the evidence was there, yet the model declined
    wrong_with_evidence     the evidence was there, yet the answer was wrong
    correct

The first three are retrieval failures; the next two are generation failures.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

STAGES = [
    "wrong_filing",
    "wrong_page",
    "evidence_missing",
    "refused_with_evidence",
    "wrong_with_evidence",
    "correct",
]
EVIDENCE_THRESHOLD = 0.5


def classify(grade: dict[str, float], k: int = 5) -> str:
    if grade.get("correct") == 1.0:
        return "correct"
    if grade.get(f"doc_hit@{k}") == 0.0:
        return "wrong_filing"
    if grade.get(f"page_hit@{k}") == 0.0:
        return "wrong_page"
    if grade.get(f"evidence_coverage@{k}", 1.0) < EVIDENCE_THRESHOLD:
        return "evidence_missing"
    if grade.get("refused") == 1.0:
        return "refused_with_evidence"
    return "wrong_with_evidence"


def analyze(run_dir: Path, k: int = 5) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["status"] == "ok" and "correct" in r["grade"]]
    stages = {r["prompt_id"]: classify(r["grade"], k) for r in rows}
    counts = Counter(stages.values())
    examples: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        stage = stages[r["prompt_id"]]
        if stage != "correct" and len(examples.setdefault(stage, [])) < 3:
            examples[stage].append(
                {
                    "id": r["prompt_id"],
                    "question": r["prompt"][:120],
                    "answer": r["answer"][:160],
                    "gold": r["gold"][:100],
                }
            )
    return {
        "run": run_dir.name,
        "cases": len(rows),
        "counts": {s: counts.get(s, 0) for s in STAGES},
        "examples": examples,
    }


def render(report: dict[str, Any]) -> str:
    n = report["cases"]
    lines = [
        f"### Error analysis · {report['run']} · {n} cases",
        "",
        "| Stage | Cases | Share |",
        "|---|---|---|",
    ]
    for stage, count in report["counts"].items():
        lines.append(f"| {stage} | {count} | {count / n:.0%} |" if n else f"| {stage} | 0 | - |")
    retrieval = sum(report["counts"][s] for s in STAGES[:3])
    generation = sum(report["counts"][s] for s in STAGES[3:5])
    lines += ["", f"Retrieval failures: {retrieval} · generation failures: {generation}"]
    return "\n".join(lines)
