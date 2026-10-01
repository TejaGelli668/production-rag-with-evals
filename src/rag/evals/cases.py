"""Eval cases: FinanceBench questions by split, plus hand-written unanswerable cases."""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag.data.financebench import PROJECT_ROOT, Evidence, load_questions
from rag.data.splits import load_split

CUSTOM_DIR = PROJECT_ROOT / "evals" / "custom"
UNANSWERABLE_PATH = CUSTOM_DIR / "unanswerable.jsonl"


class EvalCase(BaseModel):
    case_id: str
    question: str
    answerable: bool
    gold_answer: str  # for unanswerable cases: what a correct refusal should say
    justification: str | None = None
    doc_name: str | None = None  # the filing the question is about (None if not in corpus)
    evidence: list[Evidence] = []
    # tags[0] is the primary grouping key for reports.
    tags: list[str] = []
    # Where the gold label came from (human-written, human-verified, or model-drafted).
    source: str = "FinanceBench (human-written by financial analysts)"

    @property
    def gold_pages(self) -> set[tuple[str, int]]:
        return {(e.doc_name, e.evidence_page_num) for e in self.evidence}


def _reasoning_tag(label: str | None) -> str:
    """Collapse FinanceBench's free-text reasoning labels to a few buckets."""
    if not label:
        return "unlabeled"
    label = label.lower()
    if "logical" in label:
        return "logical"
    if "numerical" in label:
        return "numerical"
    return "extraction"


def load_cases(split: str) -> list[EvalCase]:
    """`dev`, `test`, `ci_smoke` (FinanceBench), or `unanswerable` (custom)."""
    if split == "unanswerable":
        return [
            EvalCase.model_validate(json.loads(line))
            for line in UNANSWERABLE_PATH.read_text().splitlines()
            if line.strip()
        ]
    questions = {q.financebench_id: q for q in load_questions()}
    return [
        EvalCase(
            case_id=q.financebench_id,
            question=q.question,
            answerable=True,
            gold_answer=q.answer,
            justification=q.justification,
            doc_name=q.doc_name,
            evidence=q.evidence,
            tags=[q.question_type, _reasoning_tag(q.question_reasoning)],
        )
        for q in (questions[i] for i in load_split(split))
    ]
