"""FinanceBench dataset: typed models, loaders, and corpus selection.

Source: https://github.com/patronus-ai/financebench (CC-BY-NC 4.0).
Files are fetched by `scripts/download_financebench.py` into `data/financebench/`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

REPO = "patronus-ai/financebench"
# Pinned for reproducibility: the upstream repo has been stable since this commit (2024-12-03).
PINNED_SHA = "cc39aeb4afdf33909ee1412188bf89035950c2eb"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "financebench"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
PDF_DIR = DATA_DIR / "pdfs"


class CorpusMode(StrEnum):
    FOCUSED = "focused"  # only documents referenced by the 150 questions
    FULL = "full"  # every document with metadata


class Evidence(BaseModel):
    evidence_text: str
    doc_name: str
    # Page index as labeled upstream. 0-based: verified against PyMuPDF page indices,
    # see docs/data_exploration.md.
    evidence_page_num: int
    evidence_text_full_page: str


class Question(BaseModel):
    financebench_id: str
    company: str
    doc_name: str
    question_type: Literal["metrics-generated", "domain-relevant", "novel-generated"]
    question_reasoning: str | None
    domain_question_num: str | None
    question: str
    answer: str
    justification: str | None
    evidence: list[Evidence]


class Document(BaseModel):
    doc_name: str
    company: str
    gics_sector: str
    doc_type: str  # 10k | 10q | 8k | earnings | 10k_annualreport
    doc_period: int
    doc_link: str

    @property
    def pdf_path(self) -> Path:
        return PDF_DIR / f"{self.doc_name}.pdf"


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    return [Question.model_validate(row) for row in _read_jsonl(path)]


def load_documents(path: Path = DOCUMENTS_PATH) -> list[Document]:
    """Load document metadata, de-duplicated by `doc_name`.

    Upstream lists FOOTLOCKER_2023_annualreport twice (doc_period 2023 and 2022);
    the first occurrence wins.
    """
    seen: dict[str, Document] = {}
    for row in _read_jsonl(path):
        doc = Document.model_validate(row)
        seen.setdefault(doc.doc_name, doc)
    return list(seen.values())


def select_documents(
    mode: CorpusMode, documents: list[Document], questions: list[Question]
) -> list[Document]:
    if mode is CorpusMode.FULL:
        return documents
    referenced = {q.doc_name for q in questions}
    return [d for d in documents if d.doc_name in referenced]
