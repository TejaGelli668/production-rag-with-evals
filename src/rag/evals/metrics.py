"""Programmatic metrics: retrieval quality, refusals, numeric match, citations.

These are deterministic and free; the LLM judges (judges.py) cover what needs judgment.
"""

from __future__ import annotations

import re

from rag.evals.cases import EvalCase
from rag.generate.citations import cited_markers
from rag.generate.prompts import INSUFFICIENT
from rag.schema import RetrievedChunk

# ---------------------------------------------------------------- retrieval


_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _words(text: str) -> set[str]:
    # Drop thousands separators so "1,577" and "1577" compare equal.
    return set(_WORD.findall(re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())))


def evidence_coverage(retrieved: list[RetrievedChunk], case: EvalCase) -> float:
    """Fraction of the gold evidence's words present anywhere in the retrieved text.

    Catches what page overlap misses: a chunk can overlap the gold page without
    containing the evidence (e.g. a table split across chunk boundaries).
    """
    evidence = set().union(*(_words(e.evidence_text) for e in case.evidence))
    if not evidence:
        return 0.0
    retrieved_words = (
        set().union(*(_words(rc.chunk.text) for rc in retrieved)) if retrieved else set()
    )
    return len(evidence & retrieved_words) / len(evidence)


def retrieval_metrics(retrieved: list[RetrievedChunk], case: EvalCase, k: int) -> dict[str, float]:
    """Doc/page hit, page recall, MRR and evidence coverage over the top-k results."""
    top = retrieved[:k]
    gold = case.gold_pages
    gold_docs = {doc for doc, _ in gold}

    def overlaps(rc: RetrievedChunk) -> set[tuple[str, int]]:
        return {(rc.chunk.doc_name, p) for p in rc.chunk.pages} & gold

    covered = set().union(*(overlaps(rc) for rc in top)) if top else set()
    first_hit = next((i for i, rc in enumerate(top, start=1) if overlaps(rc)), None)
    return {
        f"doc_hit@{k}": float(any(rc.chunk.doc_name in gold_docs for rc in top)),
        f"page_hit@{k}": float(bool(covered)),
        f"page_recall@{k}": len(covered) / len(gold) if gold else 0.0,
        f"mrr@{k}": 1.0 / first_hit if first_hit else 0.0,
        f"evidence_coverage@{k}": evidence_coverage(top, case),
    }


# ---------------------------------------------------------------- answers


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(INSUFFICIENT.lower())


_NUMERIC_ANSWER = re.compile(
    r"[\s$€(\-]*(?P<num>-?[\d,]*\.?\d+)\s*(?P<pct>%)?\)?\s*(million|billion|x)?\.?\s*", re.I
)
_NUMBER = re.compile(r"(?<![\w.])-?\$?\(?(\d[\d,]*\.?\d*|\.\d+)\)?\s*(%|percent)?", re.I)
# A reported figure may be scaled differently from the gold (e.g. $1.577B vs 1577 in millions).
_SCALES = (1, 1e3, 1e-3, 1e6, 1e-6, 100, 0.01)


def gold_number(answer: str) -> float | None:
    """The value of a purely numeric gold answer like '$1577.00', '0.96' or '65.4%'."""
    m = _NUMERIC_ANSWER.fullmatch(answer)
    if not m:
        return None
    return float(m.group("num").replace(",", ""))


def numeric_match(response: str, gold: float, rel_tol: float = 0.01) -> bool:
    """Whether any number in the response equals the gold value, up to scale and 1% tolerance.

    Deliberately lenient (any number counts), so it's a diagnostic cross-check on
    the correctness judge, not a headline metric.
    """
    for m in _NUMBER.finditer(response):
        value = float(m.group(1).replace(",", ""))
        for scale in _SCALES:
            if abs(value * scale - abs(gold)) <= max(rel_tol * abs(gold), 1e-9):
                return True
    return False


def citation_metrics(
    text: str, retrieved: list[RetrievedChunk], case: EvalCase
) -> dict[str, float]:
    markers = cited_markers(text)
    by_rank = {rc.rank: rc for rc in retrieved}
    valid = [by_rank[m] for m in markers if m in by_rank]
    cites_gold = any(
        (rc.chunk.doc_name, p) in case.gold_pages for rc in valid for p in rc.chunk.pages
    )
    return {
        "has_citation": float(bool(valid)),
        "invalid_citations": float(len(markers) - len(valid)),
        "cites_gold_page": float(cites_gold),
    }
