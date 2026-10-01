import pytest

from rag.data.financebench import Evidence
from rag.evals.cases import EvalCase, load_cases
from rag.evals.metrics import (
    citation_metrics,
    evidence_coverage,
    gold_number,
    is_refusal,
    numeric_match,
    retrieval_metrics,
)
from rag.schema import Chunk, RetrievedChunk


def chunk(rank: int, doc: str, start: int, end: int, text: str = "") -> RetrievedChunk:
    c = Chunk(
        chunk_id=f"c{rank}",
        doc_name=doc,
        chunk_index=rank,
        text=text,
        page_start=start,
        page_end=end,
        company="Acme",
        doc_type="10k",
        fiscal_year=2022,
        gics_sector="X",
    )
    return RetrievedChunk(chunk=c, score=1.0 - rank / 10, rank=rank)


def case(pages: list[tuple[str, int]], evidence_text: str = "capex 1,577") -> EvalCase:
    return EvalCase(
        case_id="q1",
        question="?",
        answerable=True,
        gold_answer="$1577.00",
        evidence=[
            Evidence(
                evidence_text=evidence_text,
                doc_name=d,
                evidence_page_num=p,
                evidence_text_full_page="",
            )
            for d, p in pages
        ],
    )


def test_retrieval_metrics_hit_recall_and_mrr():
    c = case([("A", 10), ("A", 12)])
    retrieved = [chunk(1, "B", 10, 10), chunk(2, "A", 9, 10), chunk(3, "A", 30, 30)]
    m = retrieval_metrics(retrieved, c, k=3)
    assert m["doc_hit@3"] == 1.0
    assert m["page_hit@3"] == 1.0
    assert m["page_recall@3"] == 0.5  # page 12 never retrieved
    assert m["mrr@3"] == 0.5  # first gold overlap at rank 2


def test_retrieval_metrics_respect_k():
    c = case([("A", 10)])
    retrieved = [chunk(1, "B", 1, 1), chunk(2, "A", 10, 10)]
    assert retrieval_metrics(retrieved, c, k=1)["page_hit@1"] == 0.0
    assert retrieval_metrics(retrieved, c, k=2)["page_hit@2"] == 1.0


def test_evidence_coverage_catches_page_overlap_without_the_evidence():
    c = case([("A", 10)], evidence_text="Purchases of property plant equipment (1,577)")
    on_page_but_wrong_part = [chunk(1, "A", 10, 11, "Net cash used in investing activities")]
    assert retrieval_metrics(on_page_but_wrong_part, c, k=1)["page_hit@1"] == 1.0
    assert evidence_coverage(on_page_but_wrong_part, c) < 0.2
    with_evidence = [chunk(1, "A", 10, 10, "purchases of property, plant and equipment (1577)")]
    assert evidence_coverage(with_evidence, c) == 1.0


@pytest.mark.parametrize(
    ("gold", "value"),
    [
        ("$1577.00", 1577.0),
        ("0.96", 0.96),
        ("65.4%", 65.4),
        ("$8.70", 8.7),
        ("-3.7", 3.7),
        ("1.73", 1.73),
        ("Yes, it did.", None),
        ("No. The quick ratio was 0.96", None),
    ],
)
def test_gold_number(gold, value):
    assert gold_number(gold) == (pytest.approx(value) if value is not None else None)


@pytest.mark.parametrize(
    ("response", "gold", "expected"),
    [
        ("Capex was $1,577 million [1].", 1577.0, True),
        ("Capex was $1.577 billion.", 1577.0, True),  # scale conversion
        ("The margin was 0.654.", 65.4, True),  # fraction vs percent
        ("Ratio of 0.97 [2].", 0.96, False),  # outside 1%
        ("PP&E of $8.738 billion", 8.7, True),  # within 1%
        ("No figure here.", 5.0, False),
    ],
)
def test_numeric_match(response, gold, expected):
    assert numeric_match(response, gold) is expected


def test_is_refusal():
    assert is_refusal("Insufficient information: the sources omit capex.")
    assert is_refusal("  insufficient information - nothing on 2027")
    assert not is_refusal("Capex was $1,577M. There is insufficient information on 2019.")


def test_citation_metrics():
    c = case([("A", 10)])
    retrieved = [chunk(1, "A", 10, 10), chunk(2, "B", 3, 3)]
    m = citation_metrics("X [2]. Y [1, 7].", retrieved, c)
    assert m == {"has_citation": 1.0, "invalid_citations": 1.0, "cites_gold_page": 1.0}
    assert citation_metrics("X [2].", retrieved, c)["cites_gold_page"] == 0.0


def test_unanswerable_cases_are_well_formed():
    cases = load_cases("unanswerable")
    assert len(cases) >= 30
    assert len({c.case_id for c in cases}) == len(cases)
    assert all(not c.answerable and is_refusal(c.gold_answer) for c in cases)
    assert all(c.tags and "needs human review" in c.source for c in cases)
