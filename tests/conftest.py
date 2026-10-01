import pytest

from rag.data.financebench import Question

QUESTION_TYPES = ["metrics-generated", "domain-relevant", "novel-generated"]


def make_question(i: int, question_type: str, doc_name: str = "ACME_2022_10K") -> Question:
    return Question(
        financebench_id=f"financebench_id_{i:05d}",
        company="Acme",
        doc_name=doc_name,
        question_type=question_type,
        question_reasoning=None,
        domain_question_num=None,
        question=f"question {i}",
        answer=f"answer {i}",
        justification=None,
        evidence=[],
    )


@pytest.fixture
def questions() -> list[Question]:
    """150 questions, 50 per type, mirroring FinanceBench's shape."""
    return [make_question(i, QUESTION_TYPES[i % 3]) for i in range(150)]
