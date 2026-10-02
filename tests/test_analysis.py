import json

import pytest

from rag.evals.analysis import analyze, classify


@pytest.mark.parametrize(
    ("grade", "stage"),
    [
        ({"correct": 1.0, "doc_hit@5": 0.0}, "correct"),  # right for any reason
        ({"correct": 0.0, "doc_hit@5": 0.0, "page_hit@5": 0.0}, "wrong_filing"),
        ({"correct": 0.0, "doc_hit@5": 1.0, "page_hit@5": 0.0}, "wrong_page"),
        (
            {"correct": 0.0, "doc_hit@5": 1.0, "page_hit@5": 1.0, "evidence_coverage@5": 0.3},
            "evidence_missing",
        ),
        (
            {
                "correct": 0.0,
                "refused": 1.0,
                "doc_hit@5": 1.0,
                "page_hit@5": 1.0,
                "evidence_coverage@5": 0.9,
            },
            "refused_with_evidence",
        ),
        (
            {
                "correct": 0.0,
                "refused": 0.0,
                "doc_hit@5": 1.0,
                "page_hit@5": 1.0,
                "evidence_coverage@5": 0.9,
            },
            "wrong_with_evidence",
        ),
    ],
)
def test_classify_uses_first_failed_stage(grade, stage):
    assert classify(grade) == stage


def test_analyze_counts_ok_rows_only(tmp_path):
    rows = [
        {
            "prompt_id": "a",
            "status": "ok",
            "grade": {"correct": 1.0},
            "prompt": "q",
            "answer": "x",
            "gold": "g",
        },
        {
            "prompt_id": "b",
            "status": "ok",
            "prompt": "q",
            "answer": "x",
            "gold": "g",
            "grade": {"correct": 0.0, "doc_hit@5": 0.0, "page_hit@5": 0.0},
        },
        {
            "prompt_id": "c",
            "status": "truncated",
            "grade": {"correct": 0.0},
            "prompt": "q",
            "answer": "x",
            "gold": "g",
        },
    ]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = analyze(tmp_path)
    assert report["cases"] == 2
    assert report["counts"]["correct"] == 1 and report["counts"]["wrong_filing"] == 1
