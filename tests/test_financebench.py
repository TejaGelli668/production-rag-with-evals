import json

from rag.data.financebench import CorpusMode, load_documents, select_documents
from tests.conftest import make_question


def _doc(name: str, period: int) -> dict:
    return {
        "doc_name": name,
        "company": "Acme",
        "gics_sector": "Industrials",
        "doc_type": "10k",
        "doc_period": period,
        "doc_link": "https://example.com",
    }


def test_load_documents_dedupes_keeping_first(tmp_path):
    path = tmp_path / "documents.jsonl"
    rows = [_doc("ACME_2023_10K", 2023), _doc("ACME_2023_10K", 2022), _doc("ACME_2022_10K", 2022)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    docs = load_documents(path)
    assert [d.doc_name for d in docs] == ["ACME_2023_10K", "ACME_2022_10K"]
    assert docs[0].doc_period == 2023


def test_select_documents_focused_vs_full(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(_doc(n, 2022)) for n in ["A", "B", "C"]))
    docs = load_documents(path)
    questions = [make_question(0, "domain-relevant", doc_name="B")]
    assert [d.doc_name for d in select_documents(CorpusMode.FOCUSED, docs, questions)] == ["B"]
    assert len(select_documents(CorpusMode.FULL, docs, questions)) == 3
