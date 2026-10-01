"""Retrievers.

`dense` is the real pipeline. `oracle` and `none` exist to bound the evaluation (E0):
`oracle` hands the generator the gold evidence pages, `none` gives it nothing.
"""

from __future__ import annotations

from typing import Protocol

from rag.config import RetrieverConfig
from rag.data.financebench import Document, Question
from rag.embed import Embedder
from rag.schema import Chunk, RetrievedChunk
from rag.store import Filters, QdrantStore


class Retriever(Protocol):
    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]: ...


class DenseRetriever:
    def __init__(self, cfg: RetrieverConfig, embedder: Embedder, store: QdrantStore):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store

    def retrieve(
        self, question: str, filters: Filters | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        k = top_k or self.cfg.top_k
        return self.store.search(self.embedder.embed_query(question), k, filters)


class NoRetriever:
    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]:
        return []


class OracleRetriever:
    """Returns the gold evidence pages for a known FinanceBench question."""

    def __init__(self, questions: list[Question], documents: list[Document]):
        docs = {d.doc_name: d for d in documents}
        self._by_question: dict[str, list[RetrievedChunk]] = {}
        for q in questions:
            pages = {
                (e.doc_name, e.evidence_page_num): e.evidence_text_full_page for e in q.evidence
            }
            self._by_question[q.question] = [
                RetrievedChunk(chunk=page_chunk(docs[doc_name], page, text), score=1.0, rank=rank)
                for rank, ((doc_name, page), text) in enumerate(sorted(pages.items()), start=1)
            ]

    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]:
        try:
            return self._by_question[question]
        except KeyError:
            raise KeyError("oracle retrieval only works for FinanceBench questions") from None


def page_chunk(doc: Document, page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"oracle/{doc.doc_name}/{page}",
        doc_name=doc.doc_name,
        chunk_index=-1,
        text=text,
        page_start=page,
        page_end=page,
        company=doc.company,
        doc_type=doc.doc_type,
        fiscal_year=doc.doc_period,
        gics_sector=doc.gics_sector,
    )
