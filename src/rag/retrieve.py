"""Retrievers.

`SearchRetriever` is the real pipeline: dense (or dense + BM25) candidates, optional
metadata filters inferred from the question, optional cross-encoder reranking.
`OracleRetriever` and `NoRetriever` bound the evaluation (E0): the gold evidence
pages, or no context at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from rag.config import RetrieverConfig
from rag.data.financebench import Document, Question
from rag.schema import Chunk, RetrievedChunk
from rag.store import Filters, QdrantStore

if TYPE_CHECKING:
    from rag.embed import Embedder
    from rag.query import QueryAnalyzer
    from rag.rerank import Reranker
    from rag.rewrite import QueryRewriter
    from rag.sparse import BM25Index


class Retriever(Protocol):
    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]: ...


class SearchRetriever:
    def __init__(
        self,
        cfg: RetrieverConfig,
        embedder: Embedder,
        store: QdrantStore,
        analyzer: QueryAnalyzer | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.analyzer = analyzer
        self.bm25 = bm25
        self.reranker = reranker
        self.rewriter = rewriter
        self.last_queries: list[str] = []  # the queries behind the latest retrieval, for traces

    def _filter_attempts(self, question: str, filters: Filters | None) -> list[Filters]:
        if filters:  # explicit filters from the caller win
            return [filters]
        if self.analyzer is None or self.cfg.filters == "none":
            return [{}]
        return self.analyzer.filters(question, self.cfg.filters)

    def _candidates(self, question: str, filters: Filters, pool: int) -> list[RetrievedChunk]:
        dense = self.store.search(self.embedder.embed_query(question), pool, filters or None)
        if self.bm25 is None:
            return dense
        from rag.sparse import rrf_fuse

        sparse = self.bm25.search(question, pool, filters or None)
        return rrf_fuse([dense, sparse], k=self.cfg.rrf_k)

    def retrieve(
        self, question: str, filters: Filters | None = None, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        k = top_k or self.cfg.top_k
        expands = self.bm25 or self.reranker or self.rewriter
        pool = max(self.cfg.candidates, k) if expands else k
        queries = [question] + (self.rewriter.rewrite(question) if self.rewriter else [])
        self.last_queries = queries
        candidates: list[RetrievedChunk] = []
        # Strictest filters first; relax when they match fewer than k chunks.
        # Filters always come from the original question.
        for attempt in self._filter_attempts(question, filters):
            per_query = [self._candidates(q, attempt, pool) for q in queries]
            if len(per_query) == 1:
                candidates = per_query[0]
            else:
                from rag.sparse import rrf_fuse

                candidates = rrf_fuse(per_query, k=self.cfg.rrf_k)[:pool]
            if len(candidates) >= k:
                break
        if self.reranker:
            from rag.tracing import document_attributes, span

            with span(
                "rerank", "RERANKER", **{"reranker.query": question, "reranker.top_k": k}
            ) as s:
                s.set_attributes(document_attributes(candidates, "reranker.input_documents"))
                reranked = self.reranker.rerank(question, candidates, k)
                s.set_attributes(document_attributes(reranked, "reranker.output_documents"))
            return reranked
        return [
            RetrievedChunk(chunk=rc.chunk, score=rc.score, rank=r)
            for r, rc in enumerate(candidates[:k], start=1)
        ]


class NoRetriever:
    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]:
        return []


class OracleRetriever:
    """Returns the gold evidence pages for a known FinanceBench question."""

    def __init__(
        self,
        questions: list[Question],
        documents: list[Document],
        page_text: Callable[[Document, int], str] | None = None,
    ):
        """`page_text`, if given, supplies each gold page's text from our own parser
        (E3b) instead of FinanceBench's extraction."""
        docs = {d.doc_name: d for d in documents}
        self._by_question: dict[str, list[RetrievedChunk]] = {}
        for q in questions:
            pages = {
                (e.doc_name, e.evidence_page_num): (
                    page_text(docs[e.doc_name], e.evidence_page_num)
                    if page_text
                    else e.evidence_text_full_page
                )
                for e in q.evidence
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
