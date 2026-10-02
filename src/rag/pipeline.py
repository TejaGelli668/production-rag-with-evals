"""Query-time orchestration: retrieve -> generate -> resolve citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from rag.config import PipelineConfig, Settings
from rag.generate.citations import resolve_citations
from rag.generate.llm import LLM, make_llm
from rag.generate.prompts import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    STEPWISE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rag.retrieve import NoRetriever, OracleRetriever, Retriever, SearchRetriever
from rag.schema import Answer, LLMResponse, RetrievedChunk
from rag.store import Filters, QdrantStore, make_client
from rag.tracing import document_attributes, llm_attributes, span

# ("sources", list[RetrievedChunk]) | ("token", str) | ("answer", Answer)
StreamEvent = tuple[str, Any]


class IndexNotFoundError(RuntimeError):
    pass


def make_retriever(cfg: PipelineConfig, settings: Settings) -> Retriever:
    if cfg.retriever.type == "none":
        return NoRetriever()
    if cfg.retriever.type == "oracle":
        from rag.data.financebench import load_documents, load_questions

        page_text = None
        if cfg.retriever.oracle_text == "parsed":
            from functools import cache

            from rag.ingest.parse import parse_cached

            @cache
            def pages_of(doc_name: str, pdf_path):
                return parse_cached(cfg.parser, pdf_path)

            def page_text(doc, page: int) -> str:
                return pages_of(doc.doc_name, doc.pdf_path)[page].text

        return OracleRetriever(load_questions(), load_documents(), page_text)

    from rag.embed import Embedder  # deferred: loads torch

    store = QdrantStore(make_client(settings, cfg.collection_name), cfg.collection_name)
    if not store.exists():
        raise IndexNotFoundError(
            f"index '{cfg.collection_name}' not found; run `rag ingest` with this config first"
        )
    rc = cfg.retriever
    analyzer = bm25 = reranker = None
    if rc.filters != "none":
        from rag.data.financebench import load_documents
        from rag.query import QueryAnalyzer

        analyzer = QueryAnalyzer(load_documents())
    if rc.type == "hybrid":
        from rag.sparse import BM25Index

        bm25 = BM25Index.load_or_build(cfg.collection_name, store.all_chunks)
    if rc.rerank:
        from rag.rerank import Reranker

        reranker = Reranker(rc.reranker_model)
    rewriter = None
    if rc.rewrite:
        from rag.generate.llm import make_llm
        from rag.rewrite import QueryRewriter

        rewriter = QueryRewriter(make_llm(cfg.generator, settings), rc.max_rewrites)
    return SearchRetriever(rc, Embedder(cfg.embedder), store, analyzer, bm25, reranker, rewriter)


class RAGPipeline:
    def __init__(self, cfg: PipelineConfig, retriever: Retriever, llm: LLM):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.closed_book = cfg.retriever.type == "none"
        if self.closed_book:
            self.system_prompt = CLOSED_BOOK_SYSTEM_PROMPT
        elif cfg.generator.prompt == "stepwise":
            self.system_prompt = STEPWISE_SYSTEM_PROMPT
        else:
            self.system_prompt = SYSTEM_PROMPT

    @classmethod
    def from_config(cls, cfg: PipelineConfig, settings: Settings | None = None) -> RAGPipeline:
        settings = settings or Settings()
        return cls(cfg, make_retriever(cfg, settings), make_llm(cfg.generator, settings))

    def _retrieve(
        self, question: str, filters: Filters | None
    ) -> tuple[list[RetrievedChunk], float]:
        with span("retrieve", "RETRIEVER", **{"input.value": question}) as s:
            started = time.perf_counter()
            retrieved = self.retriever.retrieve(question, filters)
            s.set_attributes(document_attributes(retrieved, "retrieval.documents"))
        return retrieved, time.perf_counter() - started

    def _answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        retrieval_latency: float,
        user_prompt: str,
        llm_response: LLMResponse,
    ) -> Answer:
        return Answer(
            question=question,
            text=llm_response.text,
            citations=resolve_citations(llm_response.text, retrieved),
            retrieved=retrieved,
            llm=llm_response,
            retrieval_latency_s=retrieval_latency,
            config_name=self.cfg.name,
            user_prompt=user_prompt,
        )

    def ask(self, question: str, filters: Filters | None = None) -> Answer:
        with span("rag.ask", "CHAIN", **{"input.value": question}) as root:
            retrieved, retrieval_latency = self._retrieve(question, filters)
            user_prompt = build_user_prompt(question, retrieved, closed_book=self.closed_book)
            with span("generate", "LLM") as s:
                llm_response = self.llm.complete(self.system_prompt, user_prompt)
                s.set_attributes(llm_attributes(self.system_prompt, user_prompt, llm_response))
            answer = self._answer(question, retrieved, retrieval_latency, user_prompt, llm_response)
            root.set_attribute("output.value", answer.text)
        return answer

    def ask_stream(self, question: str, filters: Filters | None = None) -> Iterator[StreamEvent]:
        """Like `ask`, but yields ("sources", chunks) first, then ("token", text) pieces as
        the answer is generated, then ("answer", Answer) with citations resolved."""
        with span("rag.ask", "CHAIN", **{"input.value": question, "rag.streaming": True}) as root:
            retrieved, retrieval_latency = self._retrieve(question, filters)
            yield "sources", retrieved
            user_prompt = build_user_prompt(question, retrieved, closed_book=self.closed_book)
            llm_response: LLMResponse | None = None
            with span("generate", "LLM") as s:
                for event in self.llm.stream(self.system_prompt, user_prompt):
                    if isinstance(event, LLMResponse):
                        llm_response = event
                    else:
                        yield "token", event
                if llm_response is None:
                    raise RuntimeError("LLM stream ended without a final response")
                s.set_attributes(llm_attributes(self.system_prompt, user_prompt, llm_response))
            answer = self._answer(question, retrieved, retrieval_latency, user_prompt, llm_response)
            root.set_attribute("output.value", answer.text)
            yield "answer", answer
