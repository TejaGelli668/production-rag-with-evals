"""Query-time orchestration: retrieve -> generate -> resolve citations."""

from __future__ import annotations

import time

from rag.config import PipelineConfig, Settings
from rag.embed import Embedder
from rag.generate.citations import resolve_citations
from rag.generate.llm import LLM, make_llm
from rag.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.retrieve import DenseRetriever
from rag.schema import Answer
from rag.store import Filters, QdrantStore, make_client


class IndexNotFoundError(RuntimeError):
    pass


class RAGPipeline:
    def __init__(self, cfg: PipelineConfig, retriever: DenseRetriever, llm: LLM):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm

    @classmethod
    def from_config(cls, cfg: PipelineConfig, settings: Settings | None = None) -> RAGPipeline:
        settings = settings or Settings()
        store = QdrantStore(make_client(settings), cfg.collection_name)
        if not store.exists():
            raise IndexNotFoundError(
                f"index '{cfg.collection_name}' not found; run `rag ingest` with this config first"
            )
        retriever = DenseRetriever(cfg.retriever, Embedder(cfg.embedder), store)
        return cls(cfg, retriever, make_llm(cfg.generator, settings))

    def ask(self, question: str, filters: Filters | None = None) -> Answer:
        started = time.perf_counter()
        retrieved = self.retriever.retrieve(question, filters)
        retrieval_latency = time.perf_counter() - started

        llm_response = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, retrieved))
        return Answer(
            question=question,
            text=llm_response.text,
            citations=resolve_citations(llm_response.text, retrieved),
            retrieved=retrieved,
            llm=llm_response,
            retrieval_latency_s=retrieval_latency,
            config_name=self.cfg.name,
        )
