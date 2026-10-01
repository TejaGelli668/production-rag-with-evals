"""Retrievers. Phase 1 has dense retrieval only; hybrid and reranking come in Phase 3."""

from __future__ import annotations

from rag.config import RetrieverConfig
from rag.embed import Embedder
from rag.schema import RetrievedChunk
from rag.store import Filters, QdrantStore


class DenseRetriever:
    def __init__(self, cfg: RetrieverConfig, embedder: Embedder, store: QdrantStore):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, filters: Filters | None = None) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.embed_query(question), self.cfg.top_k, filters)
