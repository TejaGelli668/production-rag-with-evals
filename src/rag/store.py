"""Qdrant vector store: embedded (on-disk) by default, or a server via QDRANT_URL."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from rag.config import Settings
from rag.schema import Chunk, RetrievedChunk

# Equality filters callers may pass to `search`, e.g. {"company": "3M", "fiscal_year": 2018}.
Filters = dict[str, str | int]


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def _to_filter(filters: Filters | None) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def delete_doc(self, doc_name: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_to_filter({"doc_name": doc_name})),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=c.chunk_id, vector=v.tolist(), payload=c.model_dump())
                    for c, v in zip(batch, vectors[i : i + batch_size], strict=True)
                ],
            )

    def search(
        self, vector: np.ndarray, top_k: int, filters: Filters | None = None
    ) -> list[RetrievedChunk]:
        hits = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=_to_filter(filters),
            with_payload=True,
        ).points
        return [
            RetrievedChunk(chunk=Chunk.model_validate(h.payload), score=h.score, rank=i + 1)
            for i, h in enumerate(hits)
        ]
