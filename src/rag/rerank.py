"""Cross-encoder reranking (E6): score each (question, chunk) pair jointly."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.embed import _device
from rag.schema import RetrievedChunk


class Reranker:
    def __init__(self, model: str, batch_size: int = 16):
        self.model = CrossEncoder(model, device=_device(), max_length=512)
        self.batch_size = batch_size

    def rerank(
        self, question: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.model.predict(
            [(question, rc.chunk.text) for rc in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: -x[0])[:top_k]
        return [
            RetrievedChunk(chunk=rc.chunk, score=float(s), rank=r)
            for r, (s, rc) in enumerate(ranked, start=1)
        ]
