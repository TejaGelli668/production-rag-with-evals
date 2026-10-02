"""BM25 keyword index over a collection's chunks, for hybrid retrieval (E4).

Built once from the vector store's payloads and cached under storage/bm25/. Filters
are applied by masking corpus scores, so a filtered query still ranks the whole
filtered set rather than post-filtering a short list.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import bm25s
import numpy as np

from rag.data.financebench import PROJECT_ROOT
from rag.schema import Chunk, RetrievedChunk

CACHE_DIR = PROJECT_ROOT / "storage" / "bm25"
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")


def _normalize(text: str) -> str:
    # Keep "1,577" and "1577" as the same token.
    return _THOUSANDS.sub("", text)


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, chunks: list[Chunk]):
        self.retriever = retriever
        self.chunks = chunks
        self._columns = {
            key: np.array([getattr(c, key) for c in chunks], dtype=object)
            for key in ("company", "fiscal_year", "doc_type", "doc_name")
        }

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Index:
        tokens = bm25s.tokenize(
            [_normalize(c.text) for c in chunks], stopwords="en", show_progress=False
        )
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever, chunks)

    @classmethod
    def load_or_build(cls, collection: str, load_chunks, cache_dir: Path = CACHE_DIR) -> BM25Index:
        """`load_chunks` is called only on a cache miss (it scrolls the vector store)."""
        path = cache_dir / collection
        if (path / "chunks.json").exists():
            chunks = [
                Chunk.model_validate(c) for c in json.loads((path / "chunks.json").read_text())
            ]
            return cls(bm25s.BM25.load(str(path / "bm25")), chunks)
        index = cls.build(load_chunks())
        path.mkdir(parents=True, exist_ok=True)
        index.retriever.save(str(path / "bm25"))
        (path / "chunks.json").write_text(json.dumps([c.model_dump() for c in index.chunks]))
        return index

    def _mask(self, filters: dict) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        for key, value in filters.items():
            values = value if isinstance(value, list) else [value]
            mask &= np.isin(self._columns[key], values)
        return mask

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievedChunk]:
        terms = bm25s.tokenize(
            [_normalize(query)], stopwords="en", return_ids=False, show_progress=False
        )[0]
        scores = self.retriever.get_scores(terms) if terms else np.zeros(len(self.chunks))
        if filters:
            scores = np.where(self._mask(filters), scores, -np.inf)
        top = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(top, start=1)
            if np.isfinite(scores[i]) and scores[i] > 0
        ]


def rrf_fuse(rankings: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank) across rankings."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rc in ranking:
            scores[rc.chunk.chunk_id] = scores.get(rc.chunk.chunk_id, 0.0) + 1.0 / (k + rc.rank)
            chunks[rc.chunk.chunk_id] = rc.chunk
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [
        RetrievedChunk(chunk=chunks[cid], score=scores[cid], rank=r)
        for r, cid in enumerate(ordered, start=1)
    ]
