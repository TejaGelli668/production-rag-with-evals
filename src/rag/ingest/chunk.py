"""Page-aware chunking.

The fixed-size chunker windows over the document's token stream, so chunks may
span page boundaries; each chunk records the page range it covers. Chunk text is
sliced from the original page text using tokenizer character offsets, which
keeps the original casing and spacing even though the tokenizer is uncased.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from rag.data.financebench import Document
from rag.schema import Chunk, Page

_CHUNK_NAMESPACE = uuid.UUID("6f1c1d2e-58a4-4b8e-9a52-2b6f6a3c9e10")


@dataclass(frozen=True)
class _Token:
    page_idx: int  # index into the pages list
    start: int  # character offsets within that page's text
    end: int


def chunk_id(index_key: str, doc_name: str, chunk_index: int) -> str:
    """Deterministic ID, so re-ingesting a document overwrites rather than duplicates."""
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{index_key}/{doc_name}/{chunk_index}"))


class FixedTokenChunker:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, size: int, overlap: int):
        if not 0 <= overlap < size:
            raise ValueError(f"overlap must be in [0, size), got overlap={overlap}, size={size}")
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required for character offsets")
        self.tokenizer = tokenizer
        self.size = size
        self.overlap = overlap

    def _tokens(self, pages: list[Page]) -> list[_Token]:
        texts = [p.text for p in pages]
        enc = self.tokenizer(
            texts, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [
            _Token(page_idx, start, end)
            for page_idx, offsets in enumerate(enc["offset_mapping"])
            for start, end in offsets
        ]

    def _windows(self, tokens: list[_Token]) -> list[list[_Token]]:
        step = self.size - self.overlap
        windows = []
        for start in range(0, len(tokens), step):
            windows.append(tokens[start : start + self.size])
            if start + self.size >= len(tokens):
                break
        return windows

    def chunk(self, pages: list[Page], doc: Document, index_key: str) -> list[Chunk]:
        return [
            self._make_chunk(window, pages, doc, index_key, i)
            for i, window in enumerate(self._windows(self._tokens(pages)))
        ]

    def _make_chunk(
        self,
        window: list[_Token],
        pages: list[Page],
        doc: Document,
        index_key: str,
        chunk_index: int,
    ) -> Chunk:
        # Slice each covered page from its first to last token in the window.
        parts: list[str] = []
        first_page, last_page = window[0].page_idx, window[-1].page_idx
        for page_idx in range(first_page, last_page + 1):
            on_page = [t for t in window if t.page_idx == page_idx]
            if on_page:
                text = pages[page_idx].text[on_page[0].start : on_page[-1].end]
                parts.append(text.strip())
        return Chunk(
            chunk_id=chunk_id(index_key, doc.doc_name, chunk_index),
            doc_name=doc.doc_name,
            chunk_index=chunk_index,
            text="\n\n".join(p for p in parts if p),
            page_start=pages[first_page].page_num,
            page_end=pages[last_page].page_num,
            company=doc.company,
            doc_type=doc.doc_type,
            fiscal_year=doc.doc_period,
            gics_sector=doc.gics_sector,
        )


class PageBoundedChunker(FixedTokenChunker):
    """Fixed-size windows that never cross a page boundary.

    Every chunk belongs to exactly one page, so a table is never glued to the
    next page's content, and short pages become single chunks.
    """

    def chunk(self, pages: list[Page], doc: Document, index_key: str) -> list[Chunk]:
        tokens = self._tokens(pages)
        by_page: dict[int, list[_Token]] = {}
        for t in tokens:
            by_page.setdefault(t.page_idx, []).append(t)
        windows = [w for page_idx in sorted(by_page) for w in self._windows(by_page[page_idx])]
        return [self._make_chunk(w, pages, doc, index_key, i) for i, w in enumerate(windows)]


def make_chunker(
    kind: str, tokenizer: PreTrainedTokenizerBase, size: int, overlap: int
) -> FixedTokenChunker:
    chunkers = {"fixed": FixedTokenChunker, "page": PageBoundedChunker}
    return chunkers[kind](tokenizer, size, overlap)
