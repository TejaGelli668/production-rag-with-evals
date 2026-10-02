import re

import pytest

from rag.data.financebench import Document
from rag.ingest.chunk import FixedTokenChunker, chunk_id
from rag.schema import Page


class WhitespaceTokenizer:
    """Offline stand-in for a fast HF tokenizer: one token per whitespace-separated word."""

    is_fast = True

    def __call__(self, texts, **_):
        return {"offset_mapping": [[m.span() for m in re.finditer(r"\S+", t)] for t in texts]}


DOC = Document(
    doc_name="ACME_2022_10K",
    company="Acme",
    gics_sector="Industrials",
    doc_type="10k",
    doc_period=2022,
    doc_link="https://example.com",
)


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_windows_overlap_and_cover_every_token():
    pages = [Page(page_num=0, text=words("w", 25))]
    chunks = FixedTokenChunker(WhitespaceTokenizer(), size=10, overlap=3).chunk(pages, DOC, "k")
    tokens = [c.text.split() for c in chunks]
    assert [len(t) for t in tokens] == [10, 10, 10, 4]
    assert tokens[1][:3] == tokens[0][-3:]  # overlap carried forward
    assert set().union(*tokens) == set(words("w", 25).split())
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]


def test_chunks_spanning_pages_record_page_range_and_keep_original_text():
    pages = [Page(page_num=0, text="Alpha Beta\nGamma"), Page(page_num=1, text="Delta  Epsilon")]
    [chunk] = FixedTokenChunker(WhitespaceTokenizer(), size=10, overlap=0).chunk(pages, DOC, "k")
    assert (chunk.page_start, chunk.page_end) == (0, 1)
    assert chunk.text == "Alpha Beta\nGamma\n\nDelta  Epsilon"
    assert chunk.page_label() == "pp. 1-2"


def test_empty_pages_are_skipped_but_page_numbers_preserved():
    pages = [Page(page_num=0, text="   "), Page(page_num=1, text=words("x", 4))]
    [chunk] = FixedTokenChunker(WhitespaceTokenizer(), size=10, overlap=0).chunk(pages, DOC, "k")
    assert (chunk.page_start, chunk.page_end) == (1, 1)
    assert chunk.page_label() == "p. 2"


def test_document_metadata_is_copied_to_chunks():
    pages = [Page(page_num=0, text="one two")]
    [chunk] = FixedTokenChunker(WhitespaceTokenizer(), size=5, overlap=0).chunk(pages, DOC, "k")
    assert (chunk.company, chunk.doc_type, chunk.fiscal_year) == ("Acme", "10k", 2022)


def test_empty_document_yields_no_chunks():
    chunker = FixedTokenChunker(WhitespaceTokenizer(), size=5, overlap=0)
    assert chunker.chunk([Page(page_num=0, text="")], DOC, "k") == []


def test_chunk_ids_are_deterministic_and_scoped_by_index():
    assert chunk_id("k", "A", 0) == chunk_id("k", "A", 0)
    assert len({chunk_id("k", "A", 0), chunk_id("k", "A", 1), chunk_id("j", "A", 0)}) == 3


def test_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        FixedTokenChunker(WhitespaceTokenizer(), size=5, overlap=5)


def test_page_bounded_chunks_never_cross_pages():
    from rag.ingest.chunk import PageBoundedChunker

    pages = [Page(page_num=0, text=words("a", 7)), Page(page_num=1, text=words("b", 3))]
    chunks = PageBoundedChunker(WhitespaceTokenizer(), size=5, overlap=1).chunk(pages, DOC, "k")
    assert [(c.page_start, c.page_end) for c in chunks] == [(0, 0), (0, 0), (1, 1)]
    assert [c.text for c in chunks] == ["a0 a1 a2 a3 a4", "a4 a5 a6", "b0 b1 b2"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
