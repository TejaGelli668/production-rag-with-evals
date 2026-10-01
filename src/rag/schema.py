"""Data types shared across ingestion, retrieval, and generation.

Page numbers are 0-based PyMuPDF page indices everywhere, matching FinanceBench's
`evidence_page_num`. Convert to 1-based only for display.
"""

from __future__ import annotations

from pydantic import BaseModel


class Page(BaseModel):
    page_num: int
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_name: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int  # inclusive
    # Document metadata, copied onto every chunk so retrieval can filter on it.
    company: str
    doc_type: str
    fiscal_year: int
    gics_sector: str

    @property
    def pages(self) -> range:
        return range(self.page_start, self.page_end + 1)

    def page_label(self) -> str:
        """Human-facing, 1-based page reference, e.g. 'p. 48' or 'pp. 48-49'."""
        if self.page_start == self.page_end:
            return f"p. {self.page_start + 1}"
        return f"pp. {self.page_start + 1}-{self.page_end + 1}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int  # 1-based position in the final context


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    latency_s: float


class Citation(BaseModel):
    marker: int  # the [n] the model wrote; 1-based index into the context sources
    doc_name: str
    page_start: int
    page_end: int


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    llm: LLMResponse
    retrieval_latency_s: float
    config_name: str
