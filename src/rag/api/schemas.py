"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag.schema import Answer, RetrievedChunk


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # Optional metadata filters; when omitted, the configured query analyzer infers them.
    filters: dict[str, str | int | list[str] | list[int]] | None = None


class Source(BaseModel):
    rank: int
    doc_name: str
    company: str
    doc_type: str
    fiscal_year: int
    page_start: int  # 0-based
    page_end: int
    page_label: str  # 1-based, for display
    score: float
    cited: bool
    text: str

    @classmethod
    def from_retrieved(cls, rc: RetrievedChunk, cited: set[int]) -> Source:
        c = rc.chunk
        return cls(
            rank=rc.rank,
            doc_name=c.doc_name,
            company=c.company,
            doc_type=c.doc_type,
            fiscal_year=c.fiscal_year,
            page_start=c.page_start,
            page_end=c.page_end,
            page_label=c.page_label(),
            score=rc.score,
            cited=rc.rank in cited,
            text=c.text,
        )


class Timings(BaseModel):
    retrieval_s: float
    generation_s: float
    total_s: float


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    refused: bool
    sources: list[Source]
    timings: Timings
    usage: Usage
    model: str
    config: str

    @classmethod
    def from_answer(cls, request_id: str, answer: Answer, total_s: float) -> AskResponse:
        from rag.evals.metrics import is_refusal

        cited = {c.marker for c in answer.citations}
        return cls(
            request_id=request_id,
            question=answer.question,
            answer=answer.text,
            refused=is_refusal(answer.text),
            sources=[Source.from_retrieved(rc, cited) for rc in answer.retrieved],
            timings=Timings(
                retrieval_s=round(answer.retrieval_latency_s, 3),
                generation_s=round(answer.llm.latency_s, 3),
                total_s=round(total_s, 3),
            ),
            usage=Usage(
                input_tokens=answer.llm.input_tokens, output_tokens=answer.llm.output_tokens
            ),
            model=answer.llm.model,
            config=answer.config_name,
        )


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    config: str
    collection: str | None
    chunks: int | None
    llm_provider: str
    llm_model: str
    llm_reachable: bool
    tracing: bool
