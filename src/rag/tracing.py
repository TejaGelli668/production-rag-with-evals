"""OpenTelemetry tracing, viewed in Arize Phoenix.

Off by default. Set `RAG_TRACING=1` (and run `make phoenix`) to send spans to the Phoenix
collector at PHOENIX_COLLECTOR_ENDPOINT (default http://localhost:6006). Spans follow the
OpenInference conventions, so Phoenix shows retrieved documents, rerank scores, prompts and
token counts natively. When tracing is off, every helper here is a cheap no-op.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rag.schema import LLMResponse, RetrievedChunk

PROJECT = "production-rag-with-evals"
_tracer = None


def setup_tracing() -> bool:
    """Register the Phoenix exporter if RAG_TRACING is set. Returns whether tracing is on."""
    global _tracer
    if os.environ.get("RAG_TRACING", "").lower() not in {"1", "true", "yes"}:
        return False
    from opentelemetry import trace
    from phoenix.otel import register

    register(
        project_name=PROJECT,
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        batch=True,
    )
    _tracer = trace.get_tracer("rag")
    return True


class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass


@contextmanager
def span(name: str, kind: str, **attributes: Any) -> Iterator[Any]:
    """A span of OpenInference `kind` (CHAIN, RETRIEVER, RERANKER, LLM, ...)."""
    if _tracer is None:
        yield _NoSpan()
        return
    with _tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for key, value in attributes.items():
            s.set_attribute(key, value)
        yield s


def document_attributes(retrieved: list[RetrievedChunk], prefix: str) -> dict[str, Any]:
    """`retrieval.documents.*` / `reranker.output_documents.*` attributes for Phoenix."""
    attrs: dict[str, Any] = {}
    for i, rc in enumerate(retrieved):
        c = rc.chunk
        base = f"{prefix}.{i}.document"
        attrs[f"{base}.id"] = c.chunk_id
        attrs[f"{base}.score"] = rc.score
        attrs[f"{base}.content"] = c.text
        attrs[f"{base}.metadata"] = json.dumps(
            {
                "doc_name": c.doc_name,
                "pages": c.page_label(),
                "company": c.company,
                "fiscal_year": c.fiscal_year,
            }
        )
    return attrs


def llm_attributes(system: str, user: str, response: LLMResponse) -> dict[str, Any]:
    return {
        "llm.model_name": response.model,
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": system,
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": user,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": response.text,
        "llm.token_count.prompt": response.input_tokens,
        "llm.token_count.completion": response.output_tokens,
        "llm.token_count.total": response.input_tokens + response.output_tokens,
    }
