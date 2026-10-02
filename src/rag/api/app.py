"""FastAPI service.

    uv run uvicorn rag.api.app:app --port 8000          # or: make serve

Endpoints
    GET  /health                    config, index size, whether the LLM is reachable
    POST /ask                       answer with sources, citations, timings, token usage
    POST /ask/stream                the same as server-sent events: sources, tokens, answer
    POST /feedback                  thumbs up/down on an answer, by request_id
    GET  /pages/{doc}/{page}.png    a rendered PDF page (0-based page index), for the UI

Every answer is logged to storage/logs/requests.jsonl and feedback to feedback.jsonl, so
feedback can later be joined to the exact answer and turned into eval cases. The pipeline
(embedder, reranker, embedded Qdrant) is not thread-safe, so requests run one at a time.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from rag.api.schemas import AskRequest, AskResponse, FeedbackRequest, Health, Source
from rag.config import CONFIGS_DIR, Settings, load_config
from rag.data.financebench import PROJECT_ROOT, load_documents
from rag.generate.llm import LLMError
from rag.pipeline import RAGPipeline
from rag.schema import RetrievedChunk

DEFAULT_SERVE_CONFIG = CONFIGS_DIR / "stack_filter_rerank.yaml"
LOG_DIR = PROJECT_ROOT / "storage" / "logs"


class _Log:
    """Append-only JSONL log, safe across request threads."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(
    pipeline_factory: Callable[[], RAGPipeline] | None = None,
    log_dir: Path = LOG_DIR,
    settings: Settings | None = None,
) -> FastAPI:
    """`pipeline_factory` is injectable for tests; by default the pipeline is built from
    RAG_CONFIG (default: the final `stack_filter_rerank` config) at startup."""
    settings = settings or Settings()
    requests_log = _Log(log_dir / "requests.jsonl")
    feedback_log = _Log(log_dir / "feedback.jsonl")
    pipeline_lock = threading.Lock()
    state: dict[str, Any] = {}

    def default_factory() -> RAGPipeline:
        cfg = load_config(Path(os.environ.get("RAG_CONFIG", DEFAULT_SERVE_CONFIG)))
        return RAGPipeline.from_config(cfg, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from rag.tracing import setup_tracing

        state["tracing"] = setup_tracing()
        state["pipeline"] = (pipeline_factory or default_factory)()
        try:  # page images need the downloaded corpus; the API works without it
            state["documents"] = {d.doc_name: d for d in load_documents()}
        except FileNotFoundError:
            state["documents"] = {}
        yield

    app = FastAPI(
        title="Production RAG With Evals",
        description="Question answering over SEC filings, with cited pages.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def pipeline() -> RAGPipeline:
        return state["pipeline"]

    def log_request(request_id: str, response: AskResponse) -> None:
        requests_log.write(
            {"request_id": request_id, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            | response.model_dump(exclude={"sources"})
            | {"sources": [s.model_dump(exclude={"text"}) for s in response.sources]}
        )

    @app.get("/health", response_model=Health)
    def health() -> Health:
        p = pipeline()
        gen = p.cfg.generator
        store = getattr(p.retriever, "store", None)
        reachable = True
        if gen.provider == "ollama":
            try:
                httpx.get(f"{settings.ollama_host}/api/version", timeout=2).raise_for_status()
            except httpx.HTTPError:
                reachable = False
        return Health(
            status="ok" if reachable else "degraded",
            config=p.cfg.name,
            collection=store.collection if store else None,
            chunks=store.count() if store else None,
            llm_provider=gen.provider,
            llm_model=gen.model,
            llm_reachable=reachable,
            tracing=state.get("tracing", False),
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest) -> AskResponse:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            with pipeline_lock:
                answer = pipeline().ask(body.question, body.filters)
        except LLMError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        response = AskResponse.from_answer(request_id, answer, time.perf_counter() - started)
        log_request(request_id, response)
        return response

    @app.post("/ask/stream")
    def ask_stream(body: AskRequest) -> StreamingResponse:
        request_id = uuid.uuid4().hex

        queue: Queue[str | None] = Queue()

        def produce() -> None:
            # One thread runs the whole pipeline generator, so trace context and the lock
            # stay on the thread that opened them; the response just drains the queue.
            started = time.perf_counter()
            try:
                with pipeline_lock:
                    for kind, payload in pipeline().ask_stream(body.question, body.filters):
                        if kind == "sources":
                            sources = [_source(rc).model_dump() for rc in payload]
                            queue.put(
                                _sse("sources", {"request_id": request_id, "sources": sources})
                            )
                        elif kind == "token":
                            queue.put(_sse("token", {"text": payload}))
                        else:
                            response = AskResponse.from_answer(
                                request_id, payload, time.perf_counter() - started
                            )
                            log_request(request_id, response)
                            queue.put(_sse("answer", response.model_dump()))
            except LLMError as e:
                queue.put(_sse("error", {"detail": str(e)}))
            except Exception:
                queue.put(_sse("error", {"detail": "internal error"}))
                raise
            finally:
                queue.put(None)

        def events() -> Iterator[str]:
            while (item := queue.get()) is not None:
                yield item

        threading.Thread(target=produce, daemon=True, name=f"ask-{request_id[:8]}").start()
        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
        )

    @app.post("/feedback", status_code=204)
    def feedback(body: FeedbackRequest) -> Response:
        feedback_log.write(body.model_dump() | {"at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return Response(status_code=204)

    @app.get("/pages/{doc_name}/{page}.png")
    def page_image(doc_name: str, page: int) -> Response:
        # doc_name must be a known filing: never build a path from raw input.
        doc = state["documents"].get(doc_name)
        if doc is None or not doc.pdf_path.exists():
            raise HTTPException(status_code=404, detail="unknown document")
        import pymupdf

        with pymupdf.open(doc.pdf_path) as pdf:
            if not 0 <= page < pdf.page_count:
                raise HTTPException(status_code=404, detail="page out of range")
            png = pdf[page].get_pixmap(dpi=110).tobytes("png")
        return Response(
            content=png, media_type="image/png", headers={"Cache-Control": "max-age=86400"}
        )

    return app


def _source(rc: RetrievedChunk) -> Source:
    return Source.from_retrieved(rc, cited=set())  # citations are known only at the end


app = create_app()
