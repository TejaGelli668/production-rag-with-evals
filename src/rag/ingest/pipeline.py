"""Ingestion: parse -> chunk -> embed -> upsert, one document at a time.

Resumable: a manifest records each fully indexed document, so an interrupted
run picks up where it stopped. A partially written document is deleted and
redone, because it never reached the manifest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from tqdm import tqdm

from rag.config import PipelineConfig, Settings
from rag.data.financebench import load_documents, load_questions, select_documents
from rag.embed import Embedder
from rag.ingest.chunk import FixedTokenChunker
from rag.ingest.parse import PARSERS
from rag.store import QdrantStore, make_client


@dataclass
class IngestStats:
    docs_total: int
    docs_indexed: int
    docs_skipped: int
    chunks_added: int
    seconds: float


class Manifest:
    def __init__(self, settings: Settings, cfg: PipelineConfig):
        settings.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.manifest_dir / f"{cfg.collection_name}.json"
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.config = cfg.model_dump(
            mode="json", include={"corpus", "parser", "chunker", "embedder"}
        )
        self.done: dict[str, int] = data.get("done", {})  # doc_name -> chunk count

    def mark_done(self, doc_name: str, n_chunks: int) -> None:
        self.done[doc_name] = n_chunks
        payload = {"config": self.config, "done": self.done}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        tmp.replace(self.path)


def ingest(cfg: PipelineConfig, settings: Settings, limit: int | None = None) -> IngestStats:
    started = time.perf_counter()
    docs = select_documents(cfg.corpus, load_documents(), load_questions())[:limit]
    manifest = Manifest(settings, cfg)
    todo = [d for d in docs if d.doc_name not in manifest.done]

    embedder = Embedder(cfg.embedder)
    chunker = FixedTokenChunker(embedder.tokenizer, cfg.chunker.size, cfg.chunker.overlap)
    parse = PARSERS[cfg.parser]
    store = QdrantStore(make_client(settings), cfg.collection_name)
    store.ensure_collection(embedder.dim)

    chunks_added = 0
    for doc in tqdm(todo, desc=f"ingest {cfg.collection_name}", unit="doc"):
        chunks = chunker.chunk(parse(doc.pdf_path), doc, cfg.index_key)
        store.delete_doc(doc.doc_name)  # clear any partial write from an interrupted run
        if chunks:
            store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
        manifest.mark_done(doc.doc_name, len(chunks))
        chunks_added += len(chunks)

    return IngestStats(
        docs_total=len(docs),
        docs_indexed=len(todo),
        docs_skipped=len(docs) - len(todo),
        chunks_added=chunks_added,
        seconds=time.perf_counter() - started,
    )
