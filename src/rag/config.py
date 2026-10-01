"""Configuration: secrets and paths from the environment, pipeline knobs from YAML.

A pipeline config describes one experiment. Only the ingest-related parts
(corpus, parser, chunker, embedder) determine the vector index, so configs that
change only retrieval or generation reuse an existing index.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.data.financebench import PROJECT_ROOT, CorpusMode

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "baseline.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Optional: when unset, the Anthropic SDK falls back to its own credential chain.
    anthropic_api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"
    # Embedded Qdrant (no server) until Phase 4; set QDRANT_URL to use a server instead.
    qdrant_path: Path = PROJECT_ROOT / "storage" / "qdrant"
    qdrant_url: str | None = None
    manifest_dir: Path = PROJECT_ROOT / "storage" / "manifests"


class ChunkerConfig(BaseModel):
    type: Literal["fixed"] = "fixed"
    size: int = Field(500, gt=0, description="tokens per chunk (embedder tokenizer)")
    overlap: int = Field(50, ge=0)


class EmbedderConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    # bge-*-v1.5 expects this instruction on queries (not on passages).
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class RetrieverConfig(BaseModel):
    type: Literal["dense"] = "dense"
    top_k: int = 5


class GeneratorConfig(BaseModel):
    provider: Literal["anthropic", "ollama"] = "ollama"
    model: str = "qwen3:14b"
    max_tokens: int = 16000
    # Anthropic only: thinking runs adaptively; effort controls its depth.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Ollama only: context window; the default (2-4k) would truncate retrieved context.
    num_ctx: int = 16384


class PipelineConfig(BaseModel):
    name: str
    corpus: CorpusMode = CorpusMode.FOCUSED
    parser: Literal["pymupdf"] = "pymupdf"
    chunker: ChunkerConfig = ChunkerConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    generator: GeneratorConfig = GeneratorConfig()

    @property
    def index_key(self) -> str:
        """Short hash of everything that shapes the vector index."""
        spec = {
            "corpus": self.corpus.value,
            "parser": self.parser,
            "chunker": self.chunker.model_dump(),
            "embedder": self.embedder.model_dump(exclude={"batch_size", "query_prefix"}),
        }
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def collection_name(self) -> str:
        return f"fb_{self.corpus.value}_{self.index_key}"


def load_config(path: Path = DEFAULT_CONFIG) -> PipelineConfig:
    return PipelineConfig.model_validate(yaml.safe_load(Path(path).read_text()))
