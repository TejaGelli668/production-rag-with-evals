"""Local sentence-transformer embeddings (MPS/CUDA when available)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from huggingface_hub.utils import logging as hf_hub_logging
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from rag.config import EmbedderConfig


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        # Long pages are tokenized whole for chunking; silence the >512-token warning,
        # plus the weight-loading progress bar and anonymous-Hub notice on every CLI call.
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        hf_logging.disable_progress_bar()
        hf_hub_logging.set_verbosity_error()
        self.model = SentenceTransformer(cfg.model, device=_device())

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.cfg.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.cfg.query_prefix + text, normalize_embeddings=True, convert_to_numpy=True
        )
