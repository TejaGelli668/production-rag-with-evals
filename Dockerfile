# One image for the API, the UI and the CLI (ingest, eval).
# Not yet built or run by the author; see docs/docker.md.
#
#   docker build -t production-rag .
#
# PyTorch comes from the CPU-only index on Linux (see [tool.uv.sources] in pyproject.toml),
# so the image carries no CUDA libraries. Embedding and reranking run on CPU in containers.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

# Dependencies first, in their own layer: code changes don't reinstall torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-group dev

# The project itself. evals/ is needed at runtime (splits select the CI corpus;
# custom cases and gate thresholds are read by `rag eval` / `rag gate`).
COPY src ./src
COPY configs ./configs
COPY evals/splits ./evals/splits
COPY evals/custom ./evals/custom
COPY evals/gate.yaml ./evals/gate.yaml
COPY scripts ./scripts
COPY ui ./ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-group dev

ENV PATH="/app/.venv/bin:$PATH"

# Non-root; data, storage and the model cache are mounted volumes.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/storage /cache/huggingface \
    && chown -R app:app /app/data /app/storage /cache
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
