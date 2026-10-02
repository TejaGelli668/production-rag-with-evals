# Docker

> ⚠️ **Untested.** The `Dockerfile` and `docker-compose.yml` were written without a Docker
> runtime on the author's machine, so neither has been built or run yet. The YAML structure
> was validated (services, ports, environment, volumes), and the same code runs in CI on
> Linux. The known risks are listed below. The tested way to run the project is the local
> setup in the [README](../README.md#run-it).

## What's in the stack

| Service | Image | Port | Role |
|---|---|---|---|
| `api` | built from `Dockerfile` | 8000 | FastAPI service (`/ask`, `/ask/stream`, `/feedback`, `/health`, `/pages`) |
| `ui` | same image | 8501 | Streamlit app |
| `qdrant` | `qdrant/qdrant:v1.19.1` | 6333 | Vector store in **server mode** (the local setup uses embedded mode) |
| `phoenix` | `arizephoenix/phoenix:version-20.19.0` | 6006 | Trace viewer; the API sends spans here |
| `ollama` | `ollama/ollama:0.35.0` | 11434 | Optional (`--profile ollama`), for Linux hosts without Ollama |

The LLM runs in **Ollama on the host** by default (`http://host.docker.internal:11434`). On a
Mac, containers can't use the Apple GPU, so a containerized `qwen3:14b` would be CPU-only and
very slow.

## Steps

```bash
ollama pull qwen3:14b                       # on the host
make data                                   # on the host: PDFs into ./data (mounted)
docker compose build
docker compose up -d qdrant phoenix
docker compose run --rm api rag ingest --config configs/stack_filter_rerank.yaml
docker compose up -d                        # api + ui
open http://localhost:8501                  # UI
open http://localhost:6006                  # traces
```

Ingestion writes vectors to the Qdrant server (`QDRANT_URL` is set in compose). Its resume
manifest lives in the `app_storage` volume, so an interrupted ingest picks up where it stopped.

To run the eval gate inside the container (the same check CI runs):

```bash
docker compose run --rm api python scripts/download_financebench.py --mode ci
docker compose run --rm api rag ingest --config configs/ci_gate.yaml
docker compose run --rm api rag eval --config configs/ci_gate.yaml --split ci_smoke --retrieval-only
docker compose run --rm api rag gate ci_gate__ci_smoke__retrieval
```

## Known risks (check these first)

1. **Indexing is slow on CPU.** Containers embed and rerank on CPU. CI measured about 70 s
   per filing on a 2-core runner, so the full 360-filing corpus would take hours. Give
   Docker more cores, or start with `corpus: focused` (84 filings) in a copy of the config.
   Each answer also spends a few extra seconds reranking on CPU.
2. **PyTorch on Apple Silicon Docker (`linux/arm64`).** Checked in the lockfile: `uv.lock` pins
   CPU-only torch wheels for both `manylinux_2_28_x86_64` and `manylinux_2_28_aarch64`, so the
   image should build natively on Apple Silicon. If it doesn't, build for amd64 with
   `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose build`; it runs under emulation and is
   slower.
3. **`./data` permissions on Linux.** The container runs as uid 1000. If your host uid
   differs and you download data from inside the container, the bind mount may not be
   writable. Run `make data` on the host instead (the default steps above do).
4. **Phoenix storage path.** `phoenix_data` is mounted at `/root/.phoenix`; if a newer image
   stores data elsewhere, traces won't survive a restart (they still work while it's running).
5. **First start downloads models.** The embedding model (~130 MB) and the reranker (~2.2 GB)
   download on first use into the `hf_cache` volume, so the first `/ask` is slow and the
   API's health check allows a 3-minute start period.
