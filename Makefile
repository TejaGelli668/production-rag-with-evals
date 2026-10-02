.PHONY: setup data data-focused splits explore ingest ask eval eval-retrieval check-judge serve ui phoenix test lint format check

setup:  ## Install dependencies (all groups) and git hooks
	uv sync
	uv run pre-commit install

data:  ## Download the full FinanceBench corpus (~700 MB, checksum-verified)
	uv run python scripts/download_financebench.py --mode full

data-focused:  ## Download only the 84 documents referenced by questions
	uv run python scripts/download_financebench.py --mode focused

splits:  ## Regenerate eval splits (deterministic; committed to evals/splits/)
	uv run python scripts/make_splits.py

explore:  ## Profile the corpus and write docs/data_exploration.md
	uv run python scripts/explore_data.py

CONFIG ?= configs/baseline.yaml

ingest:  ## Parse, chunk, embed and index the corpus for CONFIG (resumable)
	uv run rag ingest --config $(CONFIG)

ask:  ## Ask a question: make ask Q="..." or make ask ID=financebench_id_03029
	uv run rag ask $(if $(ID),--id $(ID),"$(Q)") --config $(CONFIG)

SPLIT ?= dev

eval:  ## Run CONFIG on SPLIT with judges: make eval CONFIG=configs/e0_oracle.yaml
	uv run rag eval --config $(CONFIG) --split $(SPLIT)

eval-retrieval:  ## Retrieval-only metrics at k=1..20 (no LLM calls)
	uv run rag eval --config $(CONFIG) --split $(SPLIT) --retrieval-only

check-judge:  ## Known-good / known-bad sanity checks for the judges
	uv run rag check-judge --split $(SPLIT)

SERVE_CONFIG ?= configs/stack_filter_rerank.yaml

serve:  ## Run the API (FastAPI) with the final config; set RAG_TRACING=1 to trace to Phoenix
	RAG_CONFIG=$(SERVE_CONFIG) uv run uvicorn rag.api.app:app --host 127.0.0.1 --port 8000

ui:  ## Run the Streamlit UI (needs `make serve`)
	uv run streamlit run ui/app.py --server.port 8501

phoenix:  ## Run the Phoenix trace viewer at http://localhost:6006
	uvx --from arize-phoenix phoenix serve

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test
