.PHONY: setup data data-focused splits explore ingest ask test lint format check

setup:  ## Install dependencies and git hooks
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

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test
