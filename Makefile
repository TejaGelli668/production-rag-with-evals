.PHONY: setup data data-focused splits explore test lint format check

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

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test
