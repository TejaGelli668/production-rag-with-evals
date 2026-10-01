# Production RAG With Evals

Question answering over real SEC filings (10-K, 10-Q, 8-K), built and tuned through evaluation on
the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark. Every design choice is
backed by a measured experiment.

> 🚧 **Status: Phase 1 complete**: a baseline pipeline answers questions with cited pages. Evaluation comes next. See [PLAN.md](PLAN.md) for the roadmap.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
make setup      # install dependencies + pre-commit hooks
make data       # download FinanceBench: 360 PDFs (~700 MB), checksum-verified
make explore    # profile the corpus -> docs/data_exploration.md
make check      # lint + tests

cp .env.example .env               # optional: add ANTHROPIC_API_KEY to use Claude
ollama pull qwen3:14b              # default local LLM
make ingest                        # index the corpus for configs/baseline.yaml (resumable)
make ask ID=financebench_id_03282  # a benchmark question, shown next to its gold answer
uv run rag ask "What was Netflix's FY2017 total current liabilities?" \
    --filter company=Netflix --filter fiscal_year=2017
```

Use `make data-focused` to download only the 84 filings the questions reference (~166 MB).

## Dataset

| | |
|---|---|
| Questions | 150 human-labeled questions with answers and evidence pages |
| Corpus | 360 filings · 53k pages · ~45M tokens · 40 companies · FY2015–2024 |
| Eval splits | `dev` 50 · `test` 100 (held out) · `ci_smoke` 20 (subset of dev), in [`evals/splits/`](evals/splits/) |

Findings that shape the pipeline are in [docs/data_exploration.md](docs/data_exploration.md).

**License:** FinanceBench is CC-BY-NC 4.0. The PDFs are downloaded by script and are never committed to this repo.

## How it works (baseline)

```
PDF ─► PyMuPDF pages ─► 500-token chunks (page ranges + company/year metadata)
    ─► bge-small-en-v1.5 embeddings ─► Qdrant (embedded)
question ─► dense top-5 (+ optional metadata filter) ─► LLM with numbered sources
         ─► answer with [n] citations ─► resolved to filing + page
```

Each experiment is a YAML file in [`configs/`](configs/). Only the corpus, parser, chunker and
embedder settings determine the index, so configs that change only retrieval or generation reuse it.

## Repository layout

```
src/rag/data/      dataset models, loaders, split logic
src/rag/ingest/    parsing, chunking, resumable indexing
src/rag/generate/  LLM providers (Claude, Ollama), prompts, citation resolution
src/rag/           config, embeddings, vector store, retrieval, pipeline, CLI
configs/           one YAML per experiment
scripts/           download, splits, data exploration
evals/splits/      committed eval split IDs
docs/              generated data report
tests/             unit tests
```
