# Production RAG With Evals

Question answering over real SEC filings (10-K, 10-Q, 8-K), built and tuned through evaluation on
the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark. Every design choice is
backed by a measured experiment.

> 🚧 **Status: Phase 0 complete** (setup, data, eval splits). See [PLAN.md](PLAN.md) for the roadmap.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
make setup      # install dependencies + pre-commit hooks
make data       # download FinanceBench: 360 PDFs (~700 MB), checksum-verified
make explore    # profile the corpus -> docs/data_exploration.md
make check      # lint + tests
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

## Repository layout

```
src/rag/data/      dataset models, loaders, split logic
scripts/           download, splits, data exploration
evals/splits/      committed eval split IDs
docs/              generated data report
tests/             unit tests
```
