# Production RAG With Evals

Question answering over real SEC filings (10-K, 10-Q, 8-K), built and tuned through evaluation on
the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark. Every design choice is
backed by a measured experiment.

> 🚧 **Status: Phase 3 in progress** (experiments). See [PLAN.md](PLAN.md).

## Results so far

On the full 360-filing corpus, metadata filters plus cross-encoder reranking take answer
accuracy on FinanceBench `dev` from **12% to 40%** (paired Δ +28 points, 95% CI [+16, +40]),
closing two-thirds of the gap to the oracle that is handed the gold pages. Answers stay
grounded: 93% are fully supported by their sources.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/end_to_end_correct-dark.svg">
  <img alt="Answer accuracy on dev: closed-book 0.22, E1 baseline 0.12, filter + rerank 0.40, oracle 0.54" src="docs/img/end_to_end_correct-light.svg">
</picture>

Each component was kept only if it measurably helped. Company + fiscal-year filtering was
the largest single win; BM25 hybrid search and smaller or page-bounded chunks were tested
and dropped.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/retrieval_page_hit-dark.svg">
  <img alt="Page hit@5 by retrieval configuration; filter + rerank reaches 0.54 vs 0.20 for the dense baseline" src="docs/img/retrieval_page_hit-light.svg">
</picture>

Full tables, paired significance tests and error analysis are in
[docs/experiments.md](docs/experiments.md). Caveat: generator and judge are currently the
same local model (`qwen3:14b`); see [docs/evaluation.md](docs/evaluation.md#known-limitations).

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

## Evaluation

```bash
make eval CONFIG=configs/baseline.yaml               # judged run on dev -> evals/results/
make eval-retrieval CONFIG=configs/baseline.yaml     # retrieval metrics at k=1..20, no LLM
uv run rag compare baseline__dev e0_oracle__dev       # paired differences with 95% CIs
uv run rag check-judge                                # judge sanity checks
uv run rag label e0_oracle__dev && uv run rag calibrate e0_oracle__dev   # human calibration
```

Metrics, bounds, harness guarantees and known limitations are described in
[docs/evaluation.md](docs/evaluation.md).

Each experiment is a YAML file in [`configs/`](configs/). Only the corpus, parser, chunker and
embedder settings determine the index, so configs that change only retrieval or generation reuse it.

## Repository layout

```
src/rag/data/      dataset models, loaders, split logic
src/rag/ingest/    parsing, chunking, resumable indexing
src/rag/generate/  LLM providers (Claude, Ollama), prompts, citation resolution
src/rag/evals/     cases, metrics, judges, runner, summaries, human labeling
src/rag/           config, embeddings, vector store, retrieval, pipeline, CLI
configs/           one YAML per experiment
scripts/           download, splits, data exploration
evals/splits/      committed eval split IDs
evals/custom/      hand-written cases (unanswerable questions)
evals/results/     committed run summaries; raw runs live in evals/runs/ (gitignored)
evals/judge/       judge sanity-check reports and human labels
docs/              generated data report
tests/             unit tests
```
