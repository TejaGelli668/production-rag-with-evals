# Production RAG With Evals

Question answering over real SEC filings (10-K, 10-Q, 8-K), built and tuned through evaluation on
the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark. Every design choice is
backed by a measured experiment.

> 🚧 **Status: Phase 4 in progress.** The API, UI and tracing run locally; CI and packaging are next. See [PLAN.md](PLAN.md).

## Results so far

On the full 360-filing corpus, metadata filters plus cross-encoder reranking take answer
accuracy on the **held-out FinanceBench `test` split from 15% to 49%** (paired Δ +34 points,
95% CI [+24, +44]); the split was run once, after all tuning on `dev`. Answers stay grounded:
93% are fully supported by their sources, and it declined all 30 questions in a set of
unanswerable ones (absent companies, future periods, nonexistent items).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/test_correct-dark.svg">
  <img alt="Held-out test accuracy: E1 baseline 0.15, final filter + rerank 0.49" src="docs/img/test_correct-light.svg">
</picture>

On `dev`, where every decision was made, the same change goes from 12% to 40%. That closes
two-thirds of the gap to an oracle that is handed the gold pages (54%):

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

## Run the app

Everything runs locally: the LLM is `qwen3:14b` on [Ollama](https://ollama.com), and no API keys are needed.

```bash
ollama pull qwen3:14b
make data && make ingest CONFIG=configs/stack_filter_rerank.yaml   # once: download + index
make serve      # API on http://localhost:8000 (docs at /docs)
make ui         # chat UI on http://localhost:8501
```

- **API** (`src/rag/api/`): `POST /ask` returns the answer with sources, citations, timings
  and token usage. `POST /ask/stream` streams the same as server-sent events: sources, then
  tokens, then the final answer. `POST /feedback` records 👍/👎, and `GET /health` reports
  status. Every answer and rating is logged to `storage/logs/` by `request_id`.
- **UI** (`ui/app.py`): the answer streams in token by token, each source card shows the
  filing, page and rerank score, the rendered PDF page is displayed, and there's a 👍/👎
  button.
- **Tracing**: run `make phoenix`, then `RAG_TRACING=1 make serve`. Each request appears in
  Phoenix at http://localhost:6006 as `rag.ask → retrieve → rerank` and `generate`, with
  the retrieved documents, rerank scores, prompts and token counts (OpenInference
  conventions).

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
src/rag/api/       FastAPI service (ask, streaming, feedback, health, page images)
src/rag/           config, embeddings, vector store, retrieval, pipeline, tracing, CLI
ui/                Streamlit app
configs/           one YAML per experiment
scripts/           download, splits, data exploration
evals/splits/      committed eval split IDs
evals/custom/      hand-written cases (unanswerable questions)
evals/results/     committed run summaries; raw runs live in evals/runs/ (gitignored)
evals/judge/       judge sanity-check reports and human labels
docs/              generated data report
tests/             unit tests
```
