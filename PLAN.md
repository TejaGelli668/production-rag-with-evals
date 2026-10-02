# Production RAG With Evals: Project Plan

> A question-answering system over SEC filings (10-K, 10-Q, 8-K), built and tuned through evaluation.
> Every design choice is backed by a measured experiment on the FinanceBench benchmark.

---

## 1. Goal and pitch

**Pitch:** *"I built a RAG system over real financial filings, measured it against a human-labeled benchmark, ran controlled experiments, and can show with numbers why each component is there."*

What sets it apart from a typical RAG demo:

| Typical portfolio RAG | This project |
|---|---|
| 3 toy text files | ~360 real SEC filings, 100+ pages each, full of tables |
| "Eval" = latency logging | Human-labeled ground truth: answers **and** evidence pages |
| One pipeline, untested | Experiment matrix with before/after metrics for each component |
| Several frameworks and vector DBs bolted on | One clean pipeline behind small interfaces |
| No regression safety | CI eval gate that fails a PR if quality drops |

---

## 2. Dataset: FinanceBench

**Source:** [patronus-ai/financebench](https://github.com/patronus-ai/financebench) (GitHub) / `PatronusAI/financebench` (Hugging Face)
**License:** CC-BY-NC 4.0, fine for a non-commercial portfolio. **Do not commit the PDFs to the repo.** Download them with a script.

### What's in it (checked on 2026-10-01)

| Item | Count |
|---|---|
| Labeled questions (open-source sample) | **150** |
| Companies covered by questions | 32 |
| Distinct documents referenced by questions | 84 |
| Documents in the metadata file | 361 rows / 360 unique (269 10-K, 30 8-K, 29 earnings, 27 10-Q, 5 annual reports) |
| PDFs in the GitHub repo | 368 |

**Question fields:** `financebench_id`, `company`, `doc_name`, `question_type`, `question_reasoning`, `question`, `answer`, `justification`, `evidence[]`
**Evidence fields:** `evidence_text`, `doc_name`, `evidence_page_num`, `evidence_text_full_page`
**Document fields:** `doc_name`, `company`, `gics_sector`, `doc_type`, `doc_period`, `doc_link`

**Question mix:** 50 `metrics-generated`, 50 `domain-relevant`, 50 `novel-generated`. Reasoning types are mostly *numerical reasoning* and *information extraction*, plus some *logical reasoning*.

Example: *"What is the FY2018 capital expenditure amount (in USD millions) for 3M?"* → `$1577.00`

### Corpus modes

| Mode | Documents | Purpose |
|---|---|---|
| `focused` | 84 (only the ones the questions reference) | Fast, cheap iteration during development |
| `full` | All 360 with metadata | The realistic setting: retrieval has to find the right filing among hundreds of similar ones |

Headline results are reported on **`full`**. That's the hard, honest setting, and the FinanceBench paper found that standard RAG baselines fail on most questions in it.

### Eval splits

| Split | Size | Use |
|---|---|---|
| `dev` | 50 (stratified by `question_type`) | Tuning and experiments; looked at often |
| `test` | 100 (stratified) | Held out; run only at milestones to report final numbers |
| `custom` | ~40 written by hand | Unanswerable questions (company not in corpus, future fiscal year, metric not disclosed), ambiguous questions, multi-document comparisons |
| `ci_smoke` | 20 (subset of `dev`) | Run on every PR by the CI gate |

Splits are fixed with a seed and committed as ID lists in `evals/splits/`.

---

## 3. Architecture

```
                ┌───────────────────────── Ingestion (offline) ─────────────────────────┐
 SEC PDFs ──►   │ parse (page-aware, tables) ─► chunk ─► enrich metadata ─► embed       │ ──► Qdrant
                │  (company, doc_type, fiscal_year, page_num, section)  dense + sparse  │    (hybrid)
                └───────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────── Query (online) ───────────────────────────┐
 Question ──►   │ query analysis ─► hybrid retrieve ─► rerank ─► generate with citations│ ──► Answer
                │ (company/year       (dense + BM25,     (cross-     (cited pages, or     │    + sources
                │  filter extraction)  RRF fusion)        encoder)    "insufficient info")│
                └───────────────────────────────────────────────────────────────────────┘
                         │ every step traced (latency, tokens, cost, retrieved chunks)
                         ▼
                   Arize Phoenix                     Eval harness ◄── golden splits
```

### Key design decisions

- **Page-aware throughout.** Every chunk keeps `doc_name` and `page_num`, because FinanceBench labels evidence by page. That makes it possible to measure retrieval exactly.
- **Metadata filtering.** Questions almost always name a company and fiscal year. Pulling those out and filtering on them is probably the single biggest retrieval win. We'll measure it.
- **Tables.** Financial statements are tables, and how they're parsed decides whether numerical questions succeed. Text extraction and table-aware extraction get compared.
- **Refusals by design.** When retrieval confidence is low, the system says "insufficient information" instead of guessing. The `custom` unanswerable questions measure this.
- **Plain Python core.** No LangChain or LlamaIndex in the core pipeline, so every step is easy to read and explain in an interview. Libraries are used for the components themselves: parsers, Qdrant client, embedding models.

---

## 4. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language / tooling | Python 3.12, `uv`, `ruff`, `pytest` | Fast and modern; standard choices |
| PDF parsing | `pymupdf` (baseline) vs `docling` (table-aware) | Compared in an experiment |
| Embeddings | `BAAI/bge-small-en-v1.5` locally (baseline); try `bge-m3` or an API model | Free, no API key needed for retrieval |
| Sparse / BM25 | Qdrant sparse vectors (BM25 / SPLADE) | Hybrid search inside one store |
| Vector store | **Qdrant**: embedded mode now, Docker server from Phase 4 | Native hybrid search and payload filtering |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder) | Strong open-source reranker |
| Generator LLM | Claude Sonnet (`claude-sonnet-5-5`), behind a provider interface; Ollama `qwen3:14b` until an API key is added | Strong numerical reasoning; can be swapped |
| Judge LLM | Claude Haiku (`claude-haiku-4-5`) | Cheap; checked against human labels |
| Local fallback | Ollama | Runs with no API key |
| API | FastAPI with streaming (SSE) | Production-style service |
| UI | Streamlit | Quick to build; shows answer, cited pages and PDF snippet |
| Tracing | Arize Phoenix (OpenTelemetry) | A single container; per-request traces |
| Packaging | Docker Compose (api, qdrant, ui, phoenix) | Starts with one command |
| CI | GitHub Actions | Lint, unit tests and eval gate |

---

## 5. Evaluation design

### Metrics

**Retrieval** (from `evidence[].doc_name` and `evidence_page_num`)
- **Doc Hit@k**: is the correct filing among the top-k results?
- **Page Hit@k**: is a gold evidence page among the top-k results?
- **MRR**: reciprocal rank of the first correct page

**Generation**
- **Answer correctness**: LLM judge compares the answer to the gold answer, with **numeric tolerance** (units and rounding, e.g. `$1,577M` = `$1577.00`). Grades: correct / incorrect / refused.
- **Faithfulness**: is every claim supported by the retrieved context?
- **Citation accuracy**: do the cited pages contain the evidence?
- **Refusal quality**: correct refusal on unanswerable questions vs. false refusal on answerable ones

**Operational:** latency p50/p95, tokens and $ per query, ingestion time

### Bounds (to tell retrieval failures from generation failures)
- **Closed-book** (no retrieval): lower bound; shows what the LLM already "knows"
- **Oracle context** (gold evidence pages given directly): upper bound for generation
- The **gap** between the system and the oracle is how much retrieval is costing us.

### Judge calibration
Label about 60 answers by hand, then report agreement between the judge and the human labels (accuracy and Cohen's κ). If the judge disagrees too often, fix the judge prompt before trusting any numbers.

### Error analysis
Tag each failure as one of: wrong doc / right doc but wrong page / right page but table parse broke / arithmetic error / hallucination / false refusal. Report the breakdown in the README.

---

## 6. Experiment matrix

Each step changes **one thing**, is run on `dev`, and is logged to `evals/results/` with the config and git SHA.

| # | Experiment | Hypothesis |
|---|---|---|
| E0 | Closed-book and oracle-context bounds | Sets the floor and ceiling |
| E1 | **Baseline:** pymupdf, fixed 500-token chunks (50 overlap), dense, k=5 | Reference point |
| E2 | Chunking: fixed vs recursive vs page-level vs section-aware | Page and section chunks fit the evidence labels better |
| E3 | Table-aware parsing (docling) | Big improvement on numerical-reasoning questions |
| E4 | Hybrid (dense + BM25, RRF) | Helps with exact terms (tickers, line items, "FY2019") |
| E5 | Metadata filtering (company / fiscal year / doc_type) | Large Doc Hit@k improvement in `full` mode |
| E6 | Cross-encoder reranker (retrieve 50 → rerank to 5) | Better Page Hit@5 and answer precision |
| E7 | Query rewriting / decomposition for multi-part questions | Helps comparison and ratio questions |
| E8 | Prompting: structured "find numbers → compute → answer" | Fewer arithmetic errors |
| E9 | Embedding model swap (bge-small → bge-m3 / API) | Quality vs cost trade-off |

The best configuration gets run once on `test` for the headline numbers.

---

## 7. Repository structure

```
production-rag-with-evals/
├── PLAN.md
├── README.md                    # results table, charts, architecture, demo link
├── pyproject.toml / uv.lock
├── docker-compose.yml
├── Makefile                     # make data | ingest | serve | eval | eval-ci
├── .env.example
├── configs/                     # YAML pipeline configs, one per experiment
│   ├── baseline.yaml
│   └── best.yaml
├── scripts/
│   └── download_financebench.py # fetch PDFs + jsonl into data/ (gitignored)
├── src/rag/
│   ├── config.py                # pydantic settings + YAML config loading
│   ├── ingest/                  # parsers/, chunkers/, metadata.py, pipeline.py
│   ├── retrieve/                # dense.py, sparse.py, hybrid.py, filters.py, rerank.py
│   ├── generate/                # llm.py (provider interface), prompts/, citations.py
│   ├── pipeline.py              # query → answer orchestration
│   ├── tracing.py               # OpenTelemetry / Phoenix setup
│   └── api/                     # FastAPI app, schemas, streaming
├── ui/
│   └── app.py                   # Streamlit
├── evals/
│   ├── splits/                  # dev.json, test.json, ci_smoke.json (IDs only)
│   ├── custom/                  # hand-written unanswerable / adversarial questions
│   ├── metrics/                 # retrieval.py, correctness.py, faithfulness.py
│   ├── judge/                   # judge prompts + calibration labels
│   ├── run_eval.py              # run a config on a split → results JSON
│   ├── compare.py               # diff two runs, regression check
│   └── results/                 # committed run summaries (not raw outputs)
├── tests/                       # unit tests (chunkers, metric math, numeric matching)
└── .github/workflows/
    ├── ci.yml                   # lint + unit tests
    └── eval-gate.yml            # ci_smoke eval; fail if below thresholds
```

---

## 8. Phases and milestones

### Phase 0: Setup and data (~½ day) ✅
- [x] `git init`, `uv` project (Python 3.12), ruff, pytest, pre-commit, `.gitignore` (including `data/`)
- [x] `scripts/download_financebench.py`: questions jsonl, document info jsonl, PDFs, pinned to upstream commit `cc39aeb`, with each PDF checked against its git blob SHA; re-runs skip intact files
- [x] Data exploration as a reproducible script (`scripts/explore_data.py` → [`docs/data_exploration.md`](docs/data_exploration.md)) instead of a notebook. **`evidence_page_num` is 0-based** (189/189 evidence items match the PyMuPDF page index)
- [x] Create the `dev` / `test` / `ci_smoke` splits in `evals/splits/` (seed 13, stratified by `question_type`)
- **Done when:** `make data` downloads everything reproducibly; splits are committed

Notes from Phase 0:
- `full` corpus = **360** documents: the metadata file has 361 rows, and one (`FOOTLOCKER_2023_annualreport`) appears twice; the first row is kept. 8 upstream PDFs have no metadata and are left out.
- No OCR needed; table structure is the parsing challenge. Each company has a median of 9 filings, which are the hard distractors. 23% of questions need more than one page. See the implications section of the data report.

### Phase 1: Baseline pipeline (~1–2 days) ✅
- [x] Page-aware PDF parsing (pymupdf) and a fixed-size token chunker (500 tokens, 50 overlap). Chunks may cross pages and record `page_start`/`page_end`; text is sliced by character offsets so original casing is kept. Company, doc type and fiscal year are copied onto every chunk
- [x] Qdrant in **embedded mode** (`storage/qdrant`, no Docker; set `QDRANT_URL` to use a server). `focused` is indexed: 84 filings → 18,596 chunks in ~3 min on Apple M5 Pro (MPS). Ingestion resumes from a manifest
- [x] Dense retrieval (bge-small-en-v1.5), with optional metadata filters (`--filter company=3M`)
- [x] Generation with `[n]` citations resolved to document and page, through an LLM provider interface: Ollama (`qwen3:14b`, used for now) and Claude (`claude-sonnet-5-5`, adaptive thinking, `effort: medium`, server-side refusal fallback; tested with a fake client until an API key is added)
- [x] CLI: `rag ingest`, `rag ask "..."`, `rag ask --id <financebench_id>` (shows the gold answer and marks retrieved chunks that overlap gold pages)
- **Done when:** the CLI answers a FinanceBench question with cited pages

Notes from Phase 1 (first answers from the baseline, not measured yet):
- **Chunk boundaries split tables.** 3M FY2018 capex: a chunk overlapping the gold page was retrieved, but the capex line sat in the previous chunk, so the model correctly said "insufficient information". So Page Hit@k can over-credit, and Phase 2 should also check whether the evidence text itself was retrieved.
- **No company awareness.** Boeing's tax-rate question retrieved Kraft Heinz, PayPal and Microsoft chunks. With `--filter company=Netflix --filter fiscal_year=2017`, the Netflix current-liabilities question went from a confidently wrong, cited $3,529.6M to the correct $5,466.31M. That's an early signal for E5.
- **Wrong answers can still carry citations,** so the faithfulness and citation checks in Phase 2 are essential.

### Phase 2: Eval harness (~2 days) 🚧 in progress
- [x] Retrieval metrics (Doc Hit@k, Page Hit@k, Page Recall@k, MRR), plus **evidence coverage@k** (added because Phase 1 showed page overlap over-credits retrieval)
- [x] Correctness judge (rubric with unit, scale and rounding tolerance) and faithfulness judge (claim-by-claim against the retrieved sources), using structured JSON output; a programmatic `numeric_match` cross-check; citation checks (`has_citation`, `cites_gold_page`, `invalid_citations`)
- [x] `rag eval` → `evals/runs/<run>/` (results, errors sidecar, full traces, summary with bootstrap CIs); `rag compare` → paired differences; `--retrieval-only` sweeps k=1..20 with no LLM calls
- [x] Judge sanity checks (`rag check-judge`): 4-case smoke test passes all 6 checks; full `dev` run pending
- [x] Labeling and calibration tooling (`rag label`, `rag calibrate`, Cohen's kappa)
- [ ] **Hand-label ~60 answers** (needs a human) and calibrate the judge
- [x] Drafted the `unanswerable` set (30 cases in 4 categories). **Needs human review**
- [x] E1 baseline on `dev`: 14% correct, 67% refused, Doc Hit@5 92%, Page Hit@5 22% (see `evals/results/baseline__dev.json`)
- [x] E0 bounds on `dev`: closed-book 22% correct, oracle 54%, baseline 14%. Retrieval costs ~41 points; the local generator caps out near 54%
- [x] Retrieval sweep (k=1..20) for the baseline
- [x] Full judge sanity checks on `dev`: correctness judge ≥ 98% on every check; faithfulness judge 84% both ways (treat as approximate) (see Phase 3)
- **Done when:** one command produces a full metrics report for any config

Notes from Phase 2 so far:
- The judge is currently `qwen3:14b`, the same model as the generator, because there's no API key. Runs flag this. Switch to `claude-haiku-4-5` once a key is set, and re-run the bounds and baseline so all numbers come from one judge.
- Spot check of all 16 judged baseline answers: the correctness verdicts looked right. The faithfulness judge missed an arithmetic error (`financebench_id_04254`: 1,263 + 636 reported as 2,100).
- One baseline case hit a 600 s Ollama timeout. It was logged as an infra error, not a wrong answer, and a re-run retries it.

### Phase 3: Experiments (~2–3 days) ✅
- [x] `full` corpus indexed (360 filings, 81,517 chunks); `e1_full` is the Phase 3 baseline
- [x] E2 chunking: 250-token chunks hurt (doc hit −0.20\*); page-bounded chunks change nothing, alone or stacked. **Keep 500-token windows**
- [x] E3 markdown tables (pymupdf4llm): no retrieval gain; judged on `focused` 16% vs 14% (n.s.). E3b, with gold pages from each parser: 54% vs 52% (n.s.). **Keep PyMuPDF**
- [x] E4 BM25 hybrid: hurts alone (doc hit −0.18\*), adds nothing with filters. **Dropped**
- [x] E5 company + fiscal-year filters: page hit@5 0.20 → 0.38\*. **Kept**
- [x] E6 cross-encoder rerank of top-50: with filters, page hit@5 → 0.54\*. **Kept**
- [x] E7 LLM query rewriting: page hit@5 +0.06 (CI touches 0), page hit@20 +0.10\*; no gain on multi-page questions. **Not adopted** (extra LLM call per query)
- [x] E8 stepwise prompt: refusals −14 pts\* but accuracy unchanged; the extra answers were wrong. **Not adopted**
- [ ] E9 embedding swap (bge-m3): deferred, since re-indexing 360 filings takes ~2 h locally
- [x] **Judged on `dev`: `e1_full` 12% → `stack_filter_rerank` 40% correct (Δ +0.28 [+0.16, +0.40]\*)**; oracle ceiling 54%
- [x] Error analysis (`rag analyze`): wrong-filing errors 16 → 1; the remaining failures are mostly wrong page in the right filing (19/30)
- [x] README charts (`scripts/make_charts.py`, light/dark SVG)
- [x] **Final config = `stack_filter_rerank`.** Held-out `test` (n=100, run once): **15% → 49% correct, Δ +0.34 [+0.24, +0.44]\***; wrong-filing errors 53 → 4
- [x] Unanswerable set: 30/30 correctly declined (the set still needs human review); the trade-off is a 26% refusal rate on answerable `test` questions
- [x] Full judge sanity checks on `dev`: correctness judge ≥ 98% on every check; faithfulness judge 84% both ways (treat as approximate)
- **Done when:** the results table shows a measured gain for each component that was kept

Results and reasoning are in [`docs/experiments.md`](docs/experiments.md). Infra changes: one embedded Qdrant folder per collection (embedded Qdrant allows one process per folder); an on-disk parse cache with parallel pre-parsing (`scripts/preparse.py`).

### Phase 4: Service, UI and observability ✅ (Docker deferred)
**Decisions (2026-10-02):** fully local. The LLM is Ollama `qwen3:14b` with no cloud API keys. Docker Compose moves to the very end.
- [x] FastAPI: `/ask`, `/ask/stream` (SSE: sources → tokens → answer), `/feedback`, `/health`, `/pages/{doc}/{page}.png`; request and feedback logs joined by `request_id`; one pipeline call at a time (pipeline isn't thread-safe); the streaming pipeline runs on one thread so trace context stays intact
- [x] Streamlit: streaming answer, source cards (filing, page, rerank score, cited), rendered PDF page, 👍/👎 via `st.feedback`, example questions, intro on the empty state
- [x] Phoenix tracing (opt-in `RAG_TRACING=1`): OpenInference spans `rag.ask → retrieve → rerank`, `generate`, with documents, scores, prompts and token counts; FastAPI auto-instrumented
- [x] Verified end to end on the real index: 3M capex ($1,577M ✓), Netflix liabilities ($5,466.31M ✓, wrong in Phase 1), NVIDIA (declined ✓), Boeing tax rate (signs flipped ✗, a generation error visible in the trace)
- [ ] Docker Compose (api, qdrant server, ui, phoenix): **deferred to the end**
- **Done when:** a fresh clone, `make data && make ingest && make serve` + `make ui`, works end to end

### Phase 5: CI, demo and write-up (~1 day)
- [x] `ci.yml`: lint + 81 unit tests on every push and PR (Linux, CPU-only torch)
- [x] Eval gate, **retrieval-only** (no LLM): `ci_smoke` on a 51-filing CI corpus (gold filings plus neighbouring-year distractors), thresholds in `evals/gate.yaml`. Verified to fail when filters are disabled or chunks shrink to 250 tokens. Metrics on GitHub's Linux CPU match the local Mac GPU run exactly
- [x] Built index cached between CI runs: the cold build is ~55 min on a 2-core runner, cached runs are ~2 min
- [x] Demo: GIF of the UI in the README (local app, recorded from the browser)
- [ ] Human judge calibration (`rag label` + `rag calibrate`), plus review of the drafted unanswerable set (both need a human)
- [x] README: architecture diagram (Mermaid), results, kept/dropped experiments, limitations, how to run
- [ ] Make the repo public (owner's decision)
- [ ] Last: Docker Compose
- **Done when:** the repo is public and the README tells the story with numbers

**Total:** about 8–11 focused days.

---

## 9. Budget estimate (rough)

| Item | Estimate |
|---|---|
| Embeddings and reranking | $0 (local models) |
| One full eval run on `dev` (50 Qs × generation + 2–3 judge calls) | ~$0.50–$2 |
| Experiment phase (~30–50 runs) | ~$15–40 |
| CI smoke runs (20 Qs per PR) | cents per run |

Ways to keep this down: run in `focused` mode during development, cache LLM responses by (prompt, model), and use the cheap judge model.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Table parsing on scanned or complex PDFs | Compare parsers in E3; record known failure cases |
| LLM judge disagrees with humans | Calibration set; numeric-match rules before the LLM judge |
| Overfitting to `dev` | Held-out `test` set run only at milestones |
| Large corpus makes ingestion slow | Cache parsed pages; build `focused` first |
| Dataset license (non-commercial) | Never redistribute the PDFs; download them with a script; note it in the README |
| Free-tier hosting can't fit the full index | Demo on the `focused` corpus; report `full` results from local runs |

---

## 11. Open decisions (defaults assumed, change any time)

| Decision | Default in this plan | Alternatives |
|---|---|---|
| LLM provider | Claude Sonnet (generation) + Haiku (judge), Ollama fallback | OpenAI, Gemini, fully local |
| API budget | ~$20–40 total | Lower: more local models, fewer runs |
| Live demo hosting | Hugging Face Spaces (Docker) + Qdrant Cloud free tier, `focused` corpus | Fly.io / Render, local only |
| UI | Streamlit | Next.js, if targeting full-stack roles |
| Target role emphasis | AI/ML engineer (eval rigor first) | Backend (more infra), full-stack (richer UI) |
