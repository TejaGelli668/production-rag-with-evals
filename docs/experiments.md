# Experiments

Each experiment changes one thing from its baseline and is measured on the `dev` split
(50 FinanceBench questions). Each one is a config in [`configs/`](../configs/) and can be
reproduced with `rag eval --config <file> --split dev [--retrieval-only]`.

Δ values are **paired differences against the baseline on the same 50 questions**, with
bootstrap 95% CIs; **\*** marks intervals that exclude zero. With n = 50, pass-rate CIs are
roughly ±14 points, so only large effects are distinguishable. The held-out `test` split is
reserved for the final configuration.

## E0: bounds (focused corpus, judged)

| Config | Correct | Refused | What it shows |
|---|---|---|---|
| `e0_closed_book` (no retrieval) | 22% | 16% | How much `qwen3:14b` answers from memory |
| `baseline` (E1) | 14% | 67% | Dense top-5 retrieval |
| `e0_oracle` (gold pages given) | **54%** | 14% | The generator's ceiling |

- **Retrieval costs 41 points.** The baseline sits 41 points below the oracle (Δ +0.41 [+0.24, +0.55]\*).
- **The baseline doesn't beat memory.** It's no better than closed-book (Δ +0.08 [−0.06, +0.24]):
  it mostly declines when retrieval misses (67% refused), while closed-book guesses.
- **The local generator caps out.** Even with the gold pages, `qwen3:14b` gets only 54% right,
  so a stronger generator should raise the ceiling.

## E1 on the full corpus

Moving from the 84 filings the questions reference to all 360 filings adds the realistic
distractors: other years and other filing types from the same company.

| Corpus | Doc hit@5 | Page hit@5 | Evidence coverage@5 |
|---|---|---|---|
| `focused` (84 filings) | 0.92 | 0.24 | 0.56 |
| `full` (360 filings) | **0.66** | 0.20 | 0.55 |

## E2–E6: retrieval (full corpus, retrieval-only)

Retrieval-only runs need no LLM calls, so each config is screened in about a minute (reranked
ones in a few minutes).

| Config | Change from `e1_full` | Doc hit@5 | Page hit@5 | Evidence cov.@5 | MRR@5 | Page hit@20 |
|---|---|---|---|---|---|---|
| `e1_full` | — | 0.66 | 0.20 | 0.55 | 0.14 | 0.32 |
| `e5_company` | filter: company | 0.74 | 0.22 | 0.56 | 0.14 | 0.34 |
| `e5_company_year` | filter: company + fiscal year | **0.96\*** | **0.38\*** | 0.64\* | 0.21\* | 0.62\* |
| `e6_rerank` | rerank top-50 (bge-reranker-v2-m3) | 0.70 | 0.28 | 0.61\* | 0.23\* | 0.42 |
| `e4_hybrid` | dense + BM25 (RRF) | **0.48\*** ↓ | 0.18 | 0.52 | 0.14 | 0.22\* ↓ |
| `stack_filter_hybrid` | filter + hybrid | 0.94\* | 0.38\* | 0.60\* | 0.23\* | 0.58\* |
| **`stack_filter_rerank`** | **filter + rerank** | **0.98\*** | **0.54\*** | **0.69\*** | **0.37\*** | **0.72\*** |
| `stack_full` | filter + hybrid + rerank | 0.98\* | 0.56\* | 0.69\* | 0.36\* | 0.74\* |

**Findings**

- **E5, company + year filters, is the largest single win.** Filtering by company alone barely
  helps, because the confusable filings are the *same company's* other years. The rule-based
  analyzer ([`src/rag/query.py`](../src/rag/query.py)) identifies the company in 50/50 `dev`
  questions and the gold filing's year in 48/50. When a filter matches nothing, it relaxes to
  company only, then to no filter.
- **E6, reranking, helps most on top of filtering.** Alone it is within noise on page hit
  (+0.08); after filtering, the candidate pool comes from the right filing and the
  cross-encoder picks the right page (0.38 → 0.54).
- **E4, BM25 hybrid, hurts on its own.** Filings share a great deal of boilerplate, so keyword
  matches pull in chunks from other companies and years (doc hit 0.66 → 0.48\*). With filters it
  adds nothing measurable (`stack_filter_hybrid` = filter alone; `stack_full` ≈
  `stack_filter_rerank`), so **hybrid is dropped**: it costs a 81k-chunk BM25 index for no gain.
- **Cost of reranking:** median retrieval latency goes from 0.07 s (`e1_full`) to 5.4 s
  (`e6_rerank`) and 7.6 s (`stack_filter_rerank`): 50 candidates × 512 tokens through a
  568M-parameter cross-encoder. These were measured while the GPU was also indexing, so they
  are upper bounds; idle latency is not yet measured. Shrinking the candidate pool is the
  obvious lever if latency matters.

## E3: table-aware parsing (focused corpus)

`pymupdf4llm` emits markdown tables, which keeps each row's label and values on one line
(`| Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420) |`). Plain
PyMuPDF puts every table cell on its own line. Parsing costs about 0.45 s/page (a neural layout
model), roughly 3 CPU-hours for the full corpus, so E3 was run on `focused` first.

| Config | Doc hit@5 | Page hit@5 | Evidence cov.@5 | MRR@5 |
|---|---|---|---|---|
| `baseline` (PyMuPDF text) | 0.92 | 0.24 | 0.56 | 0.18 |
| `e3_md_focused` (markdown) | 0.90 | 0.20 | 0.55 | 0.13 |

Markdown parsing does **not** improve retrieval (all differences within noise). Its expected
benefit is that the generator can read tables, which is measured by the judged runs below.

## E2: chunking (full corpus)

_Pending: indexes are building._

## Judged end-to-end runs

_Pending._
