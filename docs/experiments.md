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
benefit is that the generator can read tables. The judged run can't show that either:

| Config (judged, focused) | Correct | Refused |
|---|---|---|
| `baseline` | 14% | 67% |
| `e3_md_focused` | 16% (Δ +0.02 [−0.06, +0.12]) | 62% |

That isn't a fair test, because with plain dense retrieval 60% of questions never see the
right page, so the parser rarely gets a chance to matter.

**E3b isolates the parser** by giving the generator the gold pages, as the oracle does, but
with page text from our own parsers instead of FinanceBench's extraction:
`e3b_oracle_pymupdf` vs `e3b_oracle_md`. The pages and questions are identical; only the
table format differs. _Results pending._

## E2: chunking (full corpus)

| Config | Chunks | Doc hit@5 | Page hit@5 | Evidence cov.@5 | Page hit@20 |
|---|---|---|---|---|---|
| `e1_full`: 500-token windows (may cross pages) | 81,517 | 0.66 | 0.20 | 0.55 | 0.32 |
| `e2_page`: 500-token windows within a page | 103,211 | 0.52 (−0.14) | 0.22 (+0.02) | 0.52 (−0.02) | 0.42 (+0.10) |
| `e2_fixed250`: 250-token windows | 162,911 | **0.46\*** (−0.20) | 0.14 (−0.06) | **0.45\*** (−0.10) | 0.34 (+0.02) |

On top of the kept stack (filters + rerank), page-bounded chunks change nothing measurable:

| Config | Doc hit@5 | Page hit@5 | Evidence cov.@5 | MRR@5 | Page hit@20 |
|---|---|---|---|---|---|
| `stack_filter_rerank` (500-token windows) | 0.98 | 0.54 | 0.69 | 0.37 | 0.72 |
| `stack_page_filter_rerank` (page-bounded) | 0.96 (−0.02) | 0.52 (−0.02) | 0.72 (+0.03) | 0.38 (+0.01) | 0.72 (0.00) |

**Findings**

- **Smaller chunks hurt.** 250-token chunks double the index size and lose evidence coverage
  significantly: a table split into smaller pieces leaves less of it in any one chunk.
- **Page-bounded chunks don't help.** The Phase 1 failure (a cash-flow table split across
  two chunks) motivated E2, but forbidding cross-page windows doesn't fix within-page splits,
  and it creates many short end-of-page fragments (+27% chunks) that crowd the top 5.
- **Decision: keep 500-token windows.** With filters and reranking, chunking makes no
  measurable difference, and the original chunker produces the smallest index.

## E7: query rewriting (full corpus, retrieval-only)

The local LLM writes up to three keyword-style search queries, roughly one per line item the
question needs. Results for each query are retrieved under the original question's filters,
fused with RRF, and reranked against the original question.

| Config | Page hit@5 | Page recall@5 | MRR@5 | Page hit@10 | Page hit@20 |
|---|---|---|---|---|---|
| `stack_filter_rerank` | 0.54 | 0.51 | 0.37 | 0.66 | 0.72 |
| `e7_rewrite` | 0.60 (+0.06 [0.00, +0.14]) | 0.55 (+0.04) | 0.40 (+0.03) | 0.72 (+0.06) | **0.82 (+0.10\*)** |

- **Rewriting widens what's found:** page hit@20 improves significantly. The top-5 gain sits
  at the edge of noise (CI lower bound 0.00).
- **It didn't help the questions it targeted:** for the 11 `dev` questions whose evidence
  spans several pages, page recall@5 is unchanged (0.59 → 0.59). The gain came from
  single-page questions (0.49 → 0.54).
- **Decision: not adopted.** It adds an LLM call to every query for a top-5 gain that n = 50
  can't distinguish from zero. It's worth revisiting if the generator ever gets a larger
  context (top-10 or more), where the gain is clearer.

## Judged end-to-end runs (full corpus)

Generator and judge are both `qwen3:14b`, run locally. See the caveat in
[evaluation.md](evaluation.md#known-limitations).

| Config | Correct | Refused | Faithfulness (answered) | Cites a gold page (answered) |
|---|---|---|---|---|
| `e0_closed_book` | 22% | 16% | — | — |
| `e1_full` (dense) | 12% | 68% | 1.00 (n=16) | 0.38 |
| **`stack_filter_rerank`** | **40%** | 42% | 0.98 (n=29) | 0.52 |
| `e0_oracle` | 54% | 14% | 0.97 | 1.00 |

- **Accuracy goes from 12% to 40%:** Δ +0.28 [+0.16, +0.40]\* paired on the same 50
  questions. Refusals fall from 68% to 42% because the right page is now in the context.
- **About two-thirds of the retrieval gap is closed:** 28 of the 42 points between
  `e1_full` and the oracle.
- **Answers stay grounded:** 93% of answered questions are fully supported by the sources,
  so the extra answers are not bought with hallucination.

**Where the remaining errors are** (`rag analyze`):

| Stage | `e1_full` | `stack_filter_rerank` |
|---|---|---|
| wrong filing | 16 | **1** |
| wrong page (right filing) | 22 | 19 |
| evidence missing (gold page, partial evidence) | 0 | 2 |
| refused although evidence was retrieved | 3 | 6 |
| wrong although evidence was retrieved | 3 | 2 |
| correct | 6 | **20** |

The filter fixed "wrong filing" almost completely. **Finding the right page inside the right
filing is now the main failure** (19 of 30), followed by over-cautious refusals (6).

### E8: stepwise answer format

The model must list each figure with its source, then show the calculation, then answer.
This targets the 8 generation failures: refusals despite retrieved evidence, and arithmetic slips.

| Config (judged, full) | Correct | Refused | Fully faithful | Refused w/ evidence | Wrong w/ evidence |
|---|---|---|---|---|---|
| `stack_filter_rerank` | 40% | 42% | 93% | 6 | 2 |
| `e8_stepwise` | 40% (Δ 0.00 [−0.10, +0.08]) | **28% (Δ −0.14\*)** | 81% | 1 | **7** |

- **It turned refusals into wrong answers, not right ones.** Five questions moved from
  "refused although evidence was retrieved" to "answered wrongly although evidence was
  retrieved"; accuracy did not move, and faithfulness fell.
- **Decision: not adopted.** For financial questions a confident wrong figure is worse than
  "insufficient information", so the default prompt's more cautious behaviour wins at equal
  accuracy. Refusal rate alone would have been a misleading target here.

**Cost:** median generation latency 10.4 s → 15.9 s (longer context and answers), and
median retrieval latency 0.3 s → 6.7 s because of reranking (measured on a GPU shared with
the local LLM).
