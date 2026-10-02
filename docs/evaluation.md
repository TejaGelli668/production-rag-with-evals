# Evaluation methodology

How this project measures a RAG configuration, and how far each number can be trusted.

## Cases

| Split | Cases | Source of the gold label | Use |
|---|---|---|---|
| `dev` | 50 | FinanceBench, written by financial analysts | Tuning and experiments |
| `test` | 100 | FinanceBench | Held out; final numbers only |
| `ci_smoke` | 20 (subset of `dev`) | FinanceBench | CI regression gate |
| `unanswerable` | 30 | **Drafted by Claude from corpus coverage; needs human review** | Does the system decline when it should? |

The unanswerable cases come in four kinds: companies absent from the corpus (10), fiscal
periods after every filing (8), plausible-sounding items that don't exist (7), and details
filings never disclose (5). Each case records its provenance in `source`.

## Metrics

Each property is scored separately, never blended into one number.

**Retrieval** (programmatic, at the configured k; `--retrieval-only` reports k = 1, 3, 5, 10, 20)

| Metric | Meaning |
|---|---|
| `doc_hit@k` | Any top-k chunk comes from the right filing |
| `page_hit@k` | Any top-k chunk overlaps a gold evidence page |
| `page_recall@k` | Fraction of gold pages overlapped (23% of questions need >1 page) |
| `mrr@k` | Reciprocal rank of the first chunk overlapping a gold page |
| `evidence_coverage@k` | Fraction of the gold evidence's words present in the retrieved text |

`evidence_coverage` exists because page overlap over-credits retrieval. In Phase 1, a chunk
overlapping the 3M cash-flow page was retrieved, but the capex line sat in the neighbouring
chunk.

**Answers**

| Metric | How | Meaning |
|---|---|---|
| `correct` | LLM judge vs the gold answer; refusals score 0 | Headline accuracy. On `unanswerable`, 1 means the system declined |
| `refused` | Programmatic ("Insufficient information" prefix) | Kept separate so declining and being wrong aren't summed |
| `faithfulness` | LLM judge lists the answer's claims and checks each against the retrieved sources | Fraction supported; `fully_faithful` = all supported |
| `numeric_match` | Programmatic, for numeric gold answers only | Any number in the answer equals the gold value, up to scale and 1%. A lenient cross-check on the judge |
| `has_citation`, `cites_gold_page`, `invalid_citations` | Programmatic | Whether the answer cites, cites a gold page, or cites sources that don't exist |

Faithfulness and citations aren't scored for refusals or for the closed-book bound.

## Bounds (E0)

- **Closed-book** (`configs/e0_closed_book.yaml`): no retrieval. Shows how much the model
  answers from memory alone.
- **Oracle** (`configs/e0_oracle.yaml`): the gold evidence pages are given to the model directly.
  This is a ceiling for generation. The gap between a real configuration and the oracle is
  what retrieval is costing.

## Harness guarantees

- **Infrastructure failures are never scored.** Connection errors, judge parse failures and
  model mismatches go to `errors.jsonl`, and a re-run retries them.
- **Truncated answers are counted but left out of means.**
- **The answering model is checked.** If the response came from a different model than
  requested, the case fails as an error.
- **Every case is saved in full:** prompts, retrieved chunks, answer, citations and each
  judge's structured output, in `evals/runs/<run>/traces/`.
- **Means come with uncertainty.** Means are taken per case (averaging reps), with
  case-level bootstrap 95% CIs. `rag compare` reports paired differences on shared cases.
- **Resumable runs.** An interrupted run resumes; a run whose config or judge changed is
  refused unless `--fresh` is given.

## Judge validation

- **Sanity checks** (`rag check-judge`): gold answers must be graded correct; an empty
  answer, "I don't know" and another question's gold answer must be graded incorrect. The
  faithfulness judge must accept a gold answer against its own evidence pages and reject it
  against another question's.
- **Human calibration** (`rag label <run>`, then `rag calibrate <run>`): you label answers
  without seeing the judge's grade, and the tool reports agreement and Cohen's kappa. Below
  about 90% agreement on clear-cut cases, the judge prompt needs another iteration before its
  scores can steer decisions.

### Results for the local judge (`qwen3:14b`, 50 `dev` cases)

| Check | Pass rate |
|---|---|
| gold answer graded correct | 100% |
| empty answer graded incorrect | 98% |
| "I don't know" graded incorrect | 100% |
| another question's gold answer graded incorrect | 100% |
| faithfulness: gold answer vs its own evidence counts as supported | 84% |
| faithfulness: gold answer vs another question's evidence counts as unsupported | 84% |

The **correctness judge passes its sanity checks** (≥ 98% on every check), so the accuracy
numbers rest on solid ground, pending human calibration. The **faithfulness judge is
weaker**: it misjudges about 1 in 6 clear-cut cases in each direction, and it missed an
arithmetic error in a spot check. Treat faithfulness scores as approximate. Report
`evals/judge/checks_qwen3-14b_dev.json` lists every failure.

## Known limitations

- **The judge is the generator** (`qwen3:14b`), so it may favour its own phrasing. The project
  deliberately runs fully locally, so there is no independent cloud judge; the sanity checks
  above and human calibration (`rag label` / `rag calibrate`) are the safeguards. The judge
  can be swapped with `--judge-provider` / `--judge-model`, but judge-graded numbers from
  different judges are not comparable.
- **Noise:** with 50 `dev` cases, a pass-rate CI is roughly ±14 points. Only differences
  whose paired CI excludes zero should drive decisions, and smaller effects need the `test`
  split or more cases.
- **Memorization:** FinanceBench uses real, well-known companies, so a model can answer some
  questions from memory. The closed-book bound measures how much.
