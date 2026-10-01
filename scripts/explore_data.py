"""Profile the FinanceBench corpus and questions; write docs/data_exploration.md.

Answers the questions that shape later phases:
  * How big is the corpus (pages, tokens), and how much of it has no text layer?
  * Is `evidence_page_num` 0- or 1-based relative to PyMuPDF page indices?
  * What do questions and answers look like (numeric answers, multi-page evidence)?
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import pymupdf
from tqdm import tqdm

from rag.data.financebench import (
    PROJECT_ROOT,
    CorpusMode,
    Document,
    load_documents,
    load_questions,
    select_documents,
)
from rag.data.splits import load_split

REPORT_PATH = PROJECT_ROOT / "docs" / "data_exploration.md"
LOW_TEXT_CHARS = 200  # a page with less extractable text than this is likely an image/scan
OFFSETS = (-1, 0, 1)  # candidate PyMuPDF index = evidence_page_num + offset
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class DocStats:
    doc_name: str
    pages: list[str]
    size_mb: float


def extract(doc: Document) -> DocStats:
    with pymupdf.open(doc.pdf_path) as pdf:
        pages = [page.get_text() for page in pdf]
    return DocStats(doc.doc_name, pages, doc.pdf_path.stat().st_size / 1e6)


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def quantiles(xs: list[float]) -> str:
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=20)
    return (
        f"min {xs[0]:,.0f} · p50 {statistics.median(xs):,.0f} · "
        f"p95 {q[18]:,.0f} · max {xs[-1]:,.0f}"
    )


def is_numeric_answer(answer: str) -> bool:
    return bool(
        re.fullmatch(r"[\s$€(\-]*[\d,]+(\.\d+)?\s*%?\)?\s*(million|billion|x)?\.?\s*", answer, re.I)
    )


def main() -> None:
    questions = load_questions()
    all_docs = load_documents()
    focused = {d.doc_name for d in select_documents(CorpusMode.FOCUSED, all_docs, questions)}
    docs = [d for d in all_docs if d.pdf_path.exists()]
    missing = len(all_docs) - len(docs)

    with ProcessPoolExecutor() as pool:
        stats = {
            s.doc_name: s
            for s in tqdm(pool.map(extract, docs, chunksize=4), total=len(docs), desc="extract")
        }

    # --- corpus -----------------------------------------------------------------------------
    def corpus_row(label: str, names: set[str]) -> str:
        ss = [stats[n] for n in names if n in stats]
        pages = [p for s in ss for p in s.pages]
        chars = sum(len(p) for p in pages)
        low = sum(len(p.strip()) < LOW_TEXT_CHARS for p in pages)
        return (
            f"| {label} | {len(ss)} | {sum(s.size_mb for s in ss):,.0f} | {len(pages):,} | "
            f"{chars / 4 / 1e6:,.1f}M | {low:,} ({pct(low, len(pages))}) |"
        )

    page_counts = [len(s.pages) for s in stats.values()]
    no_text_docs = sorted(
        n
        for n, s in stats.items()
        if sum(len(p.strip()) for p in s.pages) < LOW_TEXT_CHARS * len(s.pages) * 0.2
    )

    # --- page index base ----------------------------------------------------------------------
    offset_wins: Counter[int] = Counter()
    best_scores: list[float] = []
    evidence_pages_low_text = 0
    weak_matches: list[str] = []
    n_evidence = 0
    for q in questions:
        for ev in q.evidence:
            pages = stats[ev.doc_name].pages
            target = words(ev.evidence_text_full_page)
            scores = {
                off: jaccard(target, words(pages[i]))
                for off in OFFSETS
                if 0 <= (i := ev.evidence_page_num + off) < len(pages)
            }
            n_evidence += 1
            best = max(scores, key=scores.get)
            offset_wins[best] += 1
            best_scores.append(scores[best])
            if scores[best] < 0.5:
                weak_matches.append(
                    f"{q.financebench_id} ({ev.doc_name} p{ev.evidence_page_num}): "
                    f"{scores[best]:.2f}"
                )
            if (
                0 <= ev.evidence_page_num < len(pages)
                and len(pages[ev.evidence_page_num].strip()) < LOW_TEXT_CHARS
            ):
                evidence_pages_low_text += 1
    winning_offset, wins = offset_wins.most_common(1)[0]
    base = {0: "0-based", 1: "0-based, shifted by one", -1: "1-based"}[winning_offset]

    # --- questions ----------------------------------------------------------------------------
    n_q = len(questions)
    numeric = sum(is_numeric_answer(q.answer) for q in questions)
    multi_page = sum(len({e.evidence_page_num for e in q.evidence}) > 1 for q in questions)
    per_company = Counter(q.company for q in questions)
    reasoning = Counter((q.question_reasoning or "(unlabeled)") for q in questions)
    doc_types = Counter(d.doc_type for d in all_docs)
    years = [d.doc_period for d in all_docs]
    q_words = [len(q.question.split()) for q in questions]
    a_words = [len(q.answer.split()) for q in questions]
    split_sizes = {n: len(load_split(n)) for n in ("dev", "test", "ci_smoke")}

    lines = [
        "# FinanceBench data exploration",
        "",
        '_Generated by `scripts/explore_data.py`. Only the "Implications" section is '
        "written by hand; re-runs keep it._",
        "",
        "## Corpus",
        "",
        "| Corpus | Docs | Size (MB) | Pages | ~Tokens (chars/4) | Low-text pages (<200 chars) |",
        "|---|---|---|---|---|---|",
        corpus_row("`focused`", focused),
        corpus_row("`full`", set(stats)),
        "",
        f"- Pages per document: {quantiles(page_counts)}",
        f"- Document types: {', '.join(f'{k} {v}' for k, v in doc_types.most_common())}",
        f"- Fiscal periods: {min(years)}-{max(years)}",
        f"- Documents with almost no text layer (likely scanned, need OCR): {len(no_text_docs)}"
        + (f": {', '.join(no_text_docs)}" if no_text_docs else ""),
        f"- Documents in metadata but missing locally: {missing}",
        "",
        "## Evidence page numbering",
        "",
        f"For each of the {n_evidence} evidence items, `evidence_text_full_page` was compared "
        "(word-set Jaccard) with PyMuPDF's text for page indices "
        "`evidence_page_num - 1`, `+0`, and `+1`.",
        "",
        "| Offset that matched best | Evidence items |",
        "|---|---|",
        *[
            f"| {off:+d} | {offset_wins.get(off, 0)} ({pct(offset_wins.get(off, 0), n_evidence)}) |"
            for off in OFFSETS
        ],
        "",
        f"**Conclusion: `evidence_page_num` is {base}** relative to PyMuPDF page indices "
        f"({pct(wins, n_evidence)} of items). Median best-match similarity: "
        f"{statistics.median(best_scores):.2f}.",
        "",
        f"- Evidence pages with almost no extractable text: {evidence_pages_low_text}",
        f"- Weak matches (best similarity < 0.5): {len(weak_matches)}",
        *[f"  - {w}" for w in weak_matches[:15]],
        "",
        "## Questions",
        "",
        f"- {n_q} questions over {len(focused)} documents and {len(per_company)} companies; "
        f"most-asked: {', '.join(f'{c} ({n})' for c, n in per_company.most_common(5))}",
        f"- Every question's evidence comes from its own `doc_name`: "
        f"{all(e.doc_name == q.doc_name for q in questions for e in q.evidence)}",
        f"- Evidence spans more than one page: {multi_page} ({pct(multi_page, n_q)})",
        f"- Purely numeric answers (e.g. `$1577.00`, `0.96`, `12.5%`): "
        f"{numeric} ({pct(numeric, n_q)}); "
        "the rest are sentences, often a number plus explanation",
        f"- Question length (words): {quantiles(q_words)}",
        f"- Answer length (words): {quantiles(a_words)}",
        f"- Eval splits: {', '.join(f'{k} {v}' for k, v in split_sizes.items())}",
        "",
        "| Reasoning label | Questions |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in reasoning.most_common()],
        "",
        "## Implications for the pipeline",
        "",
        "_Filled in by hand below, from the numbers above._",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_notes = ""
    if REPORT_PATH.exists():
        text = REPORT_PATH.read_text()
        marker = "_Filled in by hand below, from the numbers above._\n"
        if marker in text:
            existing_notes = text.split(marker, 1)[1].lstrip("\n")
    REPORT_PATH.write_text("\n".join(lines) + ("\n" + existing_notes if existing_notes else ""))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
