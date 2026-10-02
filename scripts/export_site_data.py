"""Export held-out results for the project site (site/data/explorer.json).

Reads the local eval runs (evals/runs/, gitignored) for the final config and the baseline on
the `test` split, plus the unanswerable set, and writes one JSON file the static site loads.
Regenerate after re-running those evals:

    uv run python scripts/export_site_data.py
"""

from __future__ import annotations

import json

from rag.data.financebench import PROJECT_ROOT
from rag.evals.analysis import classify

RUNS = PROJECT_ROOT / "evals" / "runs"
OUT = PROJECT_ROOT / "site" / "data" / "explorer.json"
FINAL, BASELINE, UNANSWERABLE = (
    "stack_filter_rerank__test",
    "e1_full__test",
    "stack_filter_rerank__unanswerable",
)
SNIPPET = 500


def _load(run: str) -> dict[str, tuple[dict, dict]]:
    """case id -> (result row, trace)."""
    rows = [
        json.loads(line)
        for line in (RUNS / run / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return {
        r["prompt_id"]: (
            r,
            json.loads((RUNS / run / "traces" / f"{r['prompt_id']}_rep0.json").read_text()),
        )
        for r in rows
    }


def _outcome(grade: dict) -> str:
    if grade.get("refused") == 1.0:
        return "declined"
    return "correct" if grade.get("correct") == 1.0 else "wrong"


def _sources(trace: dict, gold_pages: set[tuple[str, int]]) -> list[dict]:
    cited = {c["marker"] for c in trace["citations"]}
    out = []
    for rc in trace["retrieved"]:
        c = rc["chunk"]
        pages = range(c["page_start"], c["page_end"] + 1)
        out.append(
            {
                "rank": rc["rank"],
                "doc": c["doc_name"],
                "pages": f"p. {c['page_start'] + 1}"
                if c["page_start"] == c["page_end"]
                else f"pp. {c['page_start'] + 1}-{c['page_end'] + 1}",
                "score": round(rc["score"], 3),
                "cited": rc["rank"] in cited,
                "gold": any((c["doc_name"], p) in gold_pages for p in pages),
                "text": c["text"][:SNIPPET] + ("…" if len(c["text"]) > SNIPPET else ""),
            }
        )
    return out


def _answer(row: dict, trace: dict, gold_pages: set[tuple[str, int]]) -> dict:
    return {
        "text": row["answer"],
        "outcome": _outcome(row["grade"]),
        "stage": classify(row["grade"]),
        "judge": row.get("explanation", {}).get("correct"),
        "sources": _sources(trace, gold_pages),
    }


def main() -> None:
    final, baseline = _load(FINAL), _load(BASELINE)
    questions = []
    for case_id, (row, trace) in final.items():
        case = trace["case"]
        gold_pages = {(e["doc_name"], e["evidence_page_num"]) for e in case["evidence"]}
        b_row, b_trace = baseline[case_id]
        questions.append(
            {
                "id": case_id,
                "question": case["question"],
                "type": case["tags"][0],
                "gold": case["gold_answer"],
                "filing": case["doc_name"],
                "evidence": [
                    {
                        "doc": e["doc_name"],
                        "page": e["evidence_page_num"] + 1,
                        "text": e["evidence_text"][:SNIPPET]
                        + ("…" if len(e["evidence_text"]) > SNIPPET else ""),
                    }
                    for e in case["evidence"]
                ],
                "final": _answer(row, trace, gold_pages),
                "baseline": _answer(b_row, b_trace, gold_pages),
            }
        )
    unanswerable = [
        {
            "id": case_id,
            "question": trace["case"]["question"],
            "category": trace["case"]["tags"][0].replace("_", " "),
            "answer": row["answer"],
            "declined": row["grade"].get("refused") == 1.0,
        }
        for case_id, (row, trace) in _load(UNANSWERABLE).items()
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "FinanceBench (Patronus AI, CC-BY-NC 4.0), test split; generator and "
        "judge qwen3:14b via Ollama",
        "questions": questions,
        "unanswerable": unanswerable,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    n = len(questions)
    counts = {
        k: sum(q[k]["outcome"] == "correct" for q in questions) for k in ("baseline", "final")
    }
    print(
        f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {n} questions, "
        f"correct baseline {counts['baseline']} final {counts['final']}, "
        f"{len(unanswerable)} unanswerable"
    )


if __name__ == "__main__":
    main()
