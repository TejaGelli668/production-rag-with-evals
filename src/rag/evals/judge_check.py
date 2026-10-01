"""Sanity checks for the judges, run before trusting any judged score.

Known-good inputs must pass and known-bad inputs must fail:
  correctness  gold answer -> correct; "", "I don't know", another question's
               gold answer -> incorrect
  faithfulness gold answer vs its own evidence pages -> supported;
               vs another question's evidence pages -> unsupported
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from rag.data.financebench import PROJECT_ROOT, load_documents
from rag.evals.cases import EvalCase, load_cases
from rag.evals.judges import Judges, faithfulness_score
from rag.generate.prompts import format_source
from rag.retrieve import page_chunk
from rag.schema import RetrievedChunk

CHECKS_DIR = PROJECT_ROOT / "evals" / "judge"


def _evidence_sources(case: EvalCase, docs) -> str:
    pages = {(e.doc_name, e.evidence_page_num): e.evidence_text_full_page for e in case.evidence}
    return "\n\n".join(
        format_source(RetrievedChunk(chunk=page_chunk(docs[d], p, t), score=1.0, rank=i))
        for i, ((d, p), t) in enumerate(sorted(pages.items()), start=1)
    )


def check_judges(judges: Judges, split: str = "dev", limit: int | None = None) -> dict[str, Any]:
    cases = [c for c in load_cases(split) if c.answerable][:limit]
    docs = {d.doc_name: d for d in load_documents()}
    outcomes: dict[str, list[bool]] = defaultdict(list)
    failures: list[dict[str, str]] = []

    def record(check: str, case: EvalCase, passed: bool, detail: str) -> None:
        outcomes[check].append(passed)
        if not passed:
            failures.append({"check": check, "case": case.case_id, "detail": detail[:300]})

    for i, case in enumerate(tqdm(cases, desc="judge checks", unit="case")):
        other = cases[(i + 1) % len(cases)]
        correctness_inputs = {
            "gold_answer_is_correct": (case.gold_answer, "correct"),
            "empty_is_incorrect": ("", "incorrect"),
            "idk_is_incorrect": ("I don't know.", "incorrect"),
            "other_answer_is_incorrect": (other.gold_answer, "incorrect"),
        }
        for check, (response, expected) in correctness_inputs.items():
            verdict, _ = judges.correctness(
                case.question, response, case.gold_answer, case.justification
            )
            record(check, case, verdict.verdict == expected, verdict.reasoning)

        own = faithfulness_score(
            judges.faithfulness(case.question, case.gold_answer, _evidence_sources(case, docs))[0]
        )
        record("faithful_vs_own_evidence", case, own is not None and own >= 0.5, f"score={own}")
        swapped = faithfulness_score(
            judges.faithfulness(case.question, case.gold_answer, _evidence_sources(other, docs))[0]
        )
        record(
            "unfaithful_vs_other_evidence",
            case,
            swapped is None or swapped < 0.5,
            f"score={swapped}",
        )

    report = {
        "split": split,
        "cases": len(cases),
        "pass_rates": {k: sum(v) / len(v) for k, v in outcomes.items()},
        "failures": failures,
    }
    return report


def write_report(report: dict[str, Any], judge_model: str) -> str:
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    name = judge_model.replace(":", "-").replace("/", "-")
    (CHECKS_DIR / f"checks_{name}_{report['split']}.json").write_text(json.dumps(report, indent=2))
    lines = [
        f"Judge checks · `{judge_model}` · {report['cases']} {report['split']} cases",
        "",
        "| Check | Pass rate |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.0%} |" for k, v in report["pass_rates"].items()]
    return "\n".join(lines)
