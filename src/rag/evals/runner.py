"""Eval runner.

Layout of a run (`evals/runs/<config>__<split>/`, gitignored):
    meta.json       config, judge, split; a resumed run must match it
    results.jsonl   one row per (case, rep) that produced a gradable answer
    errors.jsonl    attempts that never produced one (infra/judge failures), kept out of scores
    traces/         full per-case record: prompts, retrieved chunks, answer, judge calls
    summary.json    aggregates with bootstrap 95% CIs (also copied to evals/results/)

Infra failures never land in results.jsonl, so they can't be scored as wrong answers,
and a re-run retries them. Truncated answers are kept but flagged and left out of means.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from rag.config import JudgeConfig, PipelineConfig, Settings
from rag.data.financebench import PROJECT_ROOT
from rag.evals.cases import EvalCase, load_cases
from rag.evals.judges import JudgeError, Judges, faithfulness_score
from rag.evals.metrics import (
    citation_metrics,
    gold_number,
    is_refusal,
    numeric_match,
    retrieval_metrics,
)
from rag.evals.summary import summarize, write_summary
from rag.generate.llm import LLMError, make_llm
from rag.generate.prompts import format_source
from rag.pipeline import RAGPipeline, make_retriever
from rag.retrieve import DenseRetriever
from rag.schema import Answer, LLMResponse

RUNS_DIR = PROJECT_ROOT / "evals" / "runs"
RETRIEVAL_KS = (1, 3, 5, 10, 20)


class RunMismatchError(RuntimeError):
    pass


def _usage(r: LLMResponse) -> dict[str, int]:
    return {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}


def _prepare_run_dir(run_dir: Path, meta: dict[str, Any], fresh: bool) -> set[tuple[str, int]]:
    """Create or validate the run directory; return the (case_id, rep) pairs already done."""
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing != meta:
            raise RunMismatchError(
                f"{run_dir} was produced by a different config or judge; "
                "rerun with --fresh to replace it"
            )
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    results = run_dir / "results.jsonl"
    if not results.exists():
        return set()
    rows = (json.loads(line) for line in results.read_text().splitlines() if line.strip())
    return {(r["prompt_id"], r["rep"]) for r in rows}


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _grade(
    case: EvalCase,
    answer: Answer,
    judges: Judges | None,
    closed_book: bool,
    retrieval_k: int | None,
) -> tuple[dict[str, float], dict[str, str], list[dict[str, Any]], list[LLMResponse]]:
    """Score one answer. Returns (grade, explanation, judge trace, judge responses)."""
    grade: dict[str, float] = {}
    explanation: dict[str, str] = {}
    judge_trace: list[dict[str, Any]] = []
    judge_calls: list[LLMResponse] = []
    refused = is_refusal(answer.text)
    grade["refused"] = float(refused)

    if retrieval_k and case.answerable:  # real retrieval only; oracle/closed-book are bounds
        grade |= retrieval_metrics(answer.retrieved, case, k=retrieval_k)

    if not case.answerable:
        grade["correct"] = float(refused)  # the right behaviour is to decline
        return grade, explanation, judge_trace, judge_calls

    if refused:
        grade["correct"] = 0.0
    elif judges:
        verdict, raw = judges.correctness(
            case.question, answer.text, case.gold_answer, case.justification
        )
        grade["correct"] = float(verdict.verdict == "correct")
        explanation["correct"] = verdict.reasoning
        judge_trace.append({"judge": "correctness", "output": verdict.model_dump()})
        judge_calls.append(raw)

    if (gold := gold_number(case.gold_answer)) is not None and not refused:
        grade["numeric_match"] = float(numeric_match(answer.text, gold))

    if not closed_book and not refused:
        grade |= citation_metrics(answer.text, answer.retrieved, case)
        if judges and answer.retrieved:
            sources = "\n\n".join(format_source(rc) for rc in answer.retrieved)
            verdict, raw = judges.faithfulness(case.question, answer.text, sources)
            score = faithfulness_score(verdict)
            if score is not None:
                grade["faithfulness"] = score
                grade["fully_faithful"] = float(score == 1.0)
            unsupported = [c.claim for c in verdict.claims if not c.supported]
            if unsupported:
                explanation["faithfulness"] = "Unsupported: " + " | ".join(unsupported)
            judge_trace.append({"judge": "faithfulness", "output": verdict.model_dump()})
            judge_calls.append(raw)

    return grade, explanation, judge_trace, judge_calls


def run_eval(
    cfg: PipelineConfig,
    split: str,
    judge_cfg: JudgeConfig,
    settings: Settings | None = None,
    reps: int = 1,
    limit: int | None = None,
    fresh: bool = False,
    use_judges: bool = True,
) -> Path:
    settings = settings or Settings()
    cases = load_cases(split)[:limit]
    run_dir = RUNS_DIR / f"{cfg.name}__{split}"
    meta = {
        "config": cfg.model_dump(mode="json"),
        "collection": cfg.collection_name if cfg.retriever.type == "dense" else None,
        "split": split,
        "judge": judge_cfg.model_dump(mode="json") if use_judges else None,
        "judge_is_generator": use_judges and judge_cfg.model == cfg.generator.model,
    }
    done = _prepare_run_dir(run_dir, meta, fresh)

    pipeline = RAGPipeline.from_config(cfg, settings)
    judges = Judges(make_llm(judge_cfg, settings)) if use_judges else None
    todo = [(c, rep) for c in cases for rep in range(reps) if (c.case_id, rep) not in done]

    for case, rep in tqdm(todo, desc=f"eval {cfg.name} on {split}", unit="case"):
        try:
            answer = pipeline.ask(case.question)
            if answer.llm.model != cfg.generator.model:
                raise LLMError(
                    f"served model {answer.llm.model!r} != requested {cfg.generator.model!r}"
                )
            grade, explanation, judge_trace, judge_calls = _grade(
                case,
                answer,
                judges,
                pipeline.closed_book,
                retrieval_k=cfg.retriever.top_k if cfg.retriever.type == "dense" else None,
            )
        except (LLMError, JudgeError, httpx.HTTPError) as e:
            _append(
                run_dir / "errors.jsonl",
                {
                    "prompt_id": case.case_id,
                    "rep": rep,
                    "failure_class": type(e).__name__,
                    "message": str(e)[:500],
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            continue

        truncated = answer.llm.stop_reason == "max_tokens"
        _append(
            run_dir / "results.jsonl",
            {
                "prompt_id": case.case_id,
                "rep": rep,
                "prompt": case.question,
                "tags": case.tags + ([] if case.answerable else ["unanswerable"]),
                "status": "truncated" if truncated else "ok",
                "stop_reason": answer.llm.stop_reason,
                "grade": grade,
                "explanation": explanation,
                "answer": answer.text,
                "gold": case.gold_answer,
                "model": answer.llm.model,
                "usage": _usage(answer.llm),
                "latency_s": round(answer.llm.latency_s, 3),
                "retrieval_latency_s": round(answer.retrieval_latency_s, 3),
                "judge_model": judge_calls[0].model if judge_calls else None,
                "judge_usage": {
                    "input_tokens": sum(r.input_tokens for r in judge_calls),
                    "output_tokens": sum(r.output_tokens for r in judge_calls),
                },
            },
        )
        trace = {
            "case": case.model_dump(),
            "system": pipeline.system_prompt,
            "user": answer.user_prompt,
            "answer": answer.text,
            "citations": [c.model_dump() for c in answer.citations],
            "retrieved": [rc.model_dump() for rc in answer.retrieved],
            "judges": judge_trace,
        }
        (run_dir / "traces" / f"{case.case_id}_rep{rep}.json").write_text(
            json.dumps(trace, indent=1)
        )

    write_summary(run_dir, summarize(run_dir))
    return run_dir


def run_retrieval_eval(
    cfg: PipelineConfig,
    split: str,
    settings: Settings | None = None,
    ks: tuple[int, ...] = RETRIEVAL_KS,
    limit: int | None = None,
) -> Path:
    """Retrieval metrics at several k with no LLM calls: fast and free, for Phase 3 sweeps."""
    settings = settings or Settings()
    retriever = make_retriever(cfg, settings)
    if not isinstance(retriever, DenseRetriever):
        raise ValueError("retrieval-only evaluation needs a dense retriever")
    cases = [c for c in load_cases(split)[:limit] if c.answerable]
    run_dir = RUNS_DIR / f"{cfg.name}__{split}__retrieval"
    meta = {
        "config": cfg.model_dump(mode="json"),
        "collection": cfg.collection_name,
        "split": split,
        "ks": list(ks),
    }
    _prepare_run_dir(run_dir, meta, fresh=True)

    for case in tqdm(cases, desc=f"retrieval {cfg.name} on {split}", unit="case"):
        started = time.perf_counter()
        retrieved = retriever.retrieve(case.question, top_k=max(ks))
        latency = time.perf_counter() - started
        grade: dict[str, float] = {}
        for k in ks:
            grade |= retrieval_metrics(retrieved, case, k)
        _append(
            run_dir / "results.jsonl",
            {
                "prompt_id": case.case_id,
                "rep": 0,
                "prompt": case.question,
                "tags": case.tags,
                "status": "ok",
                "grade": grade,
                "retrieval_latency_s": round(latency, 3),
            },
        )
    write_summary(run_dir, summarize(run_dir))
    return run_dir
