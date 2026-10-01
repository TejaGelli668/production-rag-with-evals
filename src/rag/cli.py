"""Command-line interface.

rag ingest [--config configs/baseline.yaml] [--limit N]
rag ask "What was 3M's FY2018 capex?" [--filter company=3M]
rag ask --id financebench_id_03029          # a benchmark question, shown with its gold answer
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from rag.config import DEFAULT_CONFIG, JudgeConfig, Settings, load_config
from rag.data.financebench import Question, load_questions
from rag.generate.llm import LLMError
from rag.pipeline import IndexNotFoundError, RAGPipeline
from rag.store import Filters


def _parse_filters(pairs: list[str]) -> Filters:
    filters: Filters = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--filter expects key=value, got {pair!r}")
        filters[key] = int(value) if value.isdigit() else value
    return filters


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag.ingest.pipeline import ingest

    cfg = load_config(args.config)
    stats = ingest(cfg, Settings(), limit=args.limit)
    print(
        f"{cfg.collection_name}: {stats.docs_indexed} docs indexed, {stats.docs_skipped} already "
        f"done, {stats.chunks_added:,} chunks added in {stats.seconds:.0f}s"
    )
    return 0


def _print_wrapped(text: str, indent: str = "  ") -> None:
    for para in text.splitlines():
        print(textwrap.fill(para, 100, initial_indent=indent, subsequent_indent=indent) or "")


def cmd_ask(args: argparse.Namespace) -> int:
    gold: Question | None = None
    if args.id:
        gold = next((q for q in load_questions() if q.financebench_id == args.id), None)
        if gold is None:
            raise SystemExit(f"unknown question id {args.id!r}")
    question = gold.question if gold else args.question
    if not question:
        raise SystemExit("give a question, or --id for a FinanceBench question")

    cfg = load_config(args.config)
    try:
        pipeline = RAGPipeline.from_config(cfg)
        answer = pipeline.ask(question, _parse_filters(args.filter))
    except (IndexNotFoundError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("\nQuestion:")
    _print_wrapped(question)
    print("\nAnswer:")
    _print_wrapped(answer.text)

    gold_pages = {(e.doc_name, e.evidence_page_num) for e in gold.evidence} if gold else set()
    print("\nSources:")
    cited = {c.marker for c in answer.citations}
    for rc in answer.retrieved:
        c = rc.chunk
        hit = any((c.doc_name, p) in gold_pages for p in c.pages)
        flags = ("cited " if rc.rank in cited else "      ") + ("✓ gold" if hit else "")
        print(f"  [{rc.rank}] {c.doc_name:28s} {c.page_label():12s} score {rc.score:.3f}  {flags}")
        if args.show_context:
            _print_wrapped(c.text[:600] + ("…" if len(c.text) > 600 else ""), indent="       ")

    if gold:
        print("\nGold answer:")
        _print_wrapped(gold.answer)
        pages = ", ".join(f"{e.doc_name} p. {e.evidence_page_num + 1}" for e in gold.evidence)
        print(f"  evidence: {pages}")

    r = answer.llm
    print(
        f"\n[{cfg.name} · {r.model} · retrieval {answer.retrieval_latency_s:.2f}s · "
        f"generation {r.latency_s:.1f}s · {r.input_tokens:,} in / {r.output_tokens:,} out tokens]"
    )
    return 0


def _judge_config(args: argparse.Namespace) -> JudgeConfig:
    cfg = JudgeConfig()
    if args.judge_provider:
        cfg.provider = args.judge_provider
    if args.judge_model:
        cfg.model = args.judge_model
    return cfg


def cmd_eval(args: argparse.Namespace) -> int:
    from rag.evals.runner import RunMismatchError, run_eval, run_retrieval_eval
    from rag.evals.summary import render_summary

    cfg = load_config(args.config)
    try:
        if args.retrieval_only:
            run_dir = run_retrieval_eval(cfg, args.split, limit=args.limit)
        else:
            run_dir = run_eval(
                cfg,
                args.split,
                _judge_config(args),
                reps=args.reps,
                limit=args.limit,
                fresh=args.fresh,
                use_judges=not args.no_judge,
            )
    except (IndexNotFoundError, RunMismatchError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(render_summary(json.loads((run_dir / "summary.json").read_text())))
    print(f"\nrun: {run_dir}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from rag.evals.summary import compare

    print(compare([_run_dir(d) for d in args.runs], args.metric or None))
    return 0


def cmd_check_judge(args: argparse.Namespace) -> int:
    from rag.evals.judge_check import check_judges, write_report
    from rag.evals.judges import Judges
    from rag.generate.llm import make_llm

    judge_cfg = _judge_config(args)
    report = check_judges(Judges(make_llm(judge_cfg, Settings())), args.split, args.limit)
    print(write_report(report, judge_cfg.model))
    for f in report["failures"][:10]:
        print(f"  ✗ {f['check']} {f['case']}: {f['detail']}")
    return 0


def _run_dir(path: Path) -> Path:
    from rag.evals.runner import RUNS_DIR

    run_dir = path if path.exists() else RUNS_DIR / path.name
    if not (run_dir / "results.jsonl").exists():
        raise SystemExit(f"error: no results in {run_dir}")
    return run_dir


def cmd_label(args: argparse.Namespace) -> int:
    from rag.evals.labeling import LABELS_PATH, label_run

    added = label_run(_run_dir(args.run))
    print(f"\n{added} labels added to {LABELS_PATH}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from rag.evals.labeling import calibrate

    report = calibrate(_run_dir(args.run))
    print(
        f"{report['run']}: {report['labeled']} labeled answers · agreement "
        f"{report['agreement']:.0%} · Cohen's kappa {report['cohens_kappa']:.2f}"
    )
    for d in report["disagreements"]:
        print(f"  ✗ {d['prompt_id']}: human={d['human']} judge={d['judge']}")
        _print_wrapped(d["judge_reasoning"], indent="      ")
    return 0


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-provider", choices=["ollama", "anthropic"])
    p.add_argument("--judge-model", help="e.g. claude-haiku-4-5 (default: local qwen3:14b)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over SEC filings")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="parse, chunk, embed and index the corpus")
    p_ingest.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_ingest.add_argument("--limit", type=int, help="only the first N documents (smoke test)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="answer a question with cited sources")
    p_ask.add_argument("question", nargs="?")
    p_ask.add_argument("--id", help="a FinanceBench question id; shows the gold answer too")
    p_ask.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_ask.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metadata filter, e.g. company=3M or fiscal_year=2018 (repeatable)",
    )
    p_ask.add_argument("--show-context", action="store_true", help="print retrieved chunk text")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run a config on an eval split and score it")
    p_eval.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_eval.add_argument(
        "--split", default="dev", choices=["dev", "test", "ci_smoke", "unanswerable"]
    )
    p_eval.add_argument("--reps", type=int, default=1)
    p_eval.add_argument("--limit", type=int, help="only the first N cases")
    p_eval.add_argument(
        "--fresh", action="store_true", help="discard previous results for this run"
    )
    p_eval.add_argument(
        "--retrieval-only", action="store_true", help="retrieval metrics at k=1..20, no LLM"
    )
    p_eval.add_argument("--no-judge", action="store_true", help="programmatic metrics only")
    _add_judge_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    p_cmp = sub.add_parser("compare", help="compare eval runs (paired on shared cases)")
    p_cmp.add_argument("runs", nargs="+", type=Path, help="run dirs or names under evals/runs/")
    p_cmp.add_argument("--metric", action="append", help="limit to these metrics (repeatable)")
    p_cmp.set_defaults(func=cmd_compare)

    p_chk = sub.add_parser("check-judge", help="known-good/known-bad sanity checks for judges")
    p_chk.add_argument("--split", default="dev")
    p_chk.add_argument("--limit", type=int)
    _add_judge_args(p_chk)
    p_chk.set_defaults(func=cmd_check_judge)

    p_lab = sub.add_parser("label", help="hand-label a run's answers for judge calibration")
    p_lab.add_argument("run", type=Path, help="run dir or name under evals/runs/")
    p_lab.set_defaults(func=cmd_label)

    p_cal = sub.add_parser("calibrate", help="judge vs human agreement on labeled answers")
    p_cal.add_argument("run", type=Path, help="run dir or name under evals/runs/")
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
