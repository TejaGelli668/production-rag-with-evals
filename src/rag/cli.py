"""Command-line interface.

rag ingest [--config configs/baseline.yaml] [--limit N]
rag ask "What was 3M's FY2018 capex?" [--filter company=3M]
rag ask --id financebench_id_03029          # a benchmark question, shown with its gold answer
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from rag.config import DEFAULT_CONFIG, Settings, load_config
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
