"""Aggregate eval runs and compare them.

Means are taken per case first (averaging reps), then across cases, with a
case-level bootstrap 95% CI. Comparisons are paired on the cases both runs share.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag.data.financebench import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
BOOTSTRAP_SAMPLES = 2000
SEED = 7

# Display order for the headline table; any other metric follows alphabetically.
HEADLINE = [
    "correct",
    "refused",
    "faithfulness",
    "fully_faithful",
    "numeric_match",
    "cites_gold_page",
    "has_citation",
    "doc_hit@5",
    "page_hit@5",
    "evidence_coverage@5",
    "mrr@5",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(values: list[float], seed: int = SEED) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0], values[0]) if values else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def per_case_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """metric -> case_id -> mean over that case's reps (status-ok rows only)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["status"] != "ok":
            continue
        for metric, value in row["grade"].items():
            acc[metric][row["prompt_id"]].append(value)
    return {m: {c: statistics.fmean(v) for c, v in cases.items()} for m, cases in acc.items()}


def _stat(values: list[float]) -> dict[str, Any]:
    lo, hi = bootstrap_ci(values)
    return {"mean": statistics.fmean(values), "ci95": [lo, hi], "n": len(values)}


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "meta.json").read_text())
    rows = _read_jsonl(run_dir / "results.jsonl")
    errors = _read_jsonl(run_dir / "errors.jsonl")
    done = {(r["prompt_id"], r["rep"]) for r in rows}
    unresolved = {(e["prompt_id"], e["rep"]) for e in errors} - done

    case_means = per_case_means(rows)
    metrics = {m: _stat(list(cases.values())) for m, cases in case_means.items()}

    tag_of = {r["prompt_id"]: r["tags"][0] for r in rows if r.get("tags")}
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted(set(tag_of.values())):
        by_tag[tag] = {
            m: _stat(vals)
            for m in ("correct", "refused", "page_hit@5", "evidence_coverage@5")
            if (vals := [v for c, v in case_means.get(m, {}).items() if tag_of.get(c) == tag])
        }

    ok = [r for r in rows if r["status"] == "ok"]
    perf: dict[str, float] = {}
    if ok and "latency_s" in ok[0]:
        lat = [r["latency_s"] for r in ok]
        perf = {
            "generation_latency_p50_s": statistics.median(lat),
            "generation_latency_p95_s": _percentile(lat, 0.95),
            "input_tokens_mean": statistics.fmean(r["usage"]["input_tokens"] for r in ok),
            "output_tokens_mean": statistics.fmean(r["usage"]["output_tokens"] for r in ok),
            "judge_input_tokens_mean": statistics.fmean(
                r["judge_usage"]["input_tokens"] for r in ok
            ),
        }
    if ok:
        perf["retrieval_latency_p50_s"] = statistics.median(r["retrieval_latency_s"] for r in ok)

    judge_models = {r.get("judge_model") for r in rows} - {None}
    if meta.get("judge"):
        judge_models.add(meta["judge"]["model"])
    return {
        "run": run_dir.name,
        "config": meta["config"]["name"],
        "split": meta["split"],
        "generator": meta["config"]["generator"]["model"],
        "retriever": meta["config"]["retriever"]["type"],
        "judge": sorted(judge_models),
        "judge_is_generator": meta.get("judge_is_generator", False),
        "cases": len({r["prompt_id"] for r in rows}),
        "rows": len(rows),
        "truncated": sum(r["status"] == "truncated" for r in rows),
        "unresolved_errors": len(unresolved),
        "metrics": dict(sorted(metrics.items(), key=lambda kv: _order(kv[0]))),
        "by_tag": by_tag,
        "perf": perf,
    }


def _order(metric: str) -> tuple[int, str]:
    return (HEADLINE.index(metric), "") if metric in HEADLINE else (len(HEADLINE), metric)


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    text = json.dumps(summary, indent=2) + "\n"
    (run_dir / "summary.json").write_text(text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_dir.name}.json").write_text(text)


def _fmt(stat: dict[str, Any]) -> str:
    lo, hi = stat["ci95"]
    return f"{stat['mean']:.3f} [{lo:.2f}, {hi:.2f}]"


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"### {summary['run']}",
        "",
        f"generator `{summary['generator']}` · retriever `{summary['retriever']}` · "
        f"judge `{', '.join(summary['judge']) or '-'}` · {summary['cases']} cases · "
        f"{summary['truncated']} truncated · {summary['unresolved_errors']} unresolved errors",
    ]
    if summary["judge_is_generator"]:
        lines.append("")
        lines.append(
            "> ⚠️ The judge is the same model as the generator, so it may favour its own answers."
        )
    lines += ["", "| Metric | Mean [95% CI] | n |", "|---|---|---|"]
    lines += [f"| {m} | {_fmt(s)} | {s['n']} |" for m, s in summary["metrics"].items()]
    if summary["perf"]:
        lines += ["", "| Perf | Value |", "|---|---|"]
        lines += [f"| {k} | {v:,.2f} |" for k, v in summary["perf"].items()]
    return "\n".join(lines)


def compare(run_dirs: list[Path], metrics: list[str] | None = None) -> str:
    """Side-by-side table; later runs also show their paired difference from the first."""
    runs = [(d.name, per_case_means(_read_jsonl(d / "results.jsonl"))) for d in run_dirs]
    names = [n for n, _ in runs]
    all_metrics = sorted(set().union(*(m.keys() for _, m in runs)), key=_order)
    metrics = [m for m in (metrics or all_metrics) if m in all_metrics]

    header = "| Metric | " + " | ".join(names) + " |"
    lines = [header, "|---" * (len(names) + 1) + "|"]
    base_name, base = runs[0]
    for m in metrics:
        cells = []
        for i, (_, run) in enumerate(runs):
            vals = run.get(m, {})
            if not vals:
                cells.append("-")
                continue
            cell = f"{statistics.fmean(vals.values()):.3f}"
            shared = sorted(set(vals) & set(base.get(m, {})))
            if i > 0 and shared:
                diffs = [vals[c] - base[m][c] for c in shared]
                lo, hi = bootstrap_ci(diffs)
                mark = " *" if lo > 0 or hi < 0 else ""
                cell += f" ({statistics.fmean(diffs):+.3f} [{lo:+.2f}, {hi:+.2f}]{mark})"
            cells.append(cell)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Differences are paired against `{base_name}` on shared cases, with bootstrap 95% CIs; "
        "`*` marks intervals that exclude zero.",
    ]
    return "\n".join(lines)
