import json

import pytest

from rag.evals.labeling import calibrate, cohens_kappa, label_run
from rag.evals.summary import bootstrap_ci, compare, per_case_means, summarize


def row(case_id, correct, rep=0, status="ok", answer="a", refused=0.0):
    return {
        "prompt_id": case_id,
        "rep": rep,
        "prompt": "q",
        "tags": ["t"],
        "status": status,
        "grade": {"correct": correct, "refused": refused},
        "answer": answer,
        "gold": "g",
        "explanation": {"correct": "because"},
        "latency_s": 1.0,
        "retrieval_latency_s": 0.1,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "judge_usage": {"input_tokens": 5, "output_tokens": 1},
        "judge_model": "j",
    }


def write_run(path, rows, errors=()):
    path.mkdir(parents=True)
    meta = {
        "config": {"name": path.name, "generator": {"model": "m"}, "retriever": {"type": "dense"}},
        "split": "dev",
        "judge": {"model": "j"},
        "judge_is_generator": False,
    }
    (path / "meta.json").write_text(json.dumps(meta))
    (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "errors.jsonl").write_text("".join(json.dumps(e) + "\n" for e in errors))
    return path


def test_per_case_means_average_reps_and_skip_truncated():
    rows = [row("a", 1.0, 0), row("a", 0.0, 1), row("b", 1.0), row("c", 0.0, status="truncated")]
    means = per_case_means(rows)["correct"]
    assert means == {"a": 0.5, "b": 1.0}


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.0, 1.0] * 25)
    assert lo < 0.5 < hi


def test_summarize_counts_truncation_and_unresolved_errors(tmp_path):
    run = write_run(
        tmp_path / "r",
        [row("a", 1.0), row("b", 0.0), row("c", 1.0, status="truncated")],
        errors=[{"prompt_id": "d", "rep": 0}, {"prompt_id": "a", "rep": 0}],  # "a" later succeeded
    )
    s = summarize(run)
    assert s["metrics"]["correct"]["mean"] == 0.5
    assert s["truncated"] == 1
    assert s["unresolved_errors"] == 1


def test_compare_reports_paired_difference(tmp_path):
    a = write_run(tmp_path / "a", [row(c, 0.0) for c in "wxyz"])
    b = write_run(tmp_path / "b", [row(c, 1.0) for c in "wxyz"])
    table = compare([a, b], ["correct"])
    assert "+1.000" in table and " *" in table


def test_cohens_kappa():
    assert cohens_kappa([("c", "c"), ("i", "i")]) == pytest.approx(1.0)
    assert cohens_kappa([("c", "i"), ("i", "c")]) == pytest.approx(-1.0)


def test_label_then_calibrate(tmp_path):
    run = write_run(
        tmp_path / "r",
        [
            row("a", 1.0, answer="x"),
            row("b", 0.0, answer="y"),
            row("c", 0.0, answer="z", refused=1.0),
        ],
    )
    labels = tmp_path / "labels.jsonl"
    answers = iter(["c", "c"])  # human says both non-refused answers are correct
    assert label_run(run, labels, input_fn=lambda _: next(answers)) == 2
    report = calibrate(run, labels)
    assert report["labeled"] == 2
    assert report["agreement"] == 0.5
    assert [d["prompt_id"] for d in report["disagreements"]] == ["b"]
