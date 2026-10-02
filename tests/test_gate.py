from rag.evals.gate import GATE_PATH, evaluate, load_gate, render


def summary(**means):
    return {"metrics": {k: {"mean": v} for k, v in means.items()}}


def test_gate_passes_and_fails_on_thresholds():
    checks = evaluate(summary(a=0.8, b=0.5), {"a": 0.7, "b": 0.6})
    assert [c.passed for c in checks] == [True, False]
    assert "❌ failed" in render(checks, "run")


def test_missing_metric_fails():
    [check] = evaluate(summary(), {"a": 0.1})
    assert not check.passed and "missing" in render([check], "run")


def test_committed_gate_file_is_valid():
    gate = load_gate(GATE_PATH)
    assert gate["split"] == "ci_smoke" and gate["config"] == "configs/ci_gate.yaml"
    assert all(0 < v <= 1 for v in gate["min"].values())
