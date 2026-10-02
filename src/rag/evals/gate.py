"""CI eval gate: compare a run's summary against committed minimum thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.data.financebench import PROJECT_ROOT

GATE_PATH = PROJECT_ROOT / "evals" / "gate.yaml"


@dataclass
class Check:
    metric: str
    value: float | None
    minimum: float

    @property
    def passed(self) -> bool:
        return self.value is not None and self.value >= self.minimum


def load_gate(path: Path = GATE_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def evaluate(summary: dict[str, Any], minimums: dict[str, float]) -> list[Check]:
    """A missing metric fails: a gate that silently skips checks protects nothing."""
    metrics = summary["metrics"]
    return [
        Check(metric, metrics[metric]["mean"] if metric in metrics else None, minimum)
        for metric, minimum in minimums.items()
    ]


def render(checks: list[Check], run: str) -> str:
    ok = all(c.passed for c in checks)
    lines = [
        f"### Eval gate: {'✅ passed' if ok else '❌ failed'} (`{run}`)",
        "",
        "| Metric | Value | Minimum | |",
        "|---|---|---|---|",
    ]
    for c in checks:
        value = "missing" if c.value is None else f"{c.value:.3f}"
        lines.append(f"| {c.metric} | {value} | {c.minimum:.2f} | {'✅' if c.passed else '❌'} |")
    return "\n".join(lines)
