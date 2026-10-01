"""Deterministic, stratified eval splits over FinanceBench questions.

Splits are stored as ID lists in `evals/splits/` and committed, so every
experiment is scored on exactly the same questions.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from rag.data.financebench import PINNED_SHA, PROJECT_ROOT, Question

SPLITS_DIR = PROJECT_ROOT / "evals" / "splits"
SEED = 13
DEV_SIZE = 50
CI_SMOKE_SIZE = 20


def stratified_sample[T](
    items: Sequence[T], size: int, key: Callable[[T], str], seed: int
) -> list[T]:
    """Sample `size` items, allocating to each stratum proportionally.

    Uses largest-remainder allocation, so stratum quotas always sum to `size`.
    Output keeps the input order, which makes the split files easy to diff.
    """
    if not 0 <= size <= len(items):
        raise ValueError(f"size must be in [0, {len(items)}], got {size}")
    strata: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        strata[key(item)].append(i)

    exact = {k: size * len(v) / len(items) for k, v in strata.items()}
    quota = {k: int(x) for k, x in exact.items()}
    by_remainder = sorted(strata, key=lambda k: (exact[k] - quota[k], k), reverse=True)
    for k in by_remainder[: size - sum(quota.values())]:
        quota[k] += 1

    rng = random.Random(seed)
    chosen: set[int] = set()
    for k in sorted(strata):
        chosen.update(rng.sample(strata[k], quota[k]))
    return [items[i] for i in sorted(chosen)]


def make_splits(questions: Sequence[Question], seed: int = SEED) -> dict[str, list[str]]:
    """Return {"dev", "test", "ci_smoke"} -> question IDs. ci_smoke is a subset of dev."""
    by_type = lambda q: q.question_type  # noqa: E731
    dev = stratified_sample(questions, DEV_SIZE, by_type, seed)
    dev_ids = {q.financebench_id for q in dev}
    test = [q for q in questions if q.financebench_id not in dev_ids]
    ci_smoke = stratified_sample(dev, CI_SMOKE_SIZE, by_type, seed)
    return {
        name: [q.financebench_id for q in split]
        for name, split in {"dev": dev, "test": test, "ci_smoke": ci_smoke}.items()
    }


def write_splits(splits: dict[str, list[str]], out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        payload = {
            "name": name,
            "source": f"patronus-ai/financebench@{PINNED_SHA}",
            "seed": SEED,
            "stratified_by": "question_type",
            "size": len(ids),
            "ids": ids,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    return json.loads((splits_dir / f"{name}.json").read_text())["ids"]
