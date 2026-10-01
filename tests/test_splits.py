from collections import Counter

import pytest

from rag.data.splits import SPLITS_DIR, load_split, make_splits, stratified_sample


def test_stratified_sample_is_proportional_and_deterministic():
    items = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    first = stratified_sample(list(enumerate(items)), 20, key=lambda x: x[1], seed=1)
    second = stratified_sample(list(enumerate(items)), 20, key=lambda x: x[1], seed=1)
    assert first == second
    assert Counter(label for _, label in first) == {"a": 12, "b": 6, "c": 2}


def test_stratified_sample_quotas_sum_to_size_with_remainders():
    items = list(enumerate(["a"] * 50 + ["b"] * 50 + ["c"] * 50))
    sample = stratified_sample(items, 50, key=lambda x: x[1], seed=0)
    counts = Counter(label for _, label in sample)
    assert len(sample) == 50
    assert sorted(counts.values()) == [16, 17, 17]


def test_stratified_sample_rejects_bad_size():
    with pytest.raises(ValueError):
        stratified_sample([1, 2], 3, key=str, seed=0)


def test_make_splits_invariants(questions):
    splits = make_splits(questions)
    dev, test, smoke = set(splits["dev"]), set(splits["test"]), set(splits["ci_smoke"])
    assert len(dev) == 50 and len(test) == 100 and len(smoke) == 20
    assert dev.isdisjoint(test)
    assert dev | test == {q.financebench_id for q in questions}
    assert smoke <= dev


def test_committed_splits_are_consistent():
    """Guards the committed split files against accidental edits."""
    if not (SPLITS_DIR / "dev.json").exists():
        pytest.skip("splits not generated yet")
    dev, test, smoke = (set(load_split(n)) for n in ("dev", "test", "ci_smoke"))
    assert (len(dev), len(test), len(smoke)) == (50, 100, 20)
    assert dev.isdisjoint(test) and smoke <= dev
