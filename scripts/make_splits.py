"""Create the committed dev / test / ci_smoke splits in evals/splits/."""

from collections import Counter

from rag.data.financebench import load_questions
from rag.data.splits import SPLITS_DIR, make_splits, write_splits


def main() -> None:
    questions = {q.financebench_id: q for q in load_questions()}
    splits = make_splits(list(questions.values()))
    write_splits(splits)
    for name, ids in splits.items():
        mix = Counter(questions[i].question_type for i in ids)
        print(f"{name:9s} n={len(ids):3d}  {dict(sorted(mix.items()))}")
    print(f"wrote {SPLITS_DIR}")


if __name__ == "__main__":
    main()
