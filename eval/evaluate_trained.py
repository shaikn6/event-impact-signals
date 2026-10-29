"""Compare the keyword and trained event classifiers on the same held-out test.

Test set = eval/eval_set.jsonl (the original 34) + eval/test_extra.jsonl
(74 more). None of these headlines are in eval/train_set.jsonl, which is
the only file the trained model sees. Both classifiers score the identical
rows; accuracy and macro-F1 are reported overall and on the original 34.
Macro-F1 averages over all 13 classes present in the test gold labels,
counting a class with no correct predictions as 0.

The embedding model needs the `embed` extra and downloads its encoder on
first use. The paired bootstrap resamples the test headlines 2,000 times
(seed 0) and reports the 95% interval of embedding minus trained accuracy.

Writes eval/results_comparison.json. Run: python eval/evaluate_trained.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from eis.classify.embedding import classify_event_embedding
from eis.classify.event_type import classify_event
from eis.classify.trained import classify_event_trained

HERE = Path(__file__).resolve().parent


def _load(name: str) -> list[dict]:
    return [json.loads(x) for x in (HERE / name).read_text().splitlines() if x.strip()]


def _score(rows: list[dict], preds: list[str]) -> dict:
    gold = [r["event_type"] for r in rows]
    classes = sorted(set(gold))
    f1s = []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, preds))
        fp = sum(g != c and p == c for g, p in zip(gold, preds))
        fn = sum(g == c and p != c for g, p in zip(gold, preds))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    correct = sum(g == p for g, p in zip(gold, preds))
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "correct": correct,
        "macro_f1": round(sum(f1s) / len(f1s), 3),
    }


def _paired_bootstrap(rows: list[dict], a: list[str], b: list[str], n: int = 2000) -> list[float]:
    """95% percentile interval of accuracy(a) - accuracy(b), resampling rows with replacement."""
    rng = random.Random(0)
    gold = [r["event_type"] for r in rows]
    diff = [(x == g) - (y == g) for g, x, y in zip(gold, a, b)]
    stats = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(n))
    return [round(stats[int(0.025 * n)], 3), round(stats[int(0.975 * n) - 1], 3)]


def main() -> None:
    original = _load("eval_set.jsonl")
    test = original + _load("test_extra.jsonl")
    train_texts = {r["text"] for r in _load("train_set.jsonl")}
    assert not train_texts & {r["text"] for r in test}, "train/test overlap"

    always_general = ["general"] * len(test)
    keyword = [classify_event(r["text"])[0].value for r in test]
    trained = [classify_event_trained(r["text"])[0].value for r in test]
    embedding = [classify_event_embedding(r["text"])[0].value for r in test]
    n_orig = len(original)

    result = {
        "train_n": len(train_texts),
        "test_n": len(test),
        "test_class_counts": dict(sorted(Counter(r["event_type"] for r in test).items())),
        "held_out_test": {
            "always_general": _score(test, always_general),
            "keyword": _score(test, keyword),
            "trained": _score(test, trained),
            "embedding": _score(test, embedding),
        },
        "embedding_minus_trained_accuracy_ci95": _paired_bootstrap(test, embedding, trained),
        "original_34": {
            "keyword": _score(original, keyword[:n_orig]),
            "trained": _score(original, trained[:n_orig]),
            "embedding": _score(original, embedding[:n_orig]),
        },
        "trained_errors": [
            {"text": r["text"], "gold": r["event_type"], "pred": p}
            for r, p in zip(test, trained)
            if r["event_type"] != p
        ],
        "embedding_errors": [
            {"text": r["text"], "gold": r["event_type"], "pred": p}
            for r, p in zip(test, embedding)
            if r["event_type"] != p
        ],
    }
    (HERE / "results_comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"train n={result['train_n']}  held-out test n={result['test_n']}")
    for split in ("held_out_test", "original_34"):
        print(split)
        for name, m in result[split].items():
            print(f"  {name:14s} acc={m['accuracy']:.3f} ({m['correct']}/{m['n']})  macroF1={m['macro_f1']:.3f}")
    print(f"trained errors: {len(result['trained_errors'])}  embedding errors: {len(result['embedding_errors'])}")
    print(f"embedding - trained accuracy, paired 95% interval: {result['embedding_minus_trained_accuracy_ci95']}")


if __name__ == "__main__":
    main()
