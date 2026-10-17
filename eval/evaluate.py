"""Evaluate the rule-based event-type classifier on a hand-labeled set.

Reads eval_set.jsonl (text + gold event_type), runs classify_event on each,
and reports overall accuracy, per-class precision/recall/F1 and a confusion
list. Writes results.json next to this file.

The set is small and hand-labeled, and it deliberately includes idiomatic
"trap" headlines ("price war", "a flood of new shows", "the board sanctions
a buyback") whose true label is `general`, so the score is not inflated by
only easy cases. This measures the keyword classifier, not a trained model.

Run: python eval/evaluate.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eis.classify.event_type import classify_event

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = [json.loads(line) for line in (HERE / "eval_set.jsonl").read_text().splitlines() if line.strip()]

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusions: list[dict[str, str]] = []
    correct = 0

    for row in rows:
        gold = row["event_type"]
        pred, _conf = classify_event(row["text"])
        pred = pred.value
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
            confusions.append({"text": row["text"], "gold": gold, "pred": pred})

    labels = sorted({r["event_type"] for r in rows} | {c["pred"] for c in confusions})
    per_class = {}
    for lab in labels:
        p_den = tp[lab] + fp[lab]
        r_den = tp[lab] + fn[lab]
        precision = tp[lab] / p_den if p_den else None
        recall = tp[lab] / r_den if r_den else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        per_class[lab] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "support": r_den,
        }

    result = {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "correct": correct,
        "per_class": per_class,
        "confusions": confusions,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"n={result['n']} accuracy={result['accuracy']} ({correct}/{len(rows)})")
    for lab, m in per_class.items():
        print(f"  {lab:24s} P={m['precision']} R={m['recall']} F1={m['f1']} (n={m['support']})")
    if confusions:
        print("Misclassified:")
        for c in confusions:
            print(f"  [{c['gold']} -> {c['pred']}] {c['text']}")


if __name__ == "__main__":
    main()
