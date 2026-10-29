"""Train the sentence-embedding event classifier.

Embeds eval/train_set.jsonl with BAAI/bge-small-en-v1.5 (pinned revision,
MIT licence, 33M parameters), picks the regularisation strength C of a
logistic regression by the same unshuffled stratified 5-fold
cross-validation as train_classifier.py (training set only; the held-out
test headlines are never read), then exports the linear head to
src/eis/classify/event_clf_embed.json. The encoder itself is not stored in
this repository; it is downloaded from Hugging Face at the pinned revision.

Needs the `embed` and `dev` extras. Run: python eval/train_embedding_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from eis.classify.embedding import MODEL_ID, REVISION, embed

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "eis" / "classify" / "event_clf_embed.json"
C_GRID = [1, 3, 10, 30, 100]


def main() -> None:
    rows = [json.loads(x) for x in (HERE / "train_set.jsonl").read_text().splitlines() if x.strip()]
    x = embed([r["text"] for r in rows])
    labels = [r["event_type"] for r in rows]

    cv = StratifiedKFold(n_splits=5)
    cv_scores = {
        c: float(cross_val_score(LogisticRegression(C=c, max_iter=5000), x, labels, cv=cv).mean())
        for c in C_GRID
    }
    # Smallest C among the best scores: the most regularised of the tied models.
    best_c = min(c for c in C_GRID if cv_scores[c] == max(cv_scores.values()))
    clf = LogisticRegression(C=best_c, max_iter=5000).fit(x, labels)

    model = {
        "classes": [str(c) for c in clf.classes_],
        "intercept": [round(float(b), 6) for b in clf.intercept_],
        "coef": np.round(clf.coef_, 6).tolist(),
        "meta": {
            "encoder": MODEL_ID,
            "revision": REVISION,
            "head": "LogisticRegression on L2-normalised CLS embeddings",
            "C": best_c,
            "cv_accuracy_by_C": {str(c): round(s, 3) for c, s in cv_scores.items()},
            "n_train": len(rows),
        },
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")) + "\n")
    print(f"C={best_c} cv={cv_scores[best_c]:.3f} -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
