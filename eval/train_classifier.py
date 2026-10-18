"""Train the TF-IDF + logistic-regression event classifier.

Fits on eval/train_set.jsonl only (the held-out test headlines in
eval_set.jsonl and test_extra.jsonl are never read here), picks the
regularisation strength C by 5-fold cross-validation inside the training
set, then exports the weights to src/eis/classify/event_clf.json.

Needs scikit-learn (the `dev` extra). Deterministic: no randomness in
liblinear/lbfgs fitting, and the CV folds are unshuffled and stratified.

Run: python eval/train_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "eis" / "classify" / "event_clf.json"
C_GRID = [1, 3, 10, 30, 100]


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)


def main() -> None:
    rows = [json.loads(x) for x in (HERE / "train_set.jsonl").read_text().splitlines() if x.strip()]
    texts = [r["text"] for r in rows]
    labels = [r["event_type"] for r in rows]

    cv = StratifiedKFold(n_splits=5)
    cv_scores = {}
    for c in C_GRID:
        pipe = make_pipeline(_vectorizer(), LogisticRegression(C=c, max_iter=2000))
        cv_scores[c] = float(cross_val_score(pipe, texts, labels, cv=cv).mean())
    best_c = max(cv_scores, key=lambda c: cv_scores[c])

    vec = _vectorizer()
    x = vec.fit_transform(texts)
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(x, labels)

    classes = [str(c) for c in clf.classes_]
    idf = vec.idf_
    terms = {
        term: [round(float(idf[i]), 5)] + [round(float(w), 5) for w in clf.coef_[:, i]]
        for term, i in vec.vocabulary_.items()
    }
    model = {
        "classes": classes,
        "intercept": [round(float(b), 5) for b in clf.intercept_],
        "terms": terms,
        "meta": {
            "model": "TfidfVectorizer(ngram_range=(1,2), sublinear_tf) + LogisticRegression",
            "C": best_c,
            "cv_accuracy_by_C": {str(c): round(s, 3) for c, s in cv_scores.items()},
            "n_train": len(rows),
            "n_terms": len(terms),
        },
    }
    OUT.write_text(json.dumps(model, separators=(",", ":")) + "\n")
    print(f"C={best_c} cv={cv_scores[best_c]:.3f} terms={len(terms)} -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
