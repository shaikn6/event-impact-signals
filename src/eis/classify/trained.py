"""Trained event-type classifier (TF-IDF + logistic regression).

Inference is pure Python on a small JSON weight file, so the runtime keeps
no scikit-learn dependency and loads no pickle. Weights are produced by
`eval/train_classifier.py`; the tokenizer, bigram, sublinear-TF and L2
steps below mirror scikit-learn's TfidfVectorizer so the exported
weights reproduce the fitted model (tests/test_trained_classifier.py
checks this against scikit-learn).

The keyword classifier in event_type.py stays the default; this one is
opt-in (see eis.pipeline). Confidence is the softmax probability of the
winning class, which is uncalibrated.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from eis.models import EventType

WEIGHTS_PATH = Path(__file__).with_name("event_clf.json")

# Same default token pattern as sklearn's TfidfVectorizer.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


@lru_cache(maxsize=1)
def _load(path: str = str(WEIGHTS_PATH)) -> dict:
    return json.loads(Path(path).read_text())


def _features(text: str, terms: dict[str, list[float]]) -> dict[str, float]:
    tokens = _TOKEN.findall(text.lower())
    grams = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
    counts: dict[str, int] = {}
    for g in grams:
        if g in terms:
            counts[g] = counts.get(g, 0) + 1
    # sublinear tf * idf, then L2-normalised. The norm covers only
    # in-vocabulary terms, which is what the fitted vectorizer does too.
    vec = {g: (1.0 + math.log(c)) * terms[g][0] for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {g: v / norm for g, v in vec.items()} if norm else {}


def classify_event_trained(text: str) -> tuple[EventType, float]:
    """Classify free text with the trained model.

    Returns:
        (event_type, probability) — probability of the predicted class in
        [0, 1]. Text with no in-vocabulary terms returns (GENERAL, 0.0).
    """
    model = _load()
    labels: list[str] = model["classes"]
    feats = _features(text, model["terms"])
    if not feats:
        return EventType.GENERAL, 0.0

    logits = list(model["intercept"])
    for g, v in feats.items():
        for k, w in enumerate(model["terms"][g][1:]):
            logits[k] += w * v
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    best = max(range(len(labels)), key=lambda k: exps[k])
    return EventType(labels[best]), exps[best] / sum(exps)
