import json
from pathlib import Path

import pytest

from eis.classify.trained import classify_event_trained
from eis.models import EventType

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _rows(name):
    return [json.loads(x) for x in (EVAL / name).read_text().splitlines() if x.strip()]


def test_trained_classifier_loads_and_predicts():
    event_type, prob = classify_event_trained("Hurricane makes landfall, thousands evacuated")
    assert event_type == EventType.NATURAL_DISASTER
    assert 0.0 < prob <= 1.0


def test_no_known_terms_returns_general_zero():
    assert classify_event_trained("zzzz qqqq") == (EventType.GENERAL, 0.0)


def test_trained_beats_always_general_on_held_out_test():
    test = _rows("eval_set.jsonl") + _rows("test_extra.jsonl")
    train_texts = {r["text"] for r in _rows("train_set.jsonl")}
    assert not train_texts & {r["text"] for r in test}

    correct = sum(classify_event_trained(r["text"])[0].value == r["event_type"] for r in test)
    always_general = sum(r["event_type"] == "general" for r in test)
    assert correct > always_general
    # Regression guard well below the measured ~77%, above the ~20% baseline.
    assert correct / len(test) > 0.6


def test_pipeline_uses_trained_model_when_flagged(monkeypatch):
    from eis.models import Article
    from eis.pipeline import build_signal

    monkeypatch.setenv("EIS_CLASSIFIER", "trained")
    sig = build_signal(Article(title="Ransomware gang locks hospital records", url="u", source="t"))
    assert sig.event_type == EventType.CYBERATTACK


def test_exported_weights_match_scikit_learn():
    """The pure-Python scorer must reproduce a freshly fitted sklearn model."""
    pytest.importorskip("sklearn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from eis.classify.trained import _load

    meta = _load()["meta"]
    train = _rows("train_set.jsonl")
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform([r["text"] for r in train])
    clf = LogisticRegression(C=meta["C"], max_iter=2000).fit(x, [r["event_type"] for r in train])
    texts = [r["text"] for r in _rows("test_extra.jsonl")]
    sk = clf.predict(vec.transform(texts))
    ours = [classify_event_trained(t)[0].value for t in texts]
    assert sum(a == b for a, b in zip(sk, ours)) >= len(texts) - 1
