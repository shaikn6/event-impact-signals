import json
from pathlib import Path

import pytest

from eis.classify import embedding
from eis.models import EventType

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _rows(name):
    return [json.loads(x) for x in (EVAL / name).read_text().splitlines() if x.strip()]


def _fake_embed(monkeypatch, cls_index):
    """Replace the encoder with the head's own weight row so no model is downloaded."""
    head = embedding._head()
    monkeypatch.setattr(embedding, "embed", lambda texts: [head["coef"][cls_index] for _ in texts])
    return head["classes"][cls_index]


def test_head_shape_matches_encoder():
    head = embedding._head()
    assert len(head["classes"]) == len(head["intercept"]) == len(head["coef"]) == 13
    assert {len(row) for row in head["coef"]} == {384}
    assert head["meta"]["revision"] == embedding.REVISION


def test_head_scores_its_own_direction_highest(monkeypatch):
    for k in range(13):
        expected = _fake_embed(monkeypatch, k)
        event_type, prob = embedding.classify_event_embedding("anything")
        assert event_type == EventType(expected)
        assert 0.0 < prob <= 1.0


def test_pipeline_uses_embedding_model_when_flagged(monkeypatch):
    from eis.models import Article
    from eis.pipeline import build_signal

    expected = _fake_embed(monkeypatch, 0)
    monkeypatch.setenv("EIS_CLASSIFIER", "embedding")
    sig = build_signal(Article(title="t", url="u", source="s"))
    assert sig.event_type == EventType(expected)


def test_real_encoder_on_held_out_test():
    """Runs only with the `embed` extra installed (downloads the encoder)."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    test = _rows("eval_set.jsonl") + _rows("test_extra.jsonl")
    correct = sum(
        embedding.classify_event_embedding(r["text"])[0].value == r["event_type"] for r in test
    )
    # Regression guard below the measured 102/108.
    assert correct / len(test) > 0.9
