"""Sentence-embedding event-type classifier (opt-in, EIS_CLASSIFIER=embedding).

A frozen BAAI/bge-small-en-v1.5 encoder turns the headline into a 384-number
vector (CLS token, L2-normalised, as the model card recommends); a logistic
regression head stored in event_clf_embed.json maps it to an event type. The
head is fitted by eval/train_embedding_classifier.py on the same 181
training headlines as the TF-IDF model. Unlike the keyword and TF-IDF
classifiers this one needs torch and transformers (the `embed` extra) and
downloads the encoder (about 130 MB) on first use.

Confidence is the softmax probability of the winning class (uncalibrated).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from eis.models import EventType

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
HEAD_PATH = Path(__file__).with_name("event_clf_embed.json")


@lru_cache(maxsize=1)
def _encoder():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    return tok, AutoModel.from_pretrained(MODEL_ID, revision=REVISION).eval()


def embed(texts: list[str]) -> list[list[float]]:
    """L2-normalised CLS embeddings, one list of 384 floats per text."""
    import torch

    tok, model = _encoder()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        cls = model(**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1).tolist()


@lru_cache(maxsize=1)
def _head() -> dict:
    return json.loads(HEAD_PATH.read_text())


def classify_event_embedding(text: str) -> tuple[EventType, float]:
    """Return (event_type, probability of that class)."""
    head = _head()
    vec = embed([text])[0]
    logits = [
        b + sum(w * v for w, v in zip(row, vec)) for row, b in zip(head["coef"], head["intercept"])
    ]
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    best = max(range(len(exps)), key=lambda k: exps[k])
    return EventType(head["classes"][best]), exps[best] / sum(exps)
