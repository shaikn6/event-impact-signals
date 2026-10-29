"""Combines ingestion, event classification, and sector-impact lookup into
a single list of EventSignal objects."""

from __future__ import annotations

import os

from eis.classify.event_type import classify_event
from eis.impact.sector_mapping import get_sector_impacts
from eis.models import Article, EventSignal


def build_signal(article: Article) -> EventSignal:
    """Classify one article and attach its sector-impact hypotheses."""
    text = f"{article.title} {article.summary}"
    # Keyword rules stay the default; EIS_CLASSIFIER=trained opts in to the
    # TF-IDF + logistic-regression model (see classify/trained.py) and
    # EIS_CLASSIFIER=embedding to the sentence-embedding model (classify/embedding.py).
    choice = os.environ.get("EIS_CLASSIFIER", "keyword")
    if choice == "trained":
        from eis.classify.trained import classify_event_trained

        event_type, confidence = classify_event_trained(text)
    elif choice == "embedding":
        from eis.classify.embedding import classify_event_embedding

        event_type, confidence = classify_event_embedding(text)
    else:
        event_type, confidence = classify_event(text)

    # get_sector_impacts already returns [] for EventType.GENERAL (and any
    # other type absent from the table) via its dict.get(..., []) default —
    # that's the single place "which event types get impacts" is decided,
    # so no separate confidence check is needed here.
    return EventSignal(
        article=article,
        event_type=event_type,
        event_confidence=confidence,
        sector_impacts=get_sector_impacts(event_type),
    )


def build_signals(articles: list[Article]) -> list[EventSignal]:
    """Classify a batch of articles. Order is preserved."""
    return [build_signal(article) for article in articles]


def filter_actionable(signals: list[EventSignal]) -> list[EventSignal]:
    """Keep only signals that produced at least one sector impact.

    "Actionable" here means "has a documented hypothesis attached" — it is
    not a claim that the signal should inform a trade.
    """
    return [s for s in signals if s.sector_impacts]
