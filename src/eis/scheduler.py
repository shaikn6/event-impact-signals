"""Periodic polling job: fetch from all sources, classify, store new signals.

This is the "live feed" half of the pipeline — a single function
(`poll_once`) that a scheduler calls repeatedly. Kept as a plain function
(not a class) so it's trivially testable and so the CLI can also call it
directly for a one-shot run without touching APScheduler at all.
"""

from __future__ import annotations

import logging
import time

from eis.ingest import gdelt, rss
from eis.pipeline import build_signals, filter_actionable
from eis.store import SignalStore

logger = logging.getLogger(__name__)

# GDELT queries chosen to match the event types this project classifies —
# broad enough to catch real events, narrow enough to stay low-noise.
# Parenthesized OR groups are required by GDELT's query syntax — an
# unparenthesized "a OR b OR c" silently returns zero results.
_GDELT_QUERIES = [
    "(war OR invasion OR airstrike)",
    "(earthquake OR hurricane OR wildfire OR flood)",
    "(pandemic OR outbreak OR epidemic)",
    "(tariff OR sanctions)",
    "federal reserve rate",
]

# GDELT asks that requests be spaced at least 5s apart; firing several
# queries back-to-back gets every request after the first rate-limited
# (a plain-text response, not JSON, which fetch_articles then correctly
# treats as "no data" — so the failure is silent unless you space requests).
_GDELT_REQUEST_SPACING_SECONDS = 5.0


def poll_once(store: SignalStore, include_reddit: bool = False) -> int:
    """Run one ingest -> classify -> store cycle. Returns count of new signals saved.

    Args:
        store:          Target SignalStore.
        include_reddit: Reddit is best-effort/unreliable without OAuth
                         (see ingest/reddit.py) — off by default so a
                         scheduled run doesn't depend on a flaky source.
    """
    articles = []
    for i, query in enumerate(_GDELT_QUERIES):
        if i > 0:
            time.sleep(_GDELT_REQUEST_SPACING_SECONDS)
        articles.extend(gdelt.fetch_articles(query, max_records=15))
    articles.extend(rss.fetch_all())

    if include_reddit:
        from eis.ingest import reddit

        for subreddit in ("worldnews", "economy"):
            articles.extend(reddit.fetch_subreddit(subreddit))

    signals = filter_actionable(build_signals(articles))
    inserted = store.save_new(signals)
    logger.info(
        "poll_once: fetched %d articles, %d actionable signals, %d new",
        len(articles),
        len(signals),
        inserted,
    )
    return inserted


def run_scheduler(interval_minutes: int = 15, db_path: str = "data/signals.db") -> None:
    """Blocking loop: poll every `interval_minutes` using APScheduler.

    interval_minutes defaults to 15 to match GDELT's own update cadence —
    polling faster than the source updates just burns requests for no new data.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    store = SignalStore(db_path)
    scheduler = BlockingScheduler()
    scheduler.add_job(poll_once, "interval", minutes=interval_minutes, args=[store])

    logger.info("Starting scheduler: polling every %d minutes", interval_minutes)
    poll_once(store)  # run once immediately rather than waiting for the first interval
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        store.close()
