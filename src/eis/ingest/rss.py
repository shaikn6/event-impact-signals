"""RSS ingestion — free, no API key, from a curated list of major outlets.

RSS complements GDELT: GDELT is broad but noisy (indexes everything,
including non-news pages); a short curated outlet list gives a higher
signal-to-noise feed for mainstream financial/world news.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import feedparser

from eis.models import Article

# Curated, free, no-auth financial/world-news RSS feeds.
DEFAULT_FEEDS: dict[str, str] = {
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "foxbusiness": "https://moxie.foxbusiness.com/google-publisher/latest.xml",
}


def fetch_feed(feed_name: str, feed_url: str, timeout: float = 10.0) -> list[Article]:
    """Fetch and parse one RSS feed into Articles.

    Returns an empty list (not an exception) on a parse/network failure,
    consistent with ingest.gdelt.fetch_articles — one bad feed should not
    break the whole ingest run.
    """
    try:
        parsed = feedparser.parse(feed_url, request_headers={"User-Agent": "event-impact-signals"})
    except OSError:  # network failure (feedparser wraps urllib's own OSError subclasses)
        return []

    if parsed.bozo and not parsed.entries:
        return []

    articles = []
    for entry in parsed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source=f"rss:{feed_name}",
                published_at=_parse_entry_date(entry),
                summary=entry.get("summary", ""),
            )
        )
    return articles


def fetch_all(feeds: dict[str, str] | None = None) -> list[Article]:
    """Fetch every feed in `feeds` (default: DEFAULT_FEEDS) and flatten the results.

    Feeds are independent, different-host I/O with no rate-limit
    relationship to each other, so they're fetched concurrently rather
    than one after another — this doesn't change what's fetched, only
    how long waiting for all of it takes.
    """
    feeds = feeds if feeds is not None else DEFAULT_FEEDS
    with ThreadPoolExecutor(max_workers=max(1, len(feeds))) as pool:
        results = pool.map(lambda item: fetch_feed(*item), feeds.items())
    return [article for feed_articles in results for article in feed_articles]


def _parse_entry_date(entry) -> datetime | None:
    parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed_time:
        return None
    try:
        return datetime(*parsed_time[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None
