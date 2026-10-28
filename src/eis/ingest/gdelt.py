"""GDELT DOC 2.0 API client — free, no API key, global news article search.

GDELT indexes news coverage from tens of thousands of outlets worldwide
and updates continuously, which makes it a good primary source for
geopolitical/disaster-type events (it's precisely this kind of event this
project classifies). See https://blog.gdeltproject.org/gdelt-doc-2-0-api/
for the full query syntax.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import UTC, datetime

import httpx

from eis.models import Article

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT answers HTTP 429 even at its documented 5s spacing (seen live on
# 4-6 of 7 queries per poll). Retry a couple of times with backoff instead
# of silently dropping the query, but cap both the count and the wait so a
# throttled cycle can't stall the poller or hammer the API.
_MAX_RETRIES = 2
_BACKOFF_SECONDS = 10.0
_MAX_RETRY_AFTER_SECONDS = 60.0


def fetch_articles(
    query: str,
    max_records: int = 25,
    timeout: float = 25.0,
    client: httpx.Client | None = None,
    stats: Counter | None = None,
) -> list[Article]:
    """Fetch recent English-language articles matching a GDELT query.

    Args:
        query:       GDELT query string, e.g. "war" or "earthquake sourcelang:eng".
                     English is forced if the caller doesn't already filter it.
        max_records: Max articles to return (GDELT caps this at 250).
        timeout:     HTTP timeout in seconds. Default is 25s, not the more
                     typical 10s — measured against the live API, GDELT's
                     own response time regularly runs ~12-14s even for a
                     successful request, so a 10s timeout was silently
                     dropping good responses as if they'd failed.
        client:      Optional shared httpx.Client — pass the same client
                     across repeated calls (e.g. scheduler.py's 5 GDELT
                     queries per poll) to reuse one connection instead of
                     a fresh TCP/TLS handshake per call.
        stats:       Optional Counter; incremented under "rate_limited" for
                     every HTTP 429 and "dropped" for a query given up on
                     after retries, so the caller can report them.

    Returns:
        List of Article. Returns an empty list (not an exception) on a
        network error or malformed response — a single flaky request
        should not take down the whole ingest pipeline.
    """
    if "sourcelang:" not in query:
        query = f"{query} sourcelang:eng"

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "sort": "datedesc",
        "format": "json",
    }

    stats = stats if stats is not None else Counter()
    getter = client.get if client is not None else httpx.get
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = getter(_BASE_URL, params=params, timeout=timeout)
        except httpx.HTTPError:
            return []
        if response.status_code != 429:
            break
        stats["rate_limited"] += 1
        if attempt == _MAX_RETRIES:
            stats["dropped"] += 1
            logger.warning(
                "GDELT 429 for %r on attempt %d; query dropped this cycle", query, attempt + 1
            )
            return []
        delay = _retry_delay(response, attempt)
        logger.warning(
            "GDELT 429 for %r on attempt %d; retrying in %.0fs", query, attempt + 1, delay
        )
        time.sleep(delay)

    try:
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []

    articles = []
    for item in payload.get("articles", []):
        articles.append(
            Article(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source="gdelt",
                published_at=_parse_gdelt_date(item.get("seendate")),
                country=item.get("sourcecountry"),
            )
        )
    return articles


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: Retry-After (seconds form) if sent, else backoff."""
    try:
        delay = float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        delay = _BACKOFF_SECONDS * 2**attempt
    return min(max(delay, 0.0), _MAX_RETRY_AFTER_SECONDS)


def _parse_gdelt_date(raw: str | None) -> datetime | None:
    """Parse GDELT's "20261002T104500Z" timestamp format."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
