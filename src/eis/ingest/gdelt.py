"""GDELT DOC 2.0 API client — free, no API key, global news article search.

GDELT indexes news coverage from tens of thousands of outlets worldwide
and updates continuously, which makes it a good primary source for
geopolitical/disaster-type events (it's precisely this kind of event this
project classifies). See https://blog.gdeltproject.org/gdelt-doc-2-0-api/
for the full query syntax.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from eis.ingest._http import fetch_json
from eis.models import Article

_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_articles(
    query: str,
    max_records: int = 25,
    timeout: float = 25.0,
    client: httpx.Client | None = None,
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

    payload = fetch_json(_BASE_URL, params=params, timeout=timeout, client=client)
    if payload is None:
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


def _parse_gdelt_date(raw: str | None) -> datetime | None:
    """Parse GDELT's "20261002T104500Z" timestamp format."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None
