"""Reddit ingestion via the public, unauthenticated .json endpoints.

HONEST CAVEAT: Reddit aggressively rate-limits and sometimes outright
blocks unauthenticated requests by IP/user-agent (observed: HTTP 403 from
some networks even with a descriptive User-Agent). This source is
best-effort and optional — the pipeline treats an empty/failed fetch the
same as "no new posts," never as an error. For reliable use, register a
free Reddit API app (https://www.reddit.com/prefs/apps) and switch this
to OAuth; that's a documented upgrade path, not implemented here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from eis.ingest._http import fetch_json
from eis.models import Article

_USER_AGENT = "event-impact-signals/0.1 (research tool; non-commercial)"


def fetch_subreddit(subreddit: str, limit: int = 15, timeout: float = 10.0) -> list[Article]:
    """Fetch today's top posts from one subreddit's public JSON feed.

    Returns an empty list on any failure (403, timeout, malformed JSON) —
    see the module docstring for why this source is unreliable without
    OAuth, and why that's handled as "no data" rather than raised.
    """
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"limit": str(limit), "t": "day"}

    payload = fetch_json(url, params=params, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    if payload is None:
        return []

    articles = []
    for child in payload.get("data", {}).get("children", []):
        post = child.get("data", {})
        articles.append(
            Article(
                title=post.get("title", ""),
                url=post.get("url") or f"https://reddit.com{post.get('permalink', '')}",
                source=f"reddit:{subreddit}",
                published_at=_parse_created_utc(post.get("created_utc")),
            )
        )
    return articles


def _parse_created_utc(created_utc: float | None) -> datetime | None:
    if created_utc is None:
        return None
    try:
        return datetime.fromtimestamp(created_utc, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None
