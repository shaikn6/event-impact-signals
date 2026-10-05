"""Shared HTTP-GET-JSON helper for the ingest modules.

gdelt.py and reddit.py both do the same thing — GET a URL, raise on a bad
status, parse JSON, and treat any failure (network error or malformed
body) as "no data" rather than an exception the caller has to handle.
This factors that one policy into one place instead of two near-identical
try/except blocks.
"""

from __future__ import annotations

import httpx


def fetch_json(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> dict | None:
    """GET `url` and return its parsed JSON body, or None on any failure.

    "Any failure" covers HTTP errors (4xx/5xx, timeouts, connection
    errors) and a non-JSON response body — both are treated identically
    by every caller here (as "no data this time"), so there's no reason
    for them to distinguish the two.

    Args:
        client: Optional shared httpx.Client, so a caller making several
                same-host requests (e.g. gdelt.fetch_articles called once
                per query) can reuse one TCP/TLS connection instead of
                paying setup cost on every call. Defaults to a one-off
                `httpx.get` when not given.
    """
    getter = client.get if client is not None else httpx.get
    try:
        response = getter(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None
