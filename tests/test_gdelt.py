import inspect

import httpx
import respx

from eis.ingest._http import fetch_json
from eis.ingest.gdelt import fetch_articles


@respx.mock
def test_fetch_articles_parses_valid_response():
    respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        return_value=httpx.Response(
            200,
            json={
                "articles": [
                    {
                        "url": "https://example.com/a",
                        "title": "War escalates",
                        "seendate": "20261002T104500Z",
                        "sourcecountry": "United States",
                    }
                ]
            },
        )
    )
    articles = fetch_articles("war")
    assert len(articles) == 1
    assert articles[0].title == "War escalates"
    assert articles[0].source == "gdelt"
    assert articles[0].country == "United States"
    assert articles[0].published_at is not None


@respx.mock
def test_fetch_articles_returns_empty_list_on_http_error():
    respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(return_value=httpx.Response(500))
    assert fetch_articles("war") == []


@respx.mock
def test_fetch_articles_returns_empty_list_on_malformed_json():
    respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        return_value=httpx.Response(200, text="not json")
    )
    assert fetch_articles("war") == []


@respx.mock
def test_fetch_articles_appends_sourcelang_if_missing():
    route = respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        return_value=httpx.Response(200, json={"articles": []})
    )
    fetch_articles("war")
    assert "sourcelang:eng" in route.calls.last.request.url.params["query"]


@respx.mock
def test_fetch_articles_handles_missing_seendate():
    respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        return_value=httpx.Response(
            200, json={"articles": [{"url": "https://example.com/a", "title": "x"}]}
        )
    )
    articles = fetch_articles("war")
    assert articles[0].published_at is None


def test_default_timeout_is_generous_enough_for_measured_gdelt_latency():
    # Regression guard: GDELT's own measured response time regularly runs
    # ~12-14s even on success, so a typical 10s default would silently
    # drop good responses as failures. See README's "Known limitations".
    default_timeout = inspect.signature(fetch_articles).parameters["timeout"].default
    assert default_timeout >= 15.0


@respx.mock
def test_fetch_articles_shares_client_across_calls():
    route = respx.get("https://api.gdeltproject.org/api/v2/doc/doc").mock(
        return_value=httpx.Response(200, json={"articles": []})
    )
    with httpx.Client() as client:
        fetch_articles("war", client=client)
        fetch_articles("earthquake", client=client)
    assert route.call_count == 2


@respx.mock
def test_fetch_json_without_client_still_works():
    # fetch_json's client param is optional — confirm the plain httpx.get
    # fallback path (used when no shared client is passed) is intact.
    respx.get("https://example.com/plain").mock(return_value=httpx.Response(200, json={"ok": 1}))
    assert fetch_json("https://example.com/plain") == {"ok": 1}
