import httpx
import respx

from eis.ingest.reddit import fetch_subreddit


@respx.mock
def test_fetch_subreddit_parses_valid_response():
    respx.get("https://www.reddit.com/r/worldnews/top.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "War escalates",
                                "url": "https://example.com/a",
                                "permalink": "/r/worldnews/comments/1/x",
                                "created_utc": 1700000000.0,
                            }
                        }
                    ]
                }
            },
        )
    )
    articles = fetch_subreddit("worldnews")
    assert len(articles) == 1
    assert articles[0].title == "War escalates"
    assert articles[0].source == "reddit:worldnews"


@respx.mock
def test_fetch_subreddit_returns_empty_list_on_403():
    # The real documented failure mode: Reddit blocking unauthenticated requests.
    respx.get("https://www.reddit.com/r/worldnews/top.json").mock(
        return_value=httpx.Response(403, text="blocked")
    )
    assert fetch_subreddit("worldnews") == []


@respx.mock
def test_fetch_subreddit_falls_back_to_permalink_when_url_missing():
    respx.get("https://www.reddit.com/r/worldnews/top.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "children": [{"data": {"title": "x", "permalink": "/r/worldnews/comments/1/x"}}]
                }
            },
        )
    )
    articles = fetch_subreddit("worldnews")
    assert articles[0].url == "https://reddit.com/r/worldnews/comments/1/x"
