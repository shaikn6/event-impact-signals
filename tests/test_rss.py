import feedparser

from eis.ingest.rss import fetch_all, fetch_feed

_SAMPLE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Test Feed</title>
<item>
  <title>Hurricane makes landfall</title>
  <link>https://example.com/hurricane</link>
  <description>Summary text</description>
  <pubDate>Fri, 02 Oct 2026 10:45:00 GMT</pubDate>
</item>
</channel></rss>"""


def test_fetch_feed_parses_entries(monkeypatch):
    original_parse = feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda *a, **k: original_parse(_SAMPLE_RSS))
    articles = fetch_feed("test_feed", "https://unused.example.com/rss")
    assert len(articles) == 1
    assert articles[0].title == "Hurricane makes landfall"
    assert articles[0].source == "rss:test_feed"
    assert articles[0].published_at is not None


def test_fetch_feed_returns_empty_list_for_malformed_xml(monkeypatch):
    original_parse = feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda *a, **k: original_parse("not xml at all"))
    articles = fetch_feed("broken", "https://unused.example.com/rss")
    assert articles == []


def test_fetch_all_flattens_multiple_feeds(monkeypatch):
    original_parse = feedparser.parse
    monkeypatch.setattr(feedparser, "parse", lambda *a, **k: original_parse(_SAMPLE_RSS))
    articles = fetch_all({"feed_a": "https://a.example.com", "feed_b": "https://b.example.com"})
    assert len(articles) == 2
    assert {a.source for a in articles} == {"rss:feed_a", "rss:feed_b"}
