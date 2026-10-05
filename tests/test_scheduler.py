from unittest.mock import patch

from eis.models import Article
from eis.scheduler import poll_once
from eis.store import SignalStore

# Every test here patches this out: poll_once sleeps between GDELT queries
# to respect its rate limit, which would otherwise add ~20s per test run.
_sleep_patch = patch("eis.scheduler.time.sleep")


@_sleep_patch
@patch("eis.ingest.rss.fetch_all", return_value=[])
@patch("eis.ingest.gdelt.fetch_articles")
def test_poll_once_saves_actionable_signals(mock_gdelt, mock_rss, mock_sleep, tmp_path):
    mock_gdelt.return_value = [
        Article(
            title="War escalates with airstrike and combat", url="https://a.com/1", source="gdelt"
        )
    ]
    store = SignalStore(tmp_path / "test.db")
    inserted = poll_once(store)
    assert inserted == 1
    assert len(store.recent()) == 1
    store.close()


@_sleep_patch
@patch("eis.ingest.rss.fetch_all", return_value=[])
@patch("eis.ingest.gdelt.fetch_articles", return_value=[])
def test_poll_once_with_no_articles_inserts_nothing(mock_gdelt, mock_rss, mock_sleep, tmp_path):
    store = SignalStore(tmp_path / "test.db")
    assert poll_once(store) == 0
    store.close()


@_sleep_patch
@patch("eis.ingest.reddit.fetch_subreddit", return_value=[])
@patch("eis.ingest.rss.fetch_all", return_value=[])
@patch("eis.ingest.gdelt.fetch_articles", return_value=[])
def test_poll_once_include_reddit_calls_reddit_fetch(
    mock_gdelt, mock_rss, mock_reddit, mock_sleep, tmp_path
):
    store = SignalStore(tmp_path / "test.db")
    poll_once(store, include_reddit=True)
    assert mock_reddit.called
    store.close()


@_sleep_patch
@patch("eis.ingest.rss.fetch_all", return_value=[])
@patch("eis.ingest.gdelt.fetch_articles", return_value=[])
def test_poll_once_does_not_call_reddit_by_default(mock_gdelt, mock_rss, mock_sleep, tmp_path):
    with patch("eis.ingest.reddit.fetch_subreddit") as mock_reddit:
        store = SignalStore(tmp_path / "test.db")
        poll_once(store)
        assert not mock_reddit.called
        store.close()
