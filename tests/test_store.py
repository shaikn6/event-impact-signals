from eis.models import Article, Direction, EventSignal, EventType, SectorImpact
from eis.store import SignalStore


def _signal(url: str, title: str = "War escalates") -> EventSignal:
    return EventSignal(
        article=Article(title=title, url=url, source="test"),
        event_type=EventType.WAR_CONFLICT,
        event_confidence=0.9,
        sector_impacts=[
            SectorImpact(sector="defense", direction=Direction.UP, confidence=0.8, rationale="r")
        ],
    )


def test_save_new_inserts_and_dedups(tmp_path):
    store = SignalStore(tmp_path / "test.db")
    inserted_first = store.save_new([_signal("https://a.com/1")])
    inserted_second = store.save_new([_signal("https://a.com/1")])  # same URL again
    assert inserted_first == 1
    assert inserted_second == 0
    store.close()


def test_recent_returns_saved_signals(tmp_path):
    store = SignalStore(tmp_path / "test.db")
    store.save_new([_signal("https://a.com/1"), _signal("https://a.com/2")])
    recent = store.recent(limit=10)
    assert len(recent) == 2
    assert {s.article.url for s in recent} == {"https://a.com/1", "https://a.com/2"}
    store.close()


def test_recent_round_trips_sector_impacts(tmp_path):
    store = SignalStore(tmp_path / "test.db")
    store.save_new([_signal("https://a.com/1")])
    [signal] = store.recent(limit=1)
    assert len(signal.sector_impacts) == 1
    assert signal.sector_impacts[0].sector == "defense"
    assert signal.sector_impacts[0].direction == Direction.UP
    store.close()


def test_recent_respects_limit(tmp_path):
    store = SignalStore(tmp_path / "test.db")
    store.save_new([_signal(f"https://a.com/{i}") for i in range(5)])
    assert len(store.recent(limit=2)) == 2
    store.close()


def test_creates_parent_directory(tmp_path):
    nested = tmp_path / "nested" / "dir" / "signals.db"
    store = SignalStore(nested)
    assert nested.parent.exists()
    store.close()
