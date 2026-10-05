from eis.models import Article, EventType
from eis.pipeline import build_signal, build_signals, filter_actionable


def _article(title: str, summary: str = "") -> Article:
    return Article(
        title=title, url=f"https://example.com/{hash(title)}", source="test", summary=summary
    )


def test_build_signal_attaches_sector_impacts_for_strong_match():
    article = _article(
        "War escalates as military strike hits region, airstrike and combat reported"
    )
    signal = build_signal(article)
    assert signal.event_type == EventType.WAR_CONFLICT
    assert len(signal.sector_impacts) > 0


def test_build_signal_attaches_sector_impacts_for_single_keyword_match():
    # A single specific-phrase hit ("hurricane") is still a meaningful
    # signal — see pipeline.build_signal's comment on why there's no
    # separate "weak match" threshold above confidence > 0.
    article = _article("Hurricane makes landfall")
    signal = build_signal(article)
    assert signal.event_type == EventType.NATURAL_DISASTER
    assert len(signal.sector_impacts) > 0


def test_build_signal_skips_sector_impacts_for_unrelated_text():
    article = _article("Local bakery wins award for best croissant")
    signal = build_signal(article)
    assert signal.event_type == EventType.GENERAL
    assert signal.sector_impacts == []


def test_build_signals_preserves_order_and_count():
    articles = [_article("Hurricane makes landfall"), _article("Local bakery wins award")]
    signals = build_signals(articles)
    assert len(signals) == 2
    assert signals[0].article.title == articles[0].title


def test_filter_actionable_drops_signals_without_impacts():
    articles = [
        _article("War escalates as military strike hits region, airstrike and combat reported"),
        _article("Local bakery wins award for best croissant"),
    ]
    signals = build_signals(articles)
    actionable = filter_actionable(signals)
    assert len(actionable) == 1
    assert actionable[0].event_type == EventType.WAR_CONFLICT


def test_to_dict_round_trips_core_fields():
    article = _article("Earthquake hits coastal region, tsunami warning issued")
    signal = build_signal(article)
    d = signal.to_dict()
    assert d["title"] == article.title
    assert d["event_type"] == EventType.NATURAL_DISASTER.value
    assert isinstance(d["sector_impacts"], list)
