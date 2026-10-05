from eis.classify.event_type import classify_event
from eis.models import EventType


def test_war_headline_classified_correctly():
    event_type, confidence = classify_event("Military strike escalates conflict near the border")
    assert event_type == EventType.WAR_CONFLICT
    assert confidence > 0


def test_natural_disaster_headline_classified_correctly():
    event_type, _ = classify_event("Hurricane makes landfall, thousands evacuated")
    assert event_type == EventType.NATURAL_DISASTER


def test_pandemic_headline_classified_correctly():
    event_type, _ = classify_event("WHO declares outbreak a public health emergency")
    assert event_type == EventType.PANDEMIC_PUBLIC_HEALTH


def test_unrelated_text_returns_general_with_zero_confidence():
    event_type, confidence = classify_event("Local bakery wins award for best croissant")
    assert event_type == EventType.GENERAL
    assert confidence == 0.0


def test_more_matching_phrases_increases_confidence():
    _, weak = classify_event("Troops deployed")
    _, strong = classify_event("Troops deployed as war escalates with airstrike and combat")
    assert strong > weak


def test_confidence_is_bounded_at_one():
    _, confidence = classify_event(
        "war invasion military strike airstrike ceasefire troops combat armed clash"
    )
    assert confidence <= 1.0


def test_tariff_vs_war_disambiguation():
    # "trade war" should not get misread as armed conflict just because
    # "war" appears as a substring.
    event_type, _ = classify_event("New trade war tariff announced on imports")
    assert event_type == EventType.TRADE_TARIFF


def test_merger_acquisition_headline_classified_correctly():
    event_type, confidence = classify_event("Tech giant agrees to acquire rival in buyout deal")
    assert event_type == EventType.MERGER_ACQUISITION
    assert confidence > 0


def test_supply_chain_disruption_headline_classified_correctly():
    event_type, confidence = classify_event("Automakers warn of chip shortage and factory shutdown")
    assert event_type == EventType.SUPPLY_CHAIN_DISRUPTION
    assert confidence > 0


def test_idiomatic_combat_phrase_does_not_trigger_war_conflict():
    # Regression test: found live — "combat" used idiomatically ("help
    # combat inflation") was misclassified as armed conflict before
    # "combat" was removed from WAR_CONFLICT's keyword list.
    event_type, confidence = classify_event("Checks announced to help combat rising costs")
    assert event_type == EventType.GENERAL
    assert confidence == 0.0


def test_merger_headline_with_plainer_buy_verb_classified_correctly():
    # Regression test: found live — real M&A headlines commonly use
    # plainer verbs ("X agrees to buy Y") that the original phrase list
    # ("to acquire", "agrees to merge", etc.) didn't cover.
    event_type, confidence = classify_event("Retail giant agrees to buy rival chain in cash deal")
    assert event_type == EventType.MERGER_ACQUISITION
    assert confidence > 0


def test_merger_in_talks_to_acquire_classified_correctly():
    event_type, _ = classify_event("Software firm in talks to acquire smaller competitor")
    assert event_type == EventType.MERGER_ACQUISITION


def test_figurative_flood_of_does_not_trigger_natural_disaster():
    event_type, _ = classify_event("Flood of new streaming shows overwhelms viewers")
    assert event_type == EventType.GENERAL


def test_figurative_drought_of_does_not_trigger_natural_disaster():
    event_type, _ = classify_event("Drought of talent hits tech hiring")
    assert event_type == EventType.GENERAL


def test_hacked_together_does_not_trigger_cyberattack():
    event_type, _ = classify_event("Hacked together prototype wins hackathon prize")
    assert event_type == EventType.GENERAL


def test_literal_flood_and_drought_still_classified():
    assert classify_event("Flood hits Texas as river rises")[0] == EventType.NATURAL_DISASTER
    assert classify_event("Severe drought threatens crops")[0] == EventType.NATURAL_DISASTER
    assert classify_event("Bank hacked, customer data stolen")[0] == EventType.CYBERATTACK
