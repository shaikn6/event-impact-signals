from eis.impact.sector_mapping import get_sector_impacts
from eis.models import Direction, EventType


def test_war_conflict_has_impacts():
    impacts = get_sector_impacts(EventType.WAR_CONFLICT)
    assert len(impacts) > 0
    assert all(0.0 <= si.confidence <= 1.0 for si in impacts)
    assert all(si.rationale for si in impacts)


def test_war_conflict_defense_up_airlines_down():
    impacts = {si.sector: si.direction for si in get_sector_impacts(EventType.WAR_CONFLICT)}
    assert impacts["defense"] == Direction.UP
    assert impacts["airlines"] == Direction.DOWN


def test_pandemic_pharma_up_travel_down():
    impacts = {
        si.sector: si.direction for si in get_sector_impacts(EventType.PANDEMIC_PUBLIC_HEALTH)
    }
    assert impacts["pharma_biotech"] == Direction.UP
    assert impacts["airlines"] == Direction.DOWN


def test_general_event_type_has_no_impacts():
    assert get_sector_impacts(EventType.GENERAL) == []


def test_every_event_type_except_general_and_earnings_has_impacts():
    for event_type in EventType:
        if event_type in (EventType.GENERAL, EventType.EARNINGS_CORPORATE):
            continue
        assert get_sector_impacts(event_type), f"{event_type} has no documented impacts"
