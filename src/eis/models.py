"""Shared data types passed between the ingest -> classify -> impact stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Categories of real-world events the classifier recognizes.

    Kept as a closed, explicit set (rather than free-text) so the sector
    impact table in impact/sector_mapping.py can be a straightforward
    lookup instead of fuzzy matching on arbitrary strings.
    """

    WAR_CONFLICT = "war_conflict"
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC_PUBLIC_HEALTH = "pandemic_public_health"
    TRADE_TARIFF = "trade_tariff"
    REGULATORY_SANCTION = "regulatory_sanction"
    MONETARY_POLICY = "monetary_policy"
    ENERGY_SHOCK = "energy_shock"
    LABOR_STRIKE = "labor_strike"
    CYBERATTACK = "cyberattack"
    EARNINGS_CORPORATE = "earnings_corporate"
    MERGER_ACQUISITION = "merger_acquisition"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    GENERAL = "general"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    MIXED = "mixed"  # depends on sub-sector or exposure, flagged rather than guessed


@dataclass(frozen=True)
class Article:
    """A single ingested item (news article, RSS entry, or Reddit post)."""

    title: str
    url: str
    source: str  # "gdelt" | "rss:<feed_name>" | "reddit:<subreddit>"
    published_at: datetime | None = None
    country: str | None = None
    summary: str = ""


@dataclass(frozen=True)
class SectorImpact:
    """One sector's predicted exposure to a classified event."""

    sector: str
    direction: Direction
    confidence: float  # 0.0-1.0, rule-strength not a calibrated probability
    rationale: str

    def to_dict(self) -> dict:
        return {
            "sector": self.sector,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EventSignal:
    """The end-to-end output: one article, classified, with its sector impacts."""

    article: Article
    event_type: EventType
    event_confidence: float
    sector_impacts: list[SectorImpact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.article.title,
            "url": self.article.url,
            "source": self.article.source,
            "country": self.article.country,
            "published_at": self.article.published_at.isoformat()
            if self.article.published_at
            else None,
            "event_type": self.event_type.value,
            "event_confidence": round(self.event_confidence, 2),
            "sector_impacts": [si.to_dict() for si in self.sector_impacts],
        }
