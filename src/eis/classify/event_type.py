"""Rule-based event-type classifier.

Deterministic keyword matching, not a trained model — this mirrors the
same tradeoff nano-finbert's SignalExtractor makes: a transparent,
auditable rule set beats an opaque classifier when every output needs a
traceable "why" for a research tool. Each category lists the phrases that
trigger it; the category with the most/strongest matches wins. Confidence
is the match strength, not a calibrated probability.
"""

from __future__ import annotations

import re

from eis.models import EventType

# Keyword groups per event type. Every phrase is matched with \b word
# boundaries (see classify_event below) so e.g. "war" matches the word
# "war" but not a substring inside "award"; "trade war" still separately
# matches TRADE_TARIFF via its own phrase entry, and phrase-length
# weighting (see classify_event) lets it outscore WAR_CONFLICT's bare
# "war" when both match the same text.
#
# KNOWN, UNCONFIRMED RISK (documented, not fixed, per this project's own
# rule of only changing keywords for a bug actually seen live — see
# README "Known limitations"): a few single words below are plausibly
# polysemous in ways that could misfire the way "combat" did (removed
# after a real live false positive) — e.g. "war" in "price war"/"bidding
# war" (business-competition idiom, not conflict), or "sanctions" used
# as a verb meaning "approves" ("the board sanctions a buyback") rather
# than the punitive-measure noun. Neither has been observed firing
# incorrectly in this project's own live test runs so far (all real
# "war"/"sanctions" hits during live testing were genuinely about
# conflict/punitive sanctions) — if one ever does, fix it the same way
# "combat" was: either remove the word or add a scoped exclusion, backed
# by the real headline that triggered it.
# Words whose figurative use ("a flood of new shows", "a drought of
# talent", "hacked together a prototype") is far more common in headlines
# than the literal event. A phrase listed here only counts when NOT
# immediately followed by the given regex — scoped exclusions that keep
# the literal meaning ("flood hits Texas") while dropping the idiom.
_NOT_FOLLOWED_BY: dict[str, str] = {
    "flood": r"\s+of\b",
    "drought": r"\s+of\b",
    "hacked": r"\s+together\b",
}

_KEYWORDS: dict[EventType, list[str]] = {
    EventType.WAR_CONFLICT: [
        "war",
        "invasion",
        "military strike",
        "airstrike",
        "ceasefire",
        "troops",
        "conflict escalates",
        "armed clash",
        "missile attack",
        # NOT "combat" alone — caught live misclassifying "help combat
        # inflation"/"combat climate change" as armed conflict. Idiomatic
        # "combat X" (fight against a problem) is far more common in
        # headlines than literal military combat, so the bare word is a
        # false-positive magnet; the more specific phrases above already
        # catch genuine war coverage.
    ],
    EventType.NATURAL_DISASTER: [
        "earthquake",
        "hurricane",
        "wildfire",
        "flood",
        "tsunami",
        "typhoon",
        "tornado",
        "volcanic eruption",
        "drought",
        "landslide",
    ],
    EventType.PANDEMIC_PUBLIC_HEALTH: [
        "pandemic",
        "outbreak",
        "virus spreads",
        "epidemic",
        "vaccine",
        "who declares",
        "quarantine",
        "public health emergency",
    ],
    EventType.TRADE_TARIFF: [
        "tariff",
        "trade war",
        "import duty",
        "export ban",
        "trade deal",
        "trade dispute",
    ],
    EventType.REGULATORY_SANCTION: [
        "sanctions",
        "antitrust",
        "regulatory crackdown",
        "fined by regulators",
        "bans exports to",
        "blacklist",
    ],
    EventType.MONETARY_POLICY: [
        "rate hike",
        "rate cut",
        "federal reserve raises",
        "federal reserve cuts",
        "central bank raises",
        "interest rates rise",
        "interest rates fall",
        "fomc",
    ],
    EventType.ENERGY_SHOCK: [
        "oil price",
        "opec",
        "gas prices surge",
        "energy crisis",
        "oil supply disruption",
        "crude oil jumps",
    ],
    EventType.LABOR_STRIKE: [
        "workers strike",
        "labor strike",
        "union walkout",
        "walkout",
        "labor dispute",
    ],
    EventType.CYBERATTACK: [
        "cyberattack",
        "data breach",
        "ransomware",
        "hacked",
        "security breach",
    ],
    EventType.EARNINGS_CORPORATE: [
        "quarterly earnings",
        "reports earnings",
        "profit rose",
        "profit fell",
        "revenue beat",
        "revenue miss",
        "guidance cut",
    ],
    EventType.MERGER_ACQUISITION: [
        "to acquire",
        "acquisition of",
        "merger with",
        "agrees to merge",
        "takeover bid",
        "buyout deal",
        "to be acquired by",
        # Added after a live test found real M&A headlines using plainer
        # verbs ("X to buy Y") that none of the phrases above would
        # catch — "to buy" alone would be too generic (share buybacks,
        # routine purchases), so these stay anchored to deal language.
        "agrees to buy",
        "to be bought by",
        "in talks to acquire",
        "completes acquisition",
        "merger deal",
    ],
    EventType.SUPPLY_CHAIN_DISRUPTION: [
        "chip shortage",
        "semiconductor shortage",
        "supply chain disruption",
        "port congestion",
        "shipping delays",
        "factory shutdown",
        "parts shortage",
    ],
}


def classify_event(text: str) -> tuple[EventType, float]:
    """Classify free text into one EventType with a match-strength score.

    Args:
        text: Article title (+ optional summary), any case.

    Returns:
        (event_type, confidence) — confidence in [0, 1], 0 when nothing matched
        (EventType.GENERAL is returned in that case).
    """
    lowered = text.lower()

    scores: dict[EventType, int] = {}
    for event_type, phrases in _KEYWORDS.items():
        # \b...\b so e.g. "war" matches the word "war" but not the
        # substring inside "award" or "warranty". Score by phrase word
        # count, not a flat +1 per match: "trade war" (2 words) should
        # outweigh the single word "war" it contains, so a real trade-war
        # headline isn't misclassified as armed conflict just because
        # "war" is also, technically, a substring match for that category.
        score = sum(
            len(phrase.split())
            for phrase in phrases
            if re.search(
                rf"\b{re.escape(phrase)}\b(?!{_NOT_FOLLOWED_BY.get(phrase, '(?!)')})", lowered
            )
        )
        if score:
            scores[event_type] = score

    if not scores:
        return EventType.GENERAL, 0.0

    # Tie-break policy (explicit, not an accident of dict order): when two
    # categories score equally, prefer whichever is listed first in
    # _KEYWORDS above. This matters for genuinely ambiguous text (e.g. a
    # single-word hit landing in two categories at once); reordering
    # _KEYWORDS changes tie-break outcomes, so treat its order as policy.
    best_type = max(scores, key=lambda k: scores[k])
    # Confidence saturates around 3 matched "phrase-words" — a single
    # keyword hit is a weak signal, three independent words of evidence
    # for the same category is strong.
    confidence = min(1.0, scores[best_type] / 3.0)
    return best_type, confidence
