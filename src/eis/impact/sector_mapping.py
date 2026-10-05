"""Event type -> sector impact mapping.

This is the core hypothesis this project surfaces: "one event, many
sectors, not all moving the same direction." Every entry below is a
documented, falsifiable rule (not a trained model) — e.g. "war raises
defense spending and oil-supply risk, so defense and energy tend up while
airlines and tourism, which are exposed to route disruption and safety
fears, tend down."

These are directional *hypotheses* based on historical patterns, not
guarantees — a given real event can break any of them (e.g. a short,
contained conflict with no supply disruption). Confidence reflects how
reliably the pattern has held historically, not a probability of a
specific trade outcome. This file is deliberately data, not code, so the
mapping can be read, debated, and extended without touching any logic.
"""

from __future__ import annotations

from eis.models import Direction, EventType, SectorImpact

# event_type -> list of (sector, direction, confidence, rationale)
_IMPACT_TABLE: dict[EventType, list[tuple[str, Direction, float, str]]] = {
    EventType.WAR_CONFLICT: [
        ("defense", Direction.UP, 0.8, "Military spending rises during active conflicts."),
        ("energy_oil_gas", Direction.UP, 0.7, "Supply-disruption risk pushes oil/gas prices up."),
        ("gold_safe_haven", Direction.UP, 0.6, "Flight-to-safety demand rises during conflict."),
        ("airlines", Direction.DOWN, 0.7, "Route disruption and fuel-cost spikes hurt margins."),
        (
            "tourism_hospitality",
            Direction.DOWN,
            0.6,
            "Safety concerns reduce travel to the region.",
        ),
        (
            "shipping_freight",
            Direction.MIXED,
            0.5,
            "Rerouting raises rates but adds cost and risk.",
        ),
    ],
    EventType.NATURAL_DISASTER: [
        (
            "insurance_reinsurance",
            Direction.DOWN,
            0.75,
            "Claims payouts rise sharply after disasters.",
        ),
        (
            "construction_materials",
            Direction.UP,
            0.7,
            "Rebuilding drives demand for materials and labor.",
        ),
        ("home_improvement_retail", Direction.UP, 0.6, "Repair and rebuilding purchases increase."),
        ("utilities", Direction.DOWN, 0.5, "Infrastructure damage raises near-term repair costs."),
        (
            "agriculture_commodities",
            Direction.MIXED,
            0.5,
            "Crop damage cuts supply but can raise prices.",
        ),
    ],
    EventType.PANDEMIC_PUBLIC_HEALTH: [
        ("pharma_biotech", Direction.UP, 0.75, "Demand for treatments/vaccines rises."),
        (
            "healthcare_services",
            Direction.UP,
            0.6,
            "Higher utilization of health services and payers.",
        ),
        (
            "remote_work_software",
            Direction.UP,
            0.6,
            "Remote collaboration tools see demand spikes.",
        ),
        (
            "ecommerce_logistics",
            Direction.UP,
            0.55,
            "Delivery demand substitutes for in-person retail.",
        ),
        (
            "airlines",
            Direction.DOWN,
            0.75,
            "Travel demand drops sharply during health emergencies.",
        ),
        ("travel_hospitality", Direction.DOWN, 0.75, "Lodging and events are directly curtailed."),
        ("brick_and_mortar_retail", Direction.DOWN, 0.5, "Foot traffic falls during outbreaks."),
    ],
    EventType.TRADE_TARIFF: [
        ("domestic_manufacturing", Direction.UP, 0.55, "Tariffs reduce competition from imports."),
        (
            "import_dependent_retail",
            Direction.DOWN,
            0.6,
            "Higher input/import costs compress margins.",
        ),
        (
            "agriculture_exports",
            Direction.DOWN,
            0.55,
            "Retaliatory tariffs often target farm exports.",
        ),
    ],
    EventType.REGULATORY_SANCTION: [
        (
            "targeted_sector_or_firm",
            Direction.DOWN,
            0.7,
            "Sanctioned/fined entities face direct costs.",
        ),
        ("domestic_competitors", Direction.UP, 0.5, "Less competition from the sanctioned entity."),
        (
            "compliance_legal_services",
            Direction.UP,
            0.45,
            "Demand for compliance/legal advisory rises.",
        ),
    ],
    EventType.MONETARY_POLICY: [
        (
            "banks",
            Direction.MIXED,
            0.5,
            "Rate hikes help net interest margin but can slow loan demand.",
        ),
        (
            "real_estate_reits",
            Direction.DOWN,
            0.6,
            "Higher rates raise financing costs, pressure valuations.",
        ),
        ("growth_tech", Direction.DOWN, 0.55, "Future cash flows are discounted more heavily."),
        ("bonds_fixed_income", Direction.DOWN, 0.6, "Existing bond prices fall as yields rise."),
    ],
    EventType.ENERGY_SHOCK: [
        ("energy_oil_gas", Direction.UP, 0.75, "Direct beneficiary of higher energy prices."),
        ("airlines", Direction.DOWN, 0.65, "Fuel is a major airline cost input."),
        ("trucking_logistics", Direction.DOWN, 0.6, "Fuel costs compress logistics margins."),
        (
            "renewable_energy",
            Direction.UP,
            0.45,
            "Higher fossil fuel prices improve relative economics.",
        ),
        ("chemicals_feedstock", Direction.DOWN, 0.5, "Oil/gas-derived feedstock costs rise."),
    ],
    EventType.LABOR_STRIKE: [
        (
            "affected_company",
            Direction.DOWN,
            0.65,
            "Production/service disruption during the strike.",
        ),
        (
            "direct_competitors",
            Direction.UP,
            0.4,
            "Customers may shift to competitors during disruption.",
        ),
    ],
    EventType.CYBERATTACK: [
        (
            "cybersecurity",
            Direction.UP,
            0.55,
            "Incidents raise demand for security spending broadly.",
        ),
        (
            "affected_company",
            Direction.DOWN,
            0.6,
            "Breach costs, liability, and reputational damage.",
        ),
    ],
    EventType.MERGER_ACQUISITION: [
        (
            "acquisition_target",
            Direction.UP,
            0.8,
            "Acquirers typically pay a premium over the pre-deal share price.",
        ),
        (
            "acquiring_company",
            Direction.MIXED,
            0.5,
            "Can rise on strategic fit or fall on deal-cost/integration risk.",
        ),
        (
            "investment_banking_legal",
            Direction.UP,
            0.55,
            "Advisory, legal, and due-diligence fees scale with deal volume.",
        ),
        (
            "direct_competitors",
            Direction.MIXED,
            0.4,
            "Can benefit from reduced competition or lose ground to a stronger combined rival.",
        ),
    ],
    EventType.SUPPLY_CHAIN_DISRUPTION: [
        (
            "shipping_logistics",
            Direction.UP,
            0.55,
            "Freight rates rise when capacity is constrained or rerouted.",
        ),
        (
            "affected_manufacturers",
            Direction.DOWN,
            0.7,
            "Production is directly constrained by missing parts or components.",
        ),
        (
            "domestic_reshoring_plays",
            Direction.UP,
            0.45,
            "Disruption renews interest in on-shore/near-shore manufacturing capacity.",
        ),
        (
            "inventory_warehousing",
            Direction.UP,
            0.4,
            "Firms build safety stock, raising demand for warehousing capacity.",
        ),
    ],
}


def get_sector_impacts(event_type: EventType) -> list[SectorImpact]:
    """Look up the documented sector-impact hypotheses for an event type.

    Returns an empty list for EventType.GENERAL or EARNINGS_CORPORATE —
    those are either too broad or already company-specific (a single
    earnings report affects one company, not a sector pattern worth
    tabulating here).
    """
    return [
        SectorImpact(sector=sector, direction=direction, confidence=confidence, rationale=rationale)
        for sector, direction, confidence, rationale in _IMPACT_TABLE.get(event_type, [])
    ]
