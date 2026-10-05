"""Lightweight SQLite persistence for live-collected signals.

WHY SQLite, not a database server: this runs as a single scheduled
process polling a handful of free APIs every few minutes — a file-based
store is the simplest thing that works and needs no extra infrastructure.
Dedup is by article URL (the natural primary key for "have we seen this
article already").
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from eis.models import Article, Direction, EventSignal, EventType, SectorImpact

DEFAULT_DB_PATH = Path("data/signals.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    url TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    country TEXT,
    published_at TEXT,
    event_type TEXT NOT NULL,
    event_confidence REAL NOT NULL,
    sector_impacts_json TEXT NOT NULL,
    collected_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SignalStore:
    """Dedup-on-insert store for EventSignal objects, keyed by article URL."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI's TestClient (and uvicorn in
        # general) can call into this store from a different thread than
        # the one that created it; all access here is still effectively
        # single-threaded per request, so this is safe.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def save_new(self, signals: list[EventSignal]) -> int:
        """Insert signals not already present (by URL). Returns count inserted."""
        inserted = 0
        for signal in signals:
            row = (
                signal.article.url,
                signal.article.title,
                signal.article.source,
                signal.article.country,
                signal.article.published_at.isoformat() if signal.article.published_at else None,
                signal.event_type.value,
                signal.event_confidence,
                json.dumps([si.to_dict() for si in signal.sector_impacts]),
            )
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO signals "
                "(url, title, source, country, published_at, event_type, "
                " event_confidence, sector_impacts_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            inserted += cursor.rowcount
        self._conn.commit()
        return inserted

    def recent(self, limit: int = 50) -> list[EventSignal]:
        """Return the most recently collected signals, newest first.

        Orders by rowid (SQLite's implicit, indexed, monotonically
        increasing row identifier) rather than the unindexed
        `collected_at` column — rows are inserted in collection order, so
        rowid DESC gives the same result without forcing a full-table sort
        on every call to this (frequently hit) method.
        """
        cursor = self._conn.execute(
            "SELECT url, title, source, country, published_at, event_type, "
            "event_confidence, sector_impacts_json FROM signals "
            "ORDER BY rowid DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_signal(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()


def _row_to_signal(row: tuple) -> EventSignal:
    url, title, source, country, published_at, event_type, event_confidence, impacts_json = row
    return EventSignal(
        article=Article(
            title=title,
            url=url,
            source=source,
            country=country,
            published_at=datetime.fromisoformat(published_at) if published_at else None,
        ),
        event_type=EventType(event_type),
        event_confidence=event_confidence,
        sector_impacts=[
            SectorImpact(
                sector=d["sector"],
                direction=Direction(d["direction"]),
                confidence=d["confidence"],
                rationale=d["rationale"],
            )
            for d in json.loads(impacts_json)
        ],
    )
