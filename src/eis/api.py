"""FastAPI app serving the collected signals.

Read-only on purpose — this app never places trades or calls any
brokerage API. It's a signal/research surface, not an execution system.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query

from eis.store import DEFAULT_DB_PATH, SignalStore


def create_app(db_path: str | Path = DEFAULT_DB_PATH) -> FastAPI:
    """Build the API over the SQLite file at `db_path`.

    A factory rather than a module-level app so `eis serve --db-path` serves
    the same file its poller writes, and so importing this module never
    creates data/signals.db as a side effect.
    """
    app = FastAPI(
        title="event-impact-signals",
        description=(
            "Research tool: classifies real-world events from live news and maps them to "
            "sectors likely to gain or lose, with a documented rationale per impact. "
            "Not investment advice — see each signal's rationale and confidence before "
            "drawing any conclusion."
        ),
        version="0.1.0",
    )
    store = SignalStore(db_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/signals")
    def list_signals(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
        """Most recently collected signals, newest first."""
        return [s.to_dict() for s in store.recent(limit=limit)]

    return app
