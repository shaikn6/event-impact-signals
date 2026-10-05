"""FastAPI app serving the collected signals.

Read-only on purpose — this app never places trades or calls any
brokerage API. It's a signal/research surface, not an execution system.
"""

from __future__ import annotations

from fastapi import FastAPI, Query

from eis.store import SignalStore

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

_store = SignalStore()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/signals")
def list_signals(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    """Most recently collected signals, newest first."""
    return [s.to_dict() for s in _store.recent(limit=limit)]
