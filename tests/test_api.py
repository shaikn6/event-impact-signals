import sys

import pytest
import uvicorn
from fastapi.testclient import TestClient

from eis import api, cli
from eis.models import Article, Direction, EventSignal, EventType, SectorImpact
from eis.store import SignalStore


@pytest.fixture
def client(tmp_path):
    store = SignalStore(tmp_path / "test.db")
    store.save_new(
        [
            EventSignal(
                article=Article(title="War escalates", url="https://a.com/1", source="test"),
                event_type=EventType.WAR_CONFLICT,
                event_confidence=0.9,
                sector_impacts=[
                    SectorImpact(
                        sector="defense", direction=Direction.UP, confidence=0.8, rationale="r"
                    )
                ],
            )
        ]
    )
    return TestClient(api.create_app(tmp_path / "test.db"))


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signals_endpoint_returns_saved_signal(client):
    response = client.get("/signals")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["title"] == "War escalates"
    assert body[0]["sector_impacts"][0]["sector"] == "defense"


def test_signals_endpoint_respects_limit_param(client):
    response = client.get("/signals?limit=1")
    assert response.status_code == 200
    assert len(response.json()) <= 1


def test_signals_endpoint_rejects_invalid_limit(client):
    response = client.get("/signals?limit=0")
    assert response.status_code == 422


def test_serve_db_path_reaches_the_api(tmp_path, monkeypatch):
    # Regression: `eis serve --db-path X` used to start the poller on X while
    # the API kept reading data/signals.db, so /signals returned [].
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "custom.db"
    store = SignalStore(db_path)
    store.save_new(
        [
            EventSignal(
                article=Article(title="Custom DB signal", url="https://a.com/2", source="test"),
                event_type=EventType.WAR_CONFLICT,
                event_confidence=0.9,
                sector_impacts=[],
            )
        ]
    )
    store.close()

    served = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: served.update(app=app))
    monkeypatch.setattr(sys, "argv", ["eis", "serve", "--no-live", "--db-path", str(db_path)])
    cli.main()

    body = TestClient(served["app"]).get("/signals").json()
    assert [s["title"] for s in body] == ["Custom DB signal"]
