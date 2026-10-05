import pytest
from fastapi.testclient import TestClient

from eis import api
from eis.models import Article, Direction, EventSignal, EventType, SectorImpact
from eis.store import SignalStore


@pytest.fixture
def client(tmp_path, monkeypatch):
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
    monkeypatch.setattr(api, "_store", store)
    return TestClient(api.app)


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
